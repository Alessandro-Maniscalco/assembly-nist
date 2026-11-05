"""Build optional helper-link SDFs from the untouched source STL surfaces.

Grids 301+ are independent of production grids. Two threads and bounded query
chunks keep this investigation from starving the ongoing native learner.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import time
import xml.etree.ElementTree as ET

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'assets_local/so101_helper_collision'
SOURCE = ROOT / 'assets_local/so101/so101_new_calib.xml'
JAWS = {'wrist_roll_follower_so101_v1', 'moving_jaw_so101_v1'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def source_names():
    root = ET.parse(SOURCE).getroot()
    names = {g.get('mesh') for g in root.findall('.//geom[@class="collision"]')}
    # Production also restores the four omitted static-base obstacles.
    names |= {g.get('mesh') for g in root.findall('worldbody/body/geom')}
    return sorted(names - JAWS)


def limits():
    rss = psutil.Process().memory_info().rss
    available = psutil.virtual_memory().available
    if rss > 3.5 * 2**30 or available < 2 * 2**30:
        raise MemoryError(f'Investigation resource limit: RSS={rss}, available={available}')
    return rss


def make_scene(mesh):
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene(nthreads=2)
    scene.add_triangles(o3d.core.Tensor(mesh.vertices.astype(np.float32)),
                        o3d.core.Tensor(mesh.faces.astype(np.uint32)))
    return scene


def source_geometry(name):
    import open3d as o3d
    import trimesh
    path = SOURCE.parent / 'assets' / f'{name}.stl'
    mesh = trimesh.load_mesh(path)
    components = mesh.split(only_watertight=False)
    if not mesh.is_winding_consistent or not all(c.is_watertight for c in components):
        raise ValueError(f'{name}: source has an unqualified open or inconsistent solid')
    solids = [c for c in components if c.volume > 1e-15]
    sheets = [c for c in components if abs(c.volume) <= 1e-15]
    if len(solids) != 1 or any(c.volume < -1e-15 for c in components):
        raise ValueError(f'{name}: overlapping/nested solids require an explicit sign audit')
    surface_scene = make_scene(mesh)
    solid_scene = make_scene(solids[0]) if sheets else surface_scene
    sheet_extent = 0.
    if sheets:
        points = np.vstack([np.vstack([c.vertices, c.triangles_center]) for c in sheets])
        sheet_extent = float(solid_scene.compute_distance(
            o3d.core.Tensor(points.astype(np.float32)), nthreads=2).numpy().max())
        if sum(len(c.faces) for c in sheets) != 2 or sheet_extent > .00006:
            raise ValueError(f'{name}: unexpected zero-volume surface, audit before use')
    info = dict(source=str(path.relative_to(ROOT)), source_sha256=sha256(path),
                vertices=len(mesh.vertices), triangles=len(mesh.faces),
                source_watertight=bool(mesh.is_watertight), winding_consistent=True,
                closed_solid_components=1, zero_volume_sheet_triangles=sum(len(c.faces) for c in sheets),
                zero_volume_sheet_sampled_extent_m=sheet_extent,
                sign_method='closed positive-volume component ray parity, three rays',
                distance_method='all original triangles, including zero-volume sheet')
    return mesh, surface_scene, solid_scene, bool(sheets), info


def distances(points, surface_scene, solid_scene, has_sheet):
    import open3d as o3d
    tensor = o3d.core.Tensor(np.asarray(points, np.float32))
    if not has_sheet:
        return surface_scene.compute_signed_distance(tensor, nthreads=2, nsamples=3).numpy()
    distance = surface_scene.compute_distance(tensor, nthreads=2).numpy()
    inside = solid_scene.compute_occupancy(tensor, nthreads=2, nsamples=3).numpy() > .5
    return distance * np.where(inside, -1., 1.)


def interpolate(grid, coordinates):
    base = np.floor(coordinates).astype(int)
    frac = coordinates - base
    result = np.zeros(len(coordinates))
    for z in (0, 1):
        for y in (0, 1):
            for x in (0, 1):
                xyz = base + [x, y, z]
                weight = np.prod(np.where(np.array([x, y, z]), frac, 1-frac), axis=1)
                result += weight * grid[xyz[:, 2], xyz[:, 1], xyz[:, 0]].astype(float)
    return result


def build(name, grid_id, spacing):
    started = time.monotonic()
    mesh, surface_scene, solid_scene, has_sheet, row = source_geometry(name)
    origin = np.floor((mesh.bounds[0] - 4*spacing) / spacing)*spacing
    shape = np.ceil((mesh.bounds[1] + 4*spacing - origin) / spacing).astype(int)+1
    target = ROOT / 'assets_local/sdf' / f'grid_{grid_id}.bin'
    if target.exists():
        raise FileExistsError(f'Refusing to replace any existing grid: {target}')
    temporary = DIRECTORY / f'grid_{grid_id}.incomplete'
    peak_rss = limits()
    plane = int(shape[0]*shape[1])
    max_quantization = 0.
    progress = time.monotonic()
    with temporary.open('wb') as stream:
        stream.write(struct.pack('<3i4d', *shape, *origin, spacing))
        total = int(np.prod(shape))
        for first in range(0, total, 100000):
            indices = np.arange(first, min(first+100000, total))
            xyz = np.column_stack([indices % shape[0], indices // shape[0] % shape[1], indices // plane])
            points = origin + xyz * spacing
            distance = distances(points, surface_scene, solid_scene, has_sheet)
            if not np.isfinite(distance).all():
                raise ValueError(f'{name}: nonfinite distance')
            quantized = distance.astype('<f2')
            max_quantization = max(max_quantization, float(np.max(np.abs(quantized.astype(float)-distance))))
            stream.write(quantized.tobytes())
            peak_rss = max(peak_rss, limits())
            if time.monotonic()-progress > 30:
                print(json.dumps(dict(mesh=name, fraction=(first+len(indices))/total,
                                      elapsed_s=time.monotonic()-started, rss_bytes=peak_rss)), flush=True)
                progress = time.monotonic()
    os.rename(temporary, target)
    row.update(sdf=dict(id=grid_id, spacing=spacing, origin=origin.tolist(), shape=shape.tolist()),
               grid=str(target.relative_to(ROOT)), grid_sha256=sha256(target),
               grid_bytes=target.stat().st_size, generation_seconds=time.monotonic()-started,
               peak_rss_bytes=peak_rss, sampled_grid_quantization_max_m=max_quantization,
               interpolation_bound_m=float(np.sqrt(3)*spacing/2 + max_quantization),
               bound_scope='Interior grid cells: 1-Lipschitz signed-distance interpolation plus observed node quantization; excludes sign/source defects and grid-box extrapolation.')
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spacing', type=float, default=.00015)
    parser.add_argument('--names', nargs='+', choices=source_names())
    args = parser.parse_args()
    if args.spacing <= 0:
        raise ValueError('Positive grid spacing required')
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = DIRECTORY / 'manifest.json'
    manifest = json.loads(path.read_text()) if path.exists() else dict(
        backend='grid_sdf', candidate_only=True, source_xml_sha256=sha256(SOURCE), meshes={})
    for grid_id, name in enumerate(source_names(), 301):
        if args.names and name not in args.names:
            continue
        if name in manifest['meshes']:
            record = manifest['meshes'][name]
            if record['sdf']['spacing'] != args.spacing or sha256(ROOT/record['grid']) != record['grid_sha256']:
                raise ValueError(f'{name}: existing candidate grid differs; do not overwrite')
            continue
        row = build(name, grid_id, args.spacing)
        manifest['meshes'][name] = row
        manifest['complete'] = set(manifest['meshes']) == set(source_names())
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(manifest, indent=2)+'\n')
        os.replace(temporary, path)
        print(json.dumps(dict(mesh=name, **row)), flush=True)


if __name__ == '__main__':
    main()
