"""Build cavity-preserving jaw SDFs from untouched, watertight helper STL."""
import hashlib
import json
from pathlib import Path
import struct
import time

import numpy as np
import open3d as o3d
import trimesh


ROOT = Path(__file__).resolve().parents[1]
SPACING = .0002
NAMES = ('wrist_roll_follower_so101_v1', 'moving_jaw_so101_v1')


def interpolate(grid, coordinates):
    base = np.floor(coordinates).astype(int)
    frac = coordinates-base
    result = np.zeros(len(coordinates))
    for z in (0, 1):
        for y in (0, 1):
            for x in (0, 1):
                xyz = base+[x, y, z]
                weights = np.prod(np.where(np.array([x, y, z]), frac, 1-frac), axis=1)
                result += weights*grid[xyz[:, 2], xyz[:, 1], xyz[:, 0]].astype(float)
    return result


def main():
    directory = ROOT/'assets_local/so101_collision'
    directory.mkdir(exist_ok=True)
    grid_dir = ROOT/'assets_local/sdf'
    grid_dir.mkdir(exist_ok=True)
    manifest = dict(backend='grid_sdf', meshes={})
    for grid_id, name in enumerate(NAMES, 201):
        started = time.monotonic()
        path = ROOT/'assets_local/so101/assets'/f'{name}.stl'
        mesh = trimesh.load_mesh(path)
        if not mesh.is_watertight or not mesh.is_winding_consistent:
            raise ValueError(f'{name}: expected closed source mesh')
        vertices = mesh.vertices.astype(np.float32)
        scene = o3d.t.geometry.RaycastingScene(nthreads=2)
        scene.add_triangles(o3d.core.Tensor(vertices), o3d.core.Tensor(mesh.faces.astype(np.uint32)))
        origin = np.floor((vertices.min(0)-4*SPACING)/SPACING)*SPACING
        shape = np.ceil((vertices.max(0)+4*SPACING-origin)/SPACING).astype(int)+1
        x, y = [origin[k]+np.arange(shape[k])*SPACING for k in range(2)]
        xy = np.stack(np.meshgrid(x, y, indexing='xy'), -1).reshape(-1, 2)
        target = grid_dir/f'grid_{grid_id}.bin'
        with target.open('wb') as stream:
            stream.write(struct.pack('<3i4d', *shape, *origin, SPACING))
            for z in range(shape[2]):
                points = np.column_stack([xy, np.full(len(xy), origin[2]+z*SPACING)]).astype(np.float32)
                distances = scene.compute_signed_distance(o3d.core.Tensor(points), nsamples=3).numpy()
                if not np.isfinite(distances).all():
                    raise ValueError(f'{name}: nonfinite signed distance')
                stream.write(distances.astype('<f2').tobytes())
        grid = np.memmap(target, dtype='<f2', mode='r', offset=44, shape=tuple(shape[::-1]))
        rng = np.random.default_rng(0)
        sample, _ = trimesh.sample.sample_surface(mesh, 10000, seed=0)
        samples = np.vstack([sample, rng.uniform(vertices.min(0), vertices.max(0), (10000, 3))]).astype(np.float32)
        reference = scene.compute_signed_distance(o3d.core.Tensor(samples), nsamples=3).numpy()
        interpolated = interpolate(grid, (samples-origin)/SPACING)
        error = np.abs(reference-interpolated)
        critical = samples[:, 2] > .07595 if name.startswith('wrist') else samples[:, 1] < -.055
        row = dict(source=str(path.relative_to(ROOT)), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   sdf=dict(id=grid_id, spacing=SPACING, origin=origin.tolist(), shape=shape.tolist()),
                   grid_mb=target.stat().st_size/1e6, elapsed_s=time.monotonic()-started,
                   sampled_error_max_m=float(error.max()), sampled_error_p99_m=float(np.quantile(error,.99)),
                   sampled_critical_error_max_m=float(error[critical].max()),
                   sampled_critical_count=int(critical.sum()),
                   validation_scope='20k source-surface and volume samples, exact triangle reference; not a worst-case certificate')
        manifest['meshes'][name] = row
        print(json.dumps(row), flush=True)
    (directory/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
