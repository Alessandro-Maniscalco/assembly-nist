"""Build cavity-preserving grid SDFs for the existing AssembleNIST plugin.

Samples vendor triangles without changing their positions. Grid interpolation is
an approximation: validate mating contacts before treating an episode as success.
"""
from pathlib import Path
import json
import struct
import argparse
import numpy as np
import open3d as o3d
import igl

ROOT = Path(__file__).resolve().parents[1]
SPACING = {'base': .00035, 'motor': .00015, 'horn': .00010,
           'screw': .00005, 'retainer': .00025, 'yoke': .00030}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parts', nargs='+', choices=list(SPACING), default=list(SPACING))
    args = ap.parse_args()
    directory = ROOT / 'assets_local/parts'
    meta = json.loads((directory / 'manifest.json').read_text())
    grid_dir = ROOT / 'assets_local/sdf'
    grid_dir.mkdir(exist_ok=True)
    for i, (name, h) in enumerate(SPACING.items(), 101):
        if name not in args.parts:
            continue
        z = np.load(directory / f'{name}.npz')
        v, f = z['vertices'].astype(np.float32), z['faces'].astype(np.uint32)
        scene = o3d.t.geometry.RaycastingScene(nthreads=4)
        scene.add_triangles(o3d.core.Tensor(v), o3d.core.Tensor(f))
        origin = np.floor((v.min(0) - 4*h) / h) * h
        shape = np.ceil((v.max(0) + 4*h - origin) / h).astype(int) + 1
        x, y = [origin[k] + np.arange(shape[k]) * h for k in range(2)]
        xy = np.stack(np.meshgrid(x, y, indexing='xy'), -1).reshape(-1, 2)
        with (grid_dir / f'grid_{i}.bin').open('wb') as fp:
            fp.write(struct.pack('<3i4d', *shape, *origin, h))
            for k in range(shape[2]):
                pts = np.column_stack((xy, np.full(len(xy), origin[2]+k*h))).astype(np.float32)
                d = scene.compute_distance(o3d.core.Tensor(pts)).numpy()
                # Ray parity on CAD seam gaps / overlapping servo solids can label
                # exterior points as interior. Winding sums handle these soups.
                winding = igl.fast_winding_number(z['vertices'], z['faces'], pts.astype(float))
                d *= np.where(winding > .5, -1, 1)
                if not np.isfinite(d).all():
                    raise ValueError(f'Nonfinite SDF for {name}')
                fp.write(d.astype('<f2').tobytes())
        meta['parts'][name]['sdf'] = dict(id=i, spacing=h, origin=origin.tolist(), shape=shape.tolist())
        print(name, tuple(shape), 'grid MB', round(np.prod(shape)*2/1e6, 2), flush=True)
    (directory / 'manifest.json').write_text(json.dumps(meta, indent=2)+'\n')


if __name__ == '__main__':
    main()
