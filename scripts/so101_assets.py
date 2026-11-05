"""Import unmodified Hiwonder render meshes and pinned helper robot assets.

Run with rhino3dm, numpy and trimesh installed. Vendor CAD stays local;
obtain follower.3dm from the official folder linked in docs/SO101.md.
"""
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import rhino3dm as rh
import trimesh

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'assets_local'
SHA = 'f675f6049f94206c7b74ab73c40485913916babff1a6f4fc145dc9040672a2f8'
REV = 'a758567c3978dfeefe282ede0500085a48fe8f78'
URL = f'https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/{REV}/Simulation/SO101/'
PARTS = {'base': [260], 'motor': list(range(14)), 'horn': [15],
         'screw': [36], 'retainer': [31], 'yoke': [30]}


def object_mesh(obj):
    g = obj.Geometry
    if isinstance(g, rh.Extrusion):
        g = g.ToBrep()
    chunks = []
    for face in g.Faces:
        m = face.GetMesh(rh.MeshType.Render)
        if m is None:
            raise ValueError('CAD face has no cached render mesh')
        v = np.array([[p.X, p.Y, p.Z] for p in m.Vertices]) / 1000
        fs = []
        for a, b, c, d in m.Faces:
            fs.append([a, b, c])
            if c != d:
                fs.append([a, c, d])
        chunks.append(trimesh.Trimesh(v, fs, process=False))
    mesh = trimesh.util.concatenate(chunks)
    # Join coincident seam vertices only; no hole filling, scaling or material cuts.
    mesh.merge_vertices(digits_vertex=9)
    return mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cad', type=Path, default=ASSETS / 'hiwonder/follower.3dm')
    ap.add_argument('--skip-helper', action='store_true')
    a = ap.parse_args()
    digest = hashlib.sha256(a.cad.read_bytes()).hexdigest()
    if digest != SHA:
        raise ValueError(f'Unreviewed CAD revision: {digest}')
    doc = rh.File3dm.Read(str(a.cad))
    if doc.Settings.ModelUnitSystem != rh.UnitSystem.Millimeters:
        raise ValueError('Expected millimetres')
    out = ASSETS / 'parts'
    out.mkdir(parents=True, exist_ok=True)
    manifest = {'cad_sha256': digest, 'units': 'metres', 'parts': {}}
    for name, indices in PARTS.items():
        meshes = [object_mesh(doc.Objects[i]) for i in indices]
        mesh = trimesh.util.concatenate(meshes)
        origin = mesh.bounds.mean(axis=0)
        mesh.vertices -= origin
        mesh.export(out / f'{name}.obj')
        np.savez_compressed(out / f'{name}.npz', vertices=mesh.vertices, faces=mesh.faces)
        manifest['parts'][name] = dict(indices=indices, object_ids=[str(doc.Objects[i].Attributes.Id) for i in indices],
                                      origin=origin.tolist(), bounds=mesh.bounds.tolist(),
                                      watertight=bool(mesh.is_watertight), volume=float(mesh.volume),
                                      faces=len(mesh.faces))
        print(name, 'dimensions mm', np.round(mesh.extents * 1000, 3),
              'watertight', mesh.is_watertight, flush=True)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if not a.skip_helper:
        helper = ASSETS / 'so101'
        helper.mkdir(parents=True, exist_ok=True)
        def fetch(name):
            target = helper / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                urllib.request.urlretrieve(URL + name, target)
            return target
        xml = fetch('so101_new_calib.xml')
        fetch('README.md')
        for m in ET.parse(xml).getroot().findall('asset/mesh'):
            fetch('assets/' + m.get('file').removeprefix('assets/'))
        (helper / 'revision.txt').write_text(REV + '\n')


if __name__ == '__main__':
    main()
