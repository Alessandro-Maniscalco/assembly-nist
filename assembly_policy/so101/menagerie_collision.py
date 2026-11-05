"""Optional pinned Menagerie jaw geometry; not selected by the frozen pilot.

Only collision shapes change. Current kinematics, TCP sites, inertia, contact
parameters, controls and visual meshes are preserved. See the local provenance.
"""
import copy
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


DIRECTORY = Path(__file__).resolve().parents[2]/'assets_local/so101_menagerie_collision'
BODIES = ('gripper', 'moving_jaw_so101_v1')
CLASSES = ('collision', 'collision_gripper', 'collision_gripper_mesh')
SHAPE_ATTRIBUTES = ('name', 'type', 'size', 'pos', 'quat', 'euler', 'fromto', 'mesh')


def replace_gripper_collision_menagerie(robot_xml, directory=DIRECTORY, *, prefix=''):
    """Apply after original mesh-name normalization and before side prefixes."""
    reference = ET.parse(directory/'so101.xml').getroot()
    needed = set()
    replacements = []
    for name in BODIES:
        source = reference.find(f'.//body[@name="{name}"]')
        current = robot_xml.find(f'.//body[@name="{name}"]')
        for field, fallback in [('pos', '0 0 0'), ('quat', '1 0 0 0')]:
            a = np.fromstring(source.get(field, fallback), sep=' ')
            b = np.fromstring(current.get(field, fallback), sep=' ')
            if field == 'quat':
                a /= np.linalg.norm(a)
                b /= np.linalg.norm(b)
                if np.dot(a, b) < 0:
                    b = -b
            if not np.allclose(a, b, atol=1e-6, rtol=0):
                raise ValueError(f'Menagerie collision frame differs for {name}: {field}')
        shapes = []
        for index, geom in enumerate(source.findall('geom')):
            if geom.get('class') not in CLASSES:
                continue
            # Do not also import Menagerie's different friction/condim/solref.
            attrs = {key: geom.get(key) for key in SHAPE_ATTRIBUTES if geom.get(key) is not None}
            attrs.setdefault('name', f'{name}_menagerie_collision_{index}')
            attrs.update({'class': 'collision', 'mass': '0'})
            shapes.append(ET.Element('geom', attrs))
            if geom.get('mesh'):
                needed.add(geom.get('mesh'))
        replacements.append((current, shapes))
    asset = robot_xml.find('asset')
    for geom_name in sorted(needed):
        source = next(element for element in reference.findall('asset/mesh')
                      if Path(element.get('file')).stem == geom_name)
        mesh = copy.deepcopy(source)
        mesh.set('name', geom_name)
        mesh.set('file', str(directory/'assets'/source.get('file')))
        asset.append(mesh)
    for body, shapes in replacements:
        for geom in list(body.findall('geom')):
            if geom.get('class') in CLASSES:
                body.remove(geom)
        body.extend(shapes)
    return []  # Interface-compatible with the SDF replacement's plugin list.
