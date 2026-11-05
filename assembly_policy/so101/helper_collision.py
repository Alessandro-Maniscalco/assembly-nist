"""Optional whole-helper mesh-to-SDF collision substitution.

This changes native collision geometry from convex hulls to source-surface
distance grids. It is an isolated candidate, never applied by scene.build_xml.
"""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / 'assets_local/so101_helper_collision'


def replace_helper_collision(root, directory=DIRECTORY):
    """Modify a complete workcell XML in place; keep all poses and visuals.

    Call after scene.build_xml, so restored static-base collision meshes are
    included. Existing jaw SDFs, target parts, tool and collision filters stay.
    """
    manifest = json.loads((Path(directory)/'manifest.json').read_text())
    if not manifest.get('complete') or manifest.get('backend') != 'grid_sdf':
        raise ValueError('Complete audited helper grids are required')
    asset = root.find('asset')
    extension = root.find('extension')
    if asset is None or extension is None:
        raise ValueError('Expected complete workcell XML')
    source_assets = {m.get('name'): m for m in asset.findall('mesh')}
    collision_geoms = []
    for body in root.findall('worldbody/body'):
        side = next((s for s in ('left', 'right') if body.get('name') == s+'_base'), None)
        if side is None:
            continue
        for geom in body.iter('geom'):
            if geom.get('class') != side+'_collision' or geom.get('type', 'mesh') != 'mesh':
                continue
            source = source_assets[geom.get('mesh')]
            name = Path(source.get('file')).stem
            if name not in manifest['meshes']:
                raise ValueError(f'Unqualified helper collision mesh {name}')
            if source.get('scale', '1 1 1') != '1 1 1':
                raise ValueError('Scaled source mesh requires a separate grid')
            collision_geoms.append((geom, name))
    if not collision_geoms:
        raise ValueError('No helper collision meshes found; refusing a silent no-op')
    plugin = ET.SubElement(extension, 'plugin', plugin='assemblenist.sdf.grid')
    for name in sorted({name for _, name in collision_geoms}):
        instance_name = 'helper_sdf_'+name
        instance = ET.SubElement(plugin, 'instance', name=instance_name)
        ET.SubElement(instance, 'config', key='id', value=str(manifest['meshes'][name]['sdf']['id']))
        mesh = ET.SubElement(asset, 'mesh', name=instance_name+'_mesh')
        ET.SubElement(mesh, 'plugin', instance=instance_name)
    for geom, name in collision_geoms:
        instance_name = 'helper_sdf_'+name
        geom.set('type', 'sdf')
        geom.set('mesh', instance_name+'_mesh')
        ET.SubElement(geom, 'plugin', instance=instance_name)
    return dict(converted_geoms=len(collision_geoms), unique_meshes=len({name for _, name in collision_geoms}),
                grid_ids=sorted({manifest['meshes'][name]['sdf']['id'] for _, name in collision_geoms}),
                candidate_only=True)
