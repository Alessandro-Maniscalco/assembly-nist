"""Replace filled convex jaw hulls with local source-CAD signed distances."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET


DIRECTORY = Path(__file__).resolve().parents[2]/'assets_local/so101_collision'


def replace_gripper_collision(robot_xml, directory=DIRECTORY, *, prefix=''):
    """Apply before side prefixes; append returned plugins to model extension.

    Preserves visual geometry and inertias. Fails if SDF assets are absent:
    silently reverting to a hull
    would close the real finger opening and invalidate tool grasps.
    """
    metadata = json.loads((directory/'manifest.json').read_text())
    manifest = metadata['meshes']
    expected = {'wrist_roll_follower_so101_v1', 'moving_jaw_so101_v1'}
    if set(manifest) != expected or metadata.get('backend') != 'grid_sdf':
        raise ValueError('Both jaw collision assets must finish generation before use')
    asset = robot_xml.find('asset')
    plugin = ET.Element('plugin', plugin='assemblenist.sdf.grid')
    for source_name, record in manifest.items():
        instance_name = prefix+source_name+'_sdf'
        instance = ET.SubElement(plugin, 'instance', name=instance_name)
        ET.SubElement(instance, 'config', key='id', value=str(record['sdf']['id']))
        mesh = ET.SubElement(asset, 'mesh', name=source_name+'_collision_sdf')
        ET.SubElement(mesh, 'plugin', instance=instance_name)
    for geom in robot_xml.iter('geom'):
        source_name = geom.get('mesh')
        if geom.get('class') != 'collision' or source_name not in manifest:
            continue
        geom.set('type', 'sdf')
        geom.set('name', source_name+'_collision_sdf')
        geom.set('mesh', source_name+'_collision_sdf')
        ET.SubElement(geom, 'plugin', instance=prefix+source_name+'_sdf')
    return [plugin]
