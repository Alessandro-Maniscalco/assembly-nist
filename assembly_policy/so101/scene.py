"""Clamped helper arms and free Hiwonder parts, using native MuJoCo contacts."""
from pathlib import Path
import copy
import json
import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from .fastener import AXIS, attributes, origins
from .tool import add_driver
from .collision import replace_gripper_collision

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'assets_local'
PARTS = ('base', 'motor', 'horn', 'screw', 'retainer', 'yoke')
OBS_BODIES = (*PARTS, 'driver', 'driver_trigger_saddle', 'driver_forward', 'driver_reverse', 'driver_bit')
JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')
COLORS = {'base': '.79 .81 .84 1', 'motor': '.13 .15 .18 1', 'horn': '.8 .81 .83 1',
          'screw': '.3 .33 .37 1', 'retainer': '.75 .79 .83 1', 'yoke': '.75 .79 .83 1'}
MASSES = {'base': .065, 'motor': .052, 'horn': .0025, 'screw': .00045, 'retainer': .021, 'yoke': .033}
# Nominal layout, metres. Each part gets independent +/- 8 mm and +/- 8 deg at reset.
LAYOUT = {'base': (0, .06), 'motor': (.12, .07), 'horn': (.09, -.035),
          'screw': (.045, -.045), 'retainer': (-.11, .055), 'yoke': (-.08, -.075)}


def metadata():
    return json.loads((ASSETS / 'parts/manifest.json').read_text())['parts']


def add(parent, tag, **a):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in a.items()})


def numbers(a):
    return ' '.join(f'{x:.10g}' for x in np.asarray(a).ravel())


def load_plugin():
    os.environ['ASSEMBLENIST_SDF_DIR'] = str(ASSETS / 'sdf')
    mujoco.mj_loadPluginLibrary(str(ROOT / 'plugin/build/libassemblenist_sdf.so'))
    mujoco.mj_loadPluginLibrary(str(ROOT / 'plugin/build/libso101_driver.so'))


def build_xml(assembled=False, contact_monitor=False):
    parts = metadata()
    target = origins(parts)
    root = ET.Element('mujoco', model='dual_so101_hiwonder')
    add(root, 'compiler', angle='radian', autolimits='true')
    add(root, 'option', timestep='.0005', integrator='implicitfast', cone='elliptic',
        iterations='50', ls_iterations='20', sdf_initpoints='40', sdf_iterations='10')
    default = add(root, 'default')
    add(default, 'geom', friction='.7 .005 .0001', solref='.002 1',
        solimp='.95 .99 .0002', condim='4')
    visual = add(root, 'visual')
    add(visual, 'global', offwidth='1440', offheight='1080')
    add(visual, 'headlight', ambient='.5 .5 .5', diffuse='.7 .7 .7')
    add(visual, 'rgba', haze='.12 .15 .2 1')
    asset = add(root, 'asset')
    add(asset, 'texture', name='sky', type='skybox', builtin='gradient',
        rgb1='.2 .24 .3', rgb2='.6 .65 .72', width='256', height='256')
    world = add(root, 'worldbody')
    add(world, 'light', pos='0 0 1.5', dir='0 0 -1', diffuse='.8 .8 .8')
    add(world, 'geom', name='table', type='plane', size='.65 .5 .03', rgba='.20 .25 .3 1')
    acts = add(root, 'actuator')
    contact = add(root, 'contact')
    pending_plugins = []
    for side, x, yaw in [('left', -.25, 0), ('right', .25, np.pi)]:
        src = ET.parse(ASSETS / 'so101/so101_new_calib.xml').getroot()
        for m in src.findall('asset/mesh'):
            m.set('name', Path(m.get('file')).stem)
            m.set('file', str(ASSETS / 'so101/assets' / m.get('file')))
        pending_plugins.extend(replace_gripper_collision(src, prefix=side+'_'))
        prefix = side + '_'
        for e in src.iter():
            for k in ('name', 'class', 'childclass', 'mesh', 'material', 'joint', 'body', 'site'):
                if e.get(k):
                    e.set(k, prefix + e.get(k))
        for e in src.findall('default'):
            for child in e:
                default.append(copy.deepcopy(child))
        for e in src.find('asset'):
            asset.append(copy.deepcopy(e))
        body = copy.deepcopy(src.find('worldbody/body'))
        body.set('pos', numbers([x, .0, .0024]))
        body.set('quat', numbers([np.cos(yaw/2), 0, 0, np.sin(yaw/2)]))
        # Restore the stationary base obstacles omitted by the upstream robot XML.
        # The adjacent rotating shoulder is a bearing connection, not a collision pair.
        for e in list(body.findall('geom')):
            solid=copy.deepcopy(e);solid.set('class',prefix+'collision')
            body.append(solid)
        add(contact,'exclude',body1=prefix+'base',body2=prefix+'shoulder')
        # Explicit static base/clamp contact approximations; moving arm meshes are upstream.
        add(body, 'geom', name=prefix+'base_collision', type='box', size='.055 .045 .002',
            pos='0 0 0', rgba='.1 .12 .15 1')
        for y in (-.052, .052):
            add(body, 'geom', name=prefix+('clamp_a' if y < 0 else 'clamp_b'),
                type='box', size='.035 .009 .018', pos=numbers([-.005, y, .015]), rgba='.08 .09 .1 1')
        world.append(body)
        for e in src.find('actuator'):
            if e.get('name') == prefix+'gripper':
                e.set('forcerange', '-.35 .35')  # conservative grasp effort; identify on hardware
            acts.append(copy.deepcopy(e))
    ext = add(root, 'extension')
    ext.extend(pending_plugins)
    plugin = add(ext, 'plugin', plugin='assemblenist.sdf.grid')
    def sdf_instance(kind, name, attrs):
        plug = add(ext, 'plugin', plugin=kind)
        instance = add(plug, 'instance', name=name)
        for k,v in attrs.items(): add(instance, 'config', key=k, value=v)
        mesh = add(asset, 'mesh', name=name+'_mesh')
        add(mesh, 'plugin', instance=name)
    bolt,nut=attributes()
    # Match the mating helix at the nominal bearing position.
    nut['phase'] = ((AXIS[2]-target['screw'][2]) / bolt['pitch']) % 1
    sdf_instance('assemblenist.sdf.bolt','male',bolt)
    sdf_instance('assemblenist.sdf.nut','female',nut)
    sdf_instance('so101.screw_head','screw_head',{})
    base_min = parts['base']['origin'][2] + parts['base']['bounds'][0][2]
    for name, p in parts.items():
        instance = add(plugin, 'instance', name=name+'_sdf')
        add(instance, 'config', key='id', value=p['sdf']['id'])
        add(asset, 'mesh', name=name+'_mesh', file=str(ASSETS / f'parts/{name}.obj'))
        sm = add(asset, 'mesh', name=name+'_sdfmesh')
        add(sm, 'plugin', instance=name+'_sdf')
        pos = np.array([*LAYOUT[name], -p['bounds'][0][2] + .002])
        if name == 'screw': pos[2]=.004
        if assembled:
            pos = target[name] - [0, 0, base_min-.0003] + [0, .08, 0]
        body = add(world, 'body', name=name, pos=numbers(pos))
        if name=='screw' and not assembled:body.set('quat','.70710678 0 .70710678 0')
        add(body, 'freejoint', name=name+'_free')
        # Bounding-box inertia estimate, explicitly provisional until weighing/identification.
        dim = np.ptp(np.array(p['bounds']), axis=0)
        inertia = MASSES[name] / 12 * np.array([dim[1]**2+dim[2]**2, dim[0]**2+dim[2]**2, dim[0]**2+dim[1]**2])
        add(body, 'inertial', pos='0 0 0', mass=MASSES[name], diaginertia=numbers(inertia))
        if name != 'screw':
            add(body, 'geom', name=name+'_visual', type='mesh', mesh=name+'_mesh',
                contype='0', conaffinity='0', group='2', rgba=COLORS[name])
            geom = add(body, 'geom', name=name+'_contact', type='sdf', mesh=name+'_sdfmesh',
                       group='3', rgba='0 .7 .8 .3')
            add(geom, 'plugin', instance=name+'_sdf')
        else:
            for instance, quat in [('male','0 1 0 0'),('screw_head','1 0 0 0')]:
                geom = add(body, 'geom', name=instance+'_contact', type='sdf', mesh=instance+'_mesh',
                           quat=quat, group='2', contype='2' if instance=='male' else '1',
                           conaffinity='3', rgba=COLORS[name], friction='.2 .001 .00001',
                           solref='.001 1', solimp='.99 .999 .00005')
                add(geom,'plugin',instance=instance)
        if name == 'motor':
            geom=add(body,'geom',name='female_contact',type='sdf',mesh='female_mesh',
                     pos=numbers(AXIS-np.array(p['origin'])),quat='0 1 0 0',
                     contype='2',conaffinity='2',group='3',rgba='0 .7 .8 .3',
                     friction='.2 .001 .00001',solref='.001 1',solimp='.99 .999 .00005')
            add(geom,'plugin',instance='female')
    add_driver(world, ext, contact)
    if contact_monitor:
        sensors=add(root,'sensor')
        add(sensors,'contact',name='min_contact_distance',data='dist',num='1',reduce='mindist')
    return ET.tostring(root, encoding='unicode')


def model(assembled=False, contact_monitor=False):
    load_plugin()
    return mujoco.MjModel.from_xml_string(build_xml(assembled,contact_monitor))
