"""Contact-operated precision driver and a provisional printable trigger sleeve.

Dimensions are design assumptions, not HOTO CAD. See docs/SO101_TOOL.md.
The sleeve is rigidly fitted to the tool; its only moving part is a passive,
spring-preloaded saddle that presses the actual forward button by contact.
"""
import xml.etree.ElementTree as ET

import numpy as np


# Local +Z is the bit direction; +Y is the button side. Metres, kilograms.
CASE_HALF_WIDTH = .010
CASE_HALF_LENGTH = .078
GRASP_Z = .015
TIP_Z = .105
SLEEVE_BACK_Y = -.015
SADDLE_FACE_Y = .018
SADDLE_TRAVEL = .002
SADDLE_PRELOAD = 2.0
SADDLE_STIFFNESS = 400.0
BUTTON_GAP = .001
BUTTON_ON = .0006
GRASP_CENTER = np.array([0., .0015, GRASP_Z])
DEFAULT_POSITION = (.12, -.12, .0145)
DEFAULT_QUATERNION = (.7071067811865476, 0., .7071067811865476, 0.)


def _add(parent, tag, **attributes):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attributes.items()})


def _numbers(values):
    return ' '.join(f'{value:.10g}' for value in values)


def add_driver(world, extension, contact, *, position=DEFAULT_POSITION,
               quaternion=DEFAULT_QUATERNION):
    """Append a free driver; return its body without adding any actuators.

    Preserves driver_free, button_forward, button_reverse and spindle names.
    Explicit contact pairing matters: the saddle must press a separate native
    button, not supply an activation signal itself. Two force levels on the
    existing gripper allow an OFF carry grasp and an ON squeeze grasp.
    """
    tool = _add(world, 'body', name='driver', pos=_numbers(position),
                quat=_numbers(quaternion))
    _add(tool, 'freejoint', name='driver_free')
    # Nominal case inertia; overlapping same-body primitives do not add mass.
    mass = .110
    side, length = 2*CASE_HALF_WIDTH, 2*CASE_HALF_LENGTH
    transverse = mass*(side**2 + length**2)/12
    _add(tool, 'inertial', pos='0 0 0', mass=mass,
         diaginertia=_numbers([transverse, transverse, mass*side**2/6]))
    grey = '.29 .32 .37 1'
    _add(tool, 'geom', name='driver_case', type='box', size='.007 .010 .078',
         rgba=grey, mass='0')
    _add(tool, 'geom', name='driver_case_cross', type='box', size='.010 .007 .078',
         rgba=grey, mass='0')
    for index, (x, y) in enumerate(((-.007, -.007), (-.007, .007),
                                   (.007, -.007), (.007, .007))):
        _add(tool, 'geom', name=f'driver_case_corner_{index}', type='cylinder',
             size='.003 .078', pos=_numbers([x, y, 0]),
             rgba=grey, mass='0')
    _add(tool, 'geom', name='driver_nose', type='cylinder', size='.006 .003',
         pos='0 0 .075', rgba='.10 .12 .15 1', mass='0')
    _add(tool, 'geom', name='driver_led_ring', type='cylinder', size='.0062 .0007',
         pos='0 0 .078', rgba='.85 .9 .8 1', contype='0', conaffinity='0', mass='0')
    for name, z in [('forward', GRASP_Z), ('reverse', -.005)]:
        button = _add(tool, 'body', name='driver_'+name, pos=_numbers([0, .011, z]))
        _add(button, 'joint', name='button_'+name, type='slide', axis='0 -1 0',
             range='0 .001', stiffness='1400', damping='1', armature='.001',
             solreflimit='.001 1', solimplimit='.99 .999 .00001')
        _add(button, 'geom', name='button_'+name, type='box',
             size='.004 .0015 .007', mass='.001', rgba='.09 .11 .15 1')

    # Split sleeve: two bolted shells around the case. Hollow rails/collars leave
    # the button corridor open. Fixed-to-tool means a fitted accessory, not a
    # robot attachment. Each geom has contact; no invisible grasp constraint.
    sleeve = _add(tool, 'body', name='driver_sleeve', pos=_numbers([0, 0, GRASP_Z]))
    _add(sleeve, 'inertial', pos='0 -.003 0', mass='.014',
         diaginertia='.0000035 .0000035 .000002')
    orange = '.88 .39 .10 1'
    for sign, name in [(-1, 'left'), (1, 'right')]:
        _add(sleeve, 'geom', name='sleeve_'+name+'_rail', type='box',
             pos=_numbers([sign*.012, 0, 0]), size='.002 .012 .025',
             rgba=orange, mass='0')
    for sign, name in [(-1, 'rear'), (1, 'front')]:
        for y, side_name in [(-.012, 'back'), (.012, 'button')]:
            if sign == -1 and side_name == 'button':
                continue  # Keep the reverse button's full swept volume open.
            _add(sleeve, 'geom', name=f'sleeve_{name}_{side_name}_bridge',
                 type='box', pos=_numbers([0, y, sign*.023]),
                 size='.010 .002 .002', rgba=orange, mass='0')
    _add(sleeve, 'geom', name='sleeve_back', type='box', pos='0 -.012 0',
         size='.014 .002 .013', rgba=orange, mass='0')
    _add(sleeve, 'geom', name='sleeve_back_pad', type='box', pos='0 -.0145 0',
         size='.013 .0005 .013', rgba='.12 .14 .17 1', mass='0',
         friction='.8 .005 .0001', condim='4')

    # Practical counterpart: two 2 mm guide pins, captive travel stops, and
    # two return springs totaling 2 N preload. Guides are represented by one
    # passive slide; there is no guide motor or simulator-only hold mechanism.
    paddle = _add(tool, 'body', name='driver_trigger_saddle',
                  pos=_numbers([0, .016, GRASP_Z]))
    _add(paddle, 'joint', name='trigger_saddle', type='slide', axis='0 -1 0',
         range=_numbers([0, SADDLE_TRAVEL]), stiffness=SADDLE_STIFFNESS,
         springref=-SADDLE_PRELOAD/SADDLE_STIFFNESS, damping='1', armature='.001',
         solreflimit='.001 1', solimplimit='.99 .999 .00001')
    _add(paddle, 'geom', name='trigger_saddle_pad', type='box',
         size='.013 .002 .013', mass='.0035', rgba='.14 .53 .72 1',
         friction='.8 .005 .0001', condim='4')
    _add(paddle, 'geom', name='trigger_saddle_pusher', type='box',
         pos='0 -.0015 0', size='.003 .001 .004', mass='.0005',
         rgba='.10 .32 .45 1')
    for sign, name in [(-1, 'left'), (1, 'right')]:
        _add(paddle, 'geom', name='trigger_guide_'+name+'_ear', type='box',
             pos=_numbers([sign*.0105, 0, 0]), size='.0015 .0015 .003',
             mass='.0001', rgba='.14 .53 .72 1')
        _add(paddle, 'geom', name='trigger_guide_'+name+'_pin', type='capsule',
             fromto=_numbers([sign*.0115, -.007, 0, sign*.0115, .001, 0]),
             size='.0007', mass='.00005', rgba='.65 .68 .71 1')
    # Sibling bodies should collide by default; an explicit pair documents and
    # preserves the internal switch contact if scene masks later change.
    _add(contact, 'pair', geom1='trigger_saddle_pusher', geom2='button_forward',
         condim='3', friction='.3 .3 .001 .0001 .0001',
         solref='.001 1', solimp='.99 .999 .00001')
    # The shell rails are mechanical guides, represented by the passive joint.
    # They cannot also be unconstrained collision stops against the paddle.
    _add(contact, 'exclude', body1='driver_sleeve', body2='driver_trigger_saddle')

    bit = _add(tool, 'body', name='driver_bit', pos='0 0 .078')
    _add(bit, 'joint', name='spindle', type='hinge', axis='0 0 1',
         damping='.00002', armature='.000002')
    _add(bit, 'geom', name='bit_shaft', type='cylinder', size='.002 .012',
         pos='0 0 .012', mass='.002', rgba='.70 .72 .75 1')
    for axis, size in [('x', '.0013 .0003 .0015'), ('y', '.0003 .0013 .0015')]:
        _add(bit, 'geom', name='bit_'+axis, type='box', size=size,
             pos='0 0 .0255', mass='.0001', rgba='.65 .67 .70 1')
    _add(tool, 'site', name='driver_tip', pos=_numbers([0, 0, TIP_Z]),
         size='.001', rgba='0 1 0 .7', group='3')
    _add(tool, 'site', name='driver_grasp', pos=_numbers(GRASP_CENTER),
         size='.002', rgba='1 .5 0 .7', group='3')
    plugin = _add(extension, 'plugin', plugin='so101.contact_driver')
    instance = _add(plugin, 'instance', name='button_motor')
    for key, value in [('forward', 'button_forward'), ('reverse', 'button_reverse'),
                       ('spindle', 'spindle')]:
        _add(instance, 'config', key=key, value=value)
    return tool
