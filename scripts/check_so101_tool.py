"""Bench-jaw mechanism qualification, not a robot grasp or assembly policy.

Runs native MuJoCo contacts with one force-controlled test-jig jaw. No object
poses are overwritten after reset and no grasp constraints are introduced.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from assembly_policy.so101.tool import add_driver, GRASP_Z


ROOT = Path(__file__).resolve().parents[1]


def make_model():
    mujoco.mj_loadPluginLibrary(str(ROOT/'plugin/build/libso101_driver.so'))
    root = ET.Element('mujoco', model='screwdriver_contact_bench_jaws')
    ET.SubElement(root, 'compiler', angle='radian', autolimits='true')
    ET.SubElement(root, 'option', timestep='.0005', integrator='implicitfast',
                  cone='elliptic', iterations='50')
    default = ET.SubElement(root, 'default')
    ET.SubElement(default, 'geom', friction='.8 .005 .0001', condim='4',
                  solref='.002 1', solimp='.95 .99 .0002')
    world = ET.SubElement(root, 'worldbody')
    extension = ET.SubElement(root, 'extension')
    contact = ET.SubElement(root, 'contact')
    add_driver(world, extension, contact, position=(0, 0, .3), quaternion=(1, 0, 0, 0))
    ET.SubElement(world, 'geom', name='bench_jaw_fixed', type='box',
                  pos=f'0 -.017 {.3+GRASP_Z}', size='.008 .002 .011')
    jaw = ET.SubElement(world, 'body', name='bench_jaw_moving', pos=f'0 .0215 {.3+GRASP_Z}')
    ET.SubElement(jaw, 'joint', name='bench_jaw_slide', type='slide', axis='0 -1 0',
                  range='-.02 .005', damping='4')
    ET.SubElement(jaw, 'geom', name='bench_jaw_moving_pad', type='box',
                  size='.008 .0035 .011', mass='.03')
    actuator = ET.SubElement(root, 'actuator')
    ET.SubElement(actuator, 'motor', name='bench_jaw_force', joint='bench_jaw_slide',
                  gear='1', ctrlrange='-8 8')
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))


def run_check():
    model = make_model()
    data = mujoco.MjData(model)
    ids = {name: model.joint(name).qposadr[0]
           for name in ('button_forward', 'button_reverse', 'trigger_saddle')}
    spindle = model.joint('spindle').dofadr[0]
    body = model.body('driver').id
    mujoco.mj_forward(model, data)
    initial = data.xpos[body].copy()
    records = []
    for label, force, seconds, torque in [('carry_off', 2.8, 1., 0.),
                                          ('carry_torque_off', 2.8, 1., .05),
                                          ('squeeze_on', 4., 1., 0.),
                                          ('release_switch_off', 2.8, 1., 0.)]:
        data.ctrl[0] = force
        data.xfrc_applied[body, 5] = torque
        maximum_button = 0.
        for _ in range(round(seconds/model.opt.timestep)):
            mujoco.mj_step(model, data)
            maximum_button = max(maximum_button, data.qpos[ids['button_forward']])
        records.append(dict(stage=label, jig_force_n=force,
                            button_forward_mm=float(data.qpos[ids['button_forward']]*1000),
                            button_reverse_mm=float(data.qpos[ids['button_reverse']]*1000),
                            saddle_mm=float(data.qpos[ids['trigger_saddle']]*1000),
                            max_button_mm=float(maximum_button*1000),
                            spindle_rad_s=float(data.qvel[spindle]),
                            displacement_mm=float(np.linalg.norm(data.xpos[body]-initial)*1000),
                            height_m=float(data.xpos[body, 2]),
                            active_contacts=int(data.ncon)))
    assert all(row['displacement_mm'] < 2 for row in records), records
    assert all(abs(row['button_reverse_mm']) < .1 for row in records), records
    assert all(row['button_forward_mm'] < .3 for row in (records[0], records[1], records[3])), records
    assert records[0]['max_button_mm'] < .6 and records[1]['max_button_mm'] < .6, records
    assert records[2]['button_forward_mm'] > .6 and records[2]['spindle_rad_s'] > 20, records
    assert abs(records[3]['spindle_rad_s']) < .1, records
    before_release = data.xpos[body, 2]
    data.ctrl[0] = -2.
    data.xfrc_applied[:] = 0
    for _ in range(500):
        mujoco.mj_step(model, data)
    drop = float(before_release-data.xpos[body, 2])
    assert drop > .05, drop  # Prove the test did not secretly constrain the tool.
    return dict(passed=True, scope='passive sleeve in ideal bench jaws; not SO101 grasp qualification',
                records=records, open_jaws_drop_m=drop, mechanism_actuators=0,
                test_jig_actuators=model.nu,
                finite=bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    report = run_check()
    output = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output+'\n')
    print(output)
