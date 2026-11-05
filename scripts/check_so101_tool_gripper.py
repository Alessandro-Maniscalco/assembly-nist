"""Position-control checks using the source SO101 jaws and free screwdriver.

The palm is a fixed test fixture. This qualifies an already-aligned grasp, not
arm reach, pickup or tightening. No object state is overwritten during stepping.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from assembly_policy.so101.collision import replace_gripper_collision
from assembly_policy.so101.tool import add_driver


ROOT = Path(__file__).resolve().parents[1]
ENCODER_STEP = 2*np.pi/4096


def make_model(decomposed=True, tool_builder=add_driver, collision_builder=replace_gripper_collision,
               query_point=False):
    root = ET.Element('mujoco', model='actual_so101_jaw_tool_fixture')
    ET.SubElement(root, 'compiler', angle='radian', autolimits='true')
    ET.SubElement(root, 'option', timestep='.0005', integrator='implicitfast',
                  gravity='0 -9.81 0', cone='elliptic', iterations='50')
    default = ET.SubElement(root, 'default')
    ET.SubElement(default, 'geom', friction='.8 .005 .0001', condim='4',
                  solref='.002 1', solimp='.95 .99 .0002')
    source = ET.parse(ROOT/'assets_local/so101/so101_new_calib.xml').getroot()
    for element in source.findall('asset/mesh'):
        element.set('name', Path(element.get('file')).stem)
        element.set('file', str(ROOT/'assets_local/so101/assets'/element.get('file')))
    plugins = collision_builder(source) if decomposed else []
    for element in source.findall('default'):
        for child in element:
            default.append(copy.deepcopy(child))
    root.append(copy.deepcopy(source.find('asset')))
    world = ET.SubElement(root, 'worldbody')
    if query_point:
        ET.SubElement(world, 'geom', name='query_point', type='sphere', size='.000001',
                      contype='0', conaffinity='0')
    gripper = copy.deepcopy(source.find('.//body[@name="gripper"]'))
    gripper.set('pos', '0 0 0')
    gripper.set('quat', '1 0 0 0')
    gripper.set('childclass', 'so101_new_calib')
    gripper.remove(gripper.find('joint'))  # Bolt the palm to the bench fixture.
    world.append(gripper)
    extension = ET.SubElement(root, 'extension')
    extension.extend(plugins)
    contact = ET.SubElement(root, 'contact')
    tool_builder(world, extension, contact, position=(.0071, -.015, -.09),
                 quaternion=(.5, -.5, -.5, -.5))
    # With the palm fixed, MuJoCo's static-parent exception otherwise collides
    # the two sides of the actual gripper bearing against one another.
    ET.SubElement(contact, 'exclude', body1='gripper', body2='moving_jaw_so101_v1')
    actuator = ET.SubElement(root, 'actuator')
    drive = copy.deepcopy(source.find('actuator/position[@name="gripper"]'))
    drive.set('forcerange', '-.35 .35')
    actuator.append(drive)
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='1000', offheight='800')
    ET.SubElement(world, 'light', pos='.2 -.2 -.4', dir='-.2 .2 .4')
    os.environ['ASSEMBLENIST_SDF_DIR'] = str(ROOT/'assets_local/sdf')
    mujoco.mj_loadPluginLibrary(str(ROOT/'plugin/build/libassemblenist_sdf.so'))
    mujoco.mj_loadPluginLibrary(str(ROOT/'plugin/build/libso101_driver.so'))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))


def run_target(model, target):
    data = mujoco.MjData(model)
    qgrip = model.joint('gripper').qposadr[0]
    data.qpos[qgrip] = .4
    data.ctrl[0] = target
    # The initial tool is aligned against the fixed finger, without gravity.
    # Gravity is then applied to check retention, rather than infer a pickup.
    model.opt.gravity[:] = 0
    closure_button_max = 0.
    for _ in range(1200):
        mujoco.mj_step(model, data)
        closure_button_max = max(closure_button_max, data.qpos[model.joint('button_forward').qposadr[0]])
    model.opt.gravity[:] = [0, -9.81, 0]
    body = model.body('driver').id
    initial = data.xpos[body].copy()
    max_depth = 0.
    for _ in range(2000):
        mujoco.mj_step(model, data)
        max_depth = max(max_depth, max((-c.dist for c in data.contact), default=0))
    return data, dict(target_rad=target, target_encoder_steps=target/ENCODER_STEP,
                      actual_rad=float(data.qpos[qgrip]),
                      button_mm=float(data.qpos[model.joint('button_forward').qposadr[0]]*1000),
                      saddle_mm=float(data.qpos[model.joint('trigger_saddle').qposadr[0]]*1000),
                      spindle_rad_s=float(data.qvel[model.joint('spindle').dofadr[0]]),
                      gravity_drift_mm=float(np.linalg.norm(data.xpos[body]-initial)*1000),
                      motor_torque_nm=float(data.actuator_force[0]),
                      closure_button_max_mm=closure_button_max*1000,
                      max_contact_depth_mm=max_depth*1000,
                      retained=bool(np.linalg.norm(data.xpos[body]-initial)<.003))


def run_sequence(model, off_tick=145, on_tick=130):
    data, carry = run_target(model, off_tick*ENCODER_STEP)
    body = model.body('driver').id
    initial = data.xpos[body].copy()
    records = [dict(stage='carry_off', **carry)]
    for stage, tick in [('press_on', on_tick), ('release_off', off_tick)]:
        data.ctrl[0] = tick*ENCODER_STEP
        depth = 0.
        for _ in range(2000):
            mujoco.mj_step(model, data)
            depth = max(depth, max((-c.dist for c in data.contact), default=0))
        records.append(dict(stage=stage, target_encoder_steps=tick,
                            target_rad=tick*ENCODER_STEP,
                            button_mm=float(data.qpos[model.joint('button_forward').qposadr[0]]*1000),
                            spindle_rad_s=float(data.qvel[model.joint('spindle').dofadr[0]]),
                            gravity_drift_mm=float(np.linalg.norm(data.xpos[body]-initial)*1000),
                            max_contact_depth_mm=depth*1000))
    assert carry['retained'] and carry['button_mm'] < .6, records
    assert carry['closure_button_max_mm'] < .6, records
    assert records[1]['button_mm'] > .6 and records[1]['spindle_rad_s'] > 20, records
    assert records[2]['button_mm'] < .6 and abs(records[2]['spindle_rad_s']) < .1, records
    assert all(row['gravity_drift_mm'] < 3 for row in records), records
    assert all(row['max_contact_depth_mm'] < .3 for row in records), records
    before = data.xpos[body].copy()
    data.ctrl[0] = 220*ENCODER_STEP
    for _ in range(500):
        mujoco.mj_step(model, data)
    drop = float(np.linalg.norm(data.xpos[body]-before))
    assert drop > .05, drop
    return dict(passed=True, scope='aligned nominal gravity carry/press/release; not pickup or torque robustness',
                records=records, open_jaws_drop_m=drop,
                encoder_step_rad=ENCODER_STEP)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first-tick', type=int, default=130)
    parser.add_argument('--last-tick', type=int, default=200)
    parser.add_argument('--tick-stride', type=int, default=2)
    parser.add_argument('--sequence', action='store_true', help='check the nominal145→130→145 sequence')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    model = make_model()
    if args.sequence:
        report = run_sequence(model)
        print(json.dumps(report, indent=2))
    else:
        records = []
        for tick in range(args.first_tick, args.last_tick+1, args.tick_stride):
            _, record = run_target(model, tick*ENCODER_STEP)
            records.append(record)
            print(json.dumps(record), flush=True)
        report = dict(scope='fixed actual palm; aligned free tool; quantized position commands',
                      encoder_step_rad=ENCODER_STEP, records=records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
