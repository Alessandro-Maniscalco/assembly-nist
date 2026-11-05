"""Replay a hashed control recipe without reloading intermediate physical states."""
import argparse
import hashlib
import json
from pathlib import Path
import signal

import mujoco
import numpy as np

from .scene import ROOT, load_plugin
from .trajectory import STATE_SPEC, Trajectory


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay(recipe_path, output, scope):
    recipe_path, output = Path(recipe_path), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    recipe = json.loads(recipe_path.read_text())
    root = recipe_path.parent
    segments = recipe['segments']
    plugin_hashes = {p.name: sha256(p) for p in (ROOT/'plugin/build').glob('*.so')}
    model_hash, previous = None, None
    for segment in segments:
        source = root/segment['source']
        for suffix, key in [('.npz', 'trajectory_sha256'), ('.json', 'report_sha256')]:
            if sha256(source.with_suffix(suffix)) != segment[key]:
                raise ValueError(f'Changed source artifact: {source}{suffix}')
        report = json.loads(source.with_suffix('.json').read_text())
        model_hash = model_hash or report['model_sha256']
        if report['model_sha256'] != model_hash or report['plugin_sha256'] != plugin_hashes:
            raise ValueError(f'Model/plugin identity mismatch: {source}')
        with np.load(source.with_suffix('.npz')) as data:
            count = segment['used_states']
            if not 0 < count <= len(data['states']):
                raise ValueError(f'Invalid selected segment length: {source}')
            if previous is not None and not np.array_equal(data['initial_state'], previous):
                raise ValueError(f'Discontinuous source physical state: {source}')
            previous = data['states'][count-1].copy()
    first = root/segments[0]['source']
    if sha256(first.with_suffix('.mjb')) != model_hash:
        raise ValueError('Archived model hash mismatch')
    load_plugin()
    model = mujoco.MjModel.from_binary_path(str(first.with_suffix('.mjb')))
    data = mujoco.MjData(model)
    initial = np.load(first.with_suffix('.npz'))
    # The source recipe begins with a constant-control settling stage.
    if not np.array_equal(initial['controls'][0], initial['controls'][1]):
        raise ValueError('Initial control must be supplied by a constant settling prefix')
    mujoco.mj_setState(model, data, initial['initial_state'], STATE_SPEC)
    data.ctrl[:] = initial['controls'][0]
    mujoco.mj_forward(model, data)
    log = Trajectory(model, data)
    details = dict(task='motor', prepared_sgs_reset=scope == 'prepared',
                   full_loose_motor_assembly=False, full_five_stage_assembly=False,
                   learned_policy=False, uninterrupted_replay=True,
                   intermediate_state_reloads=0, replay_completed=False,
                   distribution=f'{scope.capitalize()} initial state; uninterrupted scripted motor sequence; experimental 1 Nm grips',
                   recipe_file=recipe_path.name, recipe_sha256=sha256(recipe_path),
                   selected_segments=segments, boundary_deviation=[],
                   closeup_camera=dict(azimuth=180, elevation=-25, distance=.30, offset=[0, 0, .005]))
    normal_forces, push_forces = [], []
    max_joint_speed = 0.
    warning_start = data.warning.number.copy()

    def interrupted(*_):
        raise RuntimeError('External SIGTERM')

    signal.signal(signal.SIGTERM, interrupted)
    try:
        for segment in segments:
            source = root/segment['source']
            with np.load(source.with_suffix('.npz')) as archived:
                count = segment['used_states']
                controls, labels, stages = archived['controls'], archived['labels'], archived['stage']
                for i in range(count):
                    label = str(labels[stages[i]])
                    data.ctrl[:] = controls[i]
                    log.step(label)
                    force = 0.
                    for k, contact in enumerate(data.contact):
                        names = [model.body(int(model.geom_bodyid[g])).name for g in contact.geom]
                        if 'motor' in names and any(n.startswith('right_') for n in names):
                            wrench = np.zeros(6)
                            mujoco.mj_contactForce(model, data, k, wrench)
                            force += float(wrench[0])
                    normal_forces.append(force)
                    pushing = label == 'Push motor through native normal contact'
                    push_forces.append(force if pushing else 0.)
                    speed = float(np.max(np.abs(data.qvel[:12])))
                    max_joint_speed = max(max_joint_speed, speed)
                    if log.depths[-1] > .0004:
                        raise RuntimeError('Native contact depth exceeded 0.4 mm')
                    if pushing and force > 20:
                        raise RuntimeError('Side-face pushing load exceeded 20 N')
                    if speed > 5:
                        raise RuntimeError('Robot joint speed exceeded 5 rad/s')
                    if np.any(data.warning.number != warning_start):
                        raise RuntimeError('MuJoCo warning during replay')
                    if (i+1) % 10000 == 0:
                        print(f'{source.name}: {i+1}/{count} steps', flush=True)
                expected = archived['states'][count-1]
                row = dict(segment=source.name, state_index=len(log.states)-1,
                           max_fullphysics_state_difference=float(np.max(np.abs(log.states[-1]-expected))),
                           max_qpos_difference=float(np.max(np.abs(log.states[-1][1:1+model.nq]-expected[1:1+model.nq]))))
                details['boundary_deviation'].append(row)
                print(row, flush=True)
        mujoco.mj_forward(model, data)
        if max(0., -float(min(data.contact.dist, default=0))) > .0004:
            raise RuntimeError('Final refreshed native contact depth exceeded 0.4 mm')
        details['replay_completed'] = True
    except Exception as exc:
        details['failure'] = str(exc)
        print(exc, flush=True)
    finally:
        details.update(max_robot_joint_speed_rad_s=max_joint_speed,
                       max_right_motor_normal_force_N=max(normal_forces, default=0.),
                       max_side_push_normal_force_N=max(push_forces, default=0.))
        np.savez_compressed(output.with_suffix('.forces.npz'),
                            right_motor_normal_force_N=np.asarray(normal_forces),
                            side_push_normal_force_N=np.asarray(push_forces))
        log.save(output, details)
    return details


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('recipe', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--scope', required=True, choices=['loose', 'prepared'])
    args = parser.parse_args()
    result = replay(args.recipe, args.out, args.scope)
    print(json.dumps({k: v for k, v in result.items() if k not in ('selected_segments', 'boundary_deviation')}, indent=2))
