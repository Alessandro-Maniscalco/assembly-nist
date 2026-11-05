"""Qualify prepared horn/retainer SGS holds on the unchanged native workcell.

The part begins at its physically settled predecessor pose. The helper's open
hand is initialized beside it; every subsequent grasp/withdrawal uses actuators.
This is curriculum-reset preparation, not assembly from loose parts.
"""
import argparse
import json
from pathlib import Path
import signal

import mujoco
import numpy as np

from .fastener import origins
from .resets import TASK_NAMES, reset
from .robot import down_rotation, ik
from .scene import ROOT, metadata, model
from .trajectory import STATE_SPEC, Trajectory


def generate(task, seed, offset, grip, out, withdraw=True, release_check=False, return_check=False):
    m = model()
    d = mujoco.MjData(m)
    predecessor = {'horn': 'screw', 'retainer': 'yoke'}[task]
    mujoco.mj_setState(m, d, reset(m, predecessor, seed), STATE_SPEC)
    d.ctrl[:] = d.qpos[:12]
    mujoco.mj_forward(m, d)
    R = d.body('base').xmat.reshape(3, 3).copy()
    yaw = np.pi + np.arctan2(R[1, 0], R[0, 0])
    desired = d.body(task).xpos + R @ np.asarray(offset)
    d.qpos[6:11] = ik(m, d.qpos, 'right', desired, down_rotation(yaw))
    d.qpos[11] = grip
    d.qvel[6:12] = 0.
    d.ctrl[6:] = d.qpos[6:12]
    d.time = 0.
    mujoco.mj_forward(m, d)
    initial_penetration = max(0., -min(d.contact.dist, default=0.))
    if initial_penetration > .0004:
        raise ValueError(f'Initial open hand overlap {initial_penetration*1000:.3f} mm')
    log = Trajectory(m, d)
    meta = metadata()
    reference = origins(meta)
    extent = np.ptp(np.asarray(meta[task]['bounds']), axis=0)/2
    kp = np.vstack((np.zeros(3), np.diag(extent)))
    initial_part = d.body(task).xpos.copy()
    initial_base = d.body('base').xpos.copy()
    checkpoints = []
    report = dict(task=task, seed=seed, prepared_sgs_reset=True, full_loose_assembly=False,
                  source_predecessor_reset=predecessor, prepared_grasp_offset_m=list(offset),
                  initial_gripper_position_rad=grip, initial_penetration_m=initial_penetration,
                  runtime_state_changes='robot actuator controls only', checkpoints=checkpoints)
    neighbor_contact = dict(steps=0, peak_normal_force_N=0., peak_penetration_m=0.)
    jaw_ids = {m.body('right_gripper').id, m.body('right_moving_jaw_so101_v1').id}
    neighbor_ids = {m.body(name).id for name in ('base', 'motor', 'horn', 'screw', 'retainer', 'yoke') if name != task}

    def checkpoint(label):
        mujoco.mj_forward(m, d)
        base, part = d.body('base'), d.body(task)
        Rb = base.xmat.reshape(3, 3)
        goal = base.xpos + Rb @ (reference[task]-reference['base'])
        error = float(np.linalg.norm(part.xpos+kp@part.xmat.reshape(3, 3).T-
                                     (goal+kp@Rb.T), axis=1).max())
        forces = {'right_gripper': 0., 'right_moving_jaw_so101_v1': 0.}
        adjacent = {}
        for i, contact in enumerate(d.contact):
            names = {m.body(int(m.geom_bodyid[g])).name for g in contact.geom}
            for jaw in names.intersection(forces):
                force = np.zeros(6)
                mujoco.mj_contactForce(m, d, i, force)
                if task in names:
                    forces[jaw] += float(force[0])
                else:
                    for other in names-{'right_gripper', 'right_moving_jaw_so101_v1'}:
                        if other in ('base', 'motor', 'horn', 'screw', 'retainer', 'yoke'):
                            adjacent[other] = adjacent.get(other, 0.)+float(force[0])
        row = dict(stage=label, state_index=len(log.states)-1, keypoint_error_m=error,
                   jaw_normal_forces_N=forces,
                   loaded_bilateral_grasp=min(forces.values()) > .001,
                   adjacent_part_normal_forces_N=adjacent,
                   clean_grasp=(not any(force > .001 for force in adjacent.values()) and
                                neighbor_contact['steps'] == 0),
                   part_lift_m=float(part.xpos[2]-initial_part[2]),
                   base_lift_m=float(base.xpos[2]-initial_base[2]))
        predecessor_errors = {}
        for name in TASK_NAMES[:TASK_NAMES.index(task)]:
            body = d.body(name)
            extent_prev = np.ptp(np.asarray(meta[name]['bounds']), axis=0)/2
            keypoints_prev = np.vstack((np.zeros(3), np.diag(extent_prev)))
            if name == 'screw':
                keypoints_prev = np.array([[0., 0., 0.], [0., 0., .003]])
            expected = base.xpos + Rb @ (reference[name]-reference['base'])
            predecessor_errors[name] = float(np.linalg.norm(
                body.xpos+keypoints_prev@body.xmat.reshape(3, 3).T-
                (expected+keypoints_prev@Rb.T), axis=1).max())
        row['predecessor_keypoint_errors_m'] = predecessor_errors
        checkpoints.append(row)
        print(json.dumps(row), flush=True)
        return row

    def advance(target, seconds, label):
        start = d.ctrl.copy()
        n = round(seconds/m.opt.timestep)
        for tick in range(n):
            u = (tick+1)/n
            d.ctrl[:] = start + (u*u*(3-2*u))*(target-start)
            log.step(label)
            adjacent_step = False
            for i, contact in enumerate(d.contact):
                ids = {int(m.geom_bodyid[g]) for g in contact.geom}
                if ids.intersection(jaw_ids) and ids.intersection(neighbor_ids):
                    force = np.zeros(6)
                    mujoco.mj_contactForce(m, d, i, force)
                    neighbor_contact['peak_normal_force_N'] = max(neighbor_contact['peak_normal_force_N'], float(force[0]))
                    neighbor_contact['peak_penetration_m'] = max(neighbor_contact['peak_penetration_m'], -float(contact.dist))
                    adjacent_step |= force[0] > .001
            neighbor_contact['steps'] += int(adjacent_step)
            if log.depths[-1] > .0004:
                raise RuntimeError(f'{label}: {log.depths[-1]*1000:.3f} mm penetration exceeds 0.4 mm')

    old_handler = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(RuntimeError('External termination')))
    try:
        target = d.ctrl.copy()
        target[11] = -.1
        advance(target, 1., f'Close native fingers around prepared {task}')
        advance(target, .5, f'Hold prepared {task}')
        row = checkpoint('Prepared native hold')
        if not (row['loaded_bilateral_grasp'] and row['clean_grasp']):
            raise RuntimeError('Stock fingers did not form a clean loaded bilateral grasp')
        report.update(held_state_index=row['state_index'], held_keypoint_error_m=row['keypoint_error_m'],
                      held_reset_qualified=True)
        if withdraw:
            start_position = d.site('right_gripperframe').xpos.copy()
            start_R = down_rotation(yaw)
            heights = [.0005, .001, .002, .004, .008, .012]
            if return_check:
                heights += list(reversed(heights[:-1]))+[0.]
            for step, height in enumerate(heights):
                direction = 'Withdraw' if step < 6 else 'Return'
                target[6:11] = ik(m, d.qpos, 'right', start_position+np.array([0., 0., height]), start_R)
                advance(target, .5, f'{direction} prepared {task} {height*1000:g} mm')
                advance(target, .15, f'Settle {task} withdrawal')
                row = checkpoint(f'{direction} {height*1000:g} mm')
                if not (row['loaded_bilateral_grasp'] and row['clean_grasp']):
                    raise RuntimeError('Grasp lost or neighboring component contacted during withdrawal')
        if release_check:
            target[11] = .75
            advance(target, 1., f'Release prepared {task}')
            site = d.site('right_gripperframe')
            target[6:11] = ik(m, d.qpos, 'right', site.xpos+np.array([0., 0., .015]))
            advance(target, 1., 'Clear open fingers')
            advance(target, .6, f'Check prepared {task} retention')
            row = checkpoint('Release endpoint')
            report['release_endpoint_qualified'] = (row['keypoint_error_m'] < .001 and
                max(row['jaw_normal_forces_N'].values()) <= .001 and
                max(row['predecessor_keypoint_errors_m'].values(), default=0.) < .001)
    except Exception as exc:
        report['failure'] = f'{type(exc).__name__}: {exc}'
    finally:
        signal.signal(signal.SIGTERM, old_handler)
    if not log.states:
        raise RuntimeError(report.get('failure', 'No physical steps'))
    if max(log.depths) > .0004 or log.max_clock_error > 1e-8 or np.any(np.asarray(d.warning.number)-log.warning_start):
        report['held_reset_qualified'] = False
    report.setdefault('held_reset_qualified', False)
    report['adjacent_part_contact_audit'] = neighbor_contact
    report['qualified_state_indices'] = ([r['state_index'] for r in checkpoints
                                         if r['loaded_bilateral_grasp'] and r['clean_grasp']]
                                        if report['held_reset_qualified'] else [])
    return log.save(Path(out), report)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=['horn', 'retainer'], default='retainer')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--offset', default='.0193,-.024,.004')
    p.add_argument('--grip', type=float, default=.8)
    p.add_argument('--hold-only', action='store_true')
    p.add_argument('--release-check', action='store_true')
    p.add_argument('--return-check', action='store_true')
    p.add_argument('--out', type=Path, default=ROOT/'artifacts_local/prepared_parts/retainer_front')
    a = p.parse_args()
    print(json.dumps(generate(a.task, a.seed, tuple(map(float, a.offset.split(','))), a.grip,
                              a.out, not a.hold_only, a.release_check, a.return_check)), flush=True)


if __name__ == '__main__':
    main()
