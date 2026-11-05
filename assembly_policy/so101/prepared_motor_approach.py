"""Prepared SGS motor withdrawal/reinsertion through native actuator contact.

The initial state comes from a qualified, seated motor grasp. It is an explicit
prepared reset, never evidence of assembly from loose parts. After loading that
state this module changes only robot control targets and advances mj_step.
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal

import mujoco
import numpy as np

from .prepared import measure
from .fastener import origins
from .robot import down_rotation, ik
from .scene import ROOT, metadata, model
from .trajectory import STATE_SPEC, Trajectory


def generate(source, out, gaps=(.0005, .001, .002, .003, .005, .008, .010), seconds=.6,
             support=False, source_index=None, refine=False, insert_only=False):
    source, out = Path(source), Path(out)
    source_report = json.loads(source.with_suffix('.json').read_text())
    if source_index is None and not source_report.get('held_reset_qualified'):
        raise ValueError('The source motor grasp is not qualified')
    if (any(source_report['warnings']) or source_report['max_contact_penetration_m'] > .0004 or
            source_report['max_clock_step_error_s'] > 1e-8):
        raise ValueError('The source trajectory failed physical qualification')
    m = model()
    binary = np.empty(mujoco.mj_sizeModel(m), np.uint8)
    mujoco.mj_saveModel(m, buffer=binary)
    model_hash = hashlib.sha256(binary.tobytes()).hexdigest()
    if model_hash != source_report['model_sha256']:
        raise ValueError('Source prepared state and production model differ')
    source_index = source_report['held_state_index'] if source_index is None else source_index
    if source_index < 0:
        raise ValueError('Source state index must be nonnegative')
    with np.load(source.with_suffix('.npz')) as archive:
        state = archive['states'][source_index]
        controls = archive['controls'][source_index]
    d = mujoco.MjData(m)
    mujoco.mj_setState(m, d, state, STATE_SPEC)
    d.ctrl[:] = controls
    mujoco.mj_forward(m, d)
    if not measure(m, d)[1]:
        raise ValueError('Initial motor does not contact both native jaw bodies')
    if support:
        # The optional support arm is also prepared before the recorded reset.
        # Its descent and all later movement are native actuator executions.
        support_point = d.body('base').xpos + d.body('base').xmat.reshape(3, 3) @ np.array([-.05, .035, -.018])
        high = support_point.copy()
        high[2] = .08
        d.qpos[:5] = ik(m, d.qpos, 'left', high, down_rotation(2.5))
        d.qpos[5] = -.1
        d.qvel[:6] = 0.
        d.ctrl[:6] = d.qpos[:6]
        mujoco.mj_forward(m, d)
    initial_penetration = max(0., -min(d.contact.dist, default=0.))
    if initial_penetration > .0004:
        raise ValueError(f'Prepared initial overlap {initial_penetration:.6f} m')
    log = Trajectory(m, d)
    if log.plugin_hashes != source_report['plugin_sha256']:
        raise ValueError('Source prepared state and native plugins differ')
    origin = d.site('right_gripperframe').xpos.copy()
    orientation = d.site('right_gripperframe').xmat.reshape(3, 3).copy()
    direction = d.body('base').xmat.reshape(3, 3) @ np.array([0., -1., 0.])
    initial_base = d.body('base').xpos.copy()
    initial_motor = d.body('motor').xpos.copy()
    checkpoints = []
    report = dict(prepared_sgs_reset=True, full_loose_assembly=False,
                  runtime_state_changes='robot actuator controls only', task='motor',
                  source_trajectory=str(source.relative_to(ROOT)),
                  source_state_index=int(source_index),
                  prepared_left_support_above_base=bool(support),
                  initial_penetration_m=float(initial_penetration),
                  initial_motor_keypoint_error_m=measure(m, d)[0],
                  checkpoints=checkpoints)

    def checkpoint(label, gap):
        mujoco.mj_forward(m, d)
        error, bilateral, partners = measure(m, d)
        jaw_forces = {'right_gripper': 0., 'right_moving_jaw_so101_v1': 0.}
        for i, contact in enumerate(d.contact):
            names = {m.body(int(m.geom_bodyid[g])).name for g in contact.geom}
            if 'motor' not in names:
                continue
            for jaw in names.intersection(jaw_forces):
                force = np.zeros(6)
                mujoco.mj_contactForce(m, d, i, force)
                jaw_forces[jaw] += float(force[0])
        row = dict(stage=label, state_index=len(log.states)-1,
                   requested_withdrawal_m=gap, keypoint_error_m=error,
                   bilateral_motor_grasp=bilateral, contact_partners=partners,
                   jaw_normal_forces_N=jaw_forces,
                   loaded_bilateral_motor_grasp=bilateral and min(jaw_forces.values()) > .001,
                   motor_motion_m=float(np.linalg.norm(d.body('motor').xpos-initial_motor)),
                   base_motion_m=float(np.linalg.norm(d.body('base').xpos-initial_base)),
                   motor_linear_speed_m_s=float(np.linalg.norm(
                       d.qvel[m.jnt_dofadr[m.joint('motor_free').id]:][:3])))
        checkpoints.append(row)
        print(json.dumps(row), flush=True)

    def advance(target, duration, label):
        start = d.ctrl.copy()
        count = round(duration/m.opt.timestep)
        for tick in range(count):
            u = (tick+1)/count
            d.ctrl[:] = start + (u*u*(3-2*u))*(target-start)
            log.step(label)
            if log.depths[-1] > .0004:
                raise RuntimeError(f'{label}: penetration {log.depths[-1]*1000:.3f} mm exceeds 0.4 mm')

    def move(gap, label):
        target = d.ctrl.copy()
        target[6:11] = ik(m, d.qpos, 'right', origin + gap*direction, orientation)
        advance(target, seconds, label)
        advance(target, .15, label+' settle')
        checkpoint(label, gap)

    old_handler = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(RuntimeError('External termination')))
    try:
        if support:
            target = d.ctrl.copy()
            target[:5] = ik(m, d.qpos, 'left', support_point, down_rotation(2.5))
            advance(target, 2., 'Lower supporting fingers onto base foot')
            advance(target, .4, 'Settle physical base foot support')
            checkpoint('Prepared contact support', 0.)
        advance(d.ctrl.copy(), .1, 'Retain prepared seated grasp')
        checkpoint('Prepared initial hold', 0.)
        if not insert_only:
            for gap in gaps:
                move(gap, f'Withdraw motor {gap*1000:g} mm')
            for gap in [*reversed(gaps[:-1]), 0.]:
                move(gap, f'Reinsert motor to {gap*1000:g} mm gap')
        if refine:
            reference = origins(metadata())
            for attempt in range(10):
                error, held, _ = measure(m, d)
                if error < .0005 or not held:
                    break
                base_R = d.body('base').xmat.reshape(3, 3)
                goal = d.body('base').xpos + base_R @ (reference['motor']-reference['base'])
                delta = goal-d.body('motor').xpos
                delta *= min(1., .001/max(np.linalg.norm(delta), 1e-12))
                site = d.site('right_gripperframe')
                aligned = base_R @ d.body('motor').xmat.reshape(3, 3).T @ site.xmat.reshape(3, 3)
                target = d.ctrl.copy()
                target[6:11] = ik(m, d.qpos, 'right', site.xpos+delta, aligned, orientation_axes=(0, 2))
                advance(target, .5, 'Feedback insertion correction')
                checkpoint(f'Feedback insertion correction {attempt+1}', 0.)
        target = d.ctrl.copy()
        target[11] = .75
        advance(target, 1., 'Release reinserted motor')
        if refine:
            site = d.site('right_gripperframe')
            # Once open, the empty hand needs only a clearance position. A
            # prescribed six-dimensional pose would overconstrain this 5-DOF arm.
            target[6:11] = ik(m, d.qpos, 'right', site.xpos+np.array([0., 0., .015]))
            advance(target, 1., 'Clear open fingers above motor')
        advance(target, .6, 'Check retention after release')
        checkpoint('Released retention', 0.)
        last = checkpoints[-1]
        report['released_retention_qualified'] = (
            last['keypoint_error_m'] < .001 and
            not any(p.startswith(('left_', 'right_')) for p in last['contact_partners']))
    except Exception as exc:
        report['failure'] = f'{type(exc).__name__}: {exc}'
    finally:
        signal.signal(signal.SIGTERM, old_handler)
    if log.states:
        qualified = [row for row in checkpoints if row['state_index'] >= 240 and
                     row['loaded_bilateral_motor_grasp'] and
                     row['keypoint_error_m'] <= .010]
        valid_physics = (max(log.depths) <= .0004 and log.max_clock_error <= 1e-8 and
                         not np.any(np.asarray(d.warning.number)-log.warning_start))
        report['held_reset_qualified'] = bool(qualified and valid_physics)
        report['qualified_state_indices'] = [row['state_index'] for row in qualified] if valid_physics else []
        if report['held_reset_qualified']:
            held = max(qualified, key=lambda row: row['keypoint_error_m'])
            report['held_state_index'] = held['state_index']
            report['held_keypoint_error_m'] = held['keypoint_error_m']
        return log.save(out, report)
    raise RuntimeError(report.get('failure', 'No physical steps recorded'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', default='0')
    p.add_argument('--gaps-mm', default='.5,1,2,3,5,8,10')
    p.add_argument('--seconds', type=float, default=.6)
    p.add_argument('--support', action='store_true')
    p.add_argument('--refine', action='store_true')
    p.add_argument('--insert-only', action='store_true')
    p.add_argument('--source', type=Path)
    p.add_argument('--state-index', type=int)
    p.add_argument('--out', type=Path, default=ROOT/'artifacts_local/prepared_motor_approach')
    a = p.parse_args()
    gaps = tuple(float(x)/1000 for x in a.gaps_mm.split(','))
    if not all(0 < x <= .012 for x in gaps) or any(y <= x for x, y in zip(gaps, gaps[1:])):
        raise ValueError('Withdrawal gaps must increase within 0–12 mm')
    for seed in ([None] if a.source else map(int, a.seeds.split(','))):
        stem = a.source.stem if a.source else f'motor_goal_{seed:06d}'
        source = a.source.resolve() if a.source else ROOT/'artifacts_local/prepared_holds'/stem
        report = generate(source, a.out/stem, gaps, a.seconds, a.support,
                          a.state_index, a.refine, a.insert_only)
        print(json.dumps({k:v for k,v in report.items() if k != 'checkpoints'}), flush=True)


if __name__ == '__main__':
    main()
