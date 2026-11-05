"""Experimental native motor assembly, with an optional re-executed loose pickup."""
import argparse
import json
import signal
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from .scene import JOINTS, ROOT, load_plugin, metadata
from .robot import grasp_transform, ik
from .fastener import origins
from .trajectory import STATE_SPEC, Trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--stage', default='Lift motor')
    parser.add_argument('--out', type=Path, default=ROOT/'artifacts_local/dual_full_motor')
    parser.add_argument('--prepared', action='store_true', help='Initialize at the saved held state; not a loose-start claim')
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    source = args.source.with_suffix('')
    receipt = json.loads(source.with_suffix('.json').read_text())
    saved = np.load(source.with_suffix('.npz'))
    index = next(s['state_index'] for s in receipt['snapshots'] if s['stage'] == args.stage)
    load_plugin()
    m = mujoco.MjModel.from_binary_path(str(source.with_suffix('.mjb')))
    d = mujoco.MjData(m)
    mujoco.mj_setState(m, d, saved['states'][index] if args.prepared else saved['initial_state'], STATE_SPEC)
    d.ctrl[:] = saved['controls'][index if args.prepared else 0]
    mujoco.mj_forward(m, d)
    log = Trajectory(m, d)
    meta = metadata()
    ref = origins(meta)
    offset = ref['motor']-ref['base']
    keypoints = np.vstack((np.zeros(3), np.diag(np.ptp(np.asarray(meta['motor']['bounds']), axis=0)/2)))
    qi = np.r_[np.arange(5), np.arange(6, 11)]
    ranges = m.jnt_range[[m.joint(side+'_'+joint).id for side in ('left', 'right') for joint in JOINTS[:5]]]
    goal_rotation = Rotation.from_euler('x', .24, degrees=True).as_matrix()
    report = dict(source=source.name, source_state_index=index,
                  source_config=receipt.get('config', {}),
                  actuator_force_limits_Nm=m.actuator_forcerange.tolist(),
                  prepared_physical_held_start=args.prepared,
                  prefix_controls_reexecuted=not args.prepared,
                  full_loose_assembly=False, task='motor',
                  distribution='Prepared held continuation' if args.prepared else 'Nominal loose-table initialization',
                  controller='Scripted coupled relative pose and upright table placement')
    snapshots = []
    failure = None
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(RuntimeError('External SIGTERM')))

    def quality(data=d):
        base = data.body('base')
        rb = base.xmat.reshape(3, 3)
        motor = data.body('motor')
        goal = base.xpos+rb@offset
        return float(np.linalg.norm(motor.xpos+keypoints@motor.xmat.reshape(3, 3).T
                                    -goal-keypoints@rb.T, axis=1).max())

    def pairs(data=d):
        return {frozenset(m.body(int(m.geom_bodyid[g])).name for g in c.geom) for c in data.contact}

    def both_held():
        contact_pairs = pairs()
        return all(frozenset((part, jaw)) in contact_pairs
                   for part, side in (('base', 'left'), ('motor', 'right'))
                   for jaw in (side+'_gripper', side+'_moving_jaw_so101_v1'))

    def tick(label):
        log.step(label)
        if log.depths[-1] > .0004:
            raise RuntimeError(f'{label}: penetration exceeds 0.4 mm')
        if np.max(np.abs(d.qvel[:12])) > 5:
            raise RuntimeError(f'{label}: robot joint speed exceeds 5 rad/s')

    def advance(ctrl, seconds, label):
        start = d.ctrl.copy()
        count = round(seconds/m.opt.timestep)
        for k in range(count):
            u = (k+1)/count
            d.ctrl[:] = start+u*u*(3-2*u)*(ctrl-start)
            tick(label)
        mujoco.mj_forward(m, d)
        held = both_held()
        error = quality()
        if held and error < report.get('best_held_keypoint_error_m', float('inf')):
            report.update(best_held_keypoint_error_m=error, best_held_state_index=len(log.states)-1)
        snapshots.append(dict(stage=label, state_index=len(log.states)-1, time=float(d.time),
                              both_bilateral_grasps=held, keypoint_error_m=error,
                              base=d.body('base').xpos.tolist(), motor=d.body('motor').xpos.tolist()))
        np.savez_compressed(args.out.with_suffix('.latest.npz'), state=log.state(), ctrl=d.ctrl, stage=label)
        print(snapshots[-1], flush=True)

    def solve(relative_position, anchor, base_yaw, *, target_rotation=goal_rotation,
              tilt_rotation=None, height=None):
        scratch = mujoco.MjData(m)
        scratch.qpos[:] = d.qpos
        grasps = [grasp_transform(d, 'left', 'base'), grasp_transform(d, 'right', 'motor')]
        seed = np.clip(d.qpos[qi], ranges[:, 0]+1e-6, ranges[:, 1]-1e-6)

        def residual(q):
            scratch.qpos[qi] = q
            mujoco.mj_kinematics(m, scratch)
            poses = []
            for side, (gp, gr) in zip(('left', 'right'), grasps):
                site = scratch.site(side+'_gripperframe')
                rs = site.xmat.reshape(3, 3)
                poses.append((site.xpos+rs@gp, rs@gr))
            (pb, rb), (pm, rm) = poses
            relative = np.r_[rb.T@(pm-pb)-relative_position,
                             .08*Rotation.from_matrix(rb.T@rm@target_rotation.T).as_rotvec()]
            if tilt_rotation is None:
                yaw_error = np.arctan2(rb[1, 0], rb[0, 0])-base_yaw
                support = np.r_[.15*(pb-anchor), .01*np.arctan2(np.sin(yaw_error), np.cos(yaw_error))]
            else:
                support = np.r_[.04*Rotation.from_matrix(rb@tilt_rotation.T).as_rotvec()[:2],
                                pb[2]-height, .01*(pb[:2]-anchor[:2])]
            return np.r_[relative, support, .0001*(q-seed)]

        fit = least_squares(residual, seed, bounds=(ranges[:, 0], ranges[:, 1]),
                            max_nfev=300, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        error = residual(fit.x)
        if np.linalg.norm(error[:3]) > .001 or np.linalg.norm(error[3:6])/.08 > .02:
            raise RuntimeError(f'Coupled relative IK exceeds bound: {error[:6]}')
        if tilt_rotation is not None and (np.linalg.norm(error[6:8])/.04 > .04 or abs(error[8]) > .001):
            raise RuntimeError(f'Coupled placement IK exceeds bound: {error[:9]}')
        ctrl = d.ctrl.copy()
        ctrl[qi] = fit.x
        return ctrl

    def audit_retention():
        scratch = mujoco.MjData(m)
        rows, positions, rotations = [], [], []
        motor_v = m.jnt_dofadr[m.joint('motor_free').id]
        for state, ctrl in zip(log.states[-800:], log.controls[-800:]):
            mujoco.mj_setState(m, scratch, state, STATE_SPEC)
            scratch.ctrl[:] = ctrl
            mujoco.mj_forward(m, scratch)
            contact_pairs = pairs(scratch)
            contacts = []
            for part in ('motor', 'base'):
                contacts.append(any(part in pair and any(name.startswith(('left_', 'right_', 'driver'))
                                                         for name in pair) for pair in contact_pairs))
            rows.append([quality(scratch), *contacts, max(0., -float(min(scratch.contact.dist, default=0.))),
                         np.linalg.norm(scratch.qvel[motor_v:motor_v+3]),
                         np.linalg.norm(scratch.qvel[motor_v+3:motor_v+6])])
            positions.append(scratch.body('motor').xpos.copy())
            rotations.append(scratch.body('motor').xmat.reshape(3, 3).copy())
        a = np.asarray(rows)
        translation = float(np.linalg.norm(np.asarray(positions)-positions[0], axis=1).max())
        angle = float(Rotation.from_matrix(np.asarray(rotations)@rotations[0].T).magnitude().max())
        passed = (len(rows) == 800 and a[:, 0].max() < .001 and not a[:, 1:3].any()
                  and a[:, 3].max() <= .0004 and translation <= .00025 and angle <= .01)
        return dict(physics_steps=len(rows), duration_s=len(rows)*m.opt.timestep,
                    max_keypoint_error_m=float(a[:, 0].max()), motor_hand_tool_contact_steps=int(a[:, 1].sum()),
                    base_hand_tool_contact_steps=int(a[:, 2].sum()), max_penetration_m=float(a[:, 3].max()),
                    max_world_translation_m=translation, max_world_rotation_rad=angle,
                    max_linear_speed_m_s=float(a[:, 4].max()), max_angular_speed_rad_s=float(a[:, 5].max()),
                    v2_pose_gate_passed=bool(passed))

    try:
        if not args.prepared:
            for k in range(index+1):
                d.ctrl[:] = saved['controls'][k]
                tick(str(saved['labels'][saved['stage'][k]]))
            report['prefix_max_state_difference'] = float(np.max(np.abs(log.state()-saved['states'][index])))
            mujoco.mj_forward(m, d)
        if not both_held():
            raise RuntimeError('Pickup endpoint lacks both native bilateral grasps')
        base_start = d.body('base').xpos.copy()
        site = d.site('left_gripperframe')
        ctrl = d.ctrl.copy()
        ctrl[:5] = ik(m, d.qpos, 'left', site.xpos+[0, 0, .06], site.xmat.reshape(3, 3).copy())
        advance(ctrl, 2., 'Lift loose base while retaining the motor grasp')
        report['base_lift_m'] = float(d.body('base').xpos[2]-base_start[2])
        if report['base_lift_m'] < .04 or not both_held():
            raise RuntimeError('Base did not lift through its native grasp')
        base = d.body('base')
        rb = base.xmat.reshape(3, 3)
        yaw = np.arctan2(rb[1, 0], rb[0, 0])
        initial_relative = rb.T@(d.body('motor').xpos-base.xpos)
        anchor = base.xpos.copy()+[.03, .025, 0.]
        advance(solve(offset+[initial_relative[0], -.070, 0.], anchor, yaw), 2.,
                'Move held motor in front of suspended base')
        advance(solve(offset+[0., -.070, 0.], anchor, yaw), 2.,
                'Align motor ahead of base opening')
        for gap in np.r_[np.linspace(.065, .020, 10), np.linspace(.019, 0., 20)]:
            fraction = gap/.065
            anchor = np.array([.06-.03*fraction, -.04+gap, .125-(.125-base_start[2]-.06)*fraction])
            advance(solve(offset+[.0001, -gap, 0.], anchor, yaw), .18,
                    'Insert motor by coordinating both arms')
        history = []
        for _ in range(20):
            if quality() < .0003:
                break
            advance(solve(offset+[.0001, 0., 0.], anchor, yaw), .2, 'Finish seating through native contacts')
            history.append(quality())
            if len(history) >= 8 and min(history[-4:]) >= min(history[-8:-4])-1e-5:
                break
        report['held_keypoint_error_m'] = quality()
        report['held_state_index'] = len(log.states)-1
        if quality() > .001:
            raise RuntimeError('Motor did not reach its held goal before table placement')
        base_start = d.body('base').xpos.copy()
        rb = d.body('base').xmat.reshape(3, 3).copy()
        yaw = np.arctan2(rb[1, 0], rb[0, 0])
        orientation_path = Slerp([0, 1], Rotation.from_matrix(np.array([rb, Rotation.from_euler('z', yaw).as_matrix()])))

        def placement(fraction, height=None):
            target = (1-fraction)*base_start+fraction*np.array([.03, -.04, .038])
            return solve(offset+[.0001, 0., 0.], target, yaw,
                         tilt_rotation=orientation_path(fraction).as_matrix(),
                         height=target[2] if height is None else height)

        for fraction in np.linspace(.04, 1., 25):
            advance(placement(fraction), .25, 'Place both held parts upright on table')
        for height in np.arange(.03775, .0339, -.00025):
            if frozenset(('base', 'world')) in pairs():
                break
            advance(placement(1., height), .12, 'Find native base table contact')
        if frozenset(('base', 'world')) not in pairs():
            raise RuntimeError('Base did not contact the table')
        report['table_contact_state_index'] = len(log.states)-1
        ctrl = d.ctrl.copy(); ctrl[11] = .75
        advance(ctrl, .8, 'Release seated motor above supported base')
        point = d.site('right_gripperframe').xpos+d.body('base').xmat.reshape(3, 3)@np.array([.012, 0., 0.])
        ctrl = d.ctrl.copy(); ctrl[6:11] = ik(m, d.qpos, 'right', point)
        advance(ctrl, .8, 'Move open right hand outward')
        point = d.site('right_gripperframe').xpos+[0., 0., .05]
        ctrl = d.ctrl.copy(); ctrl[6:11] = ik(m, d.qpos, 'right', point)
        advance(ctrl, 1., 'Raise empty right hand')
        ctrl = d.ctrl.copy(); ctrl[5] = .75
        advance(ctrl, 1., 'Release base supported by table')
        point = d.site('left_gripperframe').xpos+[0., 0., .06]
        ctrl = d.ctrl.copy(); ctrl[:5] = ik(m, d.qpos, 'left', point)
        advance(ctrl, 1.5, 'Withdraw left hand')
        advance(d.ctrl.copy(), 1., 'Check assembly after both hands release')
        report['fine_retention_check'] = audit_retention()
        if not report['fine_retention_check']['v2_pose_gate_passed']:
            raise RuntimeError('Independent fine-step retention gate failed')
    except (Exception, KeyboardInterrupt) as exc:
        failure = f'{type(exc).__name__}: {exc}'
        print(failure, flush=True)
    finally:
        report.update(failure=failure, completed=failure is None,
                      full_loose_assembly=failure is None and not args.prepared,
                      final_keypoint_error_m=quality(), snapshots=snapshots)
        log.save(args.out, report)
        print({k: v for k, v in report.items() if k != 'snapshots'}, flush=True)
    if failure:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
