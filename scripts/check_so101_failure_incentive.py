"""One fixed loose policy episode and its zero-action control, on native physics."""
import ctypes
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts_local/failure_incentive'
POLICY = ROOT / 'runs/so101_gpu_tol1e6_20261105/reviews/01/checkpoint.pt'
CONFIG = POLICY.parents[2] / 'config.json'
DEADLINE = time.monotonic() + 560


def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')


def resource_check():
    rss = psutil.Process().memory_info().rss
    if rss > 2.5 * 1024**3:
        raise MemoryError(f'RSS limit exceeded: {rss}')
    if time.monotonic() > DEADLINE:
        raise TimeoutError('Bounded diagnostic deadline reached')
    return rss


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text())
    from assembly_policy.so101.env import BatchedAssemblyEnv, EnvConfig, HISTORY, PROPRIO_DIM, resolve_tasks
    from assembly_policy.so101.resets import signature
    tasks = resolve_tasks(config['tasks'])
    env = BatchedAssemblyEnv(tasks, [0], EnvConfig(**{**config['env'], 'nthread': 1}))
    gc.collect()
    ctypes.CDLL('libc.so.6').malloc_trim(0)
    import torch
    from assembly_policy.rl.nets import ActorCritic
    from assembly_policy.so101.obs import PointTables
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    ck = torch.load(POLICY, map_location='cpu', weights_only=False)
    cfg = ck['args']
    assert json.loads(json.dumps(cfg['env'])) == config['env']
    assert cfg['model_signature'] == signature()
    with patch('assembly_policy.so101.obs.model', return_value=env.m):
        pts = PointTables(tasks, cfg['n_part'], cfg['n_fix'], cfg['n_grip'])
    net = ActorCritic(pts.n_pts, HISTORY*PROPRIO_DIM, 12)
    net.load_state_dict(ck['model'])
    net.eval()
    bank_path = Path(cfg['bank_dir']) / 'motor_eval.npz'
    with np.load(bank_path) as bank:
        initial = bank['states'][0].copy()
    result = dict(checkpoint=str(POLICY), checkpoint_sha256=sha(POLICY), checkpoint_iteration=ck['it'],
                  checkpoint_transitions=ck['transitions'], checkpoint_args=cfg,
                  script_sha256=sha(__file__), evaluation_backend='canonical native',
                  bank_path=str(bank_path), bank_sha256=sha(bank_path), bank_entry=0,
                  initial_state_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
                  task='motor', policy_sampling_seed=501, episodes=[],
                  scope='Exactly one fixed sampled-policy episode and one zero-action baseline. No replacements or searches.')
    del ck
    gc.collect()
    ctypes.CDLL('libc.so.6').malloc_trim(0)
    native = env.engine

    class Recorder:
        def rollout(self, *args, **kwargs):
            self.pre_state = env.state[0].copy()
            trajectory, sensors = native.rollout(*args, **kwargs)
            self.trajectory = trajectory[0].copy()
            self.sensors = sensors[0].copy()
            return trajectory, sensors

        def close(self):
            native.close()

    recorder = Recorder()
    env.engine = recorder
    audit = mujoco.MjData(env.m)

    def measurements():
        velocity = recorder.trajectory[:, 1+env.m.nq:1+env.m.nq+env.m.nv]
        robot_speed = np.abs(velocity[:, :12]).max(axis=1)
        linear = np.asarray([np.linalg.norm(velocity[:, v:v+3], axis=1) for v in env.part_v]).T
        angular = np.asarray([np.linalg.norm(velocity[:, v+3:v+6], axis=1) for v in env.part_v]).T
        sensor_depth = -recorder.sensors[:, 0]
        final_depth = max(0., -min(env.data[0].contact.dist, default=0.))
        flags = dict(robot_joint_velocity=bool((robot_speed > 5).any()),
                     free_part_linear_velocity=bool((linear > 1).any()),
                     free_part_angular_velocity=bool((angular > 30).any()),
                     physics_contact_depth=bool((sensor_depth > env.ecfg.max_contact_depth).any()),
                     final_contact_depth=bool(final_depth > env.ecfg.max_contact_depth))
        first = []
        for key, mask, preintegration in (
            ('robot_joint_velocity', robot_speed > 5, False),
            ('free_part_linear_velocity', (linear > 1).any(1), False),
            ('free_part_angular_velocity', (angular > 30).any(1), False),
            ('physics_contact_depth', sensor_depth > env.ecfg.max_contact_depth, True),
        ):
            indices = np.flatnonzero(mask)
            if len(indices):
                j = int(indices[0])
                state = (recorder.pre_state if j == 0 else recorder.trajectory[j-1]) if preintegration else recorder.trajectory[j]
                first.append((float(state[0]), key, j, state))
        if flags['final_contact_depth']:
            first.append((float(env.state[0, 0]), 'final_contact_depth', env.substeps, env.state[0]))
        detail = None
        if first:
            at, reason, substep, state = sorted(first, key=lambda x: x[0])[0]
            mujoco.mj_setState(env.m, audit, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            audit.ctrl[:] = env.last_ctrl[0]
            mujoco.mj_forward(env.m, audit)
            pairs = sorted(audit.contact, key=lambda c: c.dist)
            contacts = [dict(geoms=[env.m.geom(int(g)).name for g in c.geom],
                             bodies=[env.m.body(int(env.m.geom_bodyid[g])).name for g in c.geom],
                             depth_m=float(-c.dist)) for c in pairs[:8]]
            detail = dict(time_s=at, reason=reason, substep=substep, deepest_contacts=contacts,
                          contact_recomputation='Same recorded state and controls; mj_forward for pair identity')
        return dict(flags=flags, max_robot_speed_rad_s=float(robot_speed.max()),
                    max_part_linear_speed_m_s=float(linear.max()), max_part_angular_speed_rad_s=float(angular.max()),
                    max_sensor_depth_m=float(sensor_depth.max()), final_depth_m=final_depth,
                    first_violation=detail)

    try:
        for mode in ('sampled_policy', 'zero_action'):
            env.set_states([0], initial[None], [tasks.index('motor')])
            torch.manual_seed(501)
            episode = dict(mode=mode, initial_error_m=float(env.assembly_distance()[0]),
                           control_trace=[], accumulated_reward=0., discounted_reward=0.)
            blocks, sensors, actions, controls = [], [], [], []
            for step in range(env.horizon):
                rss = resource_check()
                if mode == 'sampled_policy':
                    o = {k: torch.as_tensor(v) for k, v in env.observe().items()}
                    with torch.no_grad():
                        pc = pts(*[o[k] for k in ('task', 'tcp_p', 'tcp_R', 'part_p', 'part_q', 'finger')])
                        action = net.dist(pc, o['proprio']).sample().numpy()
                else:
                    action = np.zeros((1, 12))
                reward, done, info = env.step(action)
                cost = float(info['cost'][0])
                episode['accumulated_reward'] += float(reward[0])
                episode['discounted_reward'] += cfg['gamma']**step * float(reward[0])
                v = recorder.trajectory[-1, 1+env.m.nq:1+env.m.nq+12]
                torque = env.data[0].actuator_force
                cost_terms = dict(action=float(env.ecfg.cost_coef*np.square(np.asarray(action, float)).sum()),
                                  joint_velocity=float(env.ecfg.cost_coef*np.square(v).sum()/25),
                                  torque=float(env.ecfg.cost_coef*np.square(torque/env.m.actuator_forcerange[:, 1]).sum()))
                assert np.isclose(sum(cost_terms.values()), cost, atol=1e-12, rtol=1e-9)
                episode['control_trace'].append(dict(step=step+1, time_s=float(env.state[0, 0]),
                    reward=float(reward[0]), cost=cost, cost_terms=cost_terms, d_asm_m=float(info['d_asm'][0])))
                blocks.append(recorder.trajectory)
                sensors.append(recorder.sensors)
                actions.append(action[0].copy())
                controls.append(env.last_ctrl[0].copy())
                if done[0]:
                    episode.update({k: bool(info[k][0]) for k in ('success', 'safety', 'timeout')})
                    episode.update(terminal_step=step+1, terminal_time_s=float(env.state[0, 0]),
                                   final_error_m=float(info['d_asm'][0]), safety_measurements=measurements(), rss_bytes=rss)
                    print(json.dumps({'terminal': {k:v for k,v in episode.items() if k != 'control_trace'}}), flush=True)
                    break
            costs = [x['cost'] for x in episode['control_trace']]
            preterminal = costs[:-1] or costs
            remaining = env.horizon-len(costs)
            estimated_mean = float(np.mean(preterminal))
            episode['avoided_cost_estimate'] = dict(remaining_controls=remaining,
                mean_own_preterminal_cost=estimated_mean,
                undiscounted=remaining*estimated_mean,
                discounted_from_start=float(sum(cfg['gamma']**t for t in range(len(costs), env.horizon))*estimated_mean),
                assumption='Hypothetical unsuccessful continuation at its own mean preterminal cost; not a counterfactual rollout or proof of learned intent.')
            episode['best_error_m'] = min([episode['initial_error_m']] + [x['d_asm_m'] for x in episode['control_trace']])
            episode['progress_m'] = episode['initial_error_m']-episode['final_error_m']
            np.savez_compressed(OUT/f'{mode}.npz', initial_state=initial, physics_states=np.asarray(blocks),
                                sensors=np.asarray(sensors), actions=np.asarray(actions), controls=np.asarray(controls))
            result['episodes'].append(episode)
            write('results.json', result)
            blocks.clear(); sensors.clear(); gc.collect()
        policy, zero = result['episodes']
        result['comparison'] = dict(
            policy_minus_zero_return=policy['accumulated_reward']-zero['accumulated_reward'],
            policy_minus_zero_discounted_return=policy['discounted_reward']-zero['discounted_reward'],
            policy_preterminal_cost_horizon_return_estimate=policy['accumulated_reward']-policy['avoided_cost_estimate']['undiscounted'],
            interpretation='Termination avoids further negative cost because safety has no terminal penalty. The two observed returns and own-cost estimate test the incentive, not the causal reason training found the behavior; the zero action policy differs in effort.')
        result['completed'] = True
        write('results.json', result)
        print(json.dumps({'comparison': result['comparison']}), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
