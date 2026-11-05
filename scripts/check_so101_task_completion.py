#!/usr/bin/env python3
"""Check actual horn/retainer completion, with archived states and negative cases.

Generate the input holds with assembly_policy.so101.prepared_parts. This checker
does not alter training, banks, or physics assets. Falling episodes start above
all helper geometry and end at safety; they never start with overlapping arms.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from assembly_policy.so101.env import BatchedAssemblyEnv, EnvConfig, TASK_NAMES
from assembly_policy.so101.resets import reset
from assembly_policy.so101.scene import ROOT, model


def digest_model(m):
    binary = np.empty(mujoco.mj_sizeModel(m), np.uint8)
    mujoco.mj_saveModel(m, buffer=binary)
    return hashlib.sha256(binary.tobytes()).hexdigest()


def load_source(directory, task, seed, baseline_hash, released=False):
    stem = f'{task}_{"release" if seed == 0 else "hold"}_{seed:06d}'
    path = directory/stem
    report = json.loads(path.with_suffix('.json').read_text())
    if (not report.get('held_reset_qualified') or report['model_sha256'] != baseline_hash or
            any(report['warnings']) or report['max_contact_penetration_m'] > .0004):
        raise ValueError(f'Unqualified source: {path}')
    for name, expected in report['plugin_sha256'].items():
        if hashlib.sha256((ROOT/'plugin/build'/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Changed native plugin: {name}')
    with np.load(path.with_suffix('.npz')) as archive:
        states = archive['states']
        index = len(states)-1 if released else report['held_state_index']
        state = states[index].copy()
    return state, dict(source=str(path), source_state_index=int(index),
                       source_sha256=hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest())


def contact_flags(m, d, task):
    active = m.body(task).id
    held = False
    deep = False
    for contact in d.contact:
        bodies = m.geom_bodyid[contact.geom]
        if active in bodies:
            held |= any(m.body(int(b)).name.startswith(('left_', 'right_', 'driver')) for b in bodies)
            deep |= contact.dist < -.0004
    return held, deep


def run_cases(cases, starts, cfg, ticks, output, stop_at_all_safety=False):
    env = BatchedAssemblyEnv(['horn', 'retainer'],
                             [0 if c['task'] == 'horn' else 1 for c in cases], cfg)
    m = env.m
    for i, case in enumerate(cases):
        if case['mode'] == 'loose_zero':
            starts[i] = reset(m, case['task'], 1000)
            case.update(source='independent native loose reset', seed=1000, source_state_index=None)
        if case['mode'] == 'falling_zero':
            state = starts[i].copy()
            for name in ('base', *TASK_NAMES[:TASK_NAMES.index(case['task'])+1]):
                q = m.jnt_qposadr[m.joint(name+'_free').id]
                v = m.jnt_dofadr[m.joint(name+'_free').id]
                state[1+q+2] += case['fall_height_m']
                state[1+m.nq+v:1+m.nq+v+6] = [0., 0., -.2, 0., 0., 0.]
            starts[i] = state
            case['initialization'] = 'Raise predecessor assembly before reset; shared -0.2 m/s vertical velocity; no runtime teleport'
    env.set_states(np.arange(len(cases)), np.asarray(starts))
    for case, data in zip(cases, env.data):
        depth = max(0., -min(data.contact.dist, default=0.))
        case['initial_peak_penetration_m'] = float(depth)
        if depth > .0004:
            env.close()
            raise ValueError(f"Reject initial overlap {depth*1000:.3f} mm: {case['mode']}/{case['task']}")
    records = {k: [] for k in ('states', 'controls', 'actions', 'success', 'safety', 'unstable',
                               'distance', 'held', 'deep', 'position', 'quaternion')}
    replays = {}
    for i, case in enumerate(cases):
        if case['mode'] == 'moving_assembly':
            with np.load(case['control_archive']) as archive:
                replays[i] = archive['controls']
            if case['source_state_index']+ticks*env.substeps >= len(replays[i]):
                raise ValueError('Moving control archive is too short for the requested probe')
    clock = time.monotonic()
    ever_safety = np.zeros(len(cases), bool)
    failure = None
    try:
        for tick in range(ticks):
            action = np.zeros((len(cases), 12))
            if tick < 10:
                for i, case in enumerate(cases):
                    if case['mode'] == 'held_open':
                        action[i, 11] = 1.
            for i, controls in replays.items():
                index = cases[i]['source_state_index']+(tick+1)*env.substeps
                scale = np.tile([*cfg.action_scale_joints, cfg.gripper_scale], 2)
                action[i] = (controls[index]-env.state[i, 1:13])/scale
            _, _, info = env.step(action)
            held, deep, positions, quaternions = [], [], [], []
            for case, data in zip(cases, env.data):
                h, p = contact_flags(m, data, case['task'])
                held.append(h)
                deep.append(p)
                positions.append(data.body(case['task']).xpos.copy())
                quaternions.append(data.body(case['task']).xquat.copy())
            values = dict(states=env.state.copy(), controls=env.last_ctrl.copy(), actions=action,
                          success=info['success'], safety=info['safety'], unstable=info['unstable'],
                          distance=info['d_asm'], held=held, deep=deep,
                          position=positions, quaternion=quaternions)
            for key, value in values.items():
                records[key].append(np.asarray(value))
            ever_safety |= info['safety']
            if (tick+1) % 5 == 0:
                print(json.dumps(dict(batch=output.name, ticks=tick+1,
                                      elapsed_s=time.monotonic()-clock)), flush=True)
            if stop_at_all_safety and ever_safety.all():
                break
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
    finally:
        env.close()
    if not records['states']:
        raise RuntimeError(failure or 'No transitions')
    z = {k: np.asarray(v) for k, v in records.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_suffix('.npz'), initial_states=np.asarray(starts), **z)
    terminal_safety = np.maximum.accumulate(z['safety'], axis=0)
    eligible = (z['distance'] < .001) & ~z['held'] & ~z['deep'] & ~terminal_safety
    for i, case in enumerate(cases):
        first = lambda values: int(np.flatnonzero(values)[0]+1) if np.any(values) else None
        success = z['success'][:, i] & ~terminal_safety[:, i]
        case.update(first_success_tick=first(success), first_safety_tick=first(z['safety'][:, i]),
                    held_ticks=int(z['held'][:, i].sum()), eligible_ticks=int(eligible[:, i].sum()),
                    final_distance_m=float(z['distance'][-1, i]))
        case['eligible_window_observed'] = any(eligible[t-10:t+1, i].all()
                                              for t in range(10, len(eligible)))
        case['world_translation_span_m'] = float(np.linalg.norm(
            z['position'][:, i]-z['position'][0, i], axis=1).max())
        if cfg.stability_metric == 'pose' and success.any():
            end = int(np.flatnonzero(success)[0])
            if end < 10:
                raise AssertionError('Success before eleven poses span 0.4 s')
            pos = z['position'][end-10:end+1, i]
            quat = z['quaternion'][end-10:end+1, i]
            drift = np.linalg.norm(pos-pos[0], axis=1)
            angles = 2*np.arccos(np.clip(np.abs((quat*quat[0]).sum(1)), 0., 1.))
            case['success_window_verified'] = bool(eligible[end-10:end+1, i].all() and
                drift.max() <= cfg.stable_translation and angles.max() <= cfg.stable_rotation)
            if not case['success_window_verified']:
                raise AssertionError('Actual success violates independent window audit')
    report = dict(config=asdict(cfg), model_sha256=digest_model(m), cases=cases,
                  ticks=len(z['states']), wall_seconds=time.monotonic()-clock,
                  failure=failure, actual_env_implementation=True,
                  success_after_safety_ignored=True)
    output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive-dir', type=Path, default=ROOT/'artifacts_local/prepared_parts')
    p.add_argument('--output-dir', type=Path, default=ROOT/'artifacts_local/prepared_parts/integrated_pose')
    p.add_argument('--seeds', default='0,1,2,3')
    p.add_argument('--ticks', type=int, default=30)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--stability-metric', choices=['pose', 'velocity'], default='pose')
    p.add_argument('--max-contact-depth', type=float, default=.0004)
    p.add_argument('--fall-height', type=float, default=.5)
    p.add_argument('--case', action='append', help='Optional mode:task:seed selection; repeat to select several')
    a = p.parse_args()
    if a.ticks < 11:
        raise ValueError('At least eleven control samples are required')
    baseline_hash = digest_model(model())
    cfg = EnvConfig(nthread=a.threads, stability_metric=a.stability_metric,
                    max_contact_depth=a.max_contact_depth)
    def selected(case):
        key = f"{case['mode']}:{case['task']}:{case['seed']}"
        return a.case is None or key in a.case
    reports = []
    cases, starts = [], []
    for mode in ('held_zero', 'held_open'):
        for task in ('horn', 'retainer'):
            for seed in map(int, a.seeds.split(',')):
                state, source = load_source(a.archive_dir, task, seed, baseline_hash)
                starts.append(state)
                cases.append(dict(mode=mode, task=task, seed=seed, **source))
    for mode in ('released_zero', 'loose_zero'):
        for task in ('horn', 'retainer'):
            state, source = load_source(a.archive_dir, task, 0, baseline_hash, released=True)
            starts.append(state)
            cases.append(dict(mode=mode, task=task, seed=0, **source))
    chosen = [(case, state) for case, state in zip(cases, starts) if selected(case)]
    if chosen:
        reports.append(run_cases([c for c, _ in chosen], [s for _, s in chosen], cfg,
                                 a.ticks, a.output_dir/'regular'))
    cases, starts = [], []
    for task in ('horn', 'retainer'):
        state, source = load_source(a.archive_dir, task, 0, baseline_hash, released=True)
        starts.append(state)
        cases.append(dict(mode='falling_zero', task=task, seed=0,
                          fall_height_m=a.fall_height, **source))
    chosen = [(case, state) for case, state in zip(cases, starts) if selected(case)]
    if chosen:
        reports.append(run_cases([c for c, _ in chosen], [s for _, s in chosen], cfg,
                                 a.ticks, a.output_dir/'falling', stop_at_all_safety=True))
    moving_case = dict(mode='moving_assembly', task='horn', seed=0)
    if selected(moving_case):
        path = a.archive_dir/'retainer_withdraw'
        source_report = json.loads(path.with_suffix('.json').read_text())
        if (source_report['model_sha256'] != baseline_hash or any(source_report['warnings']) or
                source_report['max_contact_penetration_m'] > .0004):
            raise ValueError('Unqualified moving-assembly archive')
        for name, expected in source_report['plugin_sha256'].items():
            if hashlib.sha256((ROOT/'plugin/build'/name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Changed native plugin: {name}')
        with np.load(path.with_suffix('.npz')) as archive:
            state = archive['states'][8599].copy()
        moving_case.update(source=str(path), source_state_index=8599,
                           control_archive=str(path.with_suffix('.npz')),
                           description='Helper grasps retainer and lifts assembled base; active horn is unheld')
        # Stay inside the archived interval with sustained measured motion.
        reports.append(run_cases([moving_case], [state], cfg, min(a.ticks, 20), a.output_dir/'moving'))
    if not reports:
        raise ValueError('No cases matched --case selections')
    all_cases = [case for report in reports for case in report['cases']]
    checks = []
    for case in all_cases:
        expected = case['mode'] in ('held_open', 'released_zero')
        passed = expected == (case['first_success_tick'] is not None)
        if case['mode'] == 'moving_assembly':
            passed &= (case['eligible_window_observed'] and
                       case['world_translation_span_m'] > cfg.stable_translation)
        checks.append(dict(mode=case['mode'], task=case['task'], seed=case['seed'],
                           expected_success=expected, actual_success=case['first_success_tick'] is not None,
                           passed=bool(passed)))
    passed = all(check['passed'] for check in checks) and not any(r['failure'] for r in reports)
    summary = dict(passed=passed, checks=checks, baseline_model_sha256=baseline_hash,
                   production_files_or_banks_modified=False,
                   note='Initial overlaps above 0.4 mm are rejected; falling batches stop at safety, not impact.')
    (a.output_dir/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
