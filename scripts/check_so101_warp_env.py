"""Bounded qualification of the optional fine-geometry MJWarp environment.

This checker never changes the native trainer, its banks, or collision assets.
"""
import argparse
import json
import os
from pathlib import Path
import threading
import time

import mujoco
import numpy as np
import warp as wp


def watchdog(out, seconds, gpu_limit_mib=8192):
    from scripts.so101_gpu_profile import resource_status
    deadline = time.monotonic()+seconds
    def watch():
        while True:
            status = resource_status()
            if (status['rss_bytes'] > 5.5*1024**3 or status['available_bytes'] < 2*1024**3
                    or status['gpu_used_mib'] > gpu_limit_mib or time.monotonic() > deadline):
                (out/'resource_stop.json').write_text(json.dumps(status, indent=2)+'\n')
                os._exit(70)
            time.sleep(1)
    threading.Thread(target=watch, daemon=True).start()


def mechanics(args):
    from scripts.check_so101_gpu_candidate import ProbePhysics, summarize_pair
    from scripts.check_so101_tool_gripper import make_model, ENCODER_STEP
    from scripts.so101_warp_check import thread_model
    from assembly_policy.so101.scene import load_plugin
    from assembly_policy.so101.fastener import PITCH
    load_plugin()
    ProbePhysics.sdf_precision = 'fp32'
    tool = args.case == 'tool'
    if tool:
        # The isolated fixture has jaw grids but no target base. Give its first
        # grid instance the audit anchor name used by the full-scene importer.
        from assembly_policy.so101.collision import replace_gripper_collision
        def named_fixture(source):
            extensions = replace_gripper_collision(source)
            instance = extensions[0].find('instance')
            previous = instance.get('name')
            instance.set('name', 'base_sdf')
            for element in source.findall('.//plugin'):
                if element.get('instance') == previous:
                    element.set('instance', 'base_sdf')
            return extensions
        m = make_model(collision_builder=named_fixture)
        m.opt.gravity[:] = 0
        d = mujoco.MjData(m)
        d.qpos[m.joint('gripper').qposadr[0]] = .4
        d.ctrl[0] = 145*ENCODER_STEP
    else:
        m, d = thread_model(args.case == 'axial')
    mujoco.mj_forward(m, d)
    p = ProbePhysics(m, d, nworld=args.worlds)
    p.model.opt.ls_iterations = 100
    p.model.opt.deterministic = args.determinism
    if tool:
        p.tool_stats = wp.zeros((args.worlds, 3), dtype=float, device=p.device)
        p.button_q = int(m.joint('button_forward').qposadr[0])
    report = dict(case=args.case, geometry='unchanged fine SDF jaws and target/tool surfaces',
                  sdf_precision='FP64 grid coordinates; FP32 interpolation and analytic equations',
                  native_ls_iterations=int(m.opt.ls_iterations), gpu_ls_iterations=100,
                  gpu_determinism=args.determinism,
                  worlds=args.worlds, stages=[], samples=[], plugin_audit=p.plugin_audit,
                  parameter_audit=p.parameter_audit)
    report['tool_ramp_seconds'] = args.tool_ramp if tool else None
    path = args.out/(args.case+'.json')
    def save():
        path.write_text(json.dumps(report, indent=2)+'\n')
    save()
    started = time.monotonic()
    p.step(); wp.synchronize_device(p.device); mujoco.mj_step(m, d)
    report['first_step_seconds'] = time.monotonic()-started
    count = 1
    stages = [('close', 1200, 145, False), ('carry', 3200, 145, True),
              ('press', 5200, 130, True), ('release', 7200, 145, True),
              ('open', 7700, 220, True)] if tool else [(args.case, args.steps, None, None)]
    native_trace, gpu_trace = [], []
    for name, end, tick, gravity in stages:
        maximum = dict(button_m=0., depth_m=0.)
        stage_start = count
        previous_target = float(d.ctrl[0])
        if tool:
            d.ctrl[0] = tick*ENCODER_STEP
            p.data.ctrl.assign(np.broadcast_to(d.ctrl, (args.worlds, m.nu)).copy().astype(np.float32))
            m.opt.gravity[:] = [0, -9.81 if gravity else 0, 0]
            p.model.opt.gravity.assign(np.asarray(m.opt.gravity, np.float32).reshape(1, 3))
            p.tool_stats.zero_()
        while count < end:
            n = min(80, end-count)
            if tool and args.tool_ramp and name in ('press', 'release', 'open'):
                fraction = min(1., (count-stage_start+n)*m.opt.timestep/args.tool_ramp)
                fraction = .5-.5*np.cos(np.pi*fraction)
                d.ctrl[0] = previous_target+(tick*ENCODER_STEP-previous_target)*fraction
                p.data.ctrl.assign(np.broadcast_to(d.ctrl, (args.worlds, m.nu)).copy().astype(np.float32))
            p.step(n, graph=True); wp.synchronize_device(p.device)
            for _ in range(n):
                mujoco.mj_step(m, d)
                if tool:
                    maximum['button_m'] = max(maximum['button_m'], float(d.qpos[p.button_q]))
                    maximum['depth_m'] = max(maximum['depth_m'], max([0., *(-c.dist for c in d.contact)]))
            count += n
            sample = summarize_pair(m, d, p, 0, count)
            sample.update(step=count, stage=name)
            report['samples'].append(sample)
            a = p.arrays()
            native_trace.append(np.r_[d.time, d.qpos, d.qvel])
            gpu_trace.append(np.r_[a['time'][0], a['qpos'][0], a['qvel'][0]])
            if (not sample['native_finite'] or not sample['gpu_finite'] or
                    not sample['gpu_clock_matches_expected_float32'] or any(sample['overflow_bits'])):
                save()
                raise RuntimeError('Mechanical continuation failed; inspect report')
        stage = dict(name=name, native_maximum=maximum)
        if tool:
            stats = p.tool_stats.numpy()
            stage['gpu_maximum'] = dict(button_m=float(stats[:, 0].max()), depth_m=float(stats[:, 1].max()),
                                         nonfinite=bool(stats[:, 2].any()))
            qdriver = int(m.joint('driver_free').qposadr[0])
            spindle = int(m.joint('spindle').dofadr[0])
            for engine, q, v in [('native', d.qpos, d.qvel), ('gpu', a['qpos'][0], a['qvel'][0])]:
                stage[engine] = dict(button_m=float(q[p.button_q]), spindle_rad_s=float(v[spindle]),
                                     position=q[qdriver:qdriver+3].tolist())
            stage['gpu_all_worlds'] = dict(button_m=a['qpos'][:, p.button_q].tolist(),
                spindle_rad_s=a['qvel'][:, spindle].tolist(),
                position=a['qpos'][:, qdriver:qdriver+3].tolist())
        report['stages'].append(stage)
        save()
        print(json.dumps(stage), flush=True)
    report['finite'] = all(s['native_finite'] and s['gpu_finite'] for s in report['samples'])
    report['zero_overflow'] = all(not any(s['overflow_bits']) for s in report['samples'])
    report['zero_native_warnings'] = all(not any(s['native_warning_counts']) for s in report['samples'])
    report['gates'] = {}
    if tool:
        for engine in ('native', 'gpu'):
            close, carry, press, release, opened = report['stages']
            key = engine if engine == 'native' else 'gpu_all_worlds'
            position = lambda s: np.asarray(s[key]['position']).reshape(-1,3)
            drift = max(np.linalg.norm(position(s)-position(close), axis=1).max() for s in (carry, press, release))
            drop = np.linalg.norm(position(opened)-position(release), axis=1).min()
            off = lambda s: bool((np.asarray(s[key]['button_m']) < .0006).all() and
                                  (abs(np.asarray(s[key]['spindle_rad_s'])) < .1).all())
            report['gates'][engine] = dict(
                closure_off=close[engine+'_maximum']['button_m'] < .0006,
                carry_off=off(carry),
                press_on=bool((np.asarray(press[key]['button_m']) > .0006).all() and
                              (np.asarray(press[key]['spindle_rad_s']) > 20).all()),
                release_off=off(release),
                retained=bool(drift < .003), drift_m=float(drift), open_drops=bool(drop > .05), drop_m=float(drop),
                penetration_pass=all(s[engine+'_maximum']['depth_m'] < .0003 for s in report['stages']))
    else:
        for engine, rows in [('native', native_trace), ('gpu', gpu_trace)]:
            trace = np.asarray(rows)
            start = np.searchsorted(trace[:, 0], .3)
            advance = trace[-1, 1]-trace[start, 1]
            turns = (trace[-1, 2]-trace[start, 2])/(2*np.pi)
            fit = float(np.polyfit(trace[start:, 2]/(2*np.pi), trace[start:, 1], 1)[0]) if abs(turns) > .1 else None
            report['gates'][engine] = dict(advance_m=float(advance), turns=float(turns),
                fitted_pitch_m=fit, nominal_pitch_m=PITCH, endpoint_pitch_error_m=float(advance-turns*PITCH),
                pitch_fit_pass=bool(fit is not None and abs(fit-PITCH) < 5e-6) if args.case == 'thread' else None,
                axial_pass=bool(abs(advance) < 20e-6) if args.case == 'axial' else None)
    report['elapsed_seconds'] = time.monotonic()-started
    np.savez_compressed(args.out/(args.case+'_trace.npz'), native=native_trace, gpu=gpu_trace)
    p.close(); save()
    print(json.dumps({k: v for k, v in report.items() if k not in ('samples','plugin_audit','parameter_audit')}), flush=True)


def environment(args):
    from assembly_policy.so101.env import BatchedAssemblyEnv, EnvConfig, TASK_NAMES, ARM_ACTIONS
    from assembly_policy.so101.warp_env import BatchedWarpAssemblyEnv, archived_helper_factory, physics_identity
    from assembly_policy.so101.resets import check_bank
    from assembly_policy.so101.scene import model
    from scripts.check_so101_task_completion import digest_model, load_source, contact_flags
    check_bank(args.bank_dir)
    factory = archived_helper_factory(args.candidate_mjb) if args.candidate_mjb else None
    variant = 'whole_helper_sdf' if factory else 'separated_jaws'
    report = dict(case='environment', worlds=args.worlds, checks={}, phases=[], model_variant=variant,
        candidate_provenance=factory.provenance if factory else None,
        physics_provenance=physics_identity(factory) if factory else None,
        canonical_bank_manifest=json.loads((args.bank_dir/'manifest.json').read_text()))
    path = args.out/'environment.json'
    def save():
        path.write_text(json.dumps(report, indent=2)+'\n')
    def phase(name, **values):
        report['phases'].append(dict(name=name, **values)); save()
        print(json.dumps(report['phases'][-1]), flush=True)
    cfg = EnvConfig(nthread=args.native_threads, stability_metric='pose', max_contact_depth=.0004)
    phase('native init')
    native = BatchedAssemblyEnv(TASK_NAMES, np.arange(args.worlds) % 5, cfg, model_factory=factory)
    independent = []
    for task in TASK_NAMES:
        with np.load(args.bank_dir/(task+'_eval.npz')) as archive:
            independent.append(archive['states'][0])
    initial = np.asarray(independent)[np.arange(args.worlds) % 5]
    native.set_states(np.arange(args.worlds), initial)
    native_observation = native.observe()
    initial_history = native.hist.copy()
    rng = np.random.default_rng(20261105)
    action = rng.uniform(-.15, .15, (args.worlds, 12))
    started = time.monotonic(); nr, nd, ni = native.step(action)
    phase('native first', seconds=time.monotonic()-started)
    native_state, native_ctrl = native.state.copy(), native.last_ctrl.copy()
    control_ranges = native.m.actuator_ctrlrange.copy()
    native_nq = native.m.nq
    throughput = {}
    native.set_states(np.arange(args.worlds), initial)
    native.step(np.zeros((args.worlds, 12)))
    started = time.monotonic()
    for _ in range(args.controls):
        native.step(np.zeros((args.worlds, 12)))
        native.observe()
    seconds = time.monotonic()-started
    throughput['native'] = dict(seconds=seconds, controls=args.controls, transitions_s=args.worlds*args.controls/seconds)
    phase('native end-to-end benchmark', **throughput['native'])
    native.close()
    del native
    # Read/decompress reset evidence before GPU allocations and graph capture.
    cases, starts = [], []
    with np.load(args.bank_dir/'motor_train.npz') as archive:
        motor = archive['states'][archive['family'] == 4][0].copy()
    for mode in ('held_zero','held_open'):
        cases.append(dict(task='motor', mode=mode)); starts.append(motor.copy())
    baseline = digest_model(model())
    for task in ('horn','retainer'):
        for mode in ('held_zero','held_open','released_zero'):
            state, source = load_source(Path('artifacts_local/prepared_parts'), task, 0, baseline,
                                         released=mode == 'released_zero')
            cases.append(dict(task=task, mode=mode, **source)); starts.append(state)
    # Do not retain two full 64-world CPU model/data sets during GPU capture.
    import gc
    import ctypes
    gc.collect()
    ctypes.CDLL(None).malloc_trim(0)
    phase('gpu init after freeing native environment')
    gpu = BatchedWarpAssemblyEnv(TASK_NAMES, np.arange(args.worlds) % 5, cfg, proof_dir=args.out/'proof',
        model_factory=factory, model_variant=variant)
    gpu.set_states(np.arange(args.worlds), initial)
    report['gpu_options'] = gpu.engine.options
    report['initial_observation_identical'] = all(np.array_equal(native_observation[k], gpu.observe()[k]) for k in native_observation)
    started = time.monotonic(); gr, gd, gi = gpu.step(action)
    phase('gpu first', seconds=time.monotonic()-started)
    scale = np.tile([*cfg.action_scale_joints, cfg.gripper_scale], 2)
    expected_ctrl = np.clip(initial[:, 1:13]+action*scale,
                            control_ranges[:, 0], control_ranges[:, 1])
    report['checks']['relative_control_exact'] = bool(np.array_equal(expected_ctrl, gpu.last_ctrl) and np.array_equal(native_ctrl, gpu.last_ctrl))
    report['checks']['previous_action_exact'] = bool(np.array_equal(gpu.prev_action, action))
    report['checks']['ordinary_history_shift_exact'] = bool(
        np.array_equal(gpu.hist[:,:-1], initial_history[:,1:]) and
        np.array_equal(gpu.hist[:,-1], gpu.frames) and
        np.array_equal(gpu.hist[:,-1,38:50], action.astype(np.float32)))
    report['checks']['first_transition_same_class'] = bool(np.array_equal(nd, gd) and all(np.array_equal(ni[k], gi[k]) for k in ('success','timeout','safety','unstable')))
    report['first_transition_error'] = dict(reward_max=float(abs(nr-gr).max()),
        arm_qpos_max=float(abs(native_state[:, 1:13]-gpu.state[:, 1:13]).max()),
        assembly_distance_max_m=float(abs(ni['d_asm']-gi['d_asm']).max()),
        qpos_max=float(abs(native_state[:, 1:1+native_nq]-gpu.state[:, 1:1+native_nq]).max()))
    first_gpu = gpu.state.copy()
    gpu.set_states(np.arange(args.worlds), initial)
    gpu.step(action)
    report['checks']['reset_repeat_deterministic'] = bool(np.allclose(first_gpu, gpu.state, atol=1e-6, rtol=0))
    report['reset_repeat_max_error'] = float(abs(first_gpu-gpu.state).max())
    report['reset_repeat_qpos_max_error'] = float(abs(first_gpu[:,1:1+gpu.m.nq]-gpu.state[:,1:1+gpu.m.nq]).max())
    np.savez_compressed(args.out/'first_transition.npz', initial=initial, action=action,
                        native=native_state, gpu_first=first_gpu, gpu_repeat=gpu.state.copy())
    phase('first transition checked')
    gpu.set_states(np.arange(args.worlds), initial)
    gpu.step(np.zeros((args.worlds, 12)))
    started = time.monotonic()
    for _ in range(args.controls):
        gpu.step(np.zeros((args.worlds, 12)))
        gpu.observe()
    seconds = time.monotonic()-started
    throughput['gpu'] = dict(seconds=seconds, controls=args.controls, transitions_s=args.worlds*args.controls/seconds)
    phase('gpu end-to-end benchmark', **throughput['gpu'])
    report['throughput'] = throughput
    # The same actor-facing methods handle partial SGS resets and task changes.
    ids = np.arange(0, args.worlds, 2)
    tasks = (gpu.task[ids]+1) % 5
    gpu.set_states(ids, np.asarray(independent)[tasks], tasks)
    report['checks']['partial_reset_state_and_task'] = bool(np.array_equal(gpu.task[ids], tasks) and
        np.array_equal(gpu.state[ids, 1:], np.asarray(independent)[tasks, 1:]) and
        (gpu.t[ids] == 0).all() and (gpu.stable[ids] == 0).all() and (gpu.prev_action[ids] == 0).all() and
        np.array_equal(gpu.hist[ids], np.broadcast_to(gpu.frames[ids, None], gpu.hist[ids].shape)))
    gpu.set_states(np.arange(args.worlds), initial, np.arange(args.worlds) % 5)
    gpu.horizon = 2
    gpu.step(np.zeros((args.worlds, 12)))
    _, done, info = gpu.step(np.zeros((args.worlds, 12)))
    report['checks']['timeout_exactly_two_controls'] = bool(done.all() and info['timeout'].all() and not info['success'].any())
    gpu.horizon = round(cfg.horizon_s/cfg.control_dt)
    assignment = np.arange(args.worlds) % len(cases)
    gpu.set_states(np.arange(args.worlds), np.asarray(starts)[assignment],
                   [TASK_NAMES.index(cases[i]['task']) for i in assignment])
    ever = np.zeros(args.worlds, bool); safe = np.zeros(args.worlds, bool)
    records = {k: [] for k in ('state','success','safety','held','distance','position','quaternion')}
    for tick in range(30):
        action = np.zeros((args.worlds, 12))
        if tick < 10:
            for world, case in enumerate(assignment):
                if cases[case]['mode'] == 'held_open': action[world, 11] = 1
        _, done, info = gpu.step(action)
        safe |= info['safety']; ever |= info['success'] & ~safe
        held = [contact_flags(gpu.m, d, TASK_NAMES[gpu.task[i]])[0] for i, d in enumerate(gpu.data)]
        values = dict(state=gpu.state.copy(), success=info['success'], safety=info['safety'], held=held,
            distance=info['d_asm'], position=[d.body(TASK_NAMES[gpu.task[i]]).xpos.copy() for i,d in enumerate(gpu.data)],
            quaternion=[d.body(TASK_NAMES[gpu.task[i]]).xquat.copy() for i,d in enumerate(gpu.data)])
        for key, value in values.items(): records[key].append(np.asarray(value))
        if (tick+1) % 5 == 0: phase('completion cases', tick=tick+1, successful=int(ever.sum()), unsafe=int(safe.sum()))
    z = {key: np.asarray(value) for key, value in records.items()}
    np.savez_compressed(args.out/'completion.npz', assignment=assignment, **z)
    for i, case in enumerate(cases):
        selected = assignment == i
        case['worlds'] = int(selected.sum())
        case['successes'] = int(ever[selected].sum())
        case['safety'] = int(safe[selected].sum())
        case['pass'] = bool(not ever[selected].any()) if case['mode']=='held_zero' else bool(ever[selected].all())
    report['completion_cases'] = cases
    report['checks']['physical_completion'] = all(c['pass'] for c in cases)
    for world in np.flatnonzero(ever):
        end = int(np.flatnonzero(z['success'][:, world] & ~np.maximum.accumulate(z['safety'][:, world]))[0])
        if end < 10: raise AssertionError('Success before eleven pose samples')
        pos = z['position'][end-10:end+1, world]; quat = z['quaternion'][end-10:end+1, world]
        assert not z['held'][end-10:end+1, world].any()
        assert (z['distance'][end-10:end+1, world] < .001).all()
        assert np.linalg.norm(pos-pos[0], axis=1).max() <= cfg.stable_translation
        assert (2*np.arccos(np.clip(abs((quat*quat[0]).sum(1)), 0, 1))).max() <= cfg.stable_rotation
    report['checks']['success_windows_independent_audit'] = True
    # Deliberate falling initial state: no penetrating initial placement.
    falling = initial.copy()
    q = gpu.m.joint('motor_free').qposadr[0]; v = gpu.m.joint('motor_free').dofadr[0]
    falling[:, 1+q+2] += .5
    falling[:, 1+gpu.m.nq+v+2] = -1.1
    gpu.set_states(np.arange(args.worlds), falling, np.zeros(args.worlds, int))
    _, done, info = gpu.step(np.zeros((args.worlds, 12)))
    report['checks']['physical_velocity_safety'] = bool(done.all() and info['safety'].all() and not info['success'].any())
    gpu.set_states(np.arange(args.worlds), initial, np.arange(args.worlds) % 5)
    try:
        gpu.step(np.full((args.worlds,12), np.nan))
        report['checks']['nonfinite_action_rejected'] = False
    except ValueError:
        report['checks']['nonfinite_action_rejected'] = True
    original = gpu.state.copy(); gpu.state[0,1] = np.nan
    try:
        gpu.step(np.zeros((args.worlds,12)))
        report['checks']['nonfinite_state_rejected'] = False
    except FloatingPointError:
        report['checks']['nonfinite_state_rejected'] = True
    gpu.state[:] = original
    if gpu.engine.domains:
        index, low, high = gpu.engine.domains[0]
        gpu.state[0,1+index] = high+.001
        try:
            gpu.step(np.zeros((args.worlds,12)))
            report['checks']['separation_domain_rejected'] = False
        except FloatingPointError:
            report['checks']['separation_domain_rejected'] = True
    else:
        report['checks']['no_added_collision_exclusions'] = bool(
            gpu.engine.proof['added_exclusions'] == [] and
            np.array_equal(gpu.engine.physics.model.exclude_signature.numpy(), gpu.m.exclude_signature))
    report['checks']['gpu_clock_and_overflow_all_controls'] = bool(gpu.engine.calls > 30 and not gpu.engine.last_overflow.any())
    report['rollout_calls'] = gpu.engine.calls
    report['last_sensor_audit_error_m'] = gpu.engine.last_sensor_audit_error
    gpu.close()
    report['functional_pass'] = bool(report['initial_observation_identical'] and
        all(value for key,value in report['checks'].items() if key != 'reset_repeat_deterministic'))
    report['pass'] = bool(report['initial_observation_identical'] and all(report['checks'].values()))
    save(); print(json.dumps(report), flush=True)


def reset_probe(args):
    import mujoco_warp as mjw
    from assembly_policy.so101.env import EnvConfig, TASK_NAMES
    from assembly_policy.so101.warp_env import BatchedWarpAssemblyEnv, archived_helper_factory
    cfg = EnvConfig(nthread=1, stability_metric='pose', max_contact_depth=.0004,
                    control_dt=args.reset_substeps*.0005)
    factory = archived_helper_factory(args.candidate_mjb) if args.candidate_mjb else None
    env = BatchedWarpAssemblyEnv(TASK_NAMES, np.arange(args.worlds) % 5, cfg, proof_dir=args.out/'proof',
        model_factory=factory, model_variant='whole_helper_sdf' if factory else 'separated_jaws')
    p = env.engine.physics
    p.model.opt.deterministic = args.determinism
    initial=[]
    for task in TASK_NAMES:
        with np.load(args.bank_dir/(task+'_eval.npz')) as archive: initial.append(archive['states'][0])
    initial = np.asarray(initial)[np.arange(args.worlds) % 5]
    action = np.random.default_rng(20261105).uniform(-.15,.15,(args.worlds,12))
    report=dict(worlds=args.worlds, determinism=args.determinism, substeps=args.reset_substeps, repeats=[])
    states=[]
    for repeat in range(3):
        env.set_states(np.arange(args.worlds), initial)
        initial_history = env.hist.copy()
        # Deliberately contaminate fields that native Rollout also resets.
        p.data.qacc_warmstart.fill_(123)
        p.data.qfrc_applied.fill_(456)
        p.data.qacc.fill_(789)
        p.data.efc.force.fill_(100*(repeat+1))
        mjw.reset_data(p.model,p.data)
        report['native_default_reset_fields_zero'] = bool(not p.data.qacc_warmstart.numpy().any() and
            not p.data.qfrc_applied.numpy().any() and not p.data.qacc.numpy().any() and
            not p.data.nacon.numpy().any() and not p.data.nefc.numpy().any() and
            not p.data.sensordata.numpy().any() and not p.data.overflow.numpy().any())
        reward,done,info=env.step(action)
        states.append(env.state.copy())
        item=dict(repeat=repeat, ordinary_history_shift_exact=bool(np.array_equal(env.hist[:,:-1],initial_history[:,1:]) and np.array_equal(env.hist[:,-1],env.frames) and np.array_equal(env.hist[:,-1,38:50],action.astype(np.float32))), sensor_audit_error=env.engine.last_sensor_audit_error,
                  success=info['success'].tolist(), safety=info['safety'].tolist(), reward=reward.tolist())
        if repeat:
            difference=abs(states[-1]-states[0])
            index=np.unravel_index(difference.argmax(),difference.shape)
            item.update(qpos_error=float(difference[:,1:1+env.m.nq].max()),
                        qvel_error=float(difference[:,1+env.m.nq:].max()), max_error_index=list(map(int,index)))
        report['repeats'].append(item)
        (args.out/'reset.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(item),flush=True)
    np.savez_compressed(args.out/'reset.npz',initial=initial,action=action,states=states)
    env.close()



def prepare_candidate(args):
    """Archive a sensor-equipped candidate, verifying all existing mechanics."""
    import ctypes
    import gc
    import hashlib
    import xml.etree.ElementTree as ET
    from assembly_policy.so101.scene import build_xml, load_plugin, ROOT
    from assembly_policy.so101.helper_collision import replace_helper_collision
    def digest(path):
        with Path(path).open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    source = args.replay.with_suffix('')
    proof = json.loads(source.with_suffix('.json').read_text())
    if not proof.get('passed') or any(proof['warnings']) or digest(source.with_suffix('.mjb')) != proof['model_sha256']:
        raise ValueError('Expected the passed native helper-SDF replay archive')
    load_plugin()
    for name, expected in proof['plugin_sha256'].items():
        if digest(ROOT/'plugin/build'/name) != expected:
            raise ValueError('Compiled plugin changed')
    mujoco.mj_setCacheCapacity(mujoco.mj_getCache(), 0)
    xml = ET.fromstring(build_xml(contact_monitor=True))
    replacement = replace_helper_collision(xml)
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))
    mujoco.mj_clearCache(mujoco.mj_getCache()); gc.collect(); ctypes.CDLL(None).malloc_trim(0)
    original = mujoco.MjModel.from_binary_path(str(source.with_suffix('.mjb')))
    # The extra global sensor changes sensor arrays and name bookkeeping only.
    changed = []
    for name in dir(original):
        if name.startswith(('_', 'sensor', 'name_')) or name in ('names', 'names_map'):
            continue
        value = getattr(original, name)
        if isinstance(value, np.ndarray) and not np.array_equal(value, getattr(model, name)):
            changed.append(name)
    for name in dir(original.opt):
        if name.startswith('_'): continue
        value = getattr(original.opt, name)
        if not callable(value) and not np.array_equal(value, getattr(model.opt, name)):
            changed.append('opt.'+name)
    for count, kind in (('nbody',mujoco.mjtObj.mjOBJ_BODY), ('njnt',mujoco.mjtObj.mjOBJ_JOINT),
                        ('ngeom',mujoco.mjtObj.mjOBJ_GEOM), ('nsite',mujoco.mjtObj.mjOBJ_SITE),
                        ('nu',mujoco.mjtObj.mjOBJ_ACTUATOR), ('nplugin',mujoco.mjtObj.mjOBJ_PLUGIN)):
        if getattr(original,count) != getattr(model,count) or any(
            mujoco.mj_id2name(original,kind,i) != mujoco.mj_id2name(model,kind,i) for i in range(getattr(original,count))):
            changed.append(count+' semantic names')
    if changed:
        raise ValueError('Sensor candidate changed existing mechanics: '+', '.join(changed))
    if model.nsensordata != 1 or model.sensor(0).name != 'min_contact_distance':
        raise ValueError('Missing global minimum-contact sensor')
    del original; gc.collect(); ctypes.CDLL(None).malloc_trim(0)
    destination = args.out/'candidate_sensor.mjb'
    mujoco.mj_saveModel(model, str(destination))
    report = dict(model_sha256=digest(destination), source_model_sha256=proof['model_sha256'],
                  model_file=destination.name, plugin_sha256=proof['plugin_sha256'], replacement=replacement,
                  mechanical_arrays_options_names_unchanged=True, sensor='global min_contact_distance',
                  exclude_signature=model.exclude_signature.tolist(), passed=True,
                  helper_manifest_sha256=digest(ROOT/'assets_local/so101_helper_collision/manifest.json'),
                  grid_sha256={f'assets_local/sdf/grid_{int(c.get("value"))}.bin':
                      digest(ROOT/f'assets_local/sdf/grid_{int(c.get("value"))}.bin') for c in
                      xml.findall("./extension/plugin[@plugin='assemblenist.sdf.grid']/instance/config[@key='id']")})
    (args.out/'candidate_sensor.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


def tool_replay(args):
    """Replay the actual twelve recorded arm controls, without block resets."""
    import hashlib
    import mujoco_warp as mjw
    from assembly_policy.so101.scene import model, ROOT, load_plugin
    from assembly_policy.so101.warp_env import (separated_model, MixedPhysics, _control, _record,
        GPU_SOLVER_ITERATIONS, GPU_SOLVER_TOLERANCE)
    from scripts.check_so101_gpu_candidate import ProbePhysics, _record_tool, _record_contact
    from scripts.check_so101_task_completion import digest_model
    source = args.replay.with_suffix('')
    proof = json.loads(source.with_suffix('.json').read_text())
    def file_digest(path):
        with Path(path).open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    if args.candidate_mjb:
        load_plugin()
        if file_digest(args.candidate_mjb) != proof['model_sha256']:
            raise ValueError('Candidate model archive changed')
        original = mujoco.MjModel.from_binary_path(str(args.candidate_mjb))
        active_mesh = np.flatnonzero((original.geom_type == mujoco.mjtGeom.mjGEOM_MESH) &
                                     ((original.geom_contype | original.geom_conaffinity) != 0))
        if len(active_mesh):
            raise ValueError('Whole-helper candidate still contains active mesh colliders')
        manifest = json.loads((ROOT/'assets_local/so101_helper_collision/manifest.json').read_text())
        for entry in manifest['meshes'].values():
            if file_digest(ROOT/entry['grid']) != entry['grid_sha256']:
                raise ValueError('Candidate helper grid changed')
        # Binary loading avoids recompiling all visual/plugin meshes. Drop only
        # the process-local compiler asset cache, never the active model.
        import ctypes, gc
        mujoco.mj_clearCache(mujoco.mj_getCache()); gc.collect(); ctypes.CDLL(None).malloc_trim(0)
    else:
        original = model()
        if proof['model_sha256'] != digest_model(original):
            raise ValueError('Tool archive model is not the current native model')
    if any(proof['warnings']) or not proof['passed']:
        raise ValueError('Tool replay requires a qualified native archive')
    for name, digest in proof['plugin_sha256'].items():
        if hashlib.sha256((ROOT/'plugin/build'/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Native plugin changed since the archived demonstration')
    with np.load(source.with_suffix('.npz')) as archive:
        initial = archive['initial_state'].copy()
        controls = archive['controls']
        stages = archive['stage']; labels = archive['labels']
        reference = archive['states']
    if args.candidate_mjb:
        m, separation = original, dict(mode='direct whole-helper SDF model import',
            added_exclusions=[], native_exclusions=original.exclude_signature.tolist())
    else:
        m, separation = separated_model(original, args.out/'proof')
    d = mujoco.MjData(m)
    mujoco.mj_setState(m,d,initial,mujoco.mjtState.mjSTATE_FULLPHYSICS)
    d.ctrl[:] = controls[0]; mujoco.mj_forward(m,d)
    p=MixedPhysics(m,d,nworld=args.worlds)
    p.model.opt.iterations=GPU_SOLVER_ITERATIONS
    p.model.opt.ls_iterations=100
    p.model.opt.tolerance.fill_(GPU_SOLVER_TOLERANCE)
    button=int(m.joint('button_forward').qposadr[0]); spindle=int(m.joint('spindle').dofadr[0])
    driver=int(m.joint('driver_free').qposadr[0])
    stats=wp.zeros((args.worlds,3),dtype=float,device=p.device)
    inputs=wp.zeros((args.worlds,80,m.nu),dtype=float,device=p.device)
    output=wp.zeros((args.worlds,80,len(initial)),dtype=float,device=p.device)
    sensors=wp.zeros((args.worlds,80,0),dtype=float,device=p.device)
    def step(index):
        wp.launch(_control,dim=(args.worlds,m.nu),inputs=[inputs,p.data.ctrl,index],device=p.device)
        mjw.step(p.model,p.data)
        wp.launch(_record_tool,dim=args.worlds,inputs=[p.data.qpos,p.data.qvel,stats,button],device=p.device)
        wp.launch(_record_contact,dim=p.data.naconmax,inputs=[p.data.contact.dist,p.data.contact.worldid,p.data.nacon,stats],device=p.device)
        wp.launch(_record,dim=(args.worlds,len(initial)),inputs=[p.data.qpos,p.data.qvel,p.data.time,
            p.data.sensordata,output,sensors,index],device=p.device)
    # Compile once and then restore the initial state. No subsequent reset or
    # warmstart clearing occurs throughout the recorded 20.3-second motion.
    inputs.assign(np.broadcast_to(controls[:80],inputs.shape).copy().astype(np.float32))
    step(0); wp.synchronize_device(p.device)
    mjw.reset_data(p.model,p.data)
    p.data.time.fill_(float(initial[0]))
    p.data.qpos.assign(np.broadcast_to(initial[1:1+m.nq],(args.worlds,m.nq)).copy().astype(np.float32))
    p.data.qvel.assign(np.broadcast_to(initial[1+m.nq:],(args.worlds,m.nv)).copy().astype(np.float32))
    with wp.ScopedCapture(device=p.device) as capture:
        for index in range(80): step(index)
    report=dict(source=str(source), source_model_sha256=proof['model_sha256'], worlds=args.worlds,
        scripted=True, learned_policy=False, no_resets_after_initialization=True,
        source_control_period_s=float(m.opt.timestep), stages=[], checks={},
        gpu_iterations=GPU_SOLVER_ITERATIONS, source_native_iterations=int(original.opt.iterations),
        gpu_tolerance=GPU_SOLVER_TOLERANCE, source_native_tolerance=float(original.opt.tolerance),
        gpu_ls_iterations=100, source_native_ls_iterations=int(original.opt.ls_iterations),
        adapter_sha256=file_digest(ROOT/'assembly_policy/so101/warp_env.py'),
        geometry=separation, plugin_audit=p.plugin_audit, parameter_audit=p.parameter_audit,
        candidate_mjb=str(args.candidate_mjb) if args.candidate_mjb else None,
        grid_ids=sorted(p.grids), helper_velocity_max_rad_s=0.)
    expected=np.full(args.worlds,np.float32(initial[0]),np.float32)
    traces=[];clock=0; started=time.monotonic()
    domains=[(int(m.joint(s+'_gripper').qposadr[0]),m.joint(s+'_gripper').range+[-.05,.05]) for s in ('left','right')]
    for stage_id,label in enumerate(labels):
        indices=np.flatnonzero(stages==stage_id)
        if not np.array_equal(indices,np.arange(clock,clock+len(indices))):
            raise ValueError('Archive stages must be consecutive')
        stats.zero_(); end=clock+len(indices); max_arm_speed=0.
        while clock<end:
            count=min(80,end-clock)
            batch=np.zeros(inputs.shape,np.float32)
            batch[:,:count]=controls[None,clock:clock+count]
            inputs.assign(batch)
            if count==80: wp.capture_launch(capture.graph)
            else:
                for index in range(count): step(index)
            actual=output.numpy()[:,:count].copy()
            if not np.isfinite(actual).all() or p.data.overflow.numpy().any():
                raise FloatingPointError('Replay numerical/overflow failure')
            max_arm_speed=max(max_arm_speed,float(abs(actual[:,:,1+m.nq:1+m.nq+12]).max()))
            for index in range(count):
                expected=expected+np.float32(m.opt.timestep)
                if not np.array_equal(actual[:,index,0],expected): raise FloatingPointError('Replay raw clock reset')
            for index,limits in domains:
                if ((actual[:,:,1+index]<limits[0])|(actual[:,:,1+index]>limits[1])).any():
                    raise FloatingPointError('Replay left separation proof domain')
            traces.extend(actual[0]);clock+=count
        a=p.arrays(); metrics=stats.numpy()
        row=dict(name=str(label),end_step=clock,time_s=clock*m.opt.timestep,
            native_depth_m=float((proof['task_stages'] if 'task_stages' in proof else proof['stages'])[stage_id]['max_penetration_m']),
            max_arm_velocity_rad_s=max_arm_speed,
            gpu_depth_m=float(metrics[:,1].max()),gpu_button_max_m=float(metrics[:,0].max()),
            gpu_button_m=a['qpos'][:,button].tolist(),gpu_spindle_rad_s=a['qvel'][:,spindle].tolist(),
            gpu_driver_position=a['qpos'][:,driver:driver+3].tolist(),
            native_driver_position=reference[clock-1,1+driver:1+driver+3].tolist(),
            zero_overflow=not bool(p.data.overflow.numpy().any()))
        report['stages'].append(row)
        report['helper_velocity_max_rad_s']=max(report['helper_velocity_max_rad_s'],max_arm_speed)
        (args.out/'tool_replay.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(row),flush=True)
    byname={row['name']:row for row in report['stages']}
    start=np.asarray(byname['Settle loose workcell']['gpu_driver_position'])
    lift=np.asarray(byname['Hold tool OFF above table']['gpu_driver_position'])
    on=byname['Squeeze physical forward button'];off=byname['Wait for spindle coast-down']
    final=np.asarray(byname['Withdraw empty hand']['gpu_driver_position'])
    report['checks']=dict(
        pickup_lift_40mm=bool(((lift-start)[:,2]>.04).all()),
        carried_off=bool(byname['Hold tool OFF above table']['gpu_button_max_m']<.0006),
        physically_on=bool((np.asarray(on['gpu_button_m'])>.0006).all() and (np.asarray(on['gpu_spindle_rad_s'])>20).all()),
        physically_off=bool((np.asarray(off['gpu_button_m'])<.0006).all() and (abs(np.asarray(off['gpu_spindle_rad_s']))<.1).all()),
        returned_to_table=bool((abs(final[:,2]-start[:,2])<.003).all()),
        penetration_below_0_3mm=all(row['gpu_depth_m']<.0003 for row in report['stages']),
        zero_overflow=all(row['zero_overflow'] for row in report['stages']),
        helper_velocity_below_5_rad_s=report['helper_velocity_max_rad_s']<5)
    report['elapsed_seconds']=time.monotonic()-started
    report['pass']=all(report['checks'].values())
    np.savez_compressed(args.out/'tool_replay.npz',initial_state=initial,controls=controls,states=traces,labels=labels,stage=stages)
    (args.out/'tool_replay.json').write_text(json.dumps(report,indent=2)+'\n')
    p.close();print(json.dumps(report['checks']),flush=True)


def collection(args):
    """Matched repeated 64-state fixture; complete collection without actor/PPO."""
    import hashlib
    import resource
    from assembly_policy.so101.env import EnvConfig, TASK_NAMES
    from assembly_policy.so101.warp_env import BatchedWarpAssemblyEnv, archived_helper_factory, physics_identity
    from assembly_policy.so101.resets import check_bank
    from scripts.so101_gpu_profile import resource_status
    if args.candidate_mjb is None or args.worlds % 64:
        raise ValueError('Collection scaling requires the qualified archive and a multiple of 64 worlds')
    check_bank(args.bank_dir)
    loose, prepared = {}, {}
    for name in TASK_NAMES:
        with np.load(args.bank_dir/(name+'_eval.npz')) as archive:
            loose[name] = archive['states'][:8].copy()
        if name in ('motor', 'horn', 'retainer'):
            with np.load(args.bank_dir/(name+'_train.npz')) as archive:
                prepared[name] = archive['states'][archive['family'] == 4][:16].copy()
    states, tasks = [], []
    for index in range(32):
        name = TASK_NAMES[index % 5]
        states.append(loose[name][index // 5]); tasks.append(TASK_NAMES.index(name))
    for index in range(32):
        name = ('motor', 'horn', 'retainer')[index % 3]
        states.append(prepared[name][index // 3]); tasks.append(TASK_NAMES.index(name))
    fixture, fixture_tasks = np.asarray(states), np.asarray(tasks)
    states = np.tile(fixture, (args.worlds // 64, 1))
    tasks = np.tile(fixture_tasks, args.worlds // 64)
    factory = archived_helper_factory(args.candidate_mjb)
    report = dict(worlds=args.worlds, controls=args.controls,
        distribution='Repeated 64-state fixture: 32 independent loose starts plus 32 prepared TRAIN holds',
        fixture_sha256=hashlib.sha256(fixture.tobytes()+fixture_tasks.tobytes()).hexdigest(),
        physics_provenance=physics_identity(factory), actor_and_ppo_included=False)
    config = EnvConfig(nthread=1, stability_metric='pose', max_contact_depth=.0004)
    env = BatchedWarpAssemblyEnv(TASK_NAMES, tasks, config, proof_dir=args.out/'proof',
                                model_factory=factory, model_variant='whole_helper_sdf')
    ids = np.arange(args.worlds)
    action = np.zeros((args.worlds, 12))
    env.set_states(ids, states, tasks)
    env.step(action)  # Compile/capture before timing, then restore every world.
    env.set_states(ids, states, tasks)
    counts = dict(success=0, safety=0, timeout=0)
    started = time.monotonic()
    for _ in range(args.controls):
        _, done, info = env.step(action)
        for key in counts: counts[key] += int(info[key].sum())
        reset = np.flatnonzero(done)
        if len(reset): env.set_states(reset, states[reset], tasks[reset])
        observation = env.observe()
        if not all(np.isfinite(value).all() for value in observation.values()):
            raise FloatingPointError('Nonfinite scaling benchmark observation')
    seconds = time.monotonic()-started
    report.update(seconds=seconds, transitions_s=args.worlds*args.controls/seconds,
        episodes=counts, options=env.engine.options, resource_status=resource_status(),
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        zero_gpu_overflow=not bool(env.engine.last_overflow.any()),
        sensor_audit_error_m=env.engine.last_sensor_audit_error)
    env.close()
    (args.out/'collection.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['tool','thread','axial','environment','reset','tool_replay','prepare_candidate','collection'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--worlds', type=int, default=4)
    parser.add_argument('--steps', type=int, default=16001)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--bank-dir', type=Path, default=Path('data/so101/banks_near_v2'))
    parser.add_argument('--native-threads', type=int, default=16)
    parser.add_argument('--controls', type=int, default=4)
    parser.add_argument('--tool-ramp', type=float, default=0.)
    parser.add_argument('--determinism', type=int, default=0, choices=[0,1,2,3,4,5,6,7])
    parser.add_argument('--reset-substeps', type=int, default=80)
    parser.add_argument('--candidate-mjb', type=Path, help='Explicit archived whole-helper SDF model; never rebuild canonical XML')
    parser.add_argument('--replay', type=Path, default=Path('artifacts_local/tool_pickup/raised10_coast'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    watchdog(args.out, args.timeout, gpu_limit_mib=14336 if args.case == 'collection' else 8192)
    if args.case == 'prepare_candidate':
        prepare_candidate(args)
    elif args.case == 'tool_replay':
        tool_replay(args)
    elif args.case == 'reset':
        reset_probe(args)
    elif args.case == 'environment':
        environment(args)
    elif args.case == 'collection':
        collection(args)
    else:
        mechanics(args)


if __name__ == '__main__':
    main()
