"""Isolated Menagerie-jaw MJWarp qualification; never changes the pilot or trains.

The only geometry substitution is the pinned public helper-jaw collision set.
Native and GPU paths share one candidate model and recorded initial conditions.
Any optional GPU solver changes are explicit in the output, never hidden in import.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import threading
import time

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from assembly_policy.so101 import scene
from assembly_policy.so101.menagerie_collision import replace_gripper_collision_menagerie
from assembly_policy.so101.robot import home
from assembly_policy.so101.warp_physics import WarpPhysics
from scripts import so101_warp_check as reference


ROOT = scene.ROOT


@wp.kernel
def _record_tool(qpos: wp.array2d[float], qvel: wp.array2d[float], stats: wp.array2d[float], button_q: int):
    world = wp.tid()
    wp.atomic_max(stats, world, 0, qpos[world, button_q])
    for i in range(qpos.shape[1]):
        if not wp.isfinite(qpos[world, i]):
            wp.atomic_max(stats, world, 2, 1.)
    for i in range(qvel.shape[1]):
        if not wp.isfinite(qvel[world, i]):
            wp.atomic_max(stats, world, 2, 1.)


@wp.kernel
def _record_contact(dist: wp.array[float], worldid: wp.array[int], ncon: wp.array[int], stats: wp.array2d[float]):
    i = wp.tid()
    if i < ncon[0]:
        wp.atomic_max(stats, worldid[i], 1, -dist[i])


class ProbePhysics(WarpPhysics):
    """Optional read-only per-physics-step metrics inside the captured graph."""
    tool_stats = None
    sdf_precision = 'fp64'

    def __init__(self, *args, **kwargs):
        if self.sdf_precision == 'fp64':
            super().__init__(*args, **kwargs)
        else:
            from assembly_policy.so101 import warp_physics
            from scripts import so101_gpu_fp32
            original = warp_physics.user_sdf, warp_physics.user_sdf_grad
            try:
                warp_physics.user_sdf, warp_physics.user_sdf_grad = so101_gpu_fp32.user_sdf, so101_gpu_fp32.user_sdf_grad
                super().__init__(*args, **kwargs)
            finally:
                warp_physics.user_sdf, warp_physics.user_sdf_grad = original

    def step(self, steps=1, graph=False):
        if self.tool_stats is None:
            return super().step(steps, graph)
        def advance():
            mjw.step(self.model, self.data)
            wp.launch(_record_tool, dim=self.data.nworld,
                      inputs=[self.data.qpos, self.data.qvel, self.tool_stats, self.button_q], device=self.device)
            wp.launch(_record_contact, dim=self.data.naconmax,
                      inputs=[self.data.contact.dist, self.data.contact.worldid, self.data.nacon, self.tool_stats], device=self.device)
        with wp.ScopedDevice(self.device):
            if graph:
                if steps not in self.graphs:
                    with wp.ScopedCapture(device=self.device) as capture:
                        for _ in range(steps):
                            advance()
                    self.graphs[steps] = capture.graph
                wp.capture_launch(self.graphs[steps])
            else:
                for _ in range(steps):
                    advance()


def candidate_xml(assembled=False):
    original = scene.replace_gripper_collision
    try:
        scene.replace_gripper_collision = replace_gripper_collision_menagerie
        return scene.build_xml(assembled)
    finally:
        scene.replace_gripper_collision = original


def memory_status():
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(memory['MemAvailable'].split()[0])*1024
    total = int(memory['MemTotal'].split()[0])*1024
    gpu = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
        text=True, timeout=5).strip()
    return dict(host_available_bytes=available, host_used_estimate_bytes=total-available,
                process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                gpu_memory_mib_and_utilization_percent=gpu)


def guard():
    status = memory_status()
    gpu_mib = max(float(row.split(',')[0]) for row in status['gpu_memory_mib_and_utilization_percent'].splitlines())
    if status['host_used_estimate_bytes'] > 12*1024**3 or gpu_mib > 8192:
        raise RuntimeError(f'Qualification resource limit exceeded: {status}')
    return status


def resource_watchdog(out):
    """Terminate only this scratch process if concurrent work consumes the cap."""
    def watch():
        while True:
            time.sleep(2)
            try:
                guard()
            except Exception as error:
                (out/'resource_guard.json').write_text(json.dumps(dict(error=str(error)))+'\n')
                os._exit(70)
    threading.Thread(target=watch, daemon=True).start()


def summarize_pair(m, d, p, initial_time, native_steps, heterogeneous=False):
    a = p.arrays()
    expected = initial_time+native_steps*m.opt.timestep
    increments = np.full(native_steps, np.float32(m.opt.timestep), dtype=np.float32)
    increments[0] += np.float32(initial_time)
    expected_gpu = np.cumsum(increments, dtype=np.float32)[-1]
    gpu_clock_error = float(abs(a['time']-expected_gpu).max())
    # These probes never reset. Time checks expose an engine reset as well as NaNs.
    native_finite = bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
    gpu_finite = bool(np.isfinite(a['qpos']).all() and np.isfinite(a['qvel']).all())
    parts = {}
    for name in (*scene.PARTS, 'driver'):
        joint = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name+'_free')
        if joint >= 0:
            q = m.jnt_qposadr[joint]
            selected = a['qpos'][:1] if heterogeneous else a['qpos']
            parts[name] = float(np.linalg.norm(selected[:,q:q+3]-d.qpos[q:q+3], axis=1).max())
    ncon = int(p.data.nacon.numpy()[0])
    gpu_dist = p.data.contact.dist.numpy()[:ncon]
    return dict(native_finite=native_finite, gpu_finite=gpu_finite,
                native_time=float(d.time), gpu_time=a['time'].tolist(), expected_time=float(expected),
                native_time_error=float(abs(d.time-expected)),
                gpu_time_error=float(abs(a['time']-expected).max()),
                gpu_float32_expected_time=float(expected_gpu),
                gpu_float32_clock_error=gpu_clock_error,
                gpu_clock_matches_expected_float32=bool(gpu_clock_error <= 2*np.spacing(expected_gpu)),
                native_warning_counts=d.warning.number.tolist(),
                overflow_bits=p.data.overflow.numpy().tolist(),
                parts_position_max_error_m=parts,
                native_ncon=int(d.ncon), gpu_ncon_total=ncon,
                native_sampled_max_depth_m=max([0., *(-c.dist for c in d.contact)]),
                gpu_sampled_max_depth_m=float(max(0., -gpu_dist.min())) if ncon else 0.,
                native_comparison_worlds='world0' if heterogeneous else 'all duplicated worlds',
                qpos_max_error=float(abs((a['qpos'][:1] if heterogeneous else a['qpos'])-d.qpos).max()),
                qvel_max_error=float(abs((a['qvel'][:1] if heterogeneous else a['qvel'])-d.qvel).max()))


def run(args):
    scene.load_plugin()
    xml = candidate_xml()
    (args.out/'candidate_scene.xml').write_text(xml)
    if args.case in ('thread', 'axial'):
        original_build = reference.build_xml
        reference.build_xml = candidate_xml
        try:
            m, d = reference.thread_model(args.case == 'axial')
        finally:
            reference.build_xml = original_build
    elif args.case == 'tool_sequence':
        from scripts.check_so101_tool_gripper import make_model
        m = make_model(collision_builder=replace_gripper_collision_menagerie)
        m.qpos0[m.joint('driver_free').qposadr[0]] += .0006
        d = mujoco.MjData(m)
        d.qpos[m.joint('gripper').qposadr[0]] = .4
        d.ctrl[0] = 152*2*np.pi/4096
        m.opt.gravity[:] = 0
        mujoco.mj_forward(m, d)
    else:
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        home(m, d)
        if args.case == 'actuator':
            d.ctrl[:5] += .02
        elif args.case == 'freefall':
            for name in ('base', 'motor', 'horn', 'retainer', 'yoke'):
                d.qpos[m.joint(name+'_free').qposadr[0]+2] += .1
            mujoco.mj_forward(m, d)
        elif args.case == 'button_release':
            d.qfrc_applied[m.joint('button_forward').dofadr[0]] = 2
    fixture_states = None
    if args.fixture:
        if args.case != 'zero':
            raise ValueError('A heterogeneous fixture is supported only for the zero-action diagnostic')
        from assembly_policy.so101.resets import signature
        previous = scene.replace_gripper_collision
        try:
            scene.replace_gripper_collision = replace_gripper_collision_menagerie
            expected_signature = signature()
        finally:
            scene.replace_gripper_collision = previous
        with np.load(args.fixture) as archive:
            if str(archive['model_signature']) != expected_signature:
                raise ValueError('Fixture signature differs from this candidate')
            fixture_states = archive['states'][np.arange(args.worlds) % len(archive['states'])].copy()
        if fixture_states.shape[1] != 1+m.nq+m.nv:
            raise ValueError('Fixture has additional state requiring an explicit mapping')
        mujoco.mj_setState(m, d, fixture_states[0], mujoco.mjtState.mjSTATE_FULLPHYSICS)
        d.time = 0
        d.ctrl[:] = d.qpos[:m.nu]
        mujoco.mj_forward(m, d)
    initial_qpos = d.qpos.copy()
    initial_ctrl = d.ctrl.copy()
    initial_time = d.time
    mujoco.mj_saveLastXML(str(args.out/'probe_model.xml'), m)
    p = ProbePhysics(m, d, nworld=args.worlds)
    if fixture_states is not None:
        p.data.qpos.assign(fixture_states[:,1:1+m.nq].astype(np.float32))
        p.data.qvel.assign(fixture_states[:,1+m.nq:].astype(np.float32))
        p.data.ctrl.assign(fixture_states[:,1:1+m.nu].astype(np.float32))
    if args.case == 'tool_sequence':
        p.tool_stats = wp.zeros((args.worlds, 3), dtype=float, device=p.device)
        p.button_q = int(m.joint('button_forward').qposadr[0])
    options = dict(native_tolerance=float(m.opt.tolerance), native_ls_iterations=int(m.opt.ls_iterations),
                   gpu_tolerance=float(m.opt.tolerance), gpu_ls_iterations=int(m.opt.ls_iterations))
    if args.gpu_tolerance is not None:
        p.model.opt.tolerance.fill_(args.gpu_tolerance)
        options['gpu_tolerance'] = args.gpu_tolerance
    if args.gpu_ls_iterations is not None:
        p.model.opt.ls_iterations = args.gpu_ls_iterations
        options['gpu_ls_iterations'] = args.gpu_ls_iterations
    report = dict(case=args.case, worlds=args.worlds, hostname=platform.node(), pid=os.getpid(),
                  scope='scratch Menagerie helper collisions; target CAD/thread/tool/arm controls retained',
                  shared_load='native CPU collection and intermittent CUDA PPO continue concurrently',
                  candidate_xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),
                  probe_xml_sha256=hashlib.sha256((args.out/'probe_model.xml').read_bytes()).hexdigest(),
                  native_nplugin=int(m.nplugin), nq=m.nq, nv=m.nv, ngeom=m.ngeom,
                  plugin_audit=p.plugin_audit, parameter_audit=p.parameter_audit,
                  solver_options=options, resources_before=guard(), samples=[])
    report['sdf_precision'] = args.sdf_precision
    report['fixture'] = str(args.fixture) if args.fixture else None
    report['precision_detail'] = 'FP64 original formulas' if args.sdf_precision == 'fp64' else 'FP32 interpolation/analytic equations; FP64 grid coordinates and boundary classification'
    def save():
        (args.out/(args.case+'.json')).write_text(json.dumps(report, indent=2)+'\n')
    save()
    def first_step_timeout():
        (args.out/'first_step_timeout.json').write_text(json.dumps(dict(timeout_s=60, phase='first synchronized step'))+'\n')
        os._exit(124)
    timer = threading.Timer(60, first_step_timeout)
    timer.daemon = True
    timer.start()
    start = time.perf_counter()
    p.step(); wp.synchronize_device(p.device)
    timer.cancel()
    report['first_step_s'] = time.perf_counter()-start
    mujoco.mj_step(m, d)
    count = 1
    print(json.dumps(dict(first_step_s=report['first_step_s'], resources=guard())), flush=True)
    report['samples'].append(summarize_pair(m, d, p, initial_time, count, fixture_states is not None))
    save()
    native_trace, gpu_trace = [], []
    native_tool_maximum = dict(button_m=0., depth_m=0.)
    gpu_seconds = native_seconds = 0.
    if args.case == 'tool_sequence':
        stages = [('close_zero_g', 1200, 152, False), ('carry_off', 3200, 152, True),
                  ('press_on', 5200, 130, True), ('release_off', 7200, 152, True),
                  ('open_drop', 7700, 220, True)]
    else:
        stages = [(args.case, args.steps, None, None)]
    for name, end, tick, gravity in stages:
        native_stage_maximum = dict(button_m=0., depth_m=0.)
        if tick is not None:
            d.ctrl[0] = tick*2*np.pi/4096
            p.data.ctrl.assign(np.broadcast_to(d.ctrl, (args.worlds, m.nu)).copy().astype(np.float32))
            m.opt.gravity[:] = [0, -9.81 if gravity else 0, 0]
            p.model.opt.gravity.assign(np.asarray(m.opt.gravity, np.float32).reshape(1, 3))
        while count < end:
            n = min(80, end-count)
            if args.case == 'button_release':
                if count < 400:
                    n = min(n, 400-count)
                else:
                    d.qfrc_applied[:] = 0
                    p.data.qfrc_applied.zero_()
            start = time.perf_counter(); p.step(n, graph=True); wp.synchronize_device(p.device)
            gpu_seconds += time.perf_counter()-start
            start = time.perf_counter()
            for _ in range(n):
                mujoco.mj_step(m, d)
                if args.case == 'tool_sequence':
                    native_tool_maximum['button_m'] = max(native_tool_maximum['button_m'], float(d.qpos[m.joint('button_forward').qposadr[0]]))
                    native_tool_maximum['depth_m'] = max(native_tool_maximum['depth_m'], max([0., *(-c.dist for c in d.contact)]))
                    native_stage_maximum['button_m'] = max(native_stage_maximum['button_m'], float(d.qpos[m.joint('button_forward').qposadr[0]]))
                    native_stage_maximum['depth_m'] = max(native_stage_maximum['depth_m'], max([0., *(-c.dist for c in d.contact)]))
            native_seconds += time.perf_counter()-start
            count += n
            sample = summarize_pair(m, d, p, initial_time, count, fixture_states is not None)
            sample.update(stage=name, step=count)
            if args.case == 'tool_sequence':
                actual_qpos = p.data.qpos.numpy()
                sample['gpu_button_range_m'] = [float(actual_qpos[:,m.joint('button_forward').qposadr[0]].min()),
                                                float(actual_qpos[:,m.joint('button_forward').qposadr[0]].max())]
            report['samples'].append(sample)
            native_trace.append(np.r_[d.time, d.qpos, d.qvel])
            a = p.arrays(); gpu_trace.append(np.r_[a['time'][0], a['qpos'][0], a['qvel'][0]])
            if not sample['native_finite'] or not sample['gpu_finite'] or sample['native_time_error'] > 1e-6 or not sample['gpu_clock_matches_expected_float32']:
                save(); raise RuntimeError(f'Invalid physics/time continuation: {sample}')
            if count % 400 == 0 or (count-1) % 400 == 0 or count == end:
                guard(); save()
        if args.case == 'tool_sequence':
            sample['mechanics'] = {}
            a = p.arrays()
            for engine, q, v in [('native', d.qpos, d.qvel), ('gpu', a['qpos'][0], a['qvel'][0])]:
                sample['mechanics'][engine] = dict(
                    button_m=float(q[m.joint('button_forward').qposadr[0]]),
                    spindle_rad_s=float(v[m.joint('spindle').dofadr[0]]),
                    driver_position=q[m.joint('driver_free').qposadr[0]:m.joint('driver_free').qposadr[0]+3].tolist())
            stats = p.tool_stats.numpy()
            sample['native_stage_maximum_every_step'] = native_stage_maximum
            sample['gpu_stage_maximum_every_step'] = dict(button_m=float(stats[:,0].max()),
                                                         depth_m=float(stats[:,1].max()),
                                                         any_nonfinite=bool(stats[:,2].any()))
            p.tool_stats.zero_()
            save()
        print(json.dumps(dict(stage=name, steps=count, last_sample=sample)), flush=True)
    report.update(paired_gpu_seconds=gpu_seconds, paired_native_seconds=native_seconds)
    if args.case in ('thread', 'axial'):
        from assembly_policy.so101.fastener import PITCH
        report['thread_mechanics'] = {}
        for engine, rows in [('native', native_trace), ('gpu', gpu_trace)]:
            trace = np.asarray(rows)
            start = int(np.searchsorted(trace[:,0], .3))
            if start < len(trace)-1:
                advance = trace[-1,1]-trace[start,1]
                turns = (trace[-1,2]-trace[start,2])/(2*np.pi)
                sample_turns = trace[start:,2]/(2*np.pi)
                fitted_pitch = float(np.polyfit(sample_turns, trace[start:,1], 1)[0]) if np.ptp(sample_turns) > .1 else None
                report['thread_mechanics'][engine] = dict(
                    advance_m=float(advance), turns=float(turns), pitch_error_m=float(advance-turns*PITCH),
                    fit_after_s=.3, fitted_pitch_m_per_turn=fitted_pitch,
                    nominal_pitch_m_per_turn=PITCH,
                    phase_residual_range_m=float(np.ptp(trace[start:,1]-sample_turns*PITCH)),
                    endpoint_gate_5um=bool(abs(advance-turns*PITCH) < 5e-6) if args.case == 'thread' else None,
                    axial_gate_20um=bool(abs(advance) < 20e-6) if args.case == 'axial' else None)
    if args.case == 'tool_sequence':
        report['native_tool_maximum_every_step'] = native_tool_maximum
        report['gpu_tool_contact_sampling_period_s'] = m.opt.timestep
        ends = {sample['stage']: sample for sample in report['samples'] if 'mechanics' in sample}
        report['tool_gates'] = {}
        for engine in ('native', 'gpu'):
            close, carry, press, release, opened = (ends[stage]['mechanics'][engine] for stage, *_ in stages)
            pos = lambda value: np.asarray(value['driver_position'])
            drift = max(np.linalg.norm(pos(value)-pos(close)) for value in (carry, press, release))
            drop = np.linalg.norm(pos(opened)-pos(release))
            report['tool_gates'][engine] = dict(
                closure_stays_off=ends['close_zero_g'][engine+'_stage_maximum_every_step']['button_m'] < .0006,
                carry_off=carry['button_m'] < .0006 and abs(carry['spindle_rad_s']) < .1,
                press_on=press['button_m'] > .0006 and press['spindle_rad_s'] > 20,
                release_off=release['button_m'] < .0006 and abs(release['spindle_rad_s']) < .1,
                retained_max_drift_m=float(drift), retained=bool(drift < .003),
                open_drop_m=float(drop), open_drops=bool(drop > .05))
            report['tool_gates'][engine]['contact_depth_below_0_3mm'] = all(
                sample[engine+'_stage_maximum_every_step']['depth_m'] < .0003 for sample in ends.values())
    report['controls_s'] = None
    if args.benchmark_controls:
        p.step(80, graph=True); wp.synchronize_device(p.device)
        start = time.perf_counter()
        for _ in range(args.benchmark_controls):
            p.step(80, graph=True)
        wp.synchronize_device(p.device)
        report['benchmark_seconds'] = time.perf_counter()-start
        report['controls_s'] = args.worlds*args.benchmark_controls/report['benchmark_seconds']
        report['benchmark_overflow_bits'] = p.data.overflow.numpy().tolist()
        report['benchmark_finite'] = bool(np.isfinite(p.data.qpos.numpy()).all() and np.isfinite(p.data.qvel.numpy()).all())
    report['resources_after'] = guard()
    report['all_samples_finite'] = all(s['native_finite'] and s['gpu_finite'] for s in report['samples'])
    report['all_samples_zero_overflow'] = all(not any(s['overflow_bits']) for s in report['samples'])
    report['all_samples_zero_native_warnings'] = all(not any(s['native_warning_counts']) for s in report['samples'])
    np.savez_compressed(args.out/(args.case+'_trace.npz'), native=native_trace, gpu=gpu_trace,
                        initial_qpos=initial_qpos, initial_ctrl=initial_ctrl)
    p.close(); save()
    print(json.dumps({k:v for k,v in report.items() if k not in ('samples', 'parameter_audit', 'plugin_audit')}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case', choices=['sdf','zero','actuator','freefall','button_release','thread','axial','tool_sequence','native_collection'], default='zero')
    ap.add_argument('--worlds', type=int, default=16)
    ap.add_argument('--steps', type=int, default=161)
    ap.add_argument('--benchmark-controls', type=int, default=4)
    ap.add_argument('--gpu-tolerance', type=float)
    ap.add_argument('--gpu-ls-iterations', type=int)
    ap.add_argument('--native-threads', type=int, default=4)
    ap.add_argument('--sdf-precision', choices=['fp64','fp32'], default='fp64')
    ap.add_argument('--fixture', type=Path, help='Candidate-only settled five-task fixture from native_collection')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if not 1 <= args.worlds <= 256 or args.steps < 2 or args.benchmark_controls < 0:
        ap.error('Need 1<=worlds<=256, steps>=2, benchmark-controls>=0')
    args.out.mkdir(parents=True, exist_ok=True)
    resource_watchdog(args.out)
    if args.case == 'native_collection':
        from scripts import so101_benchmark
        original_collision, original_argv = scene.replace_gripper_collision, sys.argv
        before = guard()
        try:
            scene.replace_gripper_collision = replace_gripper_collision_menagerie
            sys.argv = ['so101_benchmark', '--envs', str(args.worlds), '--threads', str(args.native_threads),
                        '--steps', str(args.steps), '--warmup', '1', '--out', str(args.out/'native_collection.json')]
            so101_benchmark.main()
        finally:
            scene.replace_gripper_collision, sys.argv = original_collision, original_argv
        (args.out/'resources.json').write_text(json.dumps(dict(before=before, after=guard()), indent=2)+'\n')
        return
    ProbePhysics.sdf_precision = args.sdf_precision
    guard(); wp.init()
    if args.case == 'sdf':
        scene.load_plugin()
        reference.model = lambda: mujoco.MjModel.from_xml_string(candidate_xml())
        reference.WarpPhysics = ProbePhysics
        if args.sdf_precision == 'fp32':
            from scripts.so101_gpu_fp32 import query_sdf
            reference.query_sdf = query_sdf
        report = reference.sdf_check(args.out)
        report['sdf_precision'] = args.sdf_precision
        (args.out/'sdf.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)
        return
    run(args)


if __name__ == '__main__':
    main()
