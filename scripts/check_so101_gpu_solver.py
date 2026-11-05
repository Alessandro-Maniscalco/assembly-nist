"""Replay one archived failing world with explicit GPU settings and solver diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import mujoco_warp as mjw
from mujoco_warp._src import solver
import numpy as np
import warp as wp

from assembly_policy.so101.env import STATE_SPEC
from assembly_policy.so101.warp_env import (archived_helper_factory, MixedPhysics, physics_identity,
    GPU_SOLVER_ITERATIONS, GPU_SOLVER_TOLERANCE)
from scripts.check_so101_warp_env import watchdog


@wp.kernel
def record(niter: wp.array[int], scale: wp.array[float], grad: wp.array[float],
           decrement: wp.array[float], improvement: wp.array[float], alpha: wp.array[float],
           qacc: wp.array2d[float], previous: wp.array2d[float], diagnostics: wp.array2d[float]):
    index = niter[0]-1
    if index >= 0 and index < diagnostics.shape[0]:
        diagnostics[index, 0] = alpha[0]
        diagnostics[index, 1] = scale[0]*improvement[0]
        diagnostics[index, 2] = scale[0]*wp.sqrt(grad[0])
        diagnostics[index, 3] = scale[0]*.5*decrement[0]
        difference, magnitude = float(0.), float(0.)
        for field in range(qacc.shape[1]):
            difference = wp.max(difference, wp.abs(qacc[0, field]-previous[0, field]))
            magnitude = wp.max(magnitude, wp.abs(qacc[0, field]))
            previous[0, field] = qacc[0, field]
        diagnostics[index, 4] = difference
        diagnostics[index, 5] = magnitude


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--failure', type=Path, required=True)
    parser.add_argument('--candidate-mjb', type=Path, required=True)
    parser.add_argument('--world', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    watchdog(args.out, 240)
    started = time.monotonic()
    with np.load(args.failure) as source:
        initial = source['initial_state'][args.world].copy()
        controls = source['control'][args.world].copy()
    factory = archived_helper_factory(args.candidate_mjb)
    provenance = physics_identity(factory)
    model = factory(contact_monitor=True)
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, initial, STATE_SPEC)
    native, native_diagnostics = [], []
    for control in controls:
        data.ctrl[:] = control
        mujoco.mj_step(model, data)
        state = np.empty_like(initial)
        mujoco.mj_getState(model, data, state, STATE_SPEC)
        native.append(state)
        native_diagnostics.append([int(data.solver_niter.max()), float(data.sensordata[0]),
                                   int(data.warning.number.sum())])
    physics = MixedPhysics(model, nworld=1)
    physics.model.opt.iterations = GPU_SOLVER_ITERATIONS
    physics.model.opt.ls_iterations = 100
    physics.model.opt.contact_sensor_maxmatch = 512
    physics.model.opt.tolerance.fill_(GPU_SOLVER_TOLERANCE)
    iteration_diagnostics = wp.zeros((GPU_SOLVER_ITERATIONS, 6), dtype=float, device=physics.device)
    previous = wp.zeros((1, model.nv), dtype=float, device=physics.device)
    original = solver._solver_iteration

    def instrument(m, d, ctx, nsolving, compact=False):
        original(m, d, ctx, nsolving, compact)
        wp.launch(record, dim=1, inputs=[d.solver_niter, ctx.inertia_scale, ctx.grad_dot,
            ctx.newton_decrement, ctx.improvement, ctx.alpha, d.qacc, previous, iteration_diagnostics])

    solver._solver_iteration = instrument
    gpu, gpu_diagnostics, traces = [], [], []
    try:
        with wp.ScopedDevice(physics.device):
            def restore():
                mjw.reset_data(physics.model, physics.data)
                physics.data.time.assign(np.array([initial[0]], np.float32))
                physics.data.qpos.assign(initial[None, 1:1+model.nq].astype(np.float32))
                physics.data.qvel.assign(initial[None, 1+model.nq:].astype(np.float32))
            restore()
            physics.data.ctrl.assign(controls[0, None].astype(np.float32))
            mjw.step(physics.model, physics.data)
            wp.synchronize_device(physics.device)
            restore()
            with wp.ScopedCapture(device=physics.device) as capture:
                mjw.step(physics.model, physics.data)
            for control in controls:
                physics.data.ctrl.assign(control[None].astype(np.float32))
                iteration_diagnostics.zero_()
                previous.zero_()
                wp.capture_launch(capture.graph)
                gpu.append(np.r_[physics.data.time.numpy()[0], physics.data.qpos.numpy()[0],
                                 physics.data.qvel.numpy()[0]])
                traces.append(iteration_diagnostics.numpy())
                gpu_diagnostics.append([int(physics.data.solver_niter.numpy()[0]),
                    int(physics.data.overflow.numpy()[0]), float(physics.data.sensordata.numpy()[0, 0])])
    finally:
        solver._solver_iteration = original
        physics.close()
    native, gpu = np.asarray(native), np.asarray(gpu)
    native_diagnostics, gpu_diagnostics = np.asarray(native_diagnostics), np.asarray(gpu_diagnostics)
    report = dict(world=args.world, elapsed_seconds=time.monotonic()-started,
        input_sha256=hashlib.sha256(args.failure.read_bytes()).hexdigest(), physics_provenance=provenance,
        native_tolerance=float(model.opt.tolerance), gpu_tolerance=GPU_SOLVER_TOLERANCE,
        native_iterations=int(model.opt.iterations), gpu_iterations=GPU_SOLVER_ITERATIONS,
        native_max_iterations=int(native_diagnostics[:, 0].max()),
        native_max_iterations_scope='maximum over all constraint islands',
        gpu_max_iterations=int(gpu_diagnostics[:, 0].max()),
        gpu_first_flag_step=next((i for i, row in enumerate(gpu_diagnostics) if row[1]), None),
        max_arm_velocity_rad_s=float(abs(gpu[:, 1+model.nq:1+model.nq+12]).max()),
        max_depth_m=-float(gpu_diagnostics[:, 2].min()), part_differences={},
        method='Single-world native and instrumented GPU replay. Per-iteration diagnostics only; no physical equations changed. Exact original multiworld trajectory equality is not claimed.',
        diagnostic_columns=['alpha', 'scaled_improvement', 'scaled_gradient_norm',
                            'scaled_half_Newton_decrement', 'max_acceleration_iteration_change',
                            'max_absolute_acceleration'])
    for name in ('base', 'motor', 'horn', 'screw', 'retainer', 'yoke', 'driver'):
        q = 1+int(model.joint(name+'_free').qposadr[0])
        qa, qb = gpu[:, q+3:q+7].astype(float), native[:, q+3:q+7].astype(float)
        qa /= np.linalg.norm(qa, axis=1, keepdims=True)
        qb /= np.linalg.norm(qb, axis=1, keepdims=True)
        report['part_differences'][name] = dict(
            position_max_m=float(np.linalg.norm(gpu[:, q:q+3]-native[:, q:q+3], axis=1).max()),
            rotation_max_rad=float((2*np.arccos(np.clip(abs((qa*qb).sum(1)), 0, 1))).max()))
    report['checks'] = dict(finite=bool(np.isfinite(native).all() and np.isfinite(gpu).all()),
        zero_native_warnings=bool(not native_diagnostics[:, 2].any()),
        zero_gpu_overflow=bool(not gpu_diagnostics[:, 1].any()),
        penetration_below_0_4mm=report['max_depth_m'] < .0004,
        arm_velocity_below_5_rad_s=report['max_arm_velocity_rad_s'] < 5)
    report['pass'] = all(report['checks'].values())
    np.savez_compressed(args.out/'solver_replay.npz', initial_state=initial, controls=controls,
        native=native, native_diagnostics=native_diagnostics, gpu=gpu,
        gpu_diagnostics=gpu_diagnostics, iterations=np.asarray(traces))
    (args.out/'solver_replay.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
