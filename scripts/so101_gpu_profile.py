"""Bounded, isolated GPU diagnostics; never instantiates a training environment.

Profile cached FP64 kernels or inspect broadphase candidates before expensive
SDF collision. No geometry, solver equations, installed library or bank changes.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time


def separating_slab(m, d, out):
    """Bound actual grid-negative support over every legal gripper angle.

    Trilinear interpolation is a convex combination of eight cell samples.
    Therefore a negative point requires at least one negative cell corner.
    Cover every such cell by per-X slabs, then maximize the rotated Z value
    analytically on the angle interval. Wrist roll preserves this axial Z.
    """
    import itertools
    import mujoco
    import numpy as np
    from assembly_policy.so101 import scene
    grid_path = scene.ASSETS/'sdf/grid_202.bin'
    with grid_path.open('rb') as stream:
        shape = np.fromfile(stream, np.int32, 3)
        header = np.fromfile(stream, np.float64, 4)
        values = np.fromfile(stream, np.float16).reshape(tuple(shape[::-1]))
    negative = values < 0
    boundary_min = float(min(values[0].min(), values[-1].min(), values[:, 0].min(),
                             values[:, -1].min(), values[:, :, 0].min(), values[:, :, -1].min()))
    if boundary_min <= 0:
        raise ValueError('Exterior grid continuation requires a separate proof')
    slabs = []
    for x in range(int(shape[0])):
        z, y = np.nonzero(negative[:, :, x])
        if len(y):
            slabs.extend(itertools.product((x-1, x+1), (int(y.min())-1, int(y.max())+1),
                                           (int(z.min())-1, int(z.max())+1)))
    support = np.asarray(slabs)*header[3]+header[:3]
    result = dict(grid_sha256=hashlib.sha256(grid_path.read_bytes()).hexdigest(),
                  grid_shape_xyz=shape.tolist(), grid_spacing_m=float(header[3]),
                  boundary_min_distance_m=boundary_min, support_box_corners=len(support),
                  scope='geometric separation over joint limits, no dynamic trajectory claim', sides=[])
    for side in ('left', 'right'):
        wrist = m.body(side+'_wrist').id
        fixed = m.body(side+'_gripper').id
        moving = m.body(side+'_moving_jaw_so101_v1').id
        roll, jaw = m.joint(side+'_wrist_roll'), m.joint(side+'_gripper')
        if not np.allclose(roll.axis, [0, 0, 1], atol=1e-12) or m.jnt_bodyid[roll.id] != fixed:
            raise ValueError('Wrist-roll axial invariance no longer holds')
        d.qpos[roll.qposadr] = d.qpos[jaw.qposadr] = 0
        mujoco.mj_forward(m, d)
        R, P = d.xmat[fixed].reshape(3, 3), d.xpos[fixed]
        wrist_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == wrist and (m.geom_contype[g] or m.geom_conaffinity[g])]
        moving_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == moving and (m.geom_contype[g] or m.geom_conaffinity[g])]
        if len(moving_geoms) != 1 or m.geom_type[moving_geoms[0]] != mujoco.mjtGeom.mjGEOM_SDF:
            raise ValueError('Expected exactly one moving-jaw SDF')
        row = dict(side=side, wrist_body=m.body(wrist).name, moving_body=m.body(moving).name,
                   wrist_geoms=[], moving_geom=moving_geoms[0], roll_limits=roll.range.tolist(),
                   gripper_limits=jaw.range.tolist())
        wrist_min = np.inf
        for g in wrist_geoms:
            mesh = m.geom_dataid[g]
            vertices = m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]]
            points = (vertices@d.geom_xmat[g].reshape(3, 3).T+d.geom_xpos[g]-P)@R
            low = float(points[:, 2].min())
            wrist_min = min(wrist_min, low)
            row['wrist_geoms'].append(dict(id=g, mesh=m.mesh(mesh).name, faces=int(m.mesh_facenum[mesh]), min_axial_z_m=low))
        g = moving_geoms[0]
        points = (support@d.geom_xmat[g].reshape(3, 3).T+d.geom_xpos[g]-P)@R
        anchor, axis = (d.xanchor[jaw.id]-P)@R, d.xaxis[jaw.id]@R
        relative = points-anchor
        parallel = np.outer(relative@axis, axis)
        cosine, sine, constant = (relative-parallel)[:, 2], np.cross(axis, relative)[:, 2], (anchor+parallel)[:, 2]
        row['jaw_axis_in_roll_frame'] = axis.tolist()
        row['jaw_anchor_in_roll_frame_m'] = anchor.tolist()
        row['bounds'] = []
        for padding in (0., .05):
            lo, hi = jaw.range+np.array([-padding, padding])
            maximum = np.maximum(cosine*np.cos(lo)+sine*np.sin(lo)+constant,
                                 cosine*np.cos(hi)+sine*np.sin(hi)+constant)
            for k in range(-3, 4):
                angle = np.arctan2(sine, cosine)+k*np.pi
                keep = (angle >= lo) & (angle <= hi)
                value = cosine*np.cos(angle)+sine*np.sin(angle)+constant
                maximum[keep] = np.maximum(maximum[keep], value[keep])
            highest = float(maximum.max())
            row['bounds'].append(dict(extra_angle_rad=padding, jaw_max_axial_z_m=highest,
                                      separating_distance_lower_bound_m=float(wrist_min-highest)))
        result['sides'].append(row)
    (out/'separation.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def resource_status():
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines())
    gpu = subprocess.check_output([
        'nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True, timeout=5)
    return dict(available_bytes=int(memory['MemAvailable'].split()[0])*1024,
                rss_bytes=int(status['VmRSS'].split()[0])*1024,
                gpu_used_mib=max(float(x) for x in gpu.splitlines()))


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(pid=os.getpid(), geometry=args.geometry, mode=args.mode,
                  worlds=args.worlds, sdf_precision=args.sdf_precision,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  resource_limits=dict(rss_gib=args.rss_limit_gib, available_host_gib=2,
                                       total_gpu_mib=8192, wall_seconds=args.timeout),
                  scope='isolated read-only profiling, no training', phases=[])
    report_path = args.out/'profile.json'
    def save():
        report_path.write_text(json.dumps(report, indent=2)+'\n')
    def phase(name):
        report['phases'].append(dict(name=name, monotonic=time.monotonic(), resources=resource_status()))
        save()
        print(name, flush=True)
    def watchdog():
        deadline = time.monotonic()+args.timeout
        while True:
            try:
                state = resource_status()
                if state['available_bytes'] < 2*1024**3 or state['rss_bytes'] > args.rss_limit_gib*1024**3 or state['gpu_used_mib'] > 8192:
                    report['stopped'] = dict(reason='resource cap', phase_seconds=time.monotonic()-report['phases'][-1]['monotonic'], **state)
                elif time.monotonic() > deadline:
                    report['stopped'] = dict(reason='wall-time cap', seconds=args.timeout, **state)
                else:
                    time.sleep(1)
                    continue
                save()
                os._exit(70)
            except Exception as error:
                report['stopped'] = dict(reason='watchdog error', error=str(error))
                save()
                os._exit(71)
    threading.Thread(target=watchdog, daemon=True).start()
    phase('imports')
    import mujoco
    import numpy as np
    import warp as wp
    from mujoco_warp._src import collision_driver, collision_core, collision_sdf, forward
    from assembly_policy.so101 import scene
    from assembly_policy.so101.robot import home
    from assembly_policy.so101.warp_physics import WarpPhysics
    from scripts.check_so101_gpu_candidate import candidate_xml, ProbePhysics
    report['versions'] = dict(mujoco=mujoco.__version__, warp=wp.__version__)
    report['installed_collision_sdf_sha256'] = hashlib.sha256(Path(collision_sdf.__file__).read_bytes()).hexdigest()
    scene.load_plugin()
    xml = candidate_xml() if args.geometry == 'menagerie' else scene.build_xml()
    report['xml_sha256'] = hashlib.sha256(xml.encode()).hexdigest()
    m = mujoco.MjModel.from_xml_string(xml)
    original_model = m
    d = mujoco.MjData(m)
    home(m, d)
    if args.mode == 'separation':
        if args.geometry != 'fine':
            raise ValueError('Separation proof applies only to the fine production collision geometry')
        report['separation'] = separating_slab(m, d, args.out)
        phase('complete')
        print(json.dumps(report['separation'], indent=2), flush=True)
        return
    if args.exclude_separated:
        if args.geometry != 'fine':
            raise ValueError('The proof applies only to the fine geometry')
        import xml.etree.ElementTree as ET
        proof = separating_slab(m, d, args.out)
        if min(bound['separating_distance_lower_bound_m'] for side in proof['sides'] for bound in side['bounds']) < .001:
            raise ValueError('Separation proof has insufficient margin')
        root = ET.fromstring(xml)
        contact = root.find('contact')
        if contact is None:
            contact = ET.SubElement(root, 'contact')
        for side in proof['sides']:
            ET.SubElement(contact, 'exclude', body1=side['wrist_body'], body2=side['moving_body'])
        xml = ET.tostring(root, encoding='unicode')
        (args.out/'excluded_scene.xml').write_text(xml)
        report['excluded_xml_sha256'] = hashlib.sha256(xml.encode()).hexdigest()
        report['exclusions'] = [(side['wrist_body'], side['moving_body']) for side in proof['sides']]
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        home(m, d)
    states = None
    if args.bank_dir:
        from assembly_policy.so101.resets import check_bank
        check_bank(args.bank_dir)
        independent = []
        if args.bank_selection == 'motor-near':
            with np.load(args.bank_dir/'motor_train.npz') as archive:
                independent.extend(archive['states'][archive['family'] == 4])
            if not independent:
                raise ValueError('No held near motor starts')
        else:
            for task in ('motor', 'horn', 'screw', 'retainer', 'yoke'):
                with np.load(args.bank_dir/(task+'_eval.npz')) as archive:
                    independent.append(archive['states'][0])
        states = np.stack(independent)[np.arange(args.worlds) % len(independent)].copy()
        report['bank_dir'] = str(args.bank_dir)
        report['bank_selection'] = args.bank_selection
        report['bank_manifest_sha256'] = hashlib.sha256((args.bank_dir/'manifest.json').read_bytes()).hexdigest()
        mujoco.mj_setState(m, d, states[0], mujoco.mjtState.mjSTATE_FULLPHYSICS)
        d.time = 0
        d.ctrl[:] = d.qpos[:m.nu]
        mujoco.mj_forward(m, d)
    if args.fixture:
        if args.geometry != 'menagerie':
            raise ValueError('Existing fixture is for the Menagerie candidate only')
        with np.load(args.fixture) as archive:
            states = archive['states'][np.arange(args.worlds) % len(archive['states'])].copy()
            report['fixture_signature'] = str(archive['model_signature'])
        mujoco.mj_setState(m, d, states[0], mujoco.mjtState.mjSTATE_FULLPHYSICS)
        d.time = 0
        d.ctrl[:] = d.qpos[:m.nu]
        mujoco.mj_forward(m, d)
    phase('model import')
    if args.sdf_precision == 'mixed':
        ProbePhysics.sdf_precision = 'fp32'
        p = ProbePhysics(m, d, nworld=args.worlds)
    else:
        p = WarpPhysics(m, d, nworld=args.worlds)
    p.model.opt.ls_iterations = 100
    if states is not None:
        p.data.qpos.assign(states[:, 1:1+m.nq].astype(np.float32))
        p.data.qvel.assign(states[:, 1+m.nq:].astype(np.float32))
        p.data.ctrl.assign(states[:, 1:1+m.nu].astype(np.float32))
    report['options'] = dict(sdf_initpoints=int(m.opt.sdf_initpoints), sdf_iterations=int(m.opt.sdf_iterations),
                             main_iterations=int(m.opt.iterations), native_ls=int(m.opt.ls_iterations), gpu_ls=100,
                             tolerance=float(m.opt.tolerance), timestep=float(m.opt.timestep))
    phase('broadphase')
    forward.fwd_kinematics(p.model, p.data)
    ctx = collision_core.create_collision_context(p.data.naconmax)
    p.data.ncollision.zero_()
    collision_driver.nxn_broadphase(p.model, p.data, ctx)
    wp.synchronize_device(p.device)
    n = int(p.data.ncollision.numpy()[0])
    pairs = ctx.collision_pair.numpy()[:n]
    worlds = ctx.collision_worldid.numpy()[:n]
    counter = defaultdict(int)
    for pair, world in zip(pairs, worlds):
        counter[tuple(map(int, pair))] += 1
    report['broadphase_pairs'] = []
    for pair, count in sorted(counter.items()):
        geoms = []
        for g in pair:
            mesh = int(m.geom_dataid[g])
            geoms.append(dict(id=g, name=m.geom(g).name, body=m.body(int(m.geom_bodyid[g])).name,
                              type=int(m.geom_type[g]), mesh_faces=int(m.mesh_facenum[mesh]) if mesh >= 0 else 0))
        report['broadphase_pairs'].append(dict(geoms=geoms, worlds=count))
    report['ncollision'] = n
    report['native_initial_contacts'] = [dict(pair=[int(c.geom1), int(c.geom2)], distance=float(c.dist)) for c in d.contact]
    save()
    if args.mode == 'broadphase':
        phase('complete')
        p.close()
        return
    if args.mode == 'sdf-pairs':
        # Only this diagnostic's scratch collision buffer is filtered. This is
        # not a valid physics step and cannot qualify a changed contact model.
        report['pair_timings'] = []
        for pair in sorted(counter):
            if not any(int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_SDF) for g in pair):
                continue
            if args.pair and tuple(args.pair) != pair:
                continue
            phase('SDF pair '+str(pair))
            chosen = np.all(pairs == pair, axis=1)
            count = int(chosen.sum())
            pair_buffer = np.zeros(ctx.collision_pair.shape+(2,), dtype=np.int32)
            world_buffer = np.zeros(ctx.collision_worldid.shape, dtype=np.int32)
            pair_buffer[:count] = pairs[chosen]
            world_buffer[:count] = worlds[chosen]
            sub = collision_core.create_collision_context(p.data.naconmax)
            sub.collision_pair.assign(pair_buffer)
            sub.collision_pairid.fill_(wp.vec2i(-1, -1))
            sub.collision_worldid.assign(world_buffer)
            p.data.ncollision.fill_(count)
            p.data.nacon.zero_()
            with wp.ScopedTimer('pair', print=False, cuda_filter=wp.TIMING_KERNEL, synchronize=True) as timer:
                collision_sdf.sdf_narrowphase(p.model, p.data, sub)
            report['pair_timings'].append(dict(pair=pair, worlds=count, wall_ms=timer.elapsed,
                kernel_ms=sum(item.elapsed for item in timer.timing_results), ncon=int(p.data.nacon.numpy()[0]),
                distances=p.data.contact.dist.numpy()[:int(p.data.nacon.numpy()[0])].tolist()))
            save()
        phase('complete')
        print(json.dumps(report['pair_timings'], indent=2), flush=True)
        p.close()
        return
    phase('first step')
    start = time.perf_counter()
    p.step()
    wp.synchronize_device(p.device)
    report['first_step_seconds'] = time.perf_counter()-start
    phase('kernel timing')
    with wp.ScopedTimer('profile', print=False, cuda_filter=wp.TIMING_KERNEL, synchronize=True) as timer:
        p.step(args.steps)
    totals = defaultdict(lambda: [0, 0.])
    for item in timer.timing_results:
        totals[item.name][0] += 1
        totals[item.name][1] += item.elapsed
    total = sum(row[1] for row in totals.values())
    report['timed_steps'] = args.steps
    report['profile_wall_ms'] = timer.elapsed
    report['kernel_total_ms'] = total
    report['kernel_times'] = [dict(name=name, calls=row[0], milliseconds=row[1], percent=100*row[1]/total)
                              for name, row in sorted(totals.items(), key=lambda x: -x[1][1])]
    report['finite'] = bool(np.isfinite(p.data.qpos.numpy()).all() and np.isfinite(p.data.qvel.numpy()).all())
    report['overflow_bits'] = p.data.overflow.numpy().tolist()
    ncon = int(p.data.nacon.numpy()[0])
    grouped = defaultdict(list)
    for pair, distance in zip(p.data.contact.geom.numpy()[:ncon], p.data.contact.dist.numpy()[:ncon]):
        grouped[tuple(map(int, pair))].append(float(distance))
    report['gpu_final_contacts'] = [dict(pair=pair, count=len(distances), min_distance=min(distances), max_distance=max(distances))
                                     for pair, distances in sorted(grouped.items())]
    if args.exclude_separated:
        # Paired native continuations test the exact exclusion change, including
        # near-held jaw contacts. Both paths receive the same fixed commands.
        native_rows = []
        selected = np.stack(independent) if args.bank_dir else [None]
        for state in selected:
            before, after = mujoco.MjData(original_model), mujoco.MjData(m)
            for model, data in ((original_model, before), (m, after)):
                if state is None:
                    home(model, data)
                else:
                    mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
                    data.ctrl[:] = data.qpos[:model.nu]
                mujoco.mj_step(model, data, nstep=80)
            native_rows.append(dict(qpos_error=float(abs(before.qpos-after.qpos).max()),
                                    qvel_error=float(abs(before.qvel-after.qvel).max()),
                                    warnings_before=before.warning.number.tolist(), warnings_after=after.warning.number.tolist()))
        report['native_exclusion_80step_parity'] = native_rows
    if args.benchmark_controls:
        phase('captured 80-step throughput')
        p.step(80, graph=True)
        wp.synchronize_device(p.device)
        start = time.perf_counter()
        for _ in range(args.benchmark_controls):
            p.step(80, graph=True)
        wp.synchronize_device(p.device)
        elapsed = time.perf_counter()-start
        report['benchmark'] = dict(controls=args.benchmark_controls, physics_steps_per_control=80,
            seconds=elapsed, transitions_per_second=args.worlds*args.benchmark_controls/elapsed,
            finite=bool(np.isfinite(p.data.qpos.numpy()).all() and np.isfinite(p.data.qvel.numpy()).all()),
            overflow_bits=p.data.overflow.numpy().tolist())
    phase('complete')
    print(json.dumps(report['kernel_times'][:12], indent=2), flush=True)
    p.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', choices=('fine', 'menagerie'), default='menagerie')
    parser.add_argument('--mode', choices=('broadphase', 'profile', 'sdf-pairs', 'separation'), default='profile')
    parser.add_argument('--pair', type=int, nargs=2)
    parser.add_argument('--worlds', type=int, default=64)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--timeout', type=float, default=90)
    parser.add_argument('--rss-limit-gib', type=float, default=4.5)
    parser.add_argument('--sdf-precision', choices=('fp64', 'mixed'), default='fp64')
    parser.add_argument('--fixture', type=Path)
    parser.add_argument('--bank-dir', type=Path)
    parser.add_argument('--bank-selection', choices=('five-task-eval', 'motor-near'), default='five-task-eval')
    parser.add_argument('--exclude-separated', action='store_true')
    parser.add_argument('--benchmark-controls', type=int, default=0)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.fixture and args.bank_dir:
        parser.error('Use either --fixture or --bank-dir')
    run(args)
