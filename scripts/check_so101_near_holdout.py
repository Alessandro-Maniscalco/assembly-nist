"""Fixed-seed prepared-near holdout; never a loose-part assembly benchmark.

An explicit output directory can reuse immutable, qualified preparation from a
previous run. Preparation uses native generators without changing geometry.
"""
import argparse
from collections import deque
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts_local/near_holdout'
TASKS = ('motor', 'horn', 'retainer')
SEEDS = tuple(range(2000001, 2000005))
SOURCES = {
    'motor': ROOT/'artifacts_local/prepared_holds/motor_goal_000000.json',
    'horn': ROOT/'artifacts_local/prepared_parts/horn_edge.json',
    'retainer': ROOT/'artifacts_local/prepared_parts/retainer_front.json',
}


class ResourceLimitError(RuntimeError):
    """Abort the complete probe rather than replace a resource-failed trial."""


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2)+'\n')
    os.replace(tmp, path)


def resources(deadline):
    import psutil
    process = psutil.Process()
    rss = process.memory_info().rss
    if rss > 2.5*1024**3:
        raise ResourceLimitError(f'Probe RSS exceeds 2.5 GiB: {rss}')
    if psutil.virtual_memory().available < 2*1024**3:
        raise ResourceLimitError('Host available memory is below 2 GiB')
    if time.time() >= deadline:
        raise TimeoutError('Bounded holdout deadline reached')
    return dict(rss_bytes=rss, available_bytes=psutil.virtual_memory().available,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)


def run_child(command, deadline, env, stdout=None):
    """Watch memory even while native compilation cannot call resources()."""
    import psutil
    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout,
                            stderr=subprocess.STDOUT if stdout else None)
    try:
        child = psutil.Process(proc.pid)
        while proc.poll() is None:
            resources(deadline)
            try:
                child_rss = child.memory_info().rss
            except psutil.NoSuchProcess:
                if proc.poll() is not None:
                    break
                raise
            if child_rss > 2.5*1024**3:
                raise ResourceLimitError('Child probe RSS exceeds 2.5 GiB')
            time.sleep(.5)
        return proc.returncode
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def reuse_preparation(directory):
    directory = Path(directory).resolve()
    if directory == OUT:
        raise ValueError('Reused preparation must remain in a separate directory')
    original = json.loads((directory/'manifest.json').read_text())
    reference = json.loads((directory/'results.json').read_text())
    assert original['fixed_tasks'] == list(TASKS) and original['fixed_seeds'] == list(SEEDS)
    assert original['replacements_allowed'] is False
    assert reference['completed'] and len(reference['preparation']) == 12
    assert sha(directory/'policy.pt') == original['checkpoint_sha256'] == reference['checkpoint_sha256']
    qualified = {(r['task'], r['seed']): r for r in reference['preparation']}
    hashes, states, model_hashes = {}, {}, {}
    for task, source in original['sources'].items():
        assert Path(source['path']).resolve() == SOURCES[task].resolve()
        assert sha(source['path']) == source['sha256'], f'Changed preparation source: {task}'
    for task in TASKS:
        for seed in SEEDS:
            row = qualified[task, seed]
            assert row['qualified'], f'Unqualified archived start: {task}/{seed}'
            stem = f'{task}_{seed}'
            states[stem] = row['state_sha256']
            for suffix in ('.json', '.npz', '.mjb'):
                path = directory/(stem+suffix)
                stat = path.stat()
                identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
                if suffix == '.mjb' and identity in model_hashes:
                    digest = model_hashes[identity]
                else:
                    digest = sha(path)
                    if suffix == '.mjb':
                        model_hashes[identity] = digest
                hashes[path.name] = digest
    return dict(preparation_directory=str(directory),
                preparation_source_manifest_sha256=sha(directory/'manifest.json'),
                preparation_reference_results_sha256=sha(directory/'results.json'),
                preparation_artifact_sha256=hashes, preparation_state_sha256=states,
                reference_checkpoint_sha256=reference['checkpoint_sha256'],
                reference_summary=reference['summary'],
                reference_model_signature=reference['model_signature'])


def generate(task, seed, deadline):
    usage = resources(deadline)
    print(json.dumps({'preparation_resources': usage}), flush=True)
    source = json.loads(SOURCES[task].read_text())
    target = OUT/f'{task}_{seed}'
    if task == 'motor':
        from assembly_policy.so101.prepared import generate as prepare
        from assembly_policy.so101.scene import model
        prepare(model(), seed, target, release=False, gap=0., withdraw=False)
    else:
        from assembly_policy.so101.prepared_parts import generate as prepare
        prepare(task, seed, source['prepared_grasp_offset_m'],
                source['initial_gripper_position_rad'], target,
                withdraw=False, release_check=False, return_check=False)
    print(json.dumps({'preparation_final_resources': resources(deadline)}), flush=True)


def evaluate(deadline):
    import mujoco
    import numpy as np
    from assembly_policy.so101.env import BatchedAssemblyEnv, EnvConfig, HISTORY, PROPRIO_DIM, resolve_tasks
    from assembly_policy.so101.resets import signature

    # Compile the large visual model before importing Torch and return freed
    # compiler buffers to libc. PointTables reuses this exact read-only model,
    # avoiding a second copy of the 372 MiB visual mesh asset set.
    manifest = json.loads((OUT/'manifest.json').read_text())
    source_config = json.loads((OUT/'config.json').read_text())
    preparation_dir = Path(manifest.get('preparation_directory', OUT))
    assert sha(OUT/'policy.pt') == manifest['checkpoint_sha256'], 'Frozen checkpoint changed'
    assert sha(OUT/'config.json') == manifest['config_sha256'], 'Frozen configuration changed'
    if preparation_dir != OUT:
        assert reuse_preparation(preparation_dir) == manifest['reused_preparation'], 'Archived preparation changed'
    tasks = resolve_tasks(source_config['tasks'])
    env = BatchedAssemblyEnv(tasks, [0], EnvConfig(**{**source_config['env'], 'nthread': 1}))
    import ctypes
    gc.collect()
    ctypes.CDLL('libc.so.6').malloc_trim(0)
    print(json.dumps({'compiled_environment_resources': resources(deadline)}), flush=True)

    import torch
    from unittest.mock import patch
    from assembly_policy.rl.nets import ActorCritic
    from assembly_policy.so101.obs import PointTables

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    checkpoint = torch.load(OUT/'policy.pt', map_location='cpu', weights_only=False)
    assert checkpoint['it'] >= manifest.get('minimum_checkpoint_iteration', 0), 'Checkpoint predates requested iteration'
    cfg = checkpoint['args']
    assert json.loads(json.dumps(cfg['env'])) == source_config['env'] and cfg['tasks'] == source_config['tasks']
    assert cfg['model_signature'] == signature(), 'Canonical model changed'
    if preparation_dir != OUT:
        assert cfg['model_signature'] == manifest['reused_preparation']['reference_model_signature']
    assert cfg['env']['max_contact_depth'] == .0004
    assert cfg['env']['stability_metric'] == 'pose'
    tasks = resolve_tasks(cfg['tasks'])
    result = json.loads((OUT/'manifest.json').read_text())
    result['script_sha256'] = sha(__file__)
    result['generator_source_sha256'] = {name: sha(ROOT/'assembly_policy/so101'/name)
                                         for name in ('prepared.py', 'prepared_parts.py', 'resets.py', 'trajectory.py')}
    result.update(checkpoint_iteration=checkpoint['it'], checkpoint_transitions=checkpoint['transitions'],
                  parent_policy=cfg.get('parent_policy'), environment=cfg['env'],
                  model_signature=cfg['model_signature'], bank_signature=cfg.get('bank_signature'),
                  training_backend=cfg.get('physics_backend', 'native'), evaluation_backend='native',
                  training_physics_signature=cfg.get('physics_signature', cfg['model_signature']),
                  evaluation_seed=880301, preparation=[], episodes=[],
                  completion_scope='Production control-sampled release and eleven-pose 0.4 s retention; no claim of continuous no-contact retention')
    lineage, parent, visited = [], cfg.get('parent_policy'), set()
    while parent and parent.get('path') not in visited:
        path = Path(parent['path'])
        visited.add(str(path))
        if parent.get('sha256'):
            assert sha(path) == parent['sha256'], 'Parent checkpoint changed'
        ancestor = torch.load(path, map_location='cpu', weights_only=False)
        lineage.append(dict(path=str(path), sha256=sha(path), iteration=ancestor['it'],
                            transitions=ancestor['transitions'], model_signature=ancestor['args'].get('model_signature'),
                            bank_signature=ancestor['args'].get('bank_signature'),
                            optimizer_reused=parent.get('optimizer_reused'), sampler_reused=parent.get('sampler_reused')))
        parent = ancestor['args'].get('parent_policy')
        del ancestor
    result['parent_lineage'] = lineage
    with patch('assembly_policy.so101.obs.model', return_value=env.m):
        tables = PointTables(tasks, cfg['n_part'], cfg['n_fix'], cfg['n_grip'])
    net = ActorCritic(tables.n_pts, HISTORY*PROPRIO_DIM, 12)
    net.load_state_dict(checkpoint['model'])
    net.eval()
    del checkpoint
    gc.collect()
    ctypes.CDLL('libc.so.6').malloc_trim(0)
    print(json.dumps({'policy_and_environment_resources': resources(deadline)}), flush=True)
    assert np.allclose(env.m.actuator_forcerange[[5,11]], [[-.35,.35],[-.35,.35]], atol=0, rtol=0)
    native_engine = env.engine

    class RecordedRollout:
        """Copy native output for auditing without altering controls or stepping."""
        def __init__(self):
            self.blocks = deque(maxlen=10)
            self.sensor_depth = 0.

        def rollout(self, *args, **kwargs):
            trajectory, sensors = native_engine.rollout(*args, **kwargs)
            self.blocks.append(trajectory[0].copy())
            self.sensor_depth = max(self.sensor_depth, float(-sensors[:, :, 0].min()))
            return trajectory, sensors

        def close(self):
            native_engine.close()

    recorder = RecordedRollout()
    env.engine = recorder
    states = []
    train_states = {}
    for task in TASKS:
        with np.load(Path(cfg['bank_dir'])/f'{task}_train.npz') as archive:
            a = archive['states'].copy()
            a[:, 0] = 0
            train_states[task] = {hashlib.sha256(s.tobytes()).hexdigest() for s in a}

    def contacts(task):
        data = env.data[0]
        forces = {'right_gripper': 0., 'right_moving_jaw_so101_v1': 0.}
        touched = []
        for ci, c in enumerate(data.contact):
            names = {env.m.body(int(env.m.geom_bodyid[g])).name for g in c.geom}
            if task not in names:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(env.m, data, ci, force)
            for jaw in names.intersection(forces):
                forces[jaw] += float(force[0])
            for other in names-{task}:
                if other.startswith(('left_', 'right_', 'driver_')) or other == 'driver':
                    touched.append(other)
        return forces, sorted(set(touched))

    try:
        for task in TASKS:
            source = json.loads(SOURCES[task].read_text())
            for seed in SEEDS:
                resources(deadline)
                stem = preparation_dir/f'{task}_{seed}'
                row = dict(task=task, seed=seed, qualified=False)
                try:
                    report = json.loads(stem.with_suffix('.json').read_text())
                    assert report['model_sha256'] == source['model_sha256']
                    assert sha(stem.with_suffix('.mjb')) == source['model_sha256']
                    assert report['plugin_sha256'] == source['plugin_sha256']
                    for name, digest in report['plugin_sha256'].items():
                        assert sha(ROOT/'plugin/build'/name) == digest
                    assert report.get('held_reset_qualified') and not report.get('failure')
                    assert report['max_contact_penetration_m'] <= .0004
                    assert report['max_clock_step_error_s'] <= 1e-8
                    assert not any(report['warnings'])
                    with np.load(stem.with_suffix('.npz')) as archive:
                        all_states = archive['states']
                        assert np.isfinite(all_states).all()
                        assert np.max(np.abs(np.diff(all_states[:, 0])-env.m.opt.timestep)) < 1e-8
                        state = all_states[report['held_state_index']].copy()
                        control = archive['controls'][report['held_state_index']].copy()
                    del all_states
                    state[0] = 0.
                    state_sha = hashlib.sha256(state.tobytes()).hexdigest()
                    if preparation_dir != OUT:
                        assert state_sha == manifest['reused_preparation']['preparation_state_sha256'][stem.name]
                    assert state_sha not in train_states[task], 'Exact TRAIN state duplicate'
                    env.set_states([0], state[None], [tasks.index(task)])
                    # Generator holds use a tighter commanded aperture than the
                    # measured angle. Audit the physical loaded grasp at that
                    # command, then independently exercise actual reset semantics.
                    env.data[0].ctrl[:] = control
                    mujoco.mj_forward(env.m, env.data[0])
                    forces, touched = contacts(task)
                    assert min(forces.values()) > .001, 'No loaded bilateral grasp'
                    env.set_states([0], state[None], [tasks.index(task)])
                    zero_trace = []
                    for step in range(60):
                        resources(deadline)
                        _, done, info = env.step(np.zeros((1, 12)))
                        zero_trace.append({k: bool(info[k][0]) for k in ('success','safety','timeout')})
                        if done[0]:
                            break
                    row.update(state_sha256=state_sha, held_state_index=report['held_state_index'],
                               loaded_jaw_forces_N=forces, initial_contact_partners=touched,
                               source_peak_penetration_m=report['max_contact_penetration_m'],
                               zero_action_controls=len(zero_trace), zero_action_outcomes=zero_trace)
                    assert not any(any(r.values()) for r in zero_trace), 'Zero-action check terminated'
                    row['qualified'] = True
                    states.append((task, seed, state))
                except (ResourceLimitError, TimeoutError):
                    raise
                except Exception as exc:
                    row['rejection'] = f'{type(exc).__name__}: {exc}'
                result['preparation'].append(row)
                write(OUT/'results.json', result)
                print(json.dumps({'preparation':row}), flush=True)

        for mode in ('mean', 'sample'):
            for index, (task, seed, state) in enumerate(states):
                resources(deadline)
                episode_seed = 880301+index
                torch.manual_seed(episode_seed)
                env.set_states([0], state[None], [tasks.index(task)])
                recorder.blocks.clear()
                recorder.sensor_depth = 0.
                record = dict(mode=mode, task=task, seed=seed, policy_sampling_seed=episode_seed,
                              first_episode_only=True, actions_from_checkpoint_only=True,
                              success=False, safety=False, timeout=False)
                controls, values, trace = [], [], []
                peak_depth = 0.
                warnings_start = np.array(env.data[0].warning.number).copy()
                initial_error = float(env.assembly_distance()[0])
                try:
                    for step in range(env.horizon):
                        resources(deadline)
                        observation = {k: torch.as_tensor(v) for k,v in env.observe().items()}
                        with torch.no_grad():
                            points = tables(*[observation[k] for k in ('task','tcp_p','tcp_R','part_p','part_q','finger')])
                            dist = net.dist(points, observation['proprio'])
                            action = (dist.mean if mode == 'mean' else dist.sample()).numpy()
                        _, done, info = env.step(action)
                        controls.append(action[0].copy())
                        values.append(env.state[0].copy())
                        depth = max(0., -min(env.data[0].contact.dist, default=0.))
                        peak_depth = max(peak_depth, depth)
                        _, held = contacts(task)
                        body = env.data[0].body(task)
                        trace.append(dict(control=step+1, d_asm_m=float(info['d_asm'][0]),
                                          stable_count=int(env.stable[0]), endpoint_penetration_m=depth,
                                          contact_partners=held, position_m=body.xpos.tolist(),
                                          quaternion=body.xquat.tolist()))
                        if done[0]:
                            record.update({k:bool(info[k][0]) for k in ('success','safety','timeout')})
                            break
                    record.update(controls=len(controls), simulated_seconds=len(controls)*env.ecfg.control_dt,
                                  initial_error_m=initial_error, final_error_m=float(info['d_asm'][0]),
                                  endpoint_peak_depth_m=peak_depth,
                                  physics_sensor_peak_depth_m=recorder.sensor_depth,
                                  warnings=(np.array(env.data[0].warning.number)-warnings_start).tolist())
                    if record['safety']:
                        terminal_block = recorder.blocks[-1]
                        velocity = terminal_block[:, 1+env.m.nq:1+env.m.nq+env.m.nv]
                        record['terminal_safety_measurements'] = dict(
                            maximum_arm_joint_speed_rad_s=float(abs(velocity[:, :12]).max()),
                            maximum_part_linear_speed_m_s=max(float(np.linalg.norm(velocity[:,v:v+3],axis=1).max()) for v in env.part_v),
                            maximum_part_angular_speed_rad_s=max(float(np.linalg.norm(velocity[:,v+3:v+6],axis=1).max()) for v in env.part_v),
                            episode_peak_sensor_depth_m=recorder.sensor_depth,
                            final_depth_m=depth)
                    if record['success']:
                        window = trace[-11:]
                        p = np.array([r['position_m'] for r in window])
                        q = np.array([r['quaternion'] for r in window])
                        retention = dict(samples=len(window), span_s=(len(window)-1)*env.ecfg.control_dt,
                                         max_error_m=max(r['d_asm_m'] for r in window),
                                         any_hand_or_tool_contact=any(r['contact_partners'] for r in window),
                                         max_translation_from_first_m=float(np.linalg.norm(p-p[0],axis=1).max()),
                                         max_rotation_from_first_rad=float((2*np.arccos(np.clip(abs(q@q[0]),0,1))).max()))
                        retention['passed'] = (len(window)==11 and retention['max_error_m']<.001 and
                            not retention['any_hand_or_tool_contact'] and retention['max_translation_from_first_m']<=.00025 and
                            retention['max_rotation_from_first_rad']<=.01)
                        record['independent_success_window'] = retention
                        assert retention['passed'], 'Independent success-window audit failed'
                        audit = mujoco.MjData(env.m)
                        active_id = env.m.body(task).id
                        fine_contact_count, fine_depth = 0, 0.
                        fine_positions, fine_quaternions = [], []
                        for fine_state in np.concatenate(recorder.blocks):
                            mujoco.mj_setState(env.m, audit, fine_state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
                            mujoco.mj_forward(env.m, audit)
                            touch = False
                            for contact in audit.contact:
                                bodies = env.m.geom_bodyid[contact.geom]
                                if active_id in bodies:
                                    touch |= any(env.m.body(int(b)).name.startswith(('left_', 'right_', 'driver_')) or
                                                 env.m.body(int(b)).name == 'driver' for b in bodies)
                            fine_contact_count += int(touch)
                            fine_depth = max(fine_depth, -min(audit.contact.dist, default=0.))
                            fine_positions.append(audit.body(task).xpos.copy())
                            fine_quaternions.append(audit.body(task).xquat.copy())
                        fp, fq = np.asarray(fine_positions), np.asarray(fine_quaternions)
                        record['fine_step_release_audit'] = dict(steps=len(fp), duration_s=len(fp)*env.m.opt.timestep,
                            hand_or_tool_contact_steps=fine_contact_count, max_penetration_m=fine_depth,
                            max_translation_from_window_start_m=float(np.linalg.norm(fp-p[0],axis=1).max()),
                            max_rotation_from_window_start_rad=float((2*np.arccos(np.clip(abs(fq@q[0]),0,1))).max()),
                            scope='Supplemental every-step geometry audit; production success uses control-sampled release')
                        del audit
                except (ResourceLimitError, TimeoutError):
                    raise
                except Exception as exc:
                    record['error'] = f'{type(exc).__name__}: {exc}'
                stem = OUT/f'{mode}_{task}_{seed}'
                np.savez_compressed(stem.with_suffix('.npz'), initial_state=state,
                                    states=np.asarray(values), actions=np.asarray(controls))
                write(stem.with_suffix('.json'), dict(episode=record, control_trace=trace))
                result['episodes'].append(record)
                result['resources'] = resources(deadline)
                write(OUT/'results.json', result)
                print(json.dumps({'episode':record}), flush=True)
    finally:
        env.close()
    result['completed'] = True
    result['summary'] = {}
    for mode in ('mean', 'sample'):
        result['summary'][mode] = {}
        for task in TASKS:
            episodes = [r for r in result['episodes'] if r['mode']==mode and r['task']==task]
            result['summary'][mode][task] = dict(trials=len(episodes),
                successes=sum(r['success'] and 'error' not in r for r in episodes),
                safety_terminations=sum(r['safety'] for r in episodes),
                timeouts=sum(r['timeout'] for r in episodes), errors=sum('error' in r for r in episodes))
    write(OUT/'results.json', result)


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', default=str(ROOT/'runs/so101_overnight_near_v2_20261105/checkpoint.pt'))
    parser.add_argument('--generate', choices=TASKS)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--deadline', type=float)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--reuse-preparation', type=Path,
                        help='Read the exact qualified starts from this archived holdout; never regenerate them')
    parser.add_argument('--min-iteration', type=int, default=0)
    args = parser.parse_args()
    OUT = args.out.resolve()
    deadline = args.deadline or time.time()+24*60
    OUT.mkdir(parents=True, exist_ok=True)
    if args.generate:
        generate(args.generate,args.seed,deadline)
        return
    if args.evaluate:
        if (OUT/'results.json').exists():
            raise FileExistsError('Preserve existing first-episode evaluation results')
        evaluate(deadline)
        return
    if (OUT/'manifest.json').exists() or (OUT/'policy.pt').exists():
        raise FileExistsError('Preserve existing holdout; choose an explicit continuation step')
    # The trainer publishes with os.replace. Opening once keeps a coherent inode
    # while it publishes a later checkpoint; this copy is itself atomically published.
    with open(args.policy,'rb') as source, (OUT/'policy.pt.tmp').open('wb') as target:
        shutil.copyfileobj(source,target)
        target.flush()
        os.fsync(target.fileno())
    os.replace(OUT/'policy.pt.tmp',OUT/'policy.pt')
    shutil.copyfile(Path(args.policy).resolve().with_name('config.json'), OUT/'config.json')
    manifest = dict(distribution='HELD-OUT PREPARED NEAR', full_loose_assembly=False,
                    checkpoint_source=str(Path(args.policy).resolve()), checkpoint_sha256=sha(OUT/'policy.pt'),
                    config_sha256=sha(OUT/'config.json'),
                    minimum_checkpoint_iteration=args.min_iteration,
                    snapshot_time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                    fixed_tasks=list(TASKS), fixed_seeds=list(SEEDS), replacements_allowed=False,
                    native_gripper_limit_Nm=.35, maximum_cpu_threads=2,
                    sources={task:dict(path=str(path),sha256=sha(path)) for task,path in SOURCES.items()},
                    preparation_processes=[])
    if args.reuse_preparation:
        reused = reuse_preparation(args.reuse_preparation)
        manifest.update(preparation_directory=reused['preparation_directory'], reused_preparation=reused)
    write(OUT/'manifest.json',manifest)
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
    for task in (() if args.reuse_preparation else TASKS):
        for seed in SEEDS:
            resources(deadline)
            log = OUT/f'{task}_{seed}.log'
            with log.open('w') as stream:
                returncode=run_child([sys.executable,__file__,'--generate',task,'--seed',str(seed),
                                      '--deadline',str(deadline),'--out',str(OUT)], deadline, env, stream)
            manifest['preparation_processes'].append(dict(task=task,seed=seed,returncode=returncode))
            write(OUT/'manifest.json',manifest)
            print(json.dumps(manifest['preparation_processes'][-1]),flush=True)
    returncode=run_child([sys.executable,__file__,'--evaluate','--deadline',str(deadline),
                          '--out',str(OUT)], deadline, env)
    raise SystemExit(returncode)


if __name__=='__main__':
    try:
        main()
    except (ResourceLimitError, TimeoutError) as exc:
        write(OUT/'execution_failure.json', dict(error=f'{type(exc).__name__}: {exc}',
              time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), replacements_allowed=False))
        raise
