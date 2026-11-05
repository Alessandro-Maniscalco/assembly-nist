"""Profile native endpoint refresh without changing the learner or calling GPU code.

The diagnostic derives its split functions from the current production method's
AST. Only its three state/control/forward statements move into a per-world worker;
observation packing is otherwise the exact original method. One immutable model
and independent MjData objects follow MuJoCo's documented parallel-forward pattern.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import textwrap
import threading
import time

import mujoco
import numpy as np
import psutil

from assembly_policy.so101 import env as implementation
from assembly_policy.so101 import scene
from assembly_policy.so101.env import BatchedAssemblyEnv, EnvConfig, STATE_SPEC, TASK_NAMES


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def archive_factory(path):
    """Run the exact archive verifier without importing unused GPU libraries."""
    source_path = Path(implementation.__file__).with_name('warp_env.py')
    parsed = ast.parse(source_path.read_text())
    node = next(item for item in parsed.body if isinstance(item, ast.FunctionDef)
                and item.name == 'archived_helper_factory')
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    scope = dict(Path=Path, scene=scene, mujoco=mujoco)
    exec(compile(module, str(source_path), 'exec'), scope)
    return scope['archived_helper_factory'](path), hashlib.sha256(ast.dump(node).encode()).hexdigest()


def split_refresh(out):
    source = textwrap.dedent(inspect.getsource(BatchedAssemblyEnv._refresh))
    node = ast.parse(source).body[0]
    outer = next(item for item in node.body if isinstance(item, ast.For))
    expected = ast.parse('''
mujoco.mj_setState(self.m, d, self.state[i], STATE_SPEC)
d.ctrl[:] = self.last_ctrl[i]
mujoco.mj_forward(self.m, d)
''').body
    factored = ast.dump(node.body[0]) == ast.dump(ast.parse('self._forward_states()').body[0])
    if factored:
        forward_node = ast.parse(textwrap.dedent(inspect.getsource(BatchedAssemblyEnv._forward_world))).body[0]
        if [ast.dump(item) for item in forward_node.body[1:]] != [ast.dump(item) for item in expected]:
            raise RuntimeError('Factored native forward changed')
        node.body = node.body[1:]
    elif [ast.dump(item) for item in outer.body[:3]] != [ast.dump(item) for item in expected]:
        raise RuntimeError('Production refresh changed: inspect extraction before running')
    forward = ast.parse('def forward_one(self, i):\n    d = self.data[i]\n').body[0]
    forward.body.extend(copy.deepcopy(expected))
    if not factored: outer.body = outer.body[3:]
    node.name = 'pack_observations'
    archived = Path('artifacts_local/refresh_profile/extracted_functions.py')
    archived_node = next(item for item in ast.parse(archived.read_text()).body
                         if isinstance(item, ast.FunctionDef) and item.name == 'pack_observations')
    if ast.dump(node) != ast.dump(archived_node):
        raise RuntimeError('Observation packing differs from the preserved original')
    original = copy.deepcopy(node)
    original.name = 'original_refresh'
    original_loop = next(item for item in original.body if isinstance(item, ast.For))
    original_loop.body = copy.deepcopy(expected)+original_loop.body
    module = ast.fix_missing_locations(ast.Module(body=[forward, node, original], type_ignores=[]))
    extracted = ast.unparse(module)+'\n'
    (out/'extracted_functions.py').write_text(extracted)
    scope = dict(vars(implementation))
    exec(compile(module, '<diagnostic refresh extraction>', 'exec'), scope)
    return scope['forward_one'], scope['pack_observations'], scope['original_refresh'], source


def live_parity(model, original_refresh, out):
    """Actual production threaded methods with native rollout, isolating scheduling."""
    from assembly_policy.so101.warp_env import BatchedWarpAssemblyEnv, ENDPOINT_THREADS
    from scripts.check_so101_task_completion import load_source
    baseline = 'dbbe6bfe1cadb7501a3f2476c5d80ac537d827a41b3d223704e9481e07d3547a'
    starts, tasks = [], []
    for task in ('horn', 'retainer'):
        for released in (False, True):
            state, _ = load_source(Path('artifacts_local/prepared_parts'), task, 0, baseline, released)
            starts.append(state); tasks.append(TASK_NAMES.index(task))
    starts, tasks = np.tile(starts, (2, 1)), np.tile(tasks, 2)
    class Original(BatchedAssemblyEnv):
        _refresh = original_refresh
    class ThreadedNativeHarness(BatchedWarpAssemblyEnv):
        # Only construction differs: actual production endpoint/close methods,
        # but identical native rollout engines isolate endpoint scheduling.
        def __init__(self):
            self._endpoint_pool = None
            BatchedAssemblyEnv.__init__(self, TASK_NAMES, tasks, config,
                                        model_factory=lambda **_: model)
            self._endpoint_pool = ThreadPoolExecutor(max_workers=ENDPOINT_THREADS)
    config = EnvConfig(nthread=1, max_contact_depth=.0004, stability_metric='pose')
    serial = Original(TASK_NAMES, tasks, config, model_factory=lambda **_: model)
    threaded = ThreadedNativeHarness()
    records = []
    traces = []
    def compare(label, first_step=None, second_step=None):
        first, second = outputs(serial), outputs(threaded)
        for key in ('state','prev_action','last_ctrl','hist','task','t','stable','stable_position','stable_quaternion'):
            first['env_'+key] = getattr(serial,key).copy()
            second['env_'+key] = getattr(threaded,key).copy()
        if first_step is not None:
            for prefix, values in (('serial',first_step),('threaded',second_step)):
                destination = first if prefix == 'serial' else second
                destination['reward'], destination['done'] = values[:2]
                destination.update({'info_'+key: value for key,value in values[2].items()})
        changes = difference(first, second)
        records.append(dict(label=label, exact=not changes, differences=changes,
                            stable=serial.stable.tolist(),
                            success=None if first_step is None else first_step[2]['success'].tolist()))
        if changes:
            np.savez_compressed(out/(label+'_serial.npz'), **first)
            np.savez_compressed(out/(label+'_threaded.npz'), **second)
            raise AssertionError('Live endpoint parity failed: '+label)
    try:
        for instance in (serial, threaded): instance.set_states(np.arange(8), starts, tasks)
        compare('initial')
        for tick in range(24):
            action = np.zeros((8,12))
            # Keep first copies held; open second copies, while released starts
            # exercise actual success accumulation and full history carryover.
            if tick < 10: action[[4,6],11] = 1
            first, second = serial.step(action), threaded.step(action)
            compare(f'tick_{tick:02d}',first,second)
            traces.append(threaded.state.copy())
            if tick == 12:
                ids = np.array([0,3,5])
                replacement = np.array([2,1,0])
                for instance in (serial,threaded):
                    instance.set_states(ids, starts[replacement], tasks[replacement])
                compare('partial_reset')
        # Serial fallback executes the same production method without a pool.
        threaded._endpoint_pool.shutdown(wait=True); threaded._endpoint_pool = None
        compare('before_serial_fallback')
        compare('serial_fallback_step',serial.step(np.zeros((8,12))),threaded.step(np.zeros((8,12))))
        result = dict(exact=all(item['exact'] for item in records), checks=records,
            actual_success_observed=any(any(item['success'] or []) for item in records),
            production_endpoint_method=True, gpu_rollout=False,
            state_carryover=True, partial_reset=True, serial_fallback=True,
            limitation='Native rollout in both harnesses isolates endpoint scheduling; GPU rollout is not requalified here.')
        np.savez_compressed(out/'live_states.npz', states=np.asarray(traces), initial=starts, tasks=tasks)
        (out/'live_parity.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
    finally:
        serial.close(); threaded.close()


def fixture(bank):
    """Same 64-state ordering as check_so101_warp_env.collection, repeated twice."""
    loose, prepared = {}, {}
    for name in TASK_NAMES:
        with np.load(bank/(name+'_eval.npz')) as archive:
            loose[name] = archive['states'][:8].copy()
        if name in ('motor', 'horn', 'retainer'):
            with np.load(bank/(name+'_train.npz')) as archive:
                prepared[name] = archive['states'][archive['family'] == 4][:16].copy()
    states, tasks = [], []
    for index in range(32):
        name = TASK_NAMES[index % 5]
        states.append(loose[name][index//5]); tasks.append(TASK_NAMES.index(name))
    for index in range(32):
        name = ('motor', 'horn', 'retainer')[index % 3]
        states.append(prepared[name][index//3]); tasks.append(TASK_NAMES.index(name))
    states, tasks = np.asarray(states), np.asarray(tasks)
    fixture_hash = hashlib.sha256(states.tobytes()+tasks.tobytes()).hexdigest()
    return np.tile(states, (2, 1)), np.tile(tasks, 2), fixture_hash


def outputs(env):
    """Snapshot all policy inputs and endpoint contact/sensor/force diagnostics."""
    arrays = dict(frames=env.frames.copy(), assembly_distance=env.assembly_distance())
    arrays.update({'observation_'+key: value for key, value in env.observe().items()})
    for name in ('qpos', 'qvel', 'qacc', 'qacc_warmstart', 'ctrl', 'actuator_force',
                 'qfrc_actuator', 'qfrc_passive', 'qfrc_constraint', 'sensordata',
                 'solver_niter', 'solver_nnz'):
        arrays[name] = np.asarray([getattr(data, name).copy() for data in env.data])
    arrays['ncon'] = np.asarray([data.ncon for data in env.data])
    arrays['warnings'] = np.asarray([data.warning.number.copy() for data in env.data])
    for name in ('geom', 'dist', 'pos', 'frame', 'dim', 'friction', 'solref',
                 'solreffriction', 'solimp', 'efc_address', 'exclude'):
        arrays['contact_'+name] = np.concatenate([getattr(data.contact, name).copy() for data in env.data])
    forces = []
    for data in env.data:
        for index in range(data.ncon):
            force = np.empty(6)
            mujoco.mj_contactForce(env.m, data, index, force)
            forces.append(force)
    arrays['contact_force'] = np.asarray(forces)
    return arrays


def difference(reference, actual):
    changes = {}
    for name in reference:
        first, second = reference[name], actual[name]
        if not np.array_equal(first, second):
            record = dict(reference_shape=list(first.shape), actual_shape=list(second.shape))
            if first.shape == second.shape:
                record.update(count=int(np.count_nonzero(first != second)),
                              maximum_absolute=float(np.max(np.abs(first.astype(float)-second))))
            changes[name] = record
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('artifacts_local/whole_helper_gpu/model/candidate_sensor.mjb'))
    parser.add_argument('--bank', type=Path, default=Path('data/so101/banks_near_v2'))
    parser.add_argument('--out', type=Path, default=Path('artifacts_local/refresh_profile'))
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--rss-limit-gib', type=float, default=3)
    parser.add_argument('--production-live', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    stop_watch = threading.Event()
    process = psutil.Process()
    minima = dict(available_bytes=psutil.virtual_memory().available, maximum_rss_bytes=0)
    def watch():
        while not stop_watch.wait(.2):
            available = psutil.virtual_memory().available
            rss = process.memory_info().rss
            minima['available_bytes'] = min(minima['available_bytes'], available)
            minima['maximum_rss_bytes'] = max(minima['maximum_rss_bytes'], rss)
            if rss > args.rss_limit_gib*1024**3 or available < 2*1024**3 or time.monotonic()-started > 600:
                (args.out/'resource_stop.json').write_text(json.dumps(dict(rss=rss, available=available)))
                os._exit(70)
    threading.Thread(target=watch, daemon=True).start()
    forward, pack, original_refresh, source = split_refresh(args.out)
    states, tasks, fixture_hash = fixture(args.bank)
    factory, factory_ast_sha = archive_factory(args.model)
    env = BatchedAssemblyEnv(TASK_NAMES, tasks,
        EnvConfig(nthread=1, max_contact_depth=.0004, stability_metric='pose'), model_factory=factory)
    env.state[:] = states
    env.state[:, 0] = 0
    env.last_ctrl[:] = env.state[:, 1:13]
    env.prev_action[:] = np.random.default_rng(1234).uniform(-.1, .1, env.prev_action.shape)
    env.hist[:] = np.random.default_rng(5678).normal(0, .1, env.hist.shape)
    np.savez_compressed(args.out/'fixture.npz', states=env.state, tasks=tasks,
                        last_ctrl=env.last_ctrl, previous_action=env.prev_action, history=env.hist)
    report = dict(worlds=env.N, fixture_sha256=fixture_hash,
        distribution='64 loose and 64 prepared starts: collection fixture repeated twice',
        current_refresh_sha256=hashlib.sha256(source.encode()).hexdigest(),
        source_file_sha256=digest(inspect.getsourcefile(BatchedAssemblyEnv)),
        archive_factory_ast_sha256=factory_ast_sha,
        archive_factory_loading='Exact existing function AST, avoiding unused GPU library imports',
        model_sha256=factory.provenance['model_sha256'], mujoco_version=mujoco.__version__,
        gpu_calls=False, training_speedup_measured=False, samples=[], parity_checks=[],
        state_setup='mj_resetData then per-world identical qacc_warmstart; exact production setState/ctrl/forward',
        allocation='one read-only shared model, one MjData per world; no simultaneous access to a world',
        references=['https://mujoco.readthedocs.io/en/3.3.7/programming/simulation.html#multi-threading',
                    'https://mujoco.readthedocs.io/en/3.3.7/python.html#functions'],
        plugin_audit='Immutable grid storage with mutex-protected initialization and thread-local cached pointers; driver plugin data and force writes are per-MjData.')
    def save():
        (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    def reset(warm):
        for index, data in enumerate(env.data):
            mujoco.mj_resetData(env.m, data)
            data.qacc_warmstart[:] = warm[index]
    try:
        for case in ('zero_warmstart', 'nonzero_warmstart'):
            warm = np.zeros((env.N, env.m.nv)) if case == 'zero_warmstart' else np.random.default_rng(9012).normal(0, .1, (env.N, env.m.nv))
            reset(warm)
            original_refresh(env)
            reference = outputs(env)
            np.savez_compressed(args.out/(case+'_reference.npz'), **reference, initial_warmstart=warm)
            # Persistent pools exclude startup and first thread-local grid lookup
            # from warm timing. Pools are sequential, so at most eight workers run.
            for workers in (1, 4, 2, 8):
                pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
                def refresh_forward():
                    if pool is None:
                        for index in range(env.N): forward(env, index)
                    else:
                        list(pool.map(lambda index: forward(env, index), range(env.N)))
                try:
                    reset(warm); refresh_forward(); pack(env)
                    for repeat in range(args.repeats):
                        # Original interleaved method brackets each candidate.
                        reset(warm)
                        tick = time.perf_counter(); original_refresh(env); reference_s = time.perf_counter()-tick
                        reset(warm)
                        tick = time.perf_counter(); refresh_forward(); forward_s = time.perf_counter()-tick
                        tick = time.perf_counter(); pack(env); packing_s = time.perf_counter()-tick
                        actual = outputs(env)
                        differences = difference(reference, actual)
                        report['parity_checks'].append(dict(case=case, workers=workers, repeat=repeat,
                                                           exact=not differences, differences=differences))
                        if differences:
                            np.savez_compressed(args.out/f'{case}_w{workers}_r{repeat}_different.npz', **actual)
                        report['samples'].append(dict(case=case, workers=workers, repeat=repeat,
                            original_refresh_s=reference_s, split_forward_s=forward_s,
                            unchanged_packing_s=packing_s, split_total_s=forward_s+packing_s))
                        save()
                finally:
                    if pool is not None: pool.shutdown(wait=True)
                print(json.dumps(dict(case=case, workers=workers, exact=not differences)), flush=True)
        report['summaries'] = []
        for workers in (1, 2, 4, 8):
            samples = [item for item in report['samples'] if item['workers'] == workers]
            medians = {key: float(np.median([item[key] for item in samples])) for key in
                       ('original_refresh_s', 'split_forward_s', 'unchanged_packing_s', 'split_total_s')}
            medians.update(workers=workers,
                median_paired_refresh_speedup=float(np.median([item['original_refresh_s']/item['split_total_s'] for item in samples])))
            report['summaries'].append(medians)
        report['exact_parity'] = all(item['exact'] for item in report['parity_checks'])
        report['finite'] = all(np.isfinite(value).all() for value in reference.values())
        report['warnings_zero'] = not bool(reference['warnings'].any())
        if args.production_live:
            report['live_parity'] = live_parity(env.m, original_refresh, args.out)
        report['resource_extrema'] = minima
        report['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        report['resource_bound_pass'] = (report['peak_rss_bytes'] <= args.rss_limit_gib*1024**3 and
                                          minima['available_bytes'] >= 2*1024**3)
        report['elapsed_seconds'] = time.monotonic()-started
        report['limitations'] = ['Frozen endpoint fixture only; no GPU rollout, policy, reward or full-learner timing.',
            'Concurrent learner/evaluator may influence wall times; report all repetitions and paired references.',
            'Fresh matched MjData reset and warm-start values; live derived-state carryover is not exercised.',
            'No production code, runtime signatures, checkpoints, controls or models changed.']
        save()
        print(json.dumps(dict(exact_parity=report['exact_parity'], summaries=report['summaries'],
                              elapsed_seconds=report['elapsed_seconds'])), flush=True)
        if not report['exact_parity'] or not report['resource_bound_pass']:
            raise SystemExit('Diagnostic did not satisfy all parity/resource gates; inspect report.json')
    finally:
        env.close()
        stop_watch.set()


if __name__ == '__main__':
    main()
