"""Measure the unchanged task/learner paths; do not treat throughput as task success.

Run from the repository with its Python environment. Each invocation benchmarks
one configuration and writes timings plus exact observations for comparison.
"""
import argparse
import cProfile
import hashlib
import json
import os
from pathlib import Path
import pstats
import subprocess
import sys
import time

import numpy as np


def memory(pids):
    total = 0
    for pid in pids:
        try:
            for line in Path(f'/proc/{pid}/status').read_text().splitlines():
                if line.startswith('VmRSS:'):
                    total += int(line.split()[1]) * 1024
        except FileNotFoundError:
            pass
    return total


def cpu_seconds(pids):
    total = 0
    for pid in pids:
        try:
            fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
            total += (int(fields[11]) + int(fields[12])) / os.sysconf('SC_CLK_TCK')
        except FileNotFoundError:
            pass
    return total


def worker(run):
    from multiprocessing.connection import Client
    from assembly_policy.rl.remote import _nodelay, _serve
    deadline = time.monotonic() + 180
    while not (run / 'server.json').exists():
        if time.monotonic() > deadline:
            raise TimeoutError('No benchmark server')
        time.sleep(.05)
    info = json.loads((run / 'server.json').read_text())
    conn = _nodelay(Client((info['host'], info['port']), authkey=bytes.fromhex(info['key'])))
    try:
        _serve(conn, 1)
    finally:
        conn.close()


def learner_benchmark(a):
    """Time source PPO on repeated real observations, without a training rollout."""
    import torch
    from assembly_policy.train import Trainer, parse
    from assembly_policy.rl.nets import ActorCritic
    from assembly_policy.so101.env import ACT_DIM, PROPRIO_DIM, HISTORY, TASK_NAMES
    from assembly_policy.so101.obs import PointTables
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    tr = Trainer.__new__(Trainer)
    tr.a = parse(['--run-dir', str(a.out.parent), '--envs', str(a.envs), '--steps', str(a.steps)])
    tr.dev = torch.device('cuda')
    tr.lr = tr.a.lr
    tr.pts = PointTables(TASK_NAMES, device=tr.dev)
    tr.model = ActorCritic(tr.pts.n_pts, proprio_dim=HISTORY*PROPRIO_DIM, act_dim=ACT_DIM).to(tr.dev)
    tr.opt = torch.optim.Adam(tr.model.parameters(), lr=tr.lr)
    ob = np.load(a.learner_observations)
    N = a.envs * a.steps
    selected = np.arange(N) % len(ob['task'])
    compact = tr._obs({k: v[selected] for k, v in ob.items()})
    tr.model.norm.update(compact['prop'])
    with torch.no_grad():
        pts = tr._points(compact)
        distribution = tr.model.dist(pts, compact['prop'])
        act = distribution.sample()
        val = tr.model.value(pts, compact['prop'])
        adv = torch.linspace(-1, 1, N, device=tr.dev)
        flat = dict(**compact, act=act, logp=distribution.log_prob(act).sum(-1), val=val,
                    mu=distribution.mean, ret=val+adv, adv=adv, rew=torch.zeros_like(val),
                    valid=torch.ones(N, device=tr.dev, dtype=torch.bool))
        batch = {k: v.reshape(a.steps, a.envs, *v.shape[1:]) for k, v in flat.items()}
    del flat, compact, distribution, pts
    tr.update(batch)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    losses = tr.update(batch)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t
    if not all(np.isfinite(v) for v in losses.values()):
        raise ValueError('Nonfinite PPO benchmark losses')
    result = dict(kind='source_ppo_repeated_observation_benchmark_not_training',
                  envs=a.envs, steps=a.steps, batch=N, epochs=tr.a.epochs,
                  minibatches=tr.a.minibatches, update_s=elapsed,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                  rss_bytes=memory([os.getpid()]), losses=losses,
                  parameters=dict(actor=sum(p.numel() for p in tr.model.actor.parameters()),
                                  critic=sum(p.numel() for p in tr.model.critic.parameters()),
                                  log_std=tr.model.log_std.numel(),
                                  total=sum(p.numel() for p in tr.model.parameters())))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--backend', choices=['local', 'remote'], default='local')
    ap.add_argument('--envs', type=int, default=10)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--steps', type=int, default=8)
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--seed', type=int, default=20261105)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--fixture', type=Path, help='Saved settled states; signature must match')
    ap.add_argument('--actions', choices=['zero', 'random'], default='zero')
    ap.add_argument('--learner-observations', type=Path,
                    help='Time unchanged CUDA PPO on repeated saved observations, without physics')
    ap.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        worker(a.out)
        return
    if a.learner_observations:
        learner_benchmark(a)
        return
    if a.backend == 'remote' and a.workers > a.envs:
        ap.error('workers must not exceed environments')
    from assembly_policy.so101.env import BatchedAssemblyEnv, EnvConfig, TASK_NAMES
    from assembly_policy.so101.resets import reset, signature
    from assembly_policy.so101.scene import model
    from assembly_policy.rl.vecenv import LocalVecEnv
    from assembly_policy.rl.remote import RemoteVecEnv
    a.out.parent.mkdir(parents=True, exist_ok=True)
    model_signature = signature()
    fixture = a.fixture or a.out.with_suffix('.fixture.npz')
    if fixture.exists():
        z = np.load(fixture)
        if str(z['model_signature']) != model_signature:
            raise ValueError('Benchmark fixture belongs to another scene')
        states = z['states']
    else:
        m = model()
        states = np.stack([reset(m, name, a.seed + i) for i, name in enumerate(TASK_NAMES)])
        np.savez_compressed(fixture, states=states, model_signature=model_signature)
        del m
    # Every configuration receives the same cyclic task/reset distribution.
    next_entry = 0
    def provider(n, finished=None, outcomes=None):
        nonlocal next_entry
        ids = np.arange(next_entry, next_entry + n) % len(states)
        next_entry += n
        return ids, states[ids], ids
    children = []
    logs = []
    env = None
    try:
        start = time.perf_counter()
        cfg = EnvConfig(nthread=a.threads)
        if a.backend == 'local':
            env = LocalVecEnv(TASK_NAMES, a.envs, cfg, provider, env_type=BatchedAssemblyEnv)
        else:
            run = a.out.with_suffix('.workers')
            run.mkdir(exist_ok=True)
            (run / 'server.json').unlink(missing_ok=True)
            for i in range(a.workers):
                log = (run / f'worker_{i}.log').open('w')
                logs.append(log)
                children.append(subprocess.Popen([sys.executable, __file__, '--worker', '--out', str(run)],
                                                 stdout=log, stderr=subprocess.STDOUT))
            env = RemoteVecEnv(run, TASK_NAMES, a.envs, cfg, provider, a.workers,
                               accept_timeout=60, min_frac=1, init_wait=180)
        init_s = time.perf_counter() - start
        pids = [os.getpid()] + [p.pid for p in children]
        rng = np.random.default_rng(a.seed)
        actions = (rng.standard_normal((a.steps + a.warmup, a.envs, 12)) if a.actions == 'random'
                   else np.zeros((a.steps + a.warmup, a.envs, 12)))
        for action in actions[:a.warmup]:
            env.step(action)
        if a.backend == 'remote':
            env.reset_profile()
        profile = cProfile.Profile()
        timings, safety, unstable, done_count, valid = [], 0, 0, 0, 0
        rss = memory(pids)
        cpu_start = cpu_seconds(pids)
        start = time.perf_counter()
        profile.enable()
        for action in actions[a.warmup:]:
            t = time.perf_counter()
            reward, done, info, _, _ = env.step(action)
            timings.append(time.perf_counter() - t)
            safety += int(info['safety'].sum())
            unstable += int(info['unstable'].sum())
            done_count += int(done.sum())
            valid += int(info['valid'].sum())
            if not np.isfinite(reward).all():
                raise ValueError('Nonfinite rewards')
            rss = max(rss, memory(pids))
        profile.disable()
        wall = time.perf_counter() - start
        cpu = cpu_seconds(pids) - cpu_start
        obs = env.observe()
        if not all(np.isfinite(value).all() for value in obs.values()):
            raise ValueError('Nonfinite observations')
        result = dict(backend=a.backend, envs=a.envs, threads=a.threads, workers=len(children),
                      steps=a.steps, warmup=a.warmup, actions=a.actions, model_signature=model_signature,
                      fixture=str(fixture), init_s=init_s, wall_s=wall, cpu_s=cpu, cpu_cores_used=cpu/wall,
                      transitions=valid, transitions_s=valid/wall, step_median_s=float(np.median(timings)),
                      step_max_s=max(timings), rss_sum_peak_bytes=rss,
                      safety=safety, unstable=unstable, done=done_count,
                      observations_sha256=hashlib.sha256(b''.join(obs[k].tobytes() for k in sorted(obs))).hexdigest(),
                      plugin_sha256=hashlib.sha256(Path('plugin/assemblenist_sdf.cc').read_bytes()).hexdigest(),
                      remote_profile=env.profile() if a.backend == 'remote' else None)
        np.savez_compressed(a.out.with_suffix('.observations.npz'), **obs)
        with a.out.with_suffix('.profile.txt').open('w') as f:
            pstats.Stats(profile, stream=f).sort_stats('cumulative').print_stats(40)
        a.out.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    finally:
        if env is not None:
            env.close()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.terminate()
                child.wait(timeout=10)
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()
