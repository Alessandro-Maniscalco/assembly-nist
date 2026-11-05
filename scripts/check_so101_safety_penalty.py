"""Replay fixed archived actions to qualify optional safety-penalty semantics."""
import ctypes
import gc
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import psutil

from assembly_policy.so101.env import BatchedAssemblyEnv, EnvConfig, resolve_tasks

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts_local/failure_incentive'


def sha(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def arrays_sha(values):
    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        a = np.asarray(value)
        digest.update(key.encode())
        digest.update(str((a.shape, str(a.dtype))).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def main():
    deadline = time.monotonic()+290
    evidence = json.loads((OUT/'results.json').read_text())
    legacy_cfg = evidence['checkpoint_args']['env']
    assert 'safety_penalty' not in legacy_cfg
    cfg = EnvConfig(**{**legacy_cfg, 'nthread': 1})
    assert cfg.safety_penalty == EnvConfig().safety_penalty == 0.
    # Legacy serialized configuration still constructs with the original values.
    for key, value in legacy_cfg.items():
        assert json.loads(json.dumps(asdict(cfg)[key])) == json.loads(json.dumps(value))
    env = BatchedAssemblyEnv(resolve_tasks(evidence['checkpoint_args']['tasks']), [0], cfg)
    gc.collect()
    ctypes.CDLL('libc.so.6').malloc_trim(0)
    native = env.engine

    class Recorder:
        def rollout(self, *args, **kwargs):
            self.states, self.sensors = native.rollout(*args, **kwargs)
            return self.states, self.sensors

        def close(self):
            native.close()

    recorder = Recorder()
    env.engine = recorder
    result = dict(source_sha256=sha(ROOT/'assembly_policy/so101/env.py'),
                  script_sha256=sha(__file__), checkpoint_sha256=evidence['checkpoint_sha256'],
                  legacy_config_default_compatible=True, model_reused=True,
                  scope='Reward semantics only; no policy sampling, new learning, or improvement claim.',
                  episodes=[])
    maximum_rss = 0
    try:
        for mode in ('sampled_policy', 'zero_action'):
            path = OUT/f'{mode}.npz'
            with np.load(path) as stored:
                actions = stored['actions'].copy()
                initial = stored['initial_state'].copy()
                old_states = stored['physics_states'].copy()
                old_sensors = stored['sensors'].copy()
                old_controls = stored['controls'].copy()
            old_episode = next(x for x in evidence['episodes'] if x['mode'] == mode)
            versions = []
            for penalty in (0., 1.):
                env.ecfg.safety_penalty = penalty
                env.set_states([0], initial[None], [0])
                rows = []
                for step, action in enumerate(actions):
                    rss = psutil.Process().memory_info().rss
                    maximum_rss = max(maximum_rss, rss)
                    assert rss < 2.5*1024**3 and time.monotonic() < deadline
                    reward, done, info = env.step(action[None])
                    assert np.array_equal(recorder.states[0], old_states[step]), ('historical physics mismatch', mode, penalty, step)
                    assert np.array_equal(recorder.sensors[0], old_sensors[step]), ('historical sensor mismatch', mode, penalty, step)
                    assert np.array_equal(env.last_ctrl[0], old_controls[step])
                    assert bool(done[0]) == (step == len(actions)-1)
                    assert info['cost'][0] == old_episode['control_trace'][step]['cost']
                    if penalty == 0:
                        assert reward[0] == old_episode['control_trace'][step]['reward']
                    rows.append(dict(step=step+1, reward=float(reward[0]), done=bool(done[0]),
                                     state_sha256=arrays_sha(dict(state=env.state)),
                                     observation_sha256=arrays_sha(env.observe()), info_sha256=arrays_sha(info)))
                versions.append(rows)
            zero, one = versions
            deltas = []
            for a, b in zip(zero, one):
                for key in ('done','state_sha256','observation_sha256','info_sha256'):
                    assert a[key] == b[key], (mode, a['step'], key)
                delta = a['reward']-b['reward']
                expected = float(a['step'] == len(actions) and old_episode['safety'])
                if expected == 0:
                    assert delta == 0
                else:
                    assert abs(delta-expected) <= np.finfo(np.float32).eps
                deltas.append(delta)
            record = dict(mode=mode, controls=len(actions), archive_sha256=sha(path),
                          exact_historical_physics_sensors_controls=True,
                          exact_states_observations_done_info_between_penalties=True,
                          rewards_identical_except_terminal_safety=True,
                          terminal_penalty_reward_difference=deltas[-1],
                          reward_totals=[sum(x['reward'] for x in rows) for rows in versions],
                          per_control_versions=versions)
            result['episodes'].append(record)
            print(json.dumps({k:v for k,v in record.items() if k != 'per_control_versions'}), flush=True)
        result.update(passed=True, max_rss_bytes=maximum_rss)
        (OUT/'safety_penalty_qualification.json').write_text(json.dumps(result, indent=2)+'\n')
    finally:
        env.close()


if __name__ == '__main__':
    main()
