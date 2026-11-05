"""Archive an actual checkpoint rollout, with its reset distribution and fine-step audit."""
import argparse
import hashlib
from pathlib import Path

import mujoco
import numpy as np
import torch

from ..rl.nets import ActorCritic
from ..train import load_banks
from .env import BatchedAssemblyEnv, EnvConfig, HISTORY, PROPRIO_DIM, resolve_tasks, STATE_SPEC
from .obs import PointTables
from .resets import check_bank, signature, pooled_signature
from .trajectory import Trajectory


class CaptureRollout:
    def __init__(self, engine):self.engine=engine;self.result=None
    def rollout(self,*args,**kwargs):
        self.result=self.engine.rollout(*args,**kwargs)
        return self.result
    def close(self):self.engine.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--bank-dir',type=Path,required=True)
    p.add_argument('--task',default='motor')
    p.add_argument('--split',choices=['train','eval'],default='eval')
    p.add_argument('--entry',type=int,default=0)
    p.add_argument('--action-mode',choices=['sample','mean'],default='sample')
    p.add_argument('--seed',type=int,default=501)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(a.seed);check_bank(a.bank_dir)
    policy_bytes=a.policy.read_bytes()
    import io
    ck=torch.load(io.BytesIO(policy_bytes),map_location='cpu',weights_only=False);cfg=ck['args']
    tasks=resolve_tasks(cfg['tasks'])
    if cfg.get('model_signature')!=signature():raise ValueError('Policy physics differs')
    if cfg.get('bank_signature')!=pooled_signature(load_banks(tasks,a.bank_dir,cfg['bank_per_family'])):
        raise ValueError('Policy training bank differs')
    pts=PointTables(tasks,cfg['n_part'],cfg['n_fix'],cfg['n_grip'])
    net=ActorCritic(pts.n_pts,HISTORY*PROPRIO_DIM,12);net.load_state_dict(ck['model']);net.eval()
    with np.load(a.bank_dir/f'{a.task}_{a.split}.npz') as bank:
        state=bank['states'][a.entry].copy();family=int(bank['family'][a.entry]);seed=int(bank['seed'][a.entry])
    env=BatchedAssemblyEnv(tasks,[tasks.index(a.task)],EnvConfig(**{**cfg['env'],'nthread':1}))
    env.set_states([0],state[None]);capture=CaptureRollout(env.engine);env.engine=capture
    audit=mujoco.MjData(env.m);mujoco.mj_setState(env.m,audit,env.state[0],STATE_SPEC)
    audit.ctrl[:]=env.last_ctrl[0];mujoco.mj_forward(env.m,audit)
    log=Trajectory(env.m,audit);label='Checkpoint policy: '+a.task;log.labels=[label]
    controls=[];failure=None;terminal=None
    distribution=('Independent loose-table EVAL reset' if a.split=='eval' else
                  ('Prepared near TRAIN reset; not loose-table evaluation' if family==4 else
                   f'TRAIN reset, family {family}; not independent evaluation'))
    try:
        for step in range(env.horizon):
            o={k:torch.as_tensor(v) for k,v in env.observe().items()}
            with torch.no_grad():
                pc=pts(*[o[k] for k in ('task','tcp_p','tcp_R','part_p','part_q','finger')])
                dist=net.dist(pc,o['proprio'])
                action=(dist.sample() if a.action_mode=='sample' else dist.mean).numpy()
            reward,done,info=env.step(action)
            controls.append(dict(step=step,action=action[0].tolist(),reward=float(reward[0]),
                                 **{k:float(v[0]) for k,v in info.items()}))
            # These are the actual native Rollout outputs. Recompute contact
            # geometry on separate data for auditing, without stepping or altering them.
            for s in capture.result[0][0]:
                previous=log.states[-1][0] if log.states else log.initial_state[0]
                log.max_clock_error=max(log.max_clock_error,abs(s[0]-previous-env.m.opt.timestep))
                mujoco.mj_setState(env.m,audit,s,STATE_SPEC);audit.ctrl[:]=env.last_ctrl[0]
                mujoco.mj_forward(env.m,audit)
                if audit.ncon:
                    c=audit.contact[int(np.argmin(audit.contact.dist))]
                    depth=max(0.,-float(c.dist));pair=[int(c.geom1),int(c.geom2)]
                else:depth=0.;pair=[-1,-1]
                log.states.append(s.copy());log.controls.append(env.last_ctrl[0].copy())
                log.depths.append(depth);log.pairs.append(pair);log.stages.append(0)
            if done[0]:
                terminal={k:bool(info[k][0]) for k in ('success','safety','timeout')};break
    except Exception as error:failure=f'{type(error).__name__}: {error}'
    finally:env.close()
    if not log.states:raise RuntimeError(failure or 'No native steps recorded')
    report=log.save(a.out,dict(scripted=False,learned_policy=True,task=a.task,
        evaluation_backend='native canonical geometry',training_backend=cfg.get('physics_backend','native'),
        distribution=distribution,split=a.split,entry=a.entry,reset_family=family,reset_seed=seed,
        policy_sha256=hashlib.sha256(policy_bytes).hexdigest(),policy_iteration=ck['it'],
        policy_transitions=ck['transitions'],action_mode=a.action_mode,action_seed=a.seed,
        terminal=terminal,failure=failure,control_samples=controls,
        native_warning_counts=np.sum([d.warning.number for d in env.work],axis=0).tolist(),
        contact_audit='mj_forward on every recorded native physics state; no state alterations'))
    print({k:report[k] for k in ('distribution','physics_steps','terminal','failure','max_contact_penetration_m')},flush=True)


if __name__=='__main__':main()
