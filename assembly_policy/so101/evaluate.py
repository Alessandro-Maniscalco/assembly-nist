"""Evaluate SO101 checkpoints on the independent loose reset bank."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from ..rl.nets import ActorCritic
from .env import BatchedAssemblyEnv, EnvConfig, HISTORY, PROPRIO_DIM, resolve_tasks
from .obs import PointTables
from .resets import check_bank, signature, pooled_signature, directory_signature


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--policy',required=True);ap.add_argument('--tasks',default='all')
    ap.add_argument('--bank-dir',default='data/so101/banks');ap.add_argument('--out',required=True)
    ap.add_argument('--nthread',type=int,default=4);ap.add_argument('--trials',type=int,default=8)
    a=ap.parse_args();check_bank(a.bank_dir)
    ck=torch.load(a.policy,map_location='cpu',weights_only=False);cfg=ck['args']
    if cfg.get('model_signature')!=signature():raise ValueError('Checkpoint and evaluation model differ')
    trained=resolve_tasks(cfg['tasks']);tasks=resolve_tasks(a.tasks)
    from ..train import load_banks
    if cfg.get('bank_signature')!=pooled_signature(load_banks(trained,a.bank_dir,cfg['bank_per_family'])):
        raise ValueError('Checkpoint training bank differs; use its recorded bank directory')
    if not set(tasks)<=set(trained):raise ValueError('Evaluation task was not trained')
    pts=PointTables(trained,cfg['n_part'],cfg['n_fix'],cfg['n_grip'])
    net=ActorCritic(pts.n_pts,HISTORY*PROPRIO_DIM,12);net.load_state_dict(ck['model']);net.eval()
    env=BatchedAssemblyEnv(trained,[0],EnvConfig(**{**cfg['env'],'nthread':a.nthread}))
    result={'policy':str(Path(a.policy).resolve()),'distribution':'independent loose table',
            'evaluation_backend':'native canonical geometry','training_backend':cfg.get('physics_backend','native'),
            'model_signature':signature(),'bank_directory_signature':directory_signature(a.bank_dir),'tasks':{}}
    torch.manual_seed(501)
    try:
        for task in tasks:
            bank=np.load(Path(a.bank_dir)/f'{task}_eval.npz');records=[]
            for s in bank['states'][:a.trials]:
                env.set_states([0],s[None],[trained.index(task)])
                for _ in range(env.horizon):
                    o={k:torch.as_tensor(v) for k,v in env.observe().items()}
                    with torch.no_grad():
                        pc=pts(*[o[k] for k in ('task','tcp_p','tcp_R','part_p','part_q','finger')])
                        action=net.dist(pc,o['proprio']).sample().numpy()
                    _,done,info=env.step(action)
                    if done[0]:break
                records.append({k:bool(info[k][0]) for k in ('success','safety','timeout')})
            result['tasks'][task]={'trials':len(records),'successes':sum(x['success'] for x in records),'episodes':records}
    finally:env.close()
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
