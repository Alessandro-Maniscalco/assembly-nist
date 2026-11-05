"""Check asset provenance, task resets, observation/control plumbing, and source preservation."""
import ast
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import mujoco
import numpy as np
import torch
from assembly_policy.so101.scene import ROOT, ASSETS, model
from assembly_policy.so101.env import BatchedAssemblyEnv, EnvConfig, TASK_NAMES, STATE_SPEC, PROPRIO_DIM
from assembly_policy.so101.obs import PointTables
from assembly_policy.so101.robot import home
from assembly_policy.so101.resets import check_bank, signature, directory_signature

UPSTREAM='eb5d15e004591d380c0d521da67e025ebee37703'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bank-dir',type=Path,default=ROOT/'data/so101/banks')
    ap.add_argument('--out',type=Path,default=ROOT/'artifacts_local/preflight.json')
    ap.add_argument('--max-contact-depth',type=float,default=None)
    ap.add_argument('--stability',choices=['velocity','pose'],default='velocity')
    ap.add_argument('--physics',choices=['native','whole_helper_sdf'],default='native')
    ap.add_argument('--physics-model',type=Path)
    a=ap.parse_args();bank=a.bank_dir;check_bank(bank)
    result={'model_signature':signature(),'bank_directory_signature':directory_signature(bank),
            'bank_dir':str(bank.resolve()),'max_contact_depth':a.max_contact_depth,'stability_metric':a.stability,'checks':{}}
    result['physics_backend']=a.physics;result['physics_signature']=result['model_signature']
    def check(name,condition):
        result['checks'][name]=bool(condition)
        if not condition:raise AssertionError(name)
    check('official_cad_hash',hashlib.sha256((ASSETS/'hiwonder/follower.3dm').read_bytes()).hexdigest()==
          'f675f6049f94206c7b74ab73c40485913916babff1a6f4fc145dc9040672a2f8')
    for f in ('assembly_policy/rl/nets.py','assembly_policy/rl/sampler.py'):
        check(f+'_unchanged',subprocess.check_output(['git','show',UPSTREAM+':'+f],cwd=ROOT)==(ROOT/f).read_bytes())
    original=subprocess.check_output(['git','show',UPSTREAM+':plugin/assemblenist_sdf.cc'],cwd=ROOT,text=True)
    current=(ROOT/'plugin/assemblenist_sdf.cc').read_text()
    check('sdf_equations_unchanged',original.split('const Grid* get_grid')[0]==current.split('const Grid* get_grid')[0]
          and original.split('inline float h2f')[1]==current.split('inline float h2f')[1])
    def update(text):
        return next(f for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='Trainer'
                    for f in n.body if isinstance(f,ast.FunctionDef) and f.name=='update')
    check('ppo_update_unchanged',ast.dump(update(subprocess.check_output(['git','show',UPSTREAM+':assembly_policy/train.py'],cwd=ROOT,text=True)))==
          ast.dump(update((ROOT/'assembly_policy/train.py').read_text())))
    config=EnvConfig(nthread=1,max_contact_depth=a.max_contact_depth,stability_metric=a.stability)
    if a.physics=='whole_helper_sdf':
        if a.physics_model is None or a.max_contact_depth is None:raise ValueError('GPU preflight requires model and depth monitor')
        from assembly_policy.so101.warp_env import BatchedWarpAssemblyEnv,archived_helper_factory,physics_identity
        factory=archived_helper_factory(a.physics_model);identity=physics_identity(factory)
        result['physics_signature']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        result['physics_provenance']=identity
        env=BatchedWarpAssemblyEnv(TASK_NAMES,np.arange(5),config,model_factory=factory,model_variant='whole_helper_sdf',
                                  proof_dir=a.out.with_suffix(''))
    else:env=BatchedAssemblyEnv(TASK_NAMES,np.arange(5),config)
    states=[]
    for name in TASK_NAMES:
        tr=np.load(bank/f'{name}_train.npz');ev=np.load(bank/f'{name}_eval.npz')
        check(name+'_disjoint_seeds',not set(tr['seed'])&set(ev['seed']))
        states.append(ev['states'][0])
    env.set_states(np.arange(5),np.stack(states));o=env.observe()
    check('observation_shape',o['proprio'].shape==(5,PROPRIO_DIM*5))
    check('twelve_robot_actuators',env.m.nu==12)
    check('no_attachment_constraints',env.m.neq==0)
    for name in (*TASK_NAMES,'base','driver'):
        check(name+'_free',int(env.m.joint(name+'_free').type[0])==int(mujoco.mjtJoint.mjJNT_FREE))
    points=PointTables(TASK_NAMES)
    p=points(*[torch.tensor(o[k]) for k in ('task','tcp_p','tcp_R','part_p','part_q','finger')])
    check('points_finite_320',p.shape==(5,960) and bool(torch.isfinite(p).all()))
    reward,done,info=env.step(np.zeros((5,12)))
    check('loose_resets_not_successful',not info['success'].any())
    check('zero_control_stable',not info['safety'].any() and bool(np.isfinite(reward).all()))
    # Timeouts and autoreset consumers must receive the real terminal state.
    env.t[:]=env.horizon-1
    _,done,info=env.step(np.zeros((5,12)))
    check('timeouts_terminate',done.all() and info['timeout'].all())
    env.close()
    result['thread_check']=json.loads((ROOT/'artifacts_local/thread_check.json').read_text())
    r=result['thread_check']['rotation'];axial=result['thread_check']['axial']
    check('thread_tracks_pitch',abs(r['advance_m']-r['expected_advance_m'])<.00005 and abs(r['turns'])>1)
    check('thread_blocks_axial_push',abs(axial['advance_m'])<.0001)
    result['thread_check']={k:{kk:vv for kk,vv in v.items() if kk!='trace'} for k,v in result['thread_check'].items()}
    m=model();d=mujoco.MjData(m);home(m,d)
    for _ in range(1000):mujoco.mj_step(m,d)
    v=m.jnt_dofadr[m.joint('button_forward').id];s=m.jnt_dofadr[m.joint('spindle').id]
    check('driver_idle_without_press',abs(d.qvel[s])<.001)
    d.qfrc_applied[v]=2.
    for _ in range(500):mujoco.mj_step(m,d)
    check('driver_runs_when_physically_pressed',d.joint('button_forward').qpos[0]>.0006 and d.qvel[s]>20)
    d.qfrc_applied[v]=0
    for _ in range(1000):mujoco.mj_step(m,d)
    check('button_releases',d.joint('button_forward').qpos[0]<.0001)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(f"PASS: {len(result['checks'])} checks",flush=True)


if __name__=='__main__':main()
