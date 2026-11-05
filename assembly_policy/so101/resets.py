"""Independent loose-table reset banks with physically settled predecessor stages."""
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from .scene import model, build_xml, ASSETS, ROOT, PARTS, metadata
from .fastener import origins
from .robot import home
from .env import TASK_NAMES, STATE_SPEC


def signature():
    loose=build_xml();assembled=build_xml(True)
    digest=hashlib.sha256((loose+assembled).replace(str(ASSETS),'ASSETS').encode())
    digest.update((ASSETS/'parts/manifest.json').read_bytes())
    root=ET.fromstring(loose)
    files={Path(m.get('file')) for m in root.findall('asset/mesh') if m.get('file')}
    for plugin in root.findall('extension/plugin'):
        if plugin.get('plugin')=='assemblenist.sdf.grid':
            for config in plugin.findall('instance/config'):
                if config.get('key')=='id':files.add(ASSETS/'sdf'/f"grid_{config.get('value')}.bin")
    files.update((ROOT/'plugin/assemblenist_sdf.cc',ROOT/'plugin/so101_driver.cc'))
    for path in sorted(files):
        digest.update(str(path.relative_to(ROOT)).encode())
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def check_bank(directory):
    p=Path(directory)/'manifest.json'
    if not p.exists() or json.loads(p.read_text())['model_signature']!=signature():
        raise ValueError('Missing or stale SO101 reset banks; run python -m assembly_policy.so101.resets')


def directory_signature(directory):
    """Bind qualification to the exact train/evaluation archives and manifest."""
    digest=hashlib.sha256()
    for name in ['manifest.json',*[f'{task}_{split}.npz' for task in TASK_NAMES for split in ('train','eval')]]:
        digest.update(name.encode());digest.update((Path(directory)/name).read_bytes())
    return digest.hexdigest()


def pooled_signature(arrays):
    digest=hashlib.sha256()
    for array in arrays:
        digest.update(str((array.shape,array.dtype)).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def add_motor_trajectories(directory, paths, count=32):
    """Extract verified held/approach starts from this model's actual trajectories.

    Failed assembly attempts are valid coverage. No evaluation state comes from
    these traces. Only bilaterally held, nonterminal states are accepted.
    """
    m=model();binary=np.empty(mujoco.mj_sizeModel(m),np.uint8);mujoco.mj_saveModel(m,buffer=binary)
    model_hash=hashlib.sha256(binary.tobytes()).hexdigest();d=mujoco.MjData(m)
    from .env import BatchedAssemblyEnv,EnvConfig
    env=BatchedAssemblyEnv(['motor'],[0],EnvConfig(nthread=1))
    families={1:[],2:[],3:[],4:[]};sources=[]
    motor=m.body('motor').id
    fixed=m.body('right_gripper').id;moving=m.body('right_moving_jaw_so101_v1').id
    for path in paths:
        path=Path(path);record=json.loads(path.with_suffix('.json').read_text())
        if record['model_sha256']!=model_hash:raise ValueError(f'{path}: trajectory uses a different model')
        if any(record['warnings']) or record['max_clock_step_error_s']>1e-8:raise ValueError(f'{path}: invalid physics')
        for name,expected in record['plugin_sha256'].items():
            if hashlib.sha256((ROOT/'plugin/build'/name).read_bytes()).hexdigest()!=expected:
                raise ValueError(f'{path}: plugin binary differs')
        # NPZ members decompress on every indexing call; materialize them once.
        with np.load(path.with_suffix('.npz')) as archive:z={k:archive[k] for k in archive.files}
        sources.append({'file':path.name,'model_sha256':model_hash})
        prepared=record.get('prepared_sgs_reset',False)
        if prepared and not record.get('held_reset_qualified'):raise ValueError(f'{path}: unqualified prepared hold')
        indices=([record['held_state_index']-240+j*80 for j in range(4)] if prepared
                 else range(0,len(z['states']),80))
        for k in indices:
            s=z['states'][k]
            if not np.isfinite(s).all() or z['penetration'][k]>.0004:continue
            mujoco.mj_setState(m,d,s,STATE_SPEC);d.ctrl[:]=z['controls'][k];mujoco.mj_forward(m,d)
            touched=set()
            for c in d.contact:
                pair=m.geom_bodyid[c.geom]
                if motor in pair:touched.update(int(b) for b in pair if b!=motor)
            if not {fixed,moving}.issubset(touched):continue
            env.set_states([0],s[None]);distance=float(env.assembly_distance()[0])
            # Geometrically seated but bilaterally held is NOT terminal: the
            # task requires release and stable retention. These are useful near starts.
            label=str(z['labels'][z['stage'][k]])
            if distance<.004:family=4
            elif distance<.015:family=3
            elif 'Close' in label:family=1
            elif 'Lift' in label or 'Move held' in label or 'Approach insertion' in label:family=2
            else:continue
            families[family].append(s.copy())
    env.close();rng=np.random.default_rng(7301)
    p=Path(directory)/'motor_train.npz';old=np.load(p)
    states=[old['states']];family=[old['family']];seeds=[old['seed']];coverage={}
    for f,rows in families.items():
        if not rows:continue
        indices=rng.choice(len(rows),min(count,len(rows)),replace=False)
        chosen=np.asarray(rows)[indices];states.append(chosen);family.append(np.full(len(chosen),f))
        start=int(np.sum(old['family']==f))
        seeds.append(-100000*f-start-np.arange(len(chosen)));coverage[str(f)]=len(chosen)
    np.savez_compressed(p,states=np.concatenate(states),family=np.concatenate(family),seed=np.concatenate(seeds))
    return {'sources':sources,'motor_additional_families':coverage,'maximum_per_family':count}


def reset(m,task,seed):
    rng=np.random.default_rng(seed);d=mujoco.MjData(m);home(m,d)
    meta=metadata();target=origins(meta)
    # The base is also loose. All predecessor pieces follow its randomized pose.
    base_yaw=rng.uniform(-np.deg2rad(8),np.deg2rad(8));base_R=Rotation.from_euler('z',base_yaw).as_matrix()
    base_q=np.r_[np.cos(base_yaw/2),0,0,np.sin(base_yaw/2)]
    bq=m.jnt_qposadr[m.joint('base_free').id]
    base_pos=d.qpos[bq:bq+3].copy();base_pos[:2]+=rng.uniform(-.008,.008,2)
    previous=set(TASK_NAMES[:TASK_NAMES.index(task)])
    for name in (*PARTS,'driver'):
        q=m.jnt_qposadr[m.joint(name+'_free').id]
        if name=='base':d.qpos[q:q+3]=base_pos;d.qpos[q+3:q+7]=base_q
        elif name in previous:
            d.qpos[q:q+3]=base_pos+base_R@(target[name]-target['base'])
            d.qpos[q+3:q+7]=base_q
        else:
            d.qpos[q:q+2]+=rng.uniform(-.008,.008,2)
            # Keep the tool lying down; randomize only its table yaw.
            angle=rng.uniform(-np.deg2rad(8),np.deg2rad(8))
            yaw=np.r_[np.cos(angle/2),0,0,np.sin(angle/2)]
            quat=np.empty(4);mujoco.mju_mulQuat(quat,yaw,d.qpos[q+3:q+7]);d.qpos[q+3:q+7]=quat
    for _ in range(1000):mujoco.mj_step(m,d)
    if not np.isfinite(d.qpos).all() or abs(d.time-.5)>1e-8:
        raise ValueError(f'Divergent reset {task}/{seed}')
    if min([c.dist for c in d.contact],default=0)<-.0004:
        raise ValueError(f'Deep contact in reset {task}/{seed}')
    # Reject reset drift of predecessor placements rather than hiding it with a weld.
    base=d.body('base')
    for name in previous:
        goal=base.xpos+base.xmat.reshape(3,3)@(target[name]-target['base'])
        if np.linalg.norm(d.body(name).xpos-goal)>.002:
            raise ValueError(f'Predecessor {name} drifted in reset {task}/{seed}')
    s=np.empty(mujoco.mj_stateSize(m,STATE_SPEC));mujoco.mj_getState(m,d,s,STATE_SPEC)
    return s


_worker_model=None
def _worker(job):
    global _worker_model
    if _worker_model is None:_worker_model=model()
    task,seed=job
    try:return reset(_worker_model,task,seed),seed,None
    except ValueError as e:return None,seed,str(e)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=ROOT/'data/so101/banks')
    ap.add_argument('--count',type=int,default=64)
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--tasks',default='+'.join(TASK_NAMES))
    ap.add_argument('--trajectory',type=Path,action='append',default=[])
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    report={'model_signature':signature(),'randomization':{'xy_m':.008,'yaw_deg':8},
                      'families':['loose'],'coverage_note':'Pilot banks; held and near-insertion curricula are not generated.',
                      'tasks':{}}
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for task in a.tasks.split('+'):
            report['tasks'][task]={}
            for split,offset in [('train',0),('eval',1000000)]:
                states=[];seeds=[];rejected=[];attempt=0
                for _ in range(8):
                    count=a.count-len(states)
                    jobs=[(task,offset+10000*TASK_NAMES.index(task)+k) for k in range(attempt,attempt+count)]
                    attempt+=count
                    for state,seed,error in pool.map(_worker,jobs):
                        if error:rejected.append(error)
                        else:states.append(state);seeds.append(seed)
                    if len(states)==a.count:break
                if len(states)!=a.count:raise RuntimeError(f'{task}/{split}: only {len(states)} valid starts; {rejected[-3:]}')
                np.savez_compressed(a.out/f'{task}_{split}.npz',states=np.stack(states),family=np.zeros(len(states),np.int64),seed=seeds)
                report['tasks'][task][split]={'accepted':len(states),'rejected':len(rejected)}
                print(task,split,report['tasks'][task][split],flush=True)
    if a.trajectory:
        (a.out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        report['trajectory_coverage']=add_motor_trajectories(a.out,a.trajectory,a.count)
        report['coverage_note']='Independent loose-table banks for all five tasks; motor held/near families extracted from measured native trajectories.'
    (a.out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
