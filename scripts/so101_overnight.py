"""Bounded local CUDA learner supervisor with checkpoints and two-hour reviews.

No hardware or cloud operations. Create RUN_DIR/STOP for a graceful stop.
Numerical failures stop for review; unexpected process exits get at most two
checkpoint restarts. A falling loss is never treated as task success.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import psutil
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def write_json(path, value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def records(path):
    if not path.exists():return []
    result=[]
    for line in path.read_text().splitlines():
        try:result.append(json.loads(line))
        except json.JSONDecodeError:continue  # final line may still be flushing
    return result


def snapshot(run):
    rows=records(run/'log.jsonl')
    memory=psutil.virtual_memory();disk=shutil.disk_usage(ROOT)
    try:
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu',
                                     '--format=csv,noheader,nounits'],text=True,timeout=10).strip()
    except (OSError,subprocess.SubprocessError):gpu=None
    last=rows[-1] if rows else None
    return dict(time=datetime.now(timezone.utc).isoformat(),latest=last,iterations=len(rows),
                available_memory_gb=memory.available/2**30,disk_free_gb=disk.free/2**30,gpu_csv=gpu,
                log_age_s=time.time()-(run/'log.jsonl').stat().st_mtime if rows else None)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--end-utc',required=True,help='ISO8601 end time including +00:00')
    ap.add_argument('--envs',type=int,default=64)
    ap.add_argument('--threads',type=int,default=16)
    ap.add_argument('--steps',type=int,default=32)
    ap.add_argument('--bank-dir',type=Path,default=ROOT/'data/so101/banks')
    ap.add_argument('--init-policy',type=Path)
    ap.add_argument('--preflight',type=Path,default=ROOT/'artifacts_local/preflight.json')
    ap.add_argument('--max-contact-depth',type=float,default=None)
    ap.add_argument('--stability',choices=['velocity','pose'],default='velocity')
    ap.add_argument('--safety-penalty',type=float,default=0.)
    ap.add_argument('--physics',choices=['native','whole_helper_sdf'],default='native')
    ap.add_argument('--physics-model',type=Path)
    ap.add_argument('--review-seconds',type=int,default=7200)
    ap.add_argument('--first-review-utc',help='Optional first review ISO8601 time; subsequent reviews use --review-seconds')
    ap.add_argument('--sampler-floor-mass',type=float,default=1.6,
                    help='5 tasks x 5 source families x 640 entries x 1e-4; scale per-entry floor to pilot bank size')
    a=ap.parse_args();run=a.run_dir.resolve();run.mkdir(parents=True,exist_ok=True)
    lock=(run/'supervisor.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    end=datetime.fromisoformat(a.end_utc).timestamp()
    if end<=time.time():raise ValueError('End time is in the past')
    from assembly_policy.so101.resets import signature,check_bank,directory_signature
    check_bank(a.bank_dir)
    bank_count=sum(len(np.load(a.bank_dir/f'{name}_train.npz')['states'])
                   for name in ('motor','horn','screw','retainer','yoke'))
    eps_q=a.sampler_floor_mass/bank_count
    preflight=json.loads(a.preflight.read_text())
    physics_signature=signature()
    if a.physics=='whole_helper_sdf':
        import hashlib
        from assembly_policy.so101.warp_env import archived_helper_factory,physics_identity
        if a.physics_model is None:raise ValueError('GPU run requires qualified model archive')
        physics_signature=hashlib.sha256(json.dumps(physics_identity(archived_helper_factory(a.physics_model)),sort_keys=True).encode()).hexdigest()
    if (preflight['model_signature']!=signature() or not all(preflight['checks'].values()) or
        preflight.get('bank_directory_signature')!=directory_signature(a.bank_dir) or
        preflight.get('max_contact_depth')!=a.max_contact_depth or
        preflight.get('stability_metric','velocity')!=a.stability or
        preflight.get('safety_penalty',0.)!=a.safety_penalty or
        preflight.get('physics_backend','native')!=a.physics or
        preflight.get('physics_signature',preflight['model_signature'])!=physics_signature):
        raise ValueError('A passing preflight for the current model and exact bank is required')
    environment={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MUJOCO_GL':'egl'}
    process=None;restarts=0
    reviews=max((int(p.name) for p in (run/'reviews').glob('*') if p.is_dir() and p.name.isdigit()),default=0)
    next_review=datetime.fromisoformat(a.first_review_utc).timestamp() if a.first_review_utc else time.time()+a.review_seconds
    status={'supervisor_pid':os.getpid(),'run_dir':str(run),'end_utc':a.end_utc,
            'model_signature':signature(),'backend':a.physics+'_physics_cuda_ppo','state':'starting'}
    status['physics_signature']=physics_signature
    status['safety_penalty']=a.safety_penalty
    status['bank_dir']=str(a.bank_dir.resolve())
    status['bank_directory_signature']=directory_signature(a.bank_dir)
    status.update(bank_entries=bank_count,sampler_eps_q=eps_q,sampler_floor_mass=a.sampler_floor_mass)
    write_json(run/'supervisor.json',status)
    evaluations=[]
    def stop(reason):
        status['state']='stopping';status['reason']=reason;write_json(run/'supervisor.json',status)
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:process.wait(timeout=180)
            except subprocess.TimeoutExpired:
                process.kill();process.wait();status['forced_kill']=True
    try:
        while time.time()<end and not (run/'STOP').exists():
            if process is None or process.poll() is not None:
                if process is not None:
                    status['previous_returncode']=process.returncode
                    prior_log=run/f'trainer_attempt_{restarts}.log'
                    if prior_log.exists() and 'FloatingPointError' in prior_log.read_text()[-32768:]:
                        status.update(state='needs_review',reason='Numerical failure requires diagnosis');break
                    restarts+=1
                    if restarts>2:
                        status.update(state='needs_review',reason='Repeated unexpected trainer exits');break
                command=[sys.executable,'-m','assembly_policy.train','--suite','so101','--run-dir',str(run),
                         '--bank-dir',str(a.bank_dir.resolve()),
                         '--envs',str(a.envs),'--nthread',str(a.threads),'--steps',str(a.steps),
                         '--device','cuda','--iters','100000','--save-every','5','--keep-every','50',
                         '--eps-q',str(eps_q),
                         '--no-eval','--max-hours',str(max(.01,(end-time.time())/3600))]
                if a.init_policy:command+=['--init-policy',str(a.init_policy.resolve())]
                command+=['--so101-stability',a.stability]
                command+=['--so101-safety-penalty',str(a.safety_penalty)]
                if a.max_contact_depth is not None:command+=['--so101-max-contact-depth',str(a.max_contact_depth)]
                command+=['--so101-physics',a.physics]
                if a.physics_model:command+=['--so101-physics-model',str(a.physics_model.resolve())]
                log=(run/f'trainer_attempt_{restarts}.log').open('a')
                process=subprocess.Popen(command,cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,
                                         stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                log.close();status.update(state='running',trainer_pid=process.pid,restarts=restarts,command=command)
                attempt_started=time.time()
                write_json(run/'supervisor.json',status)
            health=snapshot(run);health['trainer_pid']=process.pid
            write_json(run/'health.json',health)
            with (run/'health.jsonl').open('a') as f:f.write(json.dumps(health)+'\n')
            row=health['latest']
            failure=None
            if row:
                for key in ('fps','pi','vf','ent','kl','std','lr'):
                    if not math.isfinite(row[key]):failure=f'Nonfinite {key}'
                if row.get('unstable',0)>0:failure='Numerical reset/nonfinite simulation state'
                if min(health['log_age_s'],time.time()-attempt_started)>1200:
                    failure='No completed learner iteration for twenty minutes'
            elif time.time()-attempt_started>1200:failure='No first learner iteration for twenty minutes'
            if health['available_memory_gb']<.75:failure='Insufficient host memory headroom'
            if health['disk_free_gb']<2:failure='Insufficient checkpoint disk headroom'
            if failure:
                stop(failure);status.update(state='needs_review',reason=failure);break
            if time.time()>=next_review:
                reviews+=1;review=run/'reviews'/f'{reviews:02d}';review.mkdir(parents=True,exist_ok=True)
                write_json(review/'health.json',health)
                if (run/'checkpoint.pt').exists():
                    shutil.copy2(run/'checkpoint.pt',review/'checkpoint.pt')
                    log=(review/'evaluation.log').open('w')
                    p=subprocess.Popen([sys.executable,'-m','assembly_policy.so101.evaluate','--policy',str(review/'checkpoint.pt'),
                                        '--bank-dir',str(a.bank_dir.resolve()),'--out',str(review/'evaluation.json'),
                                        '--nthread','1','--trials','2'],cwd=ROOT,env=environment,
                                       stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    log.close();evaluations.append(p)
                    status['evaluation_pids']=[p.pid for p in evaluations if p.poll() is None]
                next_review+=a.review_seconds
                status['reviews_started']=reviews;write_json(run/'supervisor.json',status)
            time.sleep(30)
        if status['state']!='needs_review':
            stop('User STOP file' if (run/'STOP').exists() else 'Overnight time boundary')
            status['state']='stopped'
    except BaseException as error:
        stop(type(error).__name__+': '+str(error));status['state']='needs_review';raise
    finally:
        status['finished_utc']=datetime.now(timezone.utc).isoformat()
        if (run/'STOP').exists():
            for child in evaluations:
                if child.poll() is None:child.terminate()
        write_json(run/'supervisor.json',status)
        write_json(run/'final_health.json',snapshot(run))
        lock.close()


if __name__=='__main__':main()
