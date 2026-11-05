"""Explicit SGS-style prepared motor holds; never a loose-to-assembly demonstration."""
import argparse
from pathlib import Path
import mujoco
import numpy as np
from .scene import model, metadata, ROOT
from .robot import ik, down_rotation
from .resets import reset
from .fastener import origins
from .trajectory import Trajectory, STATE_SPEC


def measure(m,d):
    meta=metadata();target=origins(meta);base=d.body('base');motor=d.body('motor')
    R=base.xmat.reshape(3,3);goal=base.xpos+R@(target['motor']-target['base'])
    extent=np.ptp(np.asarray(meta['motor']['bounds']),axis=0)/2
    kp=np.vstack((np.zeros(3),np.diag(extent)))
    error=float(np.linalg.norm(motor.xpos+kp@motor.xmat.reshape(3,3).T-(goal+kp@R.T),axis=1).max())
    partners=set()
    for c in d.contact:
        names={m.body(int(m.geom_bodyid[g])).name for g in c.geom}
        if 'motor' in names:partners|=names-{'motor'}
    bilateral={'right_gripper','right_moving_jaw_so101_v1'}<=partners
    return error,bilateral,sorted(partners)


def generate(m,seed,out,release=False,gap=0.,withdraw=False):
    d=mujoco.MjData(m)
    # This predecessor reset places the motor at its mating pose, then settles it.
    mujoco.mj_setState(m,d,reset(m,'horn',seed),STATE_SPEC)
    d.ctrl[:]=d.qpos[:12];mujoco.mj_forward(m,d)
    base=d.body('base');R=base.xmat.reshape(3,3).copy()
    if gap:
        if not 0<=gap<=.012:raise ValueError('Prepared gap must be0–12mm')
        q=m.jnt_qposadr[m.joint('motor_free').id];v=m.jnt_dofadr[m.joint('motor_free').id]
        d.qpos[q:q+3]+=R@np.array([0.,-gap,0.]);d.qvel[v:v+6]=0
        mujoco.mj_forward(m,d)
    motor=d.body('motor').xpos.copy()
    yaw=np.pi+np.arctan2(R[1,0],R[0,0])
    site=motor+R@np.array([.0126,-.005,.004])
    d.qpos[6:11]=ik(m,d.qpos,'right',site,down_rotation(yaw))
    d.qpos[11]=.15;d.qvel[6:12]=0;d.ctrl[6:]=d.qpos[6:12]
    d.time=0.;mujoco.mj_forward(m,d)
    initial=max(0.,-min(d.contact.dist,default=0.))
    if initial>.0004:raise ValueError(f'Prepared initial overlap {initial:.6f}m')
    log=Trajectory(m,d);report=dict(prepared_sgs_reset=True,full_loose_assembly=False,
        task='motor',seed=seed,source_predecessor_reset='horn',prepared_axial_gap_m=gap,initial_penetration_m=initial)
    meta=metadata();target=origins(meta);part_v=m.jnt_dofadr[m.joint('motor_free').id]
    extent=np.ptp(np.asarray(meta['motor']['bounds']),axis=0)/2;kp=np.vstack((np.zeros(3),np.diag(extent)))
    audit=mujoco.MjData(m) if release else None
    def advance(grip,seconds,label,joints=None):
        start=d.ctrl.copy();end=start.copy();end[11]=grip
        if joints is not None:end[6:11]=joints
        n=round(seconds/m.opt.timestep);tail=[]
        for k in range(n):
            u=(k+1)/n;d.ctrl[:]=start+(u*u*(3-2*u))*(end-start);log.step(label)
            if log.depths[-1]>.0004:raise RuntimeError('Prepared hold exceeded contact limit')
            if label=='Check prepared motor retention' and k>=n-round(.4/m.opt.timestep):
                # Recompute the saved post-integration state's geometry on scratch
                # data, leaving the live solver and its warm start unchanged.
                mujoco.mj_setState(m,audit,log.states[-1],STATE_SPEC)
                audit.ctrl[:]=d.ctrl;mujoco.mj_forward(m,audit)
                b=audit.body('base');Rb=b.xmat.reshape(3,3);obj=audit.body('motor');goal=b.xpos+Rb@(target['motor']-target['base'])
                err=float(np.linalg.norm(obj.xpos+kp@obj.xmat.reshape(3,3).T-(goal+kp@Rb.T),axis=1).max())
                touch=False
                for c in audit.contact:
                    names=[m.body(int(m.geom_bodyid[g])).name for g in c.geom]
                    touch|='motor' in names and any(name.startswith(('left_','right_')) for name in names)
                tail.append((err,touch,np.linalg.norm(d.qvel[part_v:part_v+3]),np.linalg.norm(d.qvel[part_v+3:part_v+6])))
        if tail:
            a=np.asarray(tail);report['fine_retention_check']=dict(physics_steps=len(tail),duration_s=len(tail)*m.opt.timestep,contact_scope='Recomputed from each saved physical state; live solver warm start unchanged',max_keypoint_error_m=float(a[:,0].max()),hand_contact_steps=int(a[:,1].sum()),max_linear_speed_m_s=float(a[:,2].max()),max_angular_speed_rad_s=float(a[:,3].max()),passed=bool(a[:,0].max()<.001 and not a[:,1].any() and a[:,2].max()<.005 and a[:,3].max()<.1))
        mujoco.mj_forward(m,d)
    try:
        advance(-.1,1.,'Close native fingers around seated motor')
        advance(-.1,.5,'Hold seated motor through contact')
        error,held,partners=measure(m,d)
        report.update(held_state_index=len(log.states)-1,held_keypoint_error_m=error,
                      bilateral_motor_grasp=held,held_contact_partners=partners,
                      held_reset_qualified=held and error<.004)
        if release:
            advance(.75,1.,'Release prepared seated motor')
            if withdraw:
                point=d.site('right_gripperframe').xpos+R@np.array([.012,0.,0.])
                q=ik(m,d.qpos,'right',point,down_rotation(yaw))
                advance(.75,.8,'Withdraw opened hand from motor',q)
            advance(.75,.6,'Check prepared motor retention')
            error,held,partners=measure(m,d)
            report.update(released_keypoint_error_m=error,released_contact_partners=partners,
                          release_endpoint_qualified=error<.001 and not any(p.startswith(('left_','right_')) for p in partners),release_check_scope='Endpoint and separately reported 800-step retention audit',hand_withdrawal=withdraw)
    except Exception as exc:report['failure']=f'{type(exc).__name__}: {exc}'
    log.save(out,report)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',default='0');p.add_argument('--out',type=Path,default=ROOT/'artifacts_local/prepared_holds');p.add_argument('--release-check',action='store_true');p.add_argument('--gap',type=float,default=0.);p.add_argument('--withdraw-on-release',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);m=model()
    for seed in map(int,a.seeds.split(',')):
        try:
            stem=f'motor_gap{round(a.gap*1e6):05d}_{seed:06d}' if a.gap else f'motor_goal_{seed:06d}'
            print(generate(m,seed,a.out/stem,a.release_check,a.gap,a.withdraw_on_release),flush=True)
        except Exception as exc:print({'seed':seed,'rejected':str(exc)},flush=True)


if __name__=='__main__':main()
