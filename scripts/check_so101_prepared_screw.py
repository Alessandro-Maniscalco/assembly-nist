"""Bounded canonical prepared screw-driving probe; never claims loose assembly.

Uses only native robot position actuators after initialization. Target parts and
screwdriver remain free. The initial threaded screw and held/aligned driver are
explicit preparation assumptions. No solver, geometry or force edits are made.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import mujoco
import numpy as np
import psutil
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from assembly_policy.so101.scene import ROOT, PARTS, load_plugin, metadata
from assembly_policy.so101.fastener import origins, PITCH
from assembly_policy.so101.robot import home, arm_indices

SPEC = mujoco.mjtState.mjSTATE_FULLPHYSICS
CANONICAL = 'dbbe6bfe1cadb7501a3f2476c5d80ac537d827a41b3d223704e9481e07d3547a'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def state(m, d):
    result = np.empty(mujoco.mj_stateSize(m, SPEC))
    mujoco.mj_getState(m, d, result, SPEC)
    return result


def set_pose(m, d, name, p, R):
    q = m.joint(name + '_free').qposadr[0]
    d.qpos[q:q+3] = p
    quat = Rotation.from_matrix(R).as_quat()
    d.qpos[q+3:q+7] = quat[[3, 0, 1, 2]]


def run(out, inspect_only=False, placement=(0., .06), screw_lift=.001, bit_phase=0., short_pulse=False, pulsed=False):
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'report.json').exists():
        raise ValueError('Choose a new output directory; existing evidence is immutable.')
    (out/'executed_script.py').write_text(Path(__file__).read_text())
    source = ROOT/'artifacts_local/tool_pickup/raised10_coast'
    source_report = json.loads(source.with_suffix('.json').read_text())
    assert digest(source.with_suffix('.mjb')) == CANONICAL == source_report['model_sha256']
    for name, expected in source_report['plugin_sha256'].items():
        assert digest(ROOT/'plugin/build'/name) == expected
    load_plugin()
    m = mujoco.MjModel.from_binary_path(str(source.with_suffix('.mjb')))
    assert m.neq == 0 and m.nu == 12 and m.opt.timestep == .0005
    for name in (*PARTS, 'driver'):
        assert m.jnt_type[m.joint(name+'_free').id] == mujoco.mjtJoint.mjJNT_FREE
    d = mujoco.MjData(m)
    home(m, d)
    with np.load(source.with_suffix('.npz')) as z:
        labels = list(z['labels'])
        k = np.flatnonzero(z['stage'] == labels.index('Hold tool OFF above table'))[-1]
        held_state = z['states'][k].copy()
    ref = mujoco.MjData(m)
    mujoco.mj_setState(m, ref, held_state, SPEC)
    mujoco.mj_forward(m, ref)
    site_R = ref.site('right_gripperframe').xmat.reshape(3, 3)
    relative_p = site_R.T@(ref.body('driver').xpos-ref.site('right_gripperframe').xpos)
    relative_R = site_R.T@ref.body('driver').xmat.reshape(3, 3)
    target = origins(metadata())
    base_p = d.body('base').xpos.copy()
    base_p[:2] = placement
    base_p[2] -= .0017  # prepare the base 0.3 mm above the table, not its loose-reset 2 mm
    # Keep the unused loose retaining frame outside the shifted prepared work area.
    rq = m.joint('retainer_free').qposadr[0]
    d.qpos[rq:rq+2] = [.12, .07]
    for name in ('base', 'motor', 'horn'):
        set_pose(m, d, name, base_p+target[name]-target['base'], np.eye(3))
    screw_bearing = base_p+target['screw']-target['base']+[0, 0, screw_lift]
    qi, _ = arm_indices(m, 'right')
    joints = [m.joint('right_'+name).id for name in ('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll')]
    bounds = m.jnt_range[joints]
    scratch = mujoco.MjData(m)
    scratch.qpos[:] = d.qpos
    tip_local = np.array([0., 0., .105])
    def tool_pose(q):
        scratch.qpos[qi[:5]] = q
        mujoco.mj_kinematics(m, scratch)
        s = scratch.site('right_gripperframe')
        R = s.xmat.reshape(3, 3)
        return s.xpos+R@relative_p, R@relative_R
    def residual(q, tip):
        p, R = tool_pose(q)
        return np.r_[p+R@tip_local-tip, .08*(R[:, 2]-[0, 0, -1])]
    candidates=[]
    seeds=[d.qpos[qi[:5]],ref.qpos[qi[:5]],np.array([0,.2,-1.2,-.5,0]),np.array([0,-.5,1.,-.5,0])]
    selected=None
    for seed in seeds:
        tip = screw_bearing+[0,0,.0009]
        sol=None
        for _ in range(3):
            sol=least_squares(residual,np.clip(seed,bounds[:,0]+1e-7,bounds[:,1]-1e-7),
                 args=(tip,),bounds=(bounds[:,0],bounds[:,1]),max_nfev=200,
                 ftol=1e-10,gtol=1e-10,xtol=1e-10)
            p,R=tool_pose(sol.x)
            yaw=np.arctan2(R[1,0],R[0,0]); yaw=(yaw+np.pi/4)%(np.pi/2)-np.pi/4
            bearing=screw_bearing+[0,0,yaw*PITCH/(2*np.pi)]
            tip=bearing+[0,0,.0009]
            seed=sol.x
        err=residual(sol.x,tip)
        row={'q':sol.x.tolist(),'position_error_m':float(np.linalg.norm(err[:3])),
             'axis_error_rad':float(np.arccos(np.clip(-R[2,2],-1,1))), 'screw_yaw_rad':float(yaw)}
        candidates.append(row)
        if row['position_error_m']<.00005 and row['axis_error_rad']<.001:
            selected=(sol.x.copy(),p.copy(),R.copy(),bearing.copy(),yaw)
            break
    report=dict(scope='prepared threaded screw and prepared held/aligned driver; no pickup, insertion, full assembly or learned-policy claim',
                model_sha256=CANONICAL,plugin_sha256=source_report['plugin_sha256'],
                script_sha256=digest(out/'executed_script.py'),
                geometry=dict(bit_cross_full_mm=[2.6,.6],recess_cross_full_mm=[3.2,.8],recess_bottom_m=.0008,recess_top_m=.0014,
                              bit_tip_initial_above_bearing_m=.0009,nominal_thread_pitch_m=PITCH),
                preparation=dict(base_xy_m=list(placement),screw_lift_m=screw_lift,held_reference_step=int(k),bit_phase_rad=bit_phase,short_pulse=short_pulse,pulsed=pulsed,
                                 assembled_predecessors=['base','motor','horn'],remaining_parts='retainer moved to x=0.12/y=0.07 m; other unused parts at default loose poses',
                                 base_floor_clearance_m=.0003,
                                 screw_yaw_adjustment='height follows pitch * yaw / (2*pi) to retain nominal helical phase'),
                ik_candidates=candidates,passed=False)
    if selected is None:
        report['failure']='No checked IK candidate reaches vertical engaged bit within 50 um / 1 mrad.'
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return
    q,p,R,bearing,yaw=selected
    d.qpos[qi[:5]]=q
    d.qpos[qi[5]]=ref.qpos[qi[5]]
    for name in ('trigger_saddle','button_forward','button_reverse'):
        d.joint(name).qpos[:]=ref.joint(name).qpos
    d.joint('spindle').qpos[:]=bit_phase
    set_pose(m,d,'driver',p,R)
    set_pose(m,d,'screw',bearing,Rotation.from_euler('z',yaw).as_matrix())
    d.qvel[:]=0
    d.ctrl[:]=d.qpos[:12]
    d.ctrl[11]=145*2*np.pi/4096
    mujoco.mj_forward(m,d)
    def contacts():
        result=[]
        for i,c in enumerate(d.contact):
            names=[m.geom(int(g)).name for g in c.geom]
            if min(c.dist,0)<-.0001 or 'screw_head_contact' in names:
                f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f)
                result.append({'pair':names,'distance_m':float(c.dist),'normal_force_n':float(f[0])})
        return result
    report['initial_contacts']=contacts()
    initial_depth=max([0.,*[-float(c.dist) for c in d.contact]])
    report['initial_max_penetration_m']=initial_depth
    report['initial_predecessor_positions']={n:d.body(n).xpos.tolist() for n in ('base','motor','horn')}
    initial=state(m,d)
    report['initial_rss_gib']=psutil.Process().memory_info().rss/2**30
    if initial_depth>.0004 or inspect_only:
        report['failure']='Initial penetration exceeds 0.4 mm' if initial_depth>.0004 else 'Inspection only; no dynamics executed'
        np.savez_compressed(out/'initial.npz',initial_state=initial,controls=d.ctrl.copy())
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return
    baseline={n:d.body(n).xpos.copy() for n in ('base','motor','horn')}
    poses=[];controls=[];metrics=[];depths=[];labels=[];couplings=[]
    initial_screw_R=d.body('screw').xmat.reshape(3,3).copy()
    first_screw_z=float(d.body('screw').xpos[2])
    warning_start=d.warning.number.copy()
    max_clock=0.;failure=None
    stages=[('Carry OFF',145,.2,0),('Press physical button',130,.6,0),
            ('Drive with axial arm feed',130,1.,.0008),('Release physical button',145,.6,.0008),('Coast OFF',145,.4,.0008)]
    if short_pulse:
        stages=[('Carry OFF',145,.2,0),('Press physical button',130,.35,0),
                ('Release physical button',145,.35,0),('Coast OFF',145,.4,0)]
    if pulsed:
        stages=[('Carry OFF',145,.2,0)]
        for cycle in range(3):
            stages += [(f'Physical pulse {cycle+1} ON',130,.04,0),
                       (f'Physical pulse {cycle+1} OFF',145,.04,0),
                       (f'Physical pulse {cycle+1} coast',145,.15,0)]
        stages.append(('Final coast OFF',145,.4,0))
    try:
        for stage,ticks,seconds,feed in stages:
            old_ctrl=d.ctrl.copy();target_ctrl=d.ctrl.copy();target_ctrl[11]=ticks*2*np.pi/4096
            if feed:
                tip=bearing+[0,0,.0009-feed]
                sol=least_squares(residual,q,args=(tip,),bounds=(bounds[:,0],bounds[:,1]),max_nfev=200,
                     ftol=1e-10,gtol=1e-10,xtol=1e-10)
                if np.linalg.norm(residual(sol.x,tip)[:3])>.00005:raise RuntimeError('Feed IK exceeds 50 um')
                target_ctrl[6:11]=sol.x
            n=round(seconds/m.opt.timestep)
            for step in range(n):
                alpha=(step+1)/n;alpha=alpha*alpha*(3-2*alpha)
                d.ctrl[:]=old_ctrl+alpha*(target_ctrl-old_ctrl)
                before=d.time;mujoco.mj_step(m,d)
                max_clock=max(max_clock,abs(d.time-before-m.opt.timestep))
                predepth=max([0.,*[-float(c.dist) for c in d.contact]])
                # Refresh endpoint contacts as well; never change physical state.
                mujoco.mj_forward(m,d)
                depth=max(predepth,max([0.,*[-float(c.dist) for c in d.contact]]))
                head_pairs=[]
                for ci,c in enumerate(d.contact):
                    names=[m.geom(int(g)).name for g in c.geom]
                    if 'screw_head_contact' in names and any(name in names for name in ('bit_x','bit_y','bit_shaft')):
                        f=np.zeros(6);mujoco.mj_contactForce(m,d,ci,f)
                        head_pairs.append([*map(int,c.geom),float(c.dist),float(f[0])])
                screw_R=d.body('screw').xmat.reshape(3,3)
                relative=initial_screw_R.T@screw_R
                angle=float(np.arctan2(relative[1,0],relative[0,0]))
                drift=[float(np.linalg.norm(d.body(name).xpos-baseline[name])) for name in baseline]
                metrics.append([d.time,d.joint('button_forward').qpos[0],d.joint('spindle').qpos[0],
                    d.joint('spindle').qvel[0],angle,d.body('screw').xpos[2]-first_screw_z,*drift,len(head_pairs),
                    sum(row[3] for row in head_pairs)])
                poses.append(state(m,d));controls.append(d.ctrl.copy());depths.append(depth);labels.append(stage);couplings.append(head_pairs)
                if not np.isfinite(poses[-1]).all() or not np.isfinite(d.qvel).all():raise RuntimeError('Nonfinite native state')
                if max_clock>1e-8:raise RuntimeError('Native clock discontinuity')
                if np.any(d.warning.number-warning_start):raise RuntimeError('Native physics warning')
                if depth>.0004:raise RuntimeError(f'Contact penetration exceeds 0.4 mm: {depth*1000:.6f} mm')
                if max(drift)>.002:raise RuntimeError('Prepared predecessor drift exceeds 2 mm')
                for name in (*PARTS,'driver'):
                    va=m.joint(name+'_free').dofadr[0]
                    if np.linalg.norm(d.qvel[va+3:va+6])>30:
                        raise RuntimeError(f'{name} angular speed exceeds production 30 rad/s guard')
                if step%100==0:
                    if psutil.Process().memory_info().rss>2.5*2**30:raise RuntimeError('RSS guard 2.5 GiB')
                    if psutil.virtual_memory().available<2*2**30:raise RuntimeError('Host available memory below 2 GiB')
                    if time.monotonic()-started>900:raise RuntimeError('15-minute runtime bound')
            print(stage,metrics[-1],flush=True)
    except Exception as error:
        failure=f'{type(error).__name__}: {error}'
    a=np.asarray(metrics)
    if len(a):
        angle=np.unwrap(a[:,4]);turns=float((angle[-1]-angle[0])/(2*np.pi));advance=float(a[-1,5]-a[0,5])
        report.update(physics_steps=len(a),failure=failure,max_penetration_m=float(max(depths)),
            max_clock_step_error_s=max_clock,warnings=(d.warning.number-warning_start).tolist(),
            max_button_m=float(a[:,1].max()),max_spindle_rad_s=float(a[:,3].max()),
            screw_turns=turns,screw_axial_advance_m=advance,helical_expected_m=turns*PITCH,
            bit_head_contact_steps=int(np.sum(a[:,9]>0)),max_predecessor_drift_m=a[:,6:9].max(axis=0).tolist(),
            end_contacts=contacts(),elapsed_s=time.monotonic()-started)
        report['passed']=bool(failure is None and turns<-.15 and advance<-.00005 and abs(advance-turns*PITCH)<.0001 and np.sum(a[:,9]>0)>100)
        np.savez_compressed(out/'trace.npz',initial_state=initial,states=np.asarray(poses),controls=np.asarray(controls),
                            penetration=np.asarray(depths),labels=np.asarray(labels),metrics=a,
                            metric_names=np.asarray(['time_s','forward_button_m','spindle_angle_rad','spindle_speed_rad_s','screw_yaw_rad','screw_z_delta_m',
                                                     'base_drift_m','motor_drift_m','horn_drift_m','bit_head_contacts','bit_head_normal_force_n']))
        (out/'bit_head_contacts.json').write_text(json.dumps(couplings)+'\n')
    else:report['failure']=failure
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


def audit(out):
    """Audit relative helix motion from complete recorded native states."""
    out = Path(out)
    report = json.loads((out/'report.json').read_text())
    if not (out/'trace.npz').exists():
        return report
    load_plugin()
    m = mujoco.MjModel.from_binary_path(str(ROOT/'artifacts_local/tool_pickup/raised10_coast.mjb'))
    with np.load(out/'trace.npz') as z:
        states=np.vstack((z['initial_state'],z['states']))
        metrics=z['metrics'].copy()
        controls=z['controls'].copy()
    q=states[:,1:1+m.nq]
    v=states[:,1+m.nq:1+m.nq+m.nv]
    def poses(name):
        a=m.joint(name+'_free').qposadr[0]
        return q[:,a:a+3],Rotation.from_quat(q[:,a+3:a+7][:,[1,2,3,0]])
    pm,Rm=poses('motor');ps,Rs=poses('screw');pb,Rb=poses('base');pt,Rt=poses('driver')
    relp=Rm.inv().apply(ps-pm)
    relR=(Rm.inv()*Rs).as_matrix()
    theta=np.unwrap(np.arctan2(relR[:,1,0],relR[:,0,0]))
    turns=(theta-theta[0])/(2*np.pi)
    advance=relp[:,2]-relp[0,2]
    pitch_residual=advance-turns*PITCH
    pred={}
    for name in ('motor','horn'):
        p,R=poses(name)
        rp=Rb.inv().apply(p-pb);rR=Rb.inv()*R
        pred[name]={'max_relative_translation_drift_m':float(np.linalg.norm(rp-rp[0],axis=1).max()),
                    'max_relative_angle_drift_rad':float((rR[0].inv()*rR).magnitude().max())}
    va={}
    for name in (*PARTS,'driver'):
        a=m.joint(name+'_free').dofadr[0]
        va[name]={'max_linear_m_s':float(np.linalg.norm(v[:,a:a+3],axis=1).max()),
                  'max_angular_rad_s':float(np.linalg.norm(v[:,a+3:a+6],axis=1).max())}
    armids=np.r_[arm_indices(m,'left')[1],arm_indices(m,'right')[1]]
    maxarm=float(np.abs(v[:,armids]).max())
    axis_dot=np.einsum('ij,ij->i',Rt.as_matrix()[:,:,2],Rs.as_matrix()[:,:,2])
    detail=dict(relative_screw_turns=float(turns[-1]),relative_screw_advance_m=float(advance[-1]),
                expected_advance_m=float(turns[-1]*PITCH),final_pitch_error_m=float(pitch_residual[-1]),
                max_pitch_error_m=float(np.abs(pitch_residual).max()),
                max_tool_screw_axis_error_rad=float(np.arccos(np.clip(-axis_dot,-1,1)).max()),
                predecessor_relative_drift=pred,free_body_velocity=va,max_arm_joint_velocity_rad_s=maxarm,
                final_button_m=float(metrics[-1,1]),final_spindle_rad_s=float(metrics[-1,3]),
                active_button_steps=int(np.sum(metrics[:,1]>.0006)),
                final_bit_head_contact_count=int(metrics[-1,9]),
                report_corrects='Relative motor-frame advancement excludes table settling from the raw world-Z trace.')
    # Call the production method directly while reusing the archived model. This
    # avoids compiling or allocating a second scene merely to inspect two states.
    from assembly_policy.so101.env import BatchedAssemblyEnv, TASK_NAMES
    view=BatchedAssemblyEnv.__new__(BatchedAssemblyEnv)
    view.m=m;view.task_names=['screw'];view.task=np.array([0]);view.N=1
    view.meta=metadata();view.origins=origins(view.meta)
    view.data=[mujoco.MjData(m)]
    snapshots={}
    for label,index in [('initial',0),('final',-1)]:
        d=view.data[0];mujoco.mj_setState(m,d,states[index],SPEC)
        d.ctrl[:]=controls[0 if index==0 else -1]
        mujoco.mj_forward(m,d)
        measured=float(BatchedAssemblyEnv.assembly_distance(view)[0])
        errors={};base=d.body('base');R=base.xmat.reshape(3,3)
        for name in TASK_NAMES[:3]:
            body=d.body(name);offset=view.origins[name]-view.origins['base']
            extent=np.ptp(np.array(view.meta[name]['bounds']),axis=0)/2
            kp=np.array([[0,0,0],[extent[0],0,0],[0,extent[1],0],[0,0,extent[2]]])
            if name=='screw':kp=np.array([[0,0,0],[0,0,.003]])
            actual=body.xpos+kp@body.xmat.reshape(3,3).T
            goal=base.xpos+(kp+offset)@R.T
            errors[name]=np.linalg.norm(actual-goal,axis=1).tolist()
        assert abs(max(max(row) for row in errors.values())-measured)<1e-12
        active=m.body('screw').id
        held=any(active in m.geom_bodyid[c.geom] and any(
            (m.body(int(body)).name.startswith(('left_','right_','driver_')) or m.body(int(body)).name=='driver')
            for body in m.geom_bodyid[c.geom]) for c in d.contact)
        pairs={frozenset((m.geom(int(c.geom1)).name,m.geom(int(c.geom2)).name)) for c in d.contact}
        thread_contact=frozenset(('male_contact','female_contact')) in pairs
        head_horn_contact=frozenset(('screw_head_contact','horn_contact')) in pairs
        snapshots[label]=dict(assembly_distance_m=measured,keypoint_errors_m=errors,
            inside_production_1mm=bool(measured<.001),screw_touching_hand_or_tool=bool(held),
            male_female_contact=thread_contact,head_horn_contact=head_horn_contact,
            release_required=True,full_screw_task_success=False,
            task_success_explanation='No release/retention completion is demonstrated; a pose or coupling result is insufficient.')
    detail['production_snapshots']=snapshots
    detail['production_full_screw_task_success']=False
    valid=report.get('failure') is None and not any(report['warnings']) and report['max_penetration_m']<=.0004
    button=float(metrics[:,1].max())>.0006 and metrics[-1,1]<.0006 and abs(metrics[-1,3])<.1
    coupling=turns[-1]<-.15 and advance[-1]<-.00005 and abs(pitch_residual[-1])<.0001 and report['bit_head_contact_steps']>100
    velocity=maxarm<5 and all(row['max_linear_m_s']<1 and row['max_angular_rad_s']<30 for row in va.values())
    detail['gates']={'full_trace_valid':bool(valid),'physical_on_off':bool(button),'thread_coupling':bool(coupling),'velocity':bool(velocity)}
    detail['passed']=all(detail['gates'].values())
    np.savez_compressed(out/'relative_audit.npz',time=states[:,0],turns=turns,advance_m=advance,pitch_residual_m=pitch_residual)
    (out/'relative_audit.json').write_text(json.dumps(detail,indent=2)+'\n')
    report.setdefault('raw_world_motion_passed',report['passed'])
    report['passed']=detail['passed'];report['relative_audit']=detail
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(detail,indent=2),flush=True)
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=ROOT/'artifacts_local/prepared_screw/vertical')
    ap.add_argument('--inspect-only',action='store_true')
    ap.add_argument('--audit-only',action='store_true')
    ap.add_argument('--base-x',type=float,default=0.)
    ap.add_argument('--base-y',type=float,default=.06)
    ap.add_argument('--screw-lift',type=float,default=.001)
    ap.add_argument('--bit-phase',type=float,default=0.)
    ap.add_argument('--short-pulse',action='store_true')
    ap.add_argument('--pulsed',action='store_true')
    a=ap.parse_args()
    if not a.audit_only:run(a.out,a.inspect_only,(a.base_x,a.base_y),a.screw_lift,a.bit_phase,a.short_pulse,a.pulsed)
    audit(a.out)
