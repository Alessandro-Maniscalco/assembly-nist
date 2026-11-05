"""Actuator-driven feasibility demonstration; never presented as a learned policy."""
from pathlib import Path
import argparse
import json
import signal
import os
import mujoco
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw
from .scene import model, ROOT, PARTS, metadata
from .robot import home, ik, ik_fixed_wrist, down_rotation
from .fastener import origins
from .trajectory import Trajectory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT/'artifacts_local/demo.mp4')
    ap.add_argument('--no-video', action='store_true')
    ap.add_argument('--grasp-offset', type=float, default=.004)
    ap.add_argument('--travel-height',type=float,default=.08)
    ap.add_argument('--servo-peak',type=float,default=None,help='Optional documented joint torque ceiling in Nm')
    ap.add_argument('--gripper-effort',type=float,default=None,help='Explicit actuator-limit experiment, Nm; keep below2.94Nm nominal stall')
    ap.add_argument('--grasp-x', type=float, default=None)
    ap.add_argument('--grasp-pad-y', type=float, default=.004, help='Site-local pad offset to balance the grasp contact line')
    ap.add_argument('--grasp-yaw', type=float, default=180., help='Degrees; site is fixed finger, not jaw midpoint')
    ap.add_argument('--arm-spacing', type=float, default=None)
    ap.add_argument('--insert', action='store_true')
    ap.add_argument('--layout',choices=['current','front','opposed'],default='current')
    ap.add_argument('--align-motor',action='store_true',help='Use measured held-part orientation when planning insertion')
    ap.add_argument('--stabilize', action='store_true')
    ap.add_argument('--support-style',choices=['foot','rear','sidefoot','horizontal'],default='foot')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--max-penetration', type=float, default=.0004)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    m = model()
    if args.layout in ('front','opposed'):
        from .layout import front_mount, opposed_mounts
        (front_mount if args.layout=='front' else opposed_mounts)(m)
    if args.arm_spacing is not None:
        m.body_pos[m.body('left_base').id,0]=-args.arm_spacing
        m.body_pos[m.body('right_base').id,0]=args.arm_spacing
    if args.servo_peak is not None:
        if not 0<args.servo_peak<=2.94:raise ValueError('Peak must remain within documented nominal servo stall torque')
        m.actuator_forcerange[:]=np.clip(m.actuator_forcerange,-args.servo_peak,args.servo_peak)
    if args.gripper_effort is not None:
        if not 0<args.gripper_effort<=2.94:raise ValueError('Effort must remain within documented nominal servo stall torque')
        m.actuator_forcerange[[5,11]]=[-args.gripper_effort,args.gripper_effort]
    d = mujoco.MjData(m); home(m,d)
    if args.seed is not None:
        from .resets import reset
        from .trajectory import STATE_SPEC
        mujoco.mj_setState(m,d,reset(m,'motor',args.seed),STATE_SPEC)
        d.ctrl[:]=d.qpos[:12];mujoco.mj_forward(m,d)
    log=Trajectory(m,d)
    def interrupted(signum, frame):
        raise RuntimeError(f'External termination signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    cam=mujoco.MjvCamera();cam.lookat[:]=[0,.02,.15]
    cam.distance=1.1;cam.azimuth=115;cam.elevation=-35
    opt=mujoco.MjvOption();opt.geomgroup[3]=0
    renderer=None if args.no_video else mujoco.Renderer(m,height=800,width=1280)
    writer=None if args.no_video else imageio.get_writer(args.out,fps=25,codec='libx264',quality=8,macro_block_size=1)
    max_height=-np.inf;motor_start=None;failure=None
    snapshots=[]
    def frame(label):
        if renderer is None:return
        renderer.update_scene(d,camera=cam,scene_option=opt)
        im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
        draw.rectangle((0,0,1280,75),fill=(21,27,36))
        draw.text((24,12),'HIWONDER ASSEMBLY  |  TWO CLAMPED SO-101 ARMS',font_size=22,fill='white')
        draw.text((24,44),'Scripted physics demonstration - no trained policy  |  '+label,font_size=17,fill=(180,213,237))
        writer.append_data(np.array(im))
    def advance(target,seconds,label):
        nonlocal max_height
        start=d.ctrl.copy();count=round(seconds/m.opt.timestep)
        for t in range(count):
            u=(t+1)/count;w=u*u*(3-2*u)
            d.ctrl[:]=start+w*(target-start)
            log.step(label)
            max_height=max(max_height,float(d.body('motor').xpos[2]))
            if log.depths[-1]>args.max_penetration:
                raise RuntimeError(f'{label}: penetration {log.depths[-1]*1000:.3f}mm exceeds {args.max_penetration*1000:.3f}mm')
            if len(log.states)%80==0:frame(label)
        latest=args.out.with_suffix('.latest.npz')
        temporary=latest.with_name(latest.stem+'.tmp.npz')
        np.savez_compressed(temporary,state=log.state(),ctrl=d.ctrl,stage=label)
        os.replace(temporary,latest)
        snapshots.append(dict(stage=label,time=float(d.time),state_index=len(log.states)-1,
                              motor=d.body('motor').xpos.tolist(),base=d.body('base').xpos.tolist()))
        print(label,'motor',d.body('motor').xpos,'tcp',d.site('right_gripperframe').xpos,
              'jaw',d.qpos[11],'peak penetration',max(log.depths),flush=True)
    def move(side,pose,seconds,label,yaw,orientation=None):
        ctrl=d.ctrl.copy();a=0 if side=='left' else 6
        axes=(0,2) if orientation is not None and side=='right' and args.align_motor and args.layout in ('front','opposed') else None
        ctrl[a:a+5]=ik(m,d.qpos,side,np.asarray(pose),down_rotation(yaw) if orientation is None else orientation,orientation_axes=axes)
        advance(ctrl,seconds,label)
    def grip(side,value,seconds,label):
        ctrl=d.ctrl.copy();ctrl[5 if side=='left' else 11]=value
        advance(ctrl,seconds,label)
    try:
        advance(d.ctrl.copy(),1.,'Settle loose parts')
        motor_start=d.body('motor').xpos.copy();base_start=d.body('base').xpos.copy()
        Rbase=d.body('base').xmat.reshape(3,3).copy()
        if args.stabilize:
            if args.support_style=='horizontal':
                support=base_start+Rbase@np.array([-.019,.015,.019])
                high=support.copy();high[2]=.14
                horizontal=np.column_stack(([0.,-1.,0.],[0.,0.,-1.],[1.,0.,0.]))
                def move_support(point,seconds,label):move('left',point,seconds,label,0,horizontal)
                grip('left',.75,.5,'Open rear gripper')
                move_support(high,3.,'Approach base from the rear')
                move_support(support,2.,'Lower fingers around base sides')
                grip('left',-.1,2.,'Grasp base horizontally')
            elif args.support_style in ('rear','sidefoot'):
                if args.support_style=='sidefoot':
                    support=base_start+Rbase@np.array([-.045,.045,-.024])
                    reference=ik(m,d.qpos,'left',support,down_rotation(-np.pi/2))
                    wrist_flex,wrist_roll=reference[3]-.15,reference[4]
                else:
                    support=base_start+Rbase@np.array([-.021,.045,.024])
                    wrist_flex=1.65;wrist_roll=ik(m,d.qpos,'left',support,down_rotation(0))[4]
                high=support.copy();high[2]=.08
                def move_support(point,seconds,label):
                    ctrl=d.ctrl.copy();ctrl[:5]=ik_fixed_wrist(m,d.qpos,'left',point,wrist_flex,wrist_roll)
                    advance(ctrl,seconds,label)
                grip('left',1.0 if args.support_style=='sidefoot' else .75,.5,'Open support gripper')
                move_support(high,3.,'Approach rear tower')
                move_support(support,2.,'Lower rear support gripper')
                grip('left',-.1,2.,'Pinch rear tower through contact')
            else:
                support=base_start+Rbase@np.array([-.05,.035,-.018])
                high=support.copy();high[2]=.08
                def move_support(point,seconds,label):move('left',point,seconds,label,2.5)
                grip('left',-.1,2.,'Close supporting fingers')
                move_support(high,3.,'Approach base foot')
                move_support(support,2.,'Press base foot against table')
                advance(d.ctrl.copy(),.5,'Support loose base through contact')
        yaw=np.deg2rad(args.grasp_yaw)
        direction=np.array([np.cos(yaw),np.sin(yaw),0.])
        offset=-.014*direction+args.grasp_pad_y*down_rotation(yaw)[:,1]
        if args.grasp_x is not None:offset[0]=args.grasp_x
        pose=motor_start+offset;pose[2]=args.travel_height
        grip('right',.75,.2,'Open pickup gripper')
        move('right',pose,3.,'Approach motor',yaw)
        pose[2]=motor_start[2]+args.grasp_offset
        move('right',pose,2.,'Lower open gripper',yaw)
        grip('right',-.1,2.,'Close through contact')
        pose[2]=args.travel_height
        move('right',pose,3.,'Lift motor',yaw)
        if d.body('motor').xpos[2]<motor_start[2]+.025:
            raise RuntimeError('Motor grasp did not produce a 25mm lift')
        if args.insert:
            # Relative held transform is measured after the physical grasp.
            held_delta=d.body('motor').xpos-d.site('right_gripperframe').xpos
            target=origins(metadata());base=d.body('base')
            goal=base.xpos+base.xmat.reshape(3,3)@(target['motor']-target['base'])
            front=base.xmat.reshape(3,3)@np.array([0.,-.070,0.])
            pose=goal+front-held_delta;pose[2]=args.travel_height
            move('right',pose,3.,'Carry motor in front of base',yaw)
            def aligned_rotation():
                if not args.align_motor:return None
                return d.body('base').xmat.reshape(3,3) @ d.body('motor').xmat.reshape(3,3).T @ d.site('right_gripperframe').xmat.reshape(3,3)
            pose=d.site('right_gripperframe').xpos+goal+front-d.body('motor').xpos
            move('right',pose,2.,'Align motor with front opening',yaw,aligned_rotation())
            # Slow straight insertion; contact can move either loose object.
            offsets=np.r_[np.linspace(.065,.020,10),np.linspace(.0195,0.,40)]
            for k,offset in enumerate(offsets):
                base=d.body('base')
                base_R=base.xmat.reshape(3,3)
                goal=base.xpos+base_R@(target['motor']-target['base'])
                desired=goal+base_R@np.array([0.,-offset,0.])
                pose=d.site('right_gripperframe').xpos+desired-d.body('motor').xpos
                move('right',pose,.18 if k<10 else .12,'Insert motor through front opening',yaw,aligned_rotation())
            if args.align_motor:
                for _ in range(6):
                    base=d.body('base');goal=base.xpos+base.xmat.reshape(3,3)@(target['motor']-target['base'])
                    correction=goal-d.body('motor').xpos
                    if np.linalg.norm(correction)<.0005:break
                    correction*=min(1.,.002/max(np.linalg.norm(correction),1e-9))
                    pose=d.site('right_gripperframe').xpos+correction
                    move('right',pose,.3,'Settle motor into reference pose',yaw,aligned_rotation())
            advance(d.ctrl.copy(),.5,'Hold inserted motor')
            grip('right',.75,2.,'Release inserted motor')
            pose=goal+front-held_delta
            move('right',pose,2.,'Withdraw from base opening',yaw)
            pose[2]=args.travel_height
            move('right',pose,2.,'Clear assembled base',yaw)
            if args.stabilize:
                if args.support_style in ('rear','sidefoot','horizontal'):grip('left',1.0,2.,'Release support grasp')
                move_support(high,2.,'Withdraw base support')
            advance(d.ctrl.copy(),1.,'Check retention after release')
            b=d.body('base');goal=b.xpos+b.xmat.reshape(3,3)@(target['motor']-target['base'])
            extent=np.ptp(np.asarray(metadata()['motor']['bounds']),axis=0)/2
            kp=np.vstack((np.zeros(3),np.diag(extent)))
            actual=d.body('motor').xpos+kp@d.body('motor').xmat.reshape(3,3).T
            expected=goal+kp@b.xmat.reshape(3,3).T
            if np.linalg.norm(actual-expected,axis=1).max()>.001:
                raise RuntimeError('Motor keypoints were not retained within the 1mm insertion tolerance')
        else:
            pose[:2]+=[-.035,.035]
            move('right',pose,2.,'Move held motor',yaw)
            pose[2]=motor_start[2]+args.grasp_offset+.004
            move('right',pose,2.,'Lower to the table',yaw)
            grip('right',.75,2.,'Release',)
            pose[2]=args.travel_height
            move('right',pose,2.,'Withdraw',yaw)
            advance(d.ctrl.copy(),.5,'Check released motor')
    except (Exception, KeyboardInterrupt) as exc:
        failure=f'{type(exc).__name__}: {exc}'
        print(failure,flush=True)
    finally:
        if renderer:
            renderer.update_scene(d,camera=cam,scene_option=opt)
            Image.fromarray(renderer.render()).save(args.out.with_suffix('.png'))
            renderer.close();writer.close()
        meta=metadata();target=origins(meta);base=d.body('base')
        goal=base.xpos+base.xmat.reshape(3,3)@(target['motor']-target['base'])
        report=log.save(args.out,dict(failure=failure,completed_script=failure is None,
            requested_insertion=args.insert,supporting_arm_used=args.stabilize,seed=args.seed,
            insertion_orientation_axes='world roll/yaw; pitch unconstrained' if args.align_motor and args.layout in ('front','opposed') else 'full requested orientation',
            motor_lift_m=None if motor_start is None else float(max_height-motor_start[2]),
            motor_start=None if motor_start is None else motor_start.tolist(),
            motor_final=d.body('motor').xpos.tolist(),motor_goal=goal.tolist(),
            motor_final_position_error_m=float(np.linalg.norm(d.body('motor').xpos-goal)),
            snapshots=snapshots,config=vars(args)|{'out':str(args.out)}))
        print({k:v for k,v in report.items() if k not in ('snapshots','stages','config')},flush=True)
    if failure:raise SystemExit(1)


if __name__=='__main__':main()
