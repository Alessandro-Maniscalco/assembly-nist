"""Native actuator-driven tool pickup probe; no learned-policy claim or attachment."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from .scene import ROOT, model
from .robot import home, ik
from .trajectory import Trajectory


def run(out,lateral_clearance=.002,grasp_height=.010):
    m=model();d=mujoco.MjData(m);home(m,d);log=Trajectory(m,d)
    # Mirror the independently checked fixed-palm grasp about the tool axis.
    # This places the palm above the tool lying on the table, with fingers down.
    relative_p=np.array([.0101,-.015,-.09])
    relative_R=np.empty(9);mujoco.mju_quat2Mat(relative_R,np.array([.5,-.5,.5,.5]))
    relative_R=relative_R.reshape(3,3)
    hand=d.body('right_gripper');site=d.site('right_gripperframe')
    local_p=hand.xmat.reshape(3,3).T@(site.xpos-hand.xpos)
    local_R=hand.xmat.reshape(3,3).T@site.xmat.reshape(3,3)
    stages=[];start=None;failure=None
    def advance(ctrl,seconds,label,expect_off=True):
        old=d.ctrl.copy();n=round(seconds/m.opt.timestep);button_max=0.
        for k in range(n):
            u=(k+1)/n;d.ctrl[:]=old+(u*u*(3-2*u))*(ctrl-old)
            log.step(label)
            button_max=max(button_max,float(d.joint('button_forward').qpos[0]))
            if log.depths[-1]>.0004:raise RuntimeError(f'{label}: contact depth exceeds0.4mm')
            if expect_off and button_max>.0006:raise RuntimeError(f'{label}: unintended tool activation')
        row=dict(stage=label,time=float(d.time),tool_position=d.body('driver').xpos.tolist(),
                 button_max_m=button_max,button_m=float(d.joint('button_forward').qpos[0]),
                 spindle_rad_s=float(d.joint('spindle').qvel[0]))
        first=not stages or stages[-1]['stage']!=label
        stages.append(row)
        if first:print(row,flush=True)
    def move(point,R,seconds,label):
        ctrl=d.ctrl.copy();ctrl[6:11]=ik(m,d.qpos,'right',point,R)
        advance(ctrl,seconds,label)
    def straight_move(point,R,seconds,label):
        start=d.site('right_gripperframe').xpos.copy()
        count=max(1,int(np.ceil(np.linalg.norm(point-start)/.004)))
        for k in range(count):move(start+(k+1)/count*(point-start),R,seconds/count,label)
    try:
        advance(d.ctrl.copy(),1.,'Settle loose workcell')
        tool=d.body('driver');start=tool.xpos.copy()
        hand_R=tool.xmat.reshape(3,3)@relative_R.T
        hand_p=tool.xpos-hand_R@relative_p
        grip_p=hand_p+hand_R@local_p;grip_R=hand_R@local_R
        grip_p[1]+=lateral_clearance
        grip_p[2]+=grasp_height
        ctrl=d.ctrl.copy();ctrl[11]=.75;advance(ctrl,.3,'Open tool pickup hand')
        move(grip_p+[0,0,.045],grip_R,3.,'Approach tool from above')
        straight_move(grip_p,grip_R,2.,'Lower open hand around sleeve')
        ctrl=d.ctrl.copy();ctrl[11]=145*2*np.pi/4096
        advance(ctrl,2.,'Close sleeve at OFF aperture')
        straight_move(grip_p+[0,0,.05],grip_R,3.,'Lift free tool through contact')
        lift=float(d.body('driver').xpos[2]-start[2])
        if lift<.04:raise RuntimeError(f'Tool lift only{lift:.5f}m')
        advance(d.ctrl.copy(),.6,'Hold tool OFF above table')
        ctrl=d.ctrl.copy();ctrl[11]=130*2*np.pi/4096
        advance(ctrl,1.,'Squeeze physical forward button',False)
        if d.joint('button_forward').qpos[0]<.0006 or d.joint('spindle').qvel[0]<20:
            raise RuntimeError('Tool failed physical activation')
        ctrl=d.ctrl.copy();ctrl[11]=145*2*np.pi/4096
        advance(ctrl,1.,'Release physical forward button',False)
        advance(d.ctrl.copy(),.4,'Wait for spindle coast-down')
        if d.joint('button_forward').qpos[0]>.0006 or abs(d.joint('spindle').qvel[0])>.1:
            raise RuntimeError('Tool failed deactivation')
        return_p=d.site('right_gripperframe').xpos.copy()
        return_p[2]+=start[2]+.001-d.body('driver').xpos[2]
        straight_move(return_p,grip_R,3.,'Return held tool to table')
        ctrl=d.ctrl.copy();ctrl[11]=.75;advance(ctrl,1.,'Release tool on table')
        straight_move(return_p+[0,.004,.045],grip_R,2.,'Withdraw empty hand')
    except Exception as error:failure=f'{type(error).__name__}: {error}'
    result=log.save(out,dict(task='tool_pickup',passed=failure is None,failure=failure,
                            task_stages=stages,lateral_clearance_m=lateral_clearance,grasp_height_m=grasp_height,
                            scope='scripted pickup/activation probe; no screw engagement'))
    print(json.dumps({k:v for k,v in result.items() if k not in ('stages','task_stages')},indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'artifacts_local/tool_pickup/current')
    p.add_argument('--lateral-clearance',type=float,default=.002)
    p.add_argument('--grasp-height',type=float,default=.010)
    a=p.parse_args();run(a.out,a.lateral_clearance,a.grasp_height)
