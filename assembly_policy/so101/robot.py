"""Kinematic planning only; execution always uses position actuators in mj_step."""
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .scene import JOINTS

HOME = np.array([0, -.7, -.5, .3, 0, .65] * 2)


def arm_indices(m, side):
    joints = [m.joint(side+'_'+j).id for j in JOINTS]
    return m.jnt_qposadr[joints], m.jnt_dofadr[joints]


def home(m, d):
    d.qpos[:12] = HOME
    d.ctrl[:12] = HOME
    mujoco.mj_forward(m, d)


def ik(m, seed, side, position, orientation=None, orientation_axes=None):
    """Solve on scratch data; optional axes explicitly control only those rotation components."""
    d = mujoco.MjData(m)
    d.qpos[:] = seed
    qi, vi = arm_indices(m, side)
    qi = qi[:5]
    site = m.site(side+'_gripperframe').id
    ranges = m.jnt_range[[m.joint(side+'_'+j).id for j in JOINTS[:5]]]
    def residual(q):
        d.qpos[qi] = q
        mujoco.mj_kinematics(m, d)
        dp = d.site_xpos[site] - position
        if orientation is None:
            return dp
        R = d.site_xmat[site].reshape(3,3)
        dr = Rotation.from_matrix(orientation @ R.T).as_rotvec()
        if orientation_axes is not None:dr=dr[list(orientation_axes)]
        return np.r_[dp, .06*dr]
    result = least_squares(residual, np.clip(seed[qi], ranges[:,0]+1e-6, ranges[:,1]-1e-6),
                           bounds=(ranges[:,0], ranges[:,1]), max_nfev=150,
                           ftol=1e-10, xtol=1e-10, gtol=1e-10)
    error = np.linalg.norm(residual(result.x)[:3])
    angle = 0 if orientation is None else np.linalg.norm(residual(result.x)[3:])/.06
    if error > .003 or angle > .03:
        raise ValueError(f'{side} IK target {position}: error {error:.4f}m, orientation {angle:.3f}rad')
    return result.x


def down_rotation(yaw=0):
    # Site origin lies on the fixed inner pad, not at the jaw midpoint.
    # Site x points along the fingers; positive site z spans the jaw opening.
    # For a centered object, place this site half the object width opposite z.
    x = np.array([0.,0.,-1.])
    z = np.array([np.cos(yaw),np.sin(yaw),0.])
    y = np.cross(z,x)
    return np.column_stack((x,y,z))


def ik_fixed_wrist(m, seed, side, position, wrist_flex, wrist_roll):
    """Position a tilted support hand with two explicit wrist angles on scratch data."""
    d=mujoco.MjData(m);d.qpos[:]=seed;qi,_=arm_indices(m,side)
    d.qpos[qi[3:5]]=[wrist_flex,wrist_roll]
    ranges=m.jnt_range[[m.joint(side+'_'+j).id for j in JOINTS[:3]]]
    site=m.site(side+'_gripperframe').id
    def residual(q):
        d.qpos[qi[:3]]=q;mujoco.mj_kinematics(m,d)
        return d.site_xpos[site]-position
    result=least_squares(residual,np.clip(seed[qi[:3]],ranges[:,0]+1e-6,ranges[:,1]-1e-6),
                         bounds=(ranges[:,0],ranges[:,1]),max_nfev=150,
                         ftol=1e-10,xtol=1e-10,gtol=1e-10)
    error=np.linalg.norm(residual(result.x))
    if error>.003:raise ValueError(f'{side} fixed-wrist IK error {error:.4f}m')
    return np.r_[result.x,wrist_flex,wrist_roll]


def grasp_transform(data, side, body):
    site=data.site(side+'_gripperframe');obj=data.body(body)
    R=site.xmat.reshape(3,3)
    return R.T@(obj.xpos-site.xpos),R.T@obj.xmat.reshape(3,3)


def relative_ik(m, seed, base_grasp, motor_grasp, relative_position, base_anchor, base_yaw, base_rotation=None):
    """Use both five-axis arms to control all six object-relative pose coordinates."""
    d=mujoco.MjData(m);d.qpos[:]=seed
    li,_=arm_indices(m,'left');ri,_=arm_indices(m,'right');qi=np.r_[li[:5],ri[:5]]
    joints=[m.joint(s+'_'+j).id for s in ('left','right') for j in JOINTS[:5]]
    ranges=m.jnt_range[joints]
    anchor_R=Rotation.from_euler('z',base_yaw).as_matrix() if base_rotation is None else base_rotation
    def objects(q):
        d.qpos[qi]=q;mujoco.mj_kinematics(m,d)
        poses=[]
        for side,(p,R) in zip(('left','right'),(base_grasp,motor_grasp)):
            site=d.site(side+'_gripperframe');Rs=site.xmat.reshape(3,3)
            poses.append((site.xpos+Rs@p,Rs@R))
        return poses
    def residual(q):
        (pb,Rb),(pm,Rm)=objects(q)
        dp=Rb.T@(pm-pb)-relative_position
        dr=Rotation.from_matrix(Rb.T@Rm).as_rotvec()
        yaw=np.arctan2(Rb[1,0],Rb[0,0]);dyaw=np.arctan2(np.sin(yaw-base_yaw),np.cos(yaw-base_yaw))
        upright=Rotation.from_matrix(Rb@anchor_R.T).as_rotvec()
        return np.r_[dp,.05*dr,.05*(pb-base_anchor),.003*upright,.0001*(q-seed[qi])]
    sol=least_squares(residual,np.clip(seed[qi],ranges[:,0]+1e-6,ranges[:,1]-1e-6),
                      bounds=(ranges[:,0],ranges[:,1]),max_nfev=200,
                      ftol=1e-10,xtol=1e-10,gtol=1e-10)
    err=residual(sol.x)
    if np.linalg.norm(err[:3])>.001 or np.linalg.norm(err[3:6])/.05>.02:
        raise ValueError(f'Coupled IK relative position {np.linalg.norm(err[:3]):.5f}m, angle {np.linalg.norm(err[3:6])/.05:.4f}rad')
    return sol.x
