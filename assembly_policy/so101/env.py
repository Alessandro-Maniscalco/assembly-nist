"""Task adapter for AssembleNIST's unchanged PPO/SimBa/SGS implementation.

All six target parts and the contact-operated tool remain dynamic. Screw mating
uses the documented nominal helical supplement in fastener.py.
"""
from dataclasses import dataclass
import mujoco
from mujoco import rollout
import numpy as np

from .scene import model, metadata, PARTS, OBS_BODIES
from .robot import home
from .fastener import origins

HISTORY = 5
PROPRIO_DIM = 53  # two TCP poses/twists (26), joints (12), actions (12), tool state (3)
ACT_DIM = 12
ARM_ACTIONS = np.array([0,1,2,3,4,6,7,8,9,10])
PREVIOUS_ACTION_START = 38
STATE_SPEC = mujoco.mjtState.mjSTATE_FULLPHYSICS
TASK_NAMES = ('motor', 'horn', 'screw', 'retainer', 'yoke')


@dataclass
class EnvConfig:
    suite: str = 'so101'
    control_dt: float = .04
    horizon_s: float = 15.
    action_scale_joints: tuple = (.035, .035, .035, .035, .035)
    gripper_scale: float = .05
    cost_coef: float = 2e-5
    nthread: int = 4
    success_threshold: float = .001
    proprio_dim: int = PROPRIO_DIM
    max_contact_depth: float | None = None  # explicit opt-in; old checkpoints retain their semantics
    stability_metric: str = 'velocity'
    stable_translation: float = .00025
    stable_rotation: float = .01
    safety_penalty: float = 0.  # explicit task-objective experiment; legacy rewards remain default


def parse_scales(value):
    if value == '0.05+0.05+0.05+0.06+0.07+0.1+0.17':
        return (.035,)*5
    vals = tuple(float(x) for x in value.replace(',', '+').split('+'))
    if len(vals) != 5:
        raise ValueError('SO101 needs five joint scales per arm')
    return vals


def resolve_tasks(value):
    if value in ('all20', 'so101', 'all'):
        return list(TASK_NAMES)
    names = value.split('+')
    if not names or any(n not in TASK_NAMES for n in names):
        raise ValueError(f'Expected SO101 tasks from {TASK_NAMES}')
    return names


class BatchedAssemblyEnv:
    def __init__(self, task_names, task_of_env, ecfg=None, *, model_factory=None):
        self.ecfg = ecfg or EnvConfig()
        if self.ecfg.stability_metric not in ('velocity','pose'):raise ValueError('Unknown stability metric')
        if self.ecfg.max_contact_depth is not None and self.ecfg.max_contact_depth<=0:
            raise ValueError('Contact depth limit must be positive')
        if not np.isfinite(self.ecfg.safety_penalty) or self.ecfg.safety_penalty<0:
            raise ValueError('Safety penalty must be finite and nonnegative')
        self.task_names = list(task_names)
        self.task = np.array(task_of_env, dtype=np.int64)
        self.N = len(self.task)
        self.m = (model_factory or model)(contact_monitor=self.ecfg.max_contact_depth is not None)
        self.data = [mujoco.MjData(self.m) for _ in range(self.N)]
        self.work = [mujoco.MjData(self.m) for _ in range(max(1,self.ecfg.nthread))]
        self.engine = rollout.Rollout(nthread=self.ecfg.nthread if self.ecfg.nthread > 1 else 0)
        self.nstate = mujoco.mj_stateSize(self.m, STATE_SPEC)
        self.state = np.zeros((self.N,self.nstate))
        self.prev_action = np.zeros((self.N,ACT_DIM))
        self.last_ctrl = np.zeros((self.N,ACT_DIM))
        self.hist = np.zeros((self.N,HISTORY,PROPRIO_DIM),np.float32)
        self.t = np.zeros(self.N,np.int64)
        self.horizon = round(self.ecfg.horizon_s/self.ecfg.control_dt)
        self.substeps = round(self.ecfg.control_dt/self.m.opt.timestep)
        self.names = (*PARTS,'driver')
        self.part_q = [self.m.jnt_qposadr[self.m.joint(n+'_free').id] for n in self.names]
        self.part_v = [self.m.jnt_dofadr[self.m.joint(n+'_free').id] for n in self.names]
        self.meta = metadata()
        self.origins = origins(self.meta)
        self.stable = np.zeros(self.N,np.int64)
        self.stable_position = np.zeros((self.N,3))
        self.stable_quaternion = np.zeros((self.N,4))
        self.frames = None

    def close(self):
        self.engine.close()

    def set_states(self, ids, states, task_ids=None):
        ids=np.asarray(ids)
        self.state[ids]=states;self.state[ids,0]=0
        if task_ids is not None:self.task[ids]=task_ids
        self.prev_action[ids]=0;self.t[ids]=0;self.stable[ids]=0
        self.last_ctrl[ids]=self.state[ids,1:13]
        self._refresh()
        self.hist[ids]=self.frames[ids,None]

    def _forward_world(self, i):
        d=self.data[i]
        mujoco.mj_setState(self.m,d,self.state[i],STATE_SPEC)
        d.ctrl[:]=self.last_ctrl[i]
        mujoco.mj_forward(self.m,d)

    def _forward_states(self):
        for i in range(self.N):self._forward_world(i)

    def _refresh(self):
        self._forward_states()
        prop=[];points_p=[];points_R=[];pp=[];pq=[]
        for i,d in enumerate(self.data):
            row=[];bp=[];br=[]
            for side in ('left','right'):
                site=self.m.site(side+'_gripperframe').id
                q=np.empty(4);mujoco.mju_mat2Quat(q,d.site_xmat[site])
                if q[0]<0:q=-q
                v=np.zeros(6);mujoco.mj_objectVelocity(self.m,d,mujoco.mjtObj.mjOBJ_SITE,site,v,0)
                row.extend(np.r_[d.site_xpos[site],q,v[3:],v[:3]])
                for b in (side+'_gripper',side+'_moving_jaw_so101_v1'):
                    bp.append(d.body(b).xpos.copy());br.append(d.body(b).xmat.reshape(3,3).copy())
            row.extend(d.qpos[:12]);row.extend(self.prev_action[i])
            row.extend([d.joint('button_forward').qpos[0],d.joint('button_reverse').qpos[0],d.joint('spindle').qvel[0]])
            prop.append(row);points_p.append(bp);points_R.append(br)
            pp.append([d.body(n).xpos.copy() for n in OBS_BODIES])
            pq.append([d.body(n).xquat.copy() for n in OBS_BODIES])
        self.frames=np.asarray(prop,np.float32)
        self.compact=dict(tcp_p=np.asarray(points_p,np.float32),tcp_R=np.asarray(points_R,np.float32),
                          part_p=np.asarray(pp,np.float32),part_q=np.asarray(pq,np.float32),
                          finger=self.state[:,[6,12]].astype(np.float32))

    def observe(self):
        return dict(proprio=self.hist.reshape(self.N,-1).copy(),task=self.task.copy(),
                    **{k:v.copy() for k,v in self.compact.items()})

    def assembly_distance(self):
        distances=[]
        for i,d in enumerate(self.data):
            active=self.task_names[self.task[i]]
            base=d.body('base');R=base.xmat.reshape(3,3)
            errors=[]
            # Require all predecessor placements; no reward for placing a horn in empty space.
            for name in TASK_NAMES[:TASK_NAMES.index(active)+1]:
                b=d.body(name)
                offset=self.origins[name]-self.origins['base']
                extent=np.ptp(np.array(self.meta[name]['bounds']),axis=0)/2
                kp=np.array([[0,0,0],[extent[0],0,0],[0,extent[1],0],[0,0,extent[2]]])
                if name=='screw':kp=np.array([[0,0,0],[0,0,.003]])
                actual=b.xpos+kp@b.xmat.reshape(3,3).T
                goal=base.xpos+(kp+offset)@R.T
                errors.append(np.linalg.norm(actual-goal,axis=1).max())
            distances.append(max(errors))
        return np.asarray(distances)

    def step(self, action):
        a=np.asarray(action,float)
        if a.shape!=(self.N,12) or not np.isfinite(a).all():raise ValueError('Expected finite (N,12) actions')
        ctrl=np.empty_like(a)
        scale=np.tile(self.ecfg.action_scale_joints,2)
        ctrl[:,ARM_ACTIONS]=np.clip(self.state[:,1+ARM_ACTIONS]+scale*a[:,ARM_ACTIONS],
                                      self.m.actuator_ctrlrange[ARM_ACTIONS,0],self.m.actuator_ctrlrange[ARM_ACTIONS,1])
        # Real SO101 aperture control supports retaining a driver while changing
        # squeeze/trigger travel. There is no independent tool activation action.
        ctrl[:,[5,11]]=np.clip(self.state[:,[6,12]]+self.ecfg.gripper_scale*a[:,[5,11]],
                              self.m.actuator_ctrlrange[[5,11],0],self.m.actuator_ctrlrange[[5,11],1])
        trajectory,sensors=self.engine.rollout(self.m,self.work,self.state,
            np.repeat(ctrl[:,None,:],self.substeps,axis=1),nstep=self.substeps)
        if not np.isfinite(trajectory).all():
            raise FloatingPointError('Nonfinite SO101 physics trajectory; no transition may enter PPO')
        self.last_ctrl=ctrl.copy()
        expected=self.state[:,0]+self.ecfg.control_dt
        self.state=trajectory[:,-1].copy()
        unstable=~np.isfinite(self.state).all(1)|(abs(self.state[:,0]-expected)>1e-6)
        if unstable.any():
            raise FloatingPointError('MuJoCo reset its simulation clock; stop and inspect physics')
        vel=trajectory[:,:,1+self.m.nq:1+self.m.nq+12]
        safety=(abs(vel)>5.).any((1,2))|unstable
        if self.ecfg.max_contact_depth is not None:
            if not np.isfinite(sensors).all():raise FloatingPointError('Nonfinite SO101 contact monitor')
            # The position-stage sensor reports pre-integration contacts. The
            # endpoint check below covers the final integrated pose as well.
            safety |= (-sensors[:,:,0].min(1)>self.ecfg.max_contact_depth)
        # Reject a thrown part before a policy can exploit coarse-step tunneling.
        for start in self.part_v:
            pv=trajectory[:,:,1+self.m.nq+start:1+self.m.nq+start+6]
            safety |= (np.linalg.norm(pv[:,:,:3],axis=2)>1.).any(1)
            safety |= (np.linalg.norm(pv[:,:,3:],axis=2)>30.).any(1)
        self.prev_action=a.copy();self.t+=1;self._refresh()
        if self.ecfg.max_contact_depth is not None:
            safety |= np.array([min(d.contact.dist,default=0)<-self.ecfg.max_contact_depth for d in self.data])
        distance=self.assembly_distance()
        settled=np.zeros(self.N,bool);positions=[];quaternions=[]
        for i,d in enumerate(self.data):
            name=self.task_names[self.task[i]]
            active_id=self.m.body(name).id
            positions.append(d.body(name).xpos.copy());quaternions.append(d.body(name).xquat.copy())
            held=any(active_id in self.m.geom_bodyid[c.geom] and any(
                (self.m.body(int(b)).name.startswith(('left_','right_','driver_')) or
                 self.m.body(int(b)).name=='driver')
                for b in self.m.geom_bodyid[c.geom]) for c in d.contact)
            v=self.part_v[self.names.index(name)]
            deep=any(c.dist<-.0004 and active_id in self.m.geom_bodyid[c.geom] for c in d.contact)
            slow=(np.linalg.norm(d.qvel[v:v+3])<.005 and np.linalg.norm(d.qvel[v+3:v+6])<.1)
            settled[i]=not held and not deep and (slow or self.ecfg.stability_metric=='pose')
            if name=='screw':
                pairs={frozenset((self.m.geom(int(c.geom1)).name,self.m.geom(int(c.geom2)).name)) for c in d.contact}
                settled[i] &= (frozenset(('male_contact','female_contact')) in pairs and
                               frozenset(('screw_head_contact','horn_contact')) in pairs)
        candidate=(distance<self.ecfg.success_threshold)&settled&~safety
        if self.ecfg.stability_metric=='pose':
            positions=np.asarray(positions);quaternions=np.asarray(quaternions)
            close=(np.linalg.norm(positions-self.stable_position,axis=1)<=self.ecfg.stable_translation)
            close &= abs((quaternions*self.stable_quaternion).sum(1))>=np.cos(self.ecfg.stable_rotation/2)
            restart=(self.stable==0)|~close
            self.stable_position[candidate&restart]=positions[candidate&restart]
            self.stable_quaternion[candidate&restart]=quaternions[candidate&restart]
            self.stable=np.where(candidate,np.where(restart,1,self.stable+1),0)
        else:self.stable=np.where(candidate,self.stable+1,0)
        # Eleven pose samples span ten intervals, i.e. an actual0.4s window.
        success=self.stable>=round(.4/self.ecfg.control_dt)+(self.ecfg.stability_metric=='pose')
        timeout=(self.t>=self.horizon)&~success&~safety
        torque=np.array([d.actuator_force.copy() for d in self.data])
        limit=self.m.actuator_forcerange[:,1]
        cost=self.ecfg.cost_coef*((a*a).sum(1)+(vel[:,-1]**2/25).sum(1)+((torque/limit)**2).sum(1))
        self.hist=np.concatenate((self.hist[:,1:],self.frames[:,None]),axis=1)
        info=dict(success=success,safety=safety,timeout=timeout,d_asm=distance,cost=cost,unstable=unstable)
        return (success-cost-self.ecfg.safety_penalty*safety).astype(np.float32),success|safety|timeout,info
