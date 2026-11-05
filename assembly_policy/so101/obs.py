"""Source-sized point observation: active part, scene context, both grippers."""
import mujoco
import numpy as np
import torch
from ..rl.obs import sample_surface, _box_surface, quat_to_mat_torch
from ..sim.robot import quat_to_mat
from .scene import model, OBS_BODIES


class PointTables:
    def __init__(self, task_names, n_part=128, n_fix=128, n_grip=64, seed=0, device='cpu'):
        if n_grip%4:raise ValueError('n_grip must be divisible by four')
        m=model();rng=np.random.default_rng(seed)
        d=mujoco.MjData(m);mujoco.mj_kinematics(m,d)
        def body_samples(name,n):
            bid=m.body(name).id;chunks=[];areas=[]
            for gid in range(m.ngeom):
                if m.body_weldid[m.geom_bodyid[gid]]!=m.body_weldid[bid]:continue
                if m.geom_group[gid]==3 or (not name.startswith('driver') and m.geom_group[gid]!=2):continue
                mid=m.geom_dataid[gid]
                if mid>=0 and int(m.geom_type[gid]) in (int(mujoco.mjtGeom.mjGEOM_MESH),int(mujoco.mjtGeom.mjGEOM_SDF)):
                    va,vn=m.mesh_vertadr[mid],m.mesh_vertnum[mid];fa,fn=m.mesh_faceadr[mid],m.mesh_facenum[mid]
                    p=sample_surface(m.mesh_vert[va:va+vn],m.mesh_face[fa:fa+fn],max(4*n,32),rng)
                    tri=m.mesh_vert[va:va+vn][m.mesh_face[fa:fa+fn]]
                    area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1).sum()/2
                else:
                    import trimesh
                    kind=int(m.geom_type[gid]);size=m.geom_size[gid]
                    if kind==int(mujoco.mjtGeom.mjGEOM_BOX):mesh=trimesh.creation.box(extents=2*size)
                    elif kind==int(mujoco.mjtGeom.mjGEOM_CYLINDER):mesh=trimesh.creation.cylinder(radius=size[0],height=2*size[1],sections=32)
                    elif kind==int(mujoco.mjtGeom.mjGEOM_CAPSULE):mesh=trimesh.creation.capsule(radius=size[0],height=2*size[1],count=[16,16])
                    else:raise ValueError(f'Unhandled point geometry {kind} on {name}')
                    p=sample_surface(mesh.vertices,mesh.faces,max(4*n,32),rng);area=mesh.area
                world=p@d.geom_xmat[gid].reshape(3,3).T+d.geom_xpos[gid]
                chunks.append((world-d.body(name).xpos)@d.body(name).xmat.reshape(3,3));areas.append(area)
            if not chunks:
                raise ValueError(f'No surface for {name}')
            p=np.concatenate(chunks)
            weights=np.concatenate([np.full(len(c),area/len(c)) for c,area in zip(chunks,areas)])
            return p[rng.choice(len(p),n,replace=len(p)<n,p=weights/weights.sum())]
        names=OBS_BODIES;points=[];which=[]
        for task in task_names:
            active=names.index(task)
            other=[j for j in range(len(names)) if j!=active]
            ids=np.array([active]*n_part+other*(n_fix//len(other))+other[:n_fix%len(other)])
            counts=np.bincount(ids,minlength=len(names))
            pools={i:body_samples(name,int(counts[i])) for i,name in enumerate(names) if counts[i]}
            used=np.zeros(len(names),int);p=[]
            for i in ids:p.append(pools[i][used[i]]);used[i]+=1
            points.append(p);which.append(ids)
        self.points=torch.tensor(np.array(points),dtype=torch.float32,device=device)
        self.which=torch.tensor(np.array(which),dtype=torch.long,device=device)
        grips=[side+'_'+part for side in ('left','right') for part in ('gripper','moving_jaw_so101_v1')]
        self.grip=torch.tensor(np.array([body_samples(b,n_grip//4) for b in grips]),dtype=torch.float32,device=device)
        self.n_pts=n_part+n_fix+n_grip

    def __call__(self,task,tcp_p,tcp_R,part_p,part_q,finger):
        B=len(task);r=quat_to_mat_torch(part_q)
        ids=self.which[task];batch=torch.arange(B,device=task.device)[:,None]
        p=torch.einsum('bnij,bnj->bni',r[batch,ids],self.points[task])+part_p[batch,ids]
        grip=torch.einsum('bgij,gpj->bgpi',tcp_R,self.grip)+tcp_p[:,:,None]
        world=torch.cat((p,grip.reshape(B,-1,3)),1)
        local=torch.einsum('bij,bnj->bni',tcp_R[:,0].transpose(1,2),world-tcp_p[:,0,None])
        return (10*local).reshape(B,-1)
