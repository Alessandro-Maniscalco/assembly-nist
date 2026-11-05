"""One isolated tapered fixed-finger pocket; native jaw/button benchmark only.

A fixed palm is the same test fixture as check_so101_tool_gripper. The tool is
free; external torque is an explicitly labeled bench load, never motor control.
The new pocket is fitted to the passive sleeve, not attached to the robot.
"""
import argparse, hashlib, json, os, time
from pathlib import Path
import xml.etree.ElementTree as ET
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import mujoco
import numpy as np
import psutil
from scipy.spatial.transform import Rotation
from assembly_policy.so101.tool import add_driver, GRASP_Z
from scripts.check_so101_tool_gripper import make_model, ENCODER_STEP


def file_hash(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main(out, rocking=False):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'report.json').exists():raise ValueError('Choose a new evidence directory')
    started=time.monotonic();source=Path(__file__).read_text();(out/'executed_script.py').write_text(source)
    reference=make_model();rd=mujoco.MjData(reference);mujoco.mj_forward(reference,rd)
    # Unchanged exact fixed-jaw visual STL, transformed into nominal tool frame.
    gi=next(g for g in range(reference.ngeom) if reference.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH
            and reference.body(int(reference.geom_bodyid[g])).name=='gripper'
            and reference.mesh(reference.geom_dataid[g]).name=='wrist_roll_follower_so101_v1')
    mi=reference.geom_dataid[gi]
    vertices=reference.mesh_vert[reference.mesh_vertadr[mi]:reference.mesh_vertadr[mi]+reference.mesh_vertnum[mi]]
    points=(vertices@rd.geom_xmat[gi].reshape(3,3).T+rd.geom_xpos[gi]-rd.body('driver').xpos)@rd.body('driver').xmat.reshape(3,3)
    slices=[];clearance=.0002;wall=.0005
    edges=np.linspace(-.01465,.00635,6)
    for lo,hi in zip(edges[:-1],edges[1:]):
        p=points[(points[:,0]>=lo-.0005)&(points[:,0]<=hi+.0005)&(points[:,1]<-.0148)]
        assert len(p)
        slices.append(dict(x0=float(lo),x1=float(hi),yback=float(p[:,1].min()-clearance),
                           zlo=float(p[:,2].min()-clearance),zhi=float(p[:,2].max()+clearance)))
    del reference,rd
    def builder(world,extension,contact,**kwargs):
        tool=add_driver(world,extension,contact,**kwargs)
        sleeve=tool.find("body[@name='driver_sleeve']")
        pocket=ET.SubElement(sleeve,'body',name='driver_fixed_finger_pocket')
        def box(name,center,half):
            center=np.asarray(center)-[0,0,GRASP_Z]
            ET.SubElement(pocket,'geom',name=name,type='box',pos=' '.join(map(str,center)),
                          size=' '.join(map(str,half)),mass='.00015',rgba='.70 .25 .09 1')
        for i,s in enumerate(slices):
            xc=(s['x0']+s['x1'])/2;hx=(s['x1']-s['x0'])/2
            yc=(s['yback']-.0147)/2;hy=(-.0147-s['yback'])/2
            box(f'fixed_pocket_lower_{i}',[xc,yc,s['zlo']-wall/2],[hx,hy,wall/2])
            box(f'fixed_pocket_upper_{i}',[xc,yc,s['zhi']+wall/2],[hx,hy,wall/2])
            box(f'fixed_pocket_back_{i}',[xc,s['yback']-wall/2,(s['zlo']+s['zhi'])/2],
                [hx,wall/2,(s['zhi']-s['zlo'])/2+wall])
        s=slices[0]
        box('fixed_pocket_distal_stop',[s['x0']-wall/2,(s['yback']-.0147)/2,(s['zlo']+s['zhi'])/2],
            [wall/2,(-.0147-s['yback'])/2,(s['zhi']-s['zlo'])/2+wall])
        if rocking:
            # Thin front pad on a passive bearing; original linear saddle,
            # pusher, button, guide and return springs remain unchanged.
            paddle=tool.find("body[@name='driver_trigger_saddle']")
            pad=paddle.find("geom[@name='trigger_saddle_pad']");paddle.remove(pad)
            rocker=ET.SubElement(paddle,'body',name='driver_trigger_rocker',pos='0 .0015 0')
            ET.SubElement(rocker,'joint',name='trigger_rocker',type='hinge',axis='0 0 -1',
                          range='-.08 .08',stiffness='.001',damping='.00001',armature='.00000001')
            pad.set('size','.013 .0005 .013');pad.set('pos','0 0 0');rocker.append(pad)
            ET.SubElement(paddle,'geom',name='rocker_backing_plate',type='box',
                          pos='0 -.00075 0',size='.010 .00025 .010',mass='.0002',rgba='.14 .53 .72 1')
            # A revolute bearing abstracts the internal pin/hole contact, just
            # as the existing saddle slide abstracts its physical guide pins.
            ET.SubElement(paddle,'geom',name='rocker_bearing_pin',type='cylinder',
                          pos='0 .0015 0',size='.0005 .0145',mass='.0001',
                          contype='0',conaffinity='0',rgba='.65 .68 .71 1')
        return tool
    m=make_model(tool_builder=builder);d=mujoco.MjData(m)
    assert m.neq==0 and m.nu==1 and np.array_equal(m.actuator_forcerange[0],[-.35,.35])
    m.opt.gravity[:]=[0,-9.81,0]  # benchmark default, explicit for provenance
    qa=m.joint('driver_free').qposadr[0];grip=m.joint('gripper').qposadr[0]
    d.qpos[grip]=.65;mujoco.mj_forward(m,d)
    initial=d.qpos.copy();R=d.body('driver').xmat.reshape(3,3).copy()
    geometry=[]
    # Open mouth faces +tool X. Translating the tool toward -X withdraws the
    # finger through that mouth; this checks every 0.25 mm along a 40 mm path.
    for distance in np.linspace(0,.04,161):
        d.qpos[:]=initial;d.qpos[qa:qa+3]-=R[:,0]*distance;mujoco.mj_forward(m,d)
        contacts=[]
        for c in d.contact:
            bodies=[m.body(int(m.geom_bodyid[g])).name for g in c.geom]
            if any(b in ('gripper','moving_jaw_so101_v1') for b in bodies) and any(b.startswith('driver') for b in bodies):
                contacts.append(c)
        depth=max([0.,*[-float(c.dist) for c in contacts]])
        geometry.append([float(distance),depth,len(contacts)])
    report=dict(scope='isolated fixed-palm contact benchmark; not full arm or assembly',
        source_sha256=file_hash(out/'executed_script.py'),candidate=('tapered fixed-finger pocket plus passive rocking contact pad' if rocking else 'tapered fixed-finger-only sleeve pocket; flat moving saddle unchanged'),
        rocking_pad=rocking,rocker_assumptions=({'axis':'local -Z, parallel to stock jaw hinge','range_rad':[-.08,.08],
            'stiffness_nm_rad':.001,'damping_nm_s_rad':.00001,'armature_kg_m2':1e-8,'pad_thickness_m':.001,
            'pad_front_face_unchanged':True,'bearing':'passive hinge abstraction; pin/hole detail unmodeled',
            'geometric_backing_clearance_at_limit_m':.0015-.0005*np.cos(.08)-.013*np.sin(.08)+.0005,
            'hardware_validation':'All hinge dimensions, friction, stiffness, mass and travel are unmeasured assumptions.'} if rocking else None),
        clearance_m=clearance,wall_m=wall,slices=slices,added_mass_kg=.0027 if rocking else .0024,
        unchanged=['stock jaw CAD','button mechanism','motor plugin','actuator limits','solver settings'],
        geometry_path=geometry,geometry_path_max_penetration_m=max(row[1] for row in geometry),
        entry_exit_scope='sampled static straight path with jaws open; no automatic release or dynamic insertion claim',
        passed=False,stages=[])
    mujoco.mj_saveModel(m,filename=str(out/'candidate.mjb'))
    report['model_sha256']=file_hash(out/'candidate.mjb')
    report['plugin_sha256']={p.name:file_hash(p) for p in Path('plugin/build').glob('*.so')}
    if report['geometry_path_max_penetration_m']>.0003:
        report['failure']='Fixed-finger pocket collides on its proposed open-jaw entry/exit path; no dynamics promoted.'
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return
    # Native gradual closure in zero gravity follows the established bench.
    d=mujoco.MjData(m);d.qpos[grip]=.4;d.ctrl[0]=145*ENCODER_STEP
    m.opt.gravity[:]=0
    body=m.body('driver').id;warning=d.warning.number.copy();trace=[];controls=[];depths=[];stageids=[];labels=[]
    spec=mujoco.mjtState.mjSTATE_FULLPHYSICS
    initial_state=np.empty(mujoco.mj_stateSize(m,spec));mujoco.mj_getState(m,d,initial_state,spec)
    gravities=[];bench_loads=[]
    start_pose=None;failure=None;max_clock=0.
    def advance(label,tick,seconds,torque=0,expect_off=False):
        nonlocal start_pose,max_clock
        labels.append(label);d.ctrl[0]=tick*ENCODER_STEP
        # World Y is the nominal screwdriver axis in this existing bench.
        d.xfrc_applied[body,4]=torque
        depth=0.;button=0.;angle=0.;drift=0.
        for k in range(round(seconds/m.opt.timestep)):
            before=d.time;mujoco.mj_step(m,d);max_clock=max(max_clock,abs(d.time-before-m.opt.timestep))
            md=max([0.,*[-float(c.dist) for c in d.contact]]);depth=max(depth,md)
            state=np.empty(mujoco.mj_stateSize(m,spec));mujoco.mj_getState(m,d,state,spec)
            trace.append(state);controls.append(d.ctrl.copy());depths.append(md);stageids.append(len(labels)-1)
            gravities.append(m.opt.gravity.copy());bench_loads.append(torque)
            button=max(button,float(d.joint('button_forward').qpos[0]))
            if start_pose is not None:
                drift=max(drift,float(np.linalg.norm(d.body('driver').xpos-start_pose[0])))
                angle=max(angle,float(Rotation.from_matrix(start_pose[1].T@d.body('driver').xmat.reshape(3,3)).magnitude()))
            if not np.isfinite(state).all() or max_clock>1e-8:raise RuntimeError('Nonfinite state or clock reset')
            if np.any(d.warning.number-warning):raise RuntimeError('Native warning')
            if depth>.0003:raise RuntimeError(f'{label}: depth exceeds existing 0.3 mm bench gate')
            if expect_off and button>.0006:raise RuntimeError(f'{label}: unintended physical activation')
            if start_pose is not None and (drift>.003 or angle>.01):
                raise RuntimeError(f'{label}: retention drift exceeds 3 mm / 0.01 rad design gate')
            if k%100==0:
                if psutil.Process().memory_info().rss>2.5*2**30 or psutil.virtual_memory().available<2*2**30:raise RuntimeError('Memory guard')
                if time.monotonic()-started>750:raise RuntimeError('12.5 minute bound')
        row=dict(stage=label,target_ticks=tick,external_bench_torque_nm=torque,max_depth_m=depth,max_button_m=button,
                 final_button_m=float(d.joint('button_forward').qpos[0]),spindle_rad_s=float(d.joint('spindle').qvel[0]),
                 max_drift_m=drift,max_angle_rad=angle,rocker_rad=float(d.joint('trigger_rocker').qpos[0]) if rocking else None)
        report['stages'].append(row);print(row,flush=True)
    try:
        advance('Closure OFF',145,.6,expect_off=True)
        m.opt.gravity[:]=[0,-9.81,0]
        advance('Gravity settle OFF',145,1.,expect_off=True)
        start_pose=(d.body('driver').xpos.copy(),d.body('driver').xmat.reshape(3,3).copy())
        advance('Physical button ON',130,1.)
        if d.joint('button_forward').qpos[0]<.0006 or d.joint('spindle').qvel[0]<20:raise RuntimeError('Physical ON failed')
        advance('Physical button OFF',145,1.)
        if d.joint('button_forward').qpos[0]>.0006 or abs(d.joint('spindle').qvel[0])>.1:raise RuntimeError('Physical OFF failed')
        snapshot=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(m,d,snapshot,mujoco.mjtState.mjSTATE_INTEGRATION)
        # Independent signed load branches from the same physical OFF state.
        for sign in (1,-1):
            d=mujoco.MjData(m);mujoco.mj_setState(m,d,snapshot,mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
            advance(f'OFF reaction {sign:+d}',145,1.,sign*.05,True)
            advance(f'OFF unload {sign:+d}',145,.5,0,True)
            advance(f'ON reaction {sign:+d}',130,1.,sign*.05)
            if d.joint('button_forward').qpos[0]<.0006:raise RuntimeError('Physical ON under reaction failed')
            advance(f'OFF after reaction {sign:+d}',145,.6,0)
        report['passed']=True
    except Exception as error:failure=f'{type(error).__name__}: {error}'
    report.update(failure=failure,physics_steps=len(trace),max_clock_step_error_s=max_clock,
                  warnings=(d.warning.number-warning).tolist(),elapsed_s=time.monotonic()-started,
                  max_penetration_m=max(depths,default=0),peak_rss_gib=psutil.Process().memory_info().rss/2**30)
    np.savez_compressed(out/'trace.npz',initial_state=initial_state,states=trace,controls=controls,penetration=depths,
                        stages=stageids,labels=labels,gravity=gravities,external_bench_torque_nm=bench_loads)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='geometry_path'},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=Path('artifacts_local/prepared_screw/fixed_finger_pocket/candidate'))
    p.add_argument('--rocking',action='store_true')
    a=p.parse_args();main(a.out,a.rocking)
