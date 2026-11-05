"""Qualification probes for the isolated GPU physics prototype; never trains."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import warp as wp
from mujoco_warp._src.types import vec_pluginattr
from assembly_policy.so101.scene import ROOT, ASSETS, model, build_xml, metadata, numbers
from assembly_policy.so101.fastener import AXIS, PITCH
from assembly_policy.so101.robot import home
from assembly_policy.so101.warp_physics import WarpPhysics, query_sdf


def native_queries(out):
    """Compile verbatim native math into a local probe, without registering plugins."""
    source = (ROOT/'plugin/assemblenist_sdf.cc').read_text().split('template <int N>')[0]
    head = (ROOT/'plugin/so101_driver.cc').read_text().split('double box(')[1].split('\n}\n\nmjPLUGIN_LIB_INIT')[0]
    source += '\n double box('+head+'\n}\n'
    source += '''
extern "C" void evaluate(int kind, const double* p, const double* a, double* out) {
  if(kind==0) out[0]=grid_eval(p,get_grid(int(a[0])),out+1);
  if(kind==1) out[0]=bolt_eval(p,a,out+1);
  if(kind==2) out[0]=nut_eval(p,a,out+1);
  if(kind==3) {out[0]=head(p);for(int k=0;k<3;k++){double q[3]={p[0],p[1],p[2]};q[k]+=1e-7;out[k+1]=(head(q)-out[0])/1e-7;}}
}
'''
    cpp, library = out/'native_reference.cpp', out/'native_reference.so'
    cpp.write_text(source)
    mjdir = Path(mujoco.__file__).parent
    subprocess.run(['g++', '-O3', '-std=c++17', '-mf16c', '-shared', '-fPIC',
                    '-I'+str(mjdir/'include'), str(cpp), '-o', str(library),
                    str(next(mjdir.glob('libmujoco.so*'))), '-Wl,-rpath,'+str(mjdir)], check=True)
    lib = ctypes.CDLL(str(library))
    evaluate = lib.evaluate
    arr = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
    evaluate.argtypes = [ctypes.c_int, arr, arr, arr]
    evaluate.restype = None
    return lib, evaluate, hashlib.sha256(source.encode()).hexdigest()


def sdf_check(out, count=2000):
    m = model(); p = WarpPhysics(m, nworld=1)
    library, evaluate, sha = native_queries(out)
    attrs = p.model.plugin_attr.numpy()
    kinds = p.model.plugin.numpy()
    rng = np.random.default_rng(20261105)
    points, attributes, selectors, labels, reference = [], [], [], [], []
    geom_ids = np.flatnonzero(m.geom_plugin >= 0)
    for row, gid in enumerate(geom_ids):
        a = attrs[row].copy(); kind = int(kinds[row])
        instance = int(m.geom_plugin[gid])
        start = m.plugin_attradr[instance]
        end = m.plugin_attradr[instance+1] if instance+1<m.nplugin else len(m.plugin_attr)
        config = [float(s) for s in bytes(m.plugin_attr[start:end]).split(b'\0') if s]
        native_attr = np.zeros(128, np.float64); native_attr[:len(config)] = config
        if kind == 0:
            n, oh, _ = p.grids[int(a[0])]
            lo, hi = oh[:3]-.002, oh[:3]+(n-1)*oh[3]+.002
        elif kind in (1, 2):
            extent = float(a[3] if kind == 1 else a[5])*1.5
            lo, hi = [-extent, -extent, -.004], [extent, extent, .012]
        else:
            lo, hi = [-.004, -.004, -.001], [.004, .004, .002]
        q = rng.uniform(lo, hi, (count, 3)).astype(np.float32)
        # Explicit boundaries also exercise the upper-grid index guard.
        if kind == 0:
            q[:2] = np.array([oh[:3], oh[:3]+(n-1)*oh[3]], np.float32)
        for point in q:
            answer = np.empty(4, np.float64)
            evaluate(kind, point.astype(np.float64), native_attr, answer)
            reference.append(answer)
        points.extend(q); attributes.extend([a]*count); selectors.extend([kind]*count)
        labels.extend([m.geom(int(gid)).name]*count)
    points = np.asarray(points, np.float32)
    attr_array = wp.array(np.asarray(attributes), dtype=vec_pluginattr, device=p.device)
    output = wp.zeros(len(points), dtype=wp.vec4, device=p.device)
    wp.launch(query_sdf, dim=len(points), inputs=[wp.array(points, dtype=wp.vec3, device=p.device),
              attr_array, wp.array(selectors, dtype=int, device=p.device), output], device=p.device)
    actual, expected = output.numpy(), np.asarray(reference)
    error = abs(actual-expected)
    report = {'native_math_sha256': sha, 'queries': len(points), 'parts': {},
              'nq':m.nq,'nv':m.nv,'ngeom':m.ngeom,'plugin_audit':p.plugin_audit}
    for label in sorted(set(labels)):
        mask = np.asarray(labels) == label
        dist = error[mask, 0]; grad = np.linalg.norm(error[mask, 1:], axis=1)
        report['parts'][label] = dict(distance_max_m=float(dist.max()), distance_p99_m=float(np.quantile(dist,.99)),
                                     gradient_max=float(grad.max()), gradient_p99=float(np.quantile(grad,.99)),
                                     finite=bool(np.isfinite(actual[mask]).all()))
    report['pass'] = bool(np.isfinite(actual).all() and error[:,0].max()<2e-6 and
                          all(v['gradient_p99'] < .05 for v in report['parts'].values()))
    np.savez_compressed(out/'sdf_queries.npz', points=points, actual=actual, expected=expected,
                        labels=labels, selectors=selectors)
    p.close()
    return report


def thread_model(push_only):
    xml = ET.fromstring(build_xml(True)); w = xml.find('worldbody')
    xml.remove(xml.find('contact'))
    for e in list(w):
        if e.get('name') not in ('motor','screw'): w.remove(e)
    motor = w.find("body[@name='motor']"); motor.remove(motor.find('freejoint')); motor.set('pos','0 0 0')
    screw = w.find("body[@name='screw']"); screw.remove(screw.find('freejoint'))
    screw.set('pos', numbers(AXIS-np.asarray(metadata()['motor']['origin'])+[0,0,.0045]))
    ET.SubElement(screw,'joint',name='depth',type='slide',axis='0 0 1',damping='.3')
    ET.SubElement(screw,'joint',name='turn',type='hinge',axis='0 0 1',armature='.000002')
    xml.find('actuator').clear()
    for e in list(xml.find('extension')):
        if e.get('plugin')=='so101.contact_driver': xml.find('extension').remove(e)
    ET.SubElement(xml.find('actuator'),'velocity',joint='turn',kv='.0003',forcerange='-.002 .002')
    m = mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode')); m.opt.timestep=.0001
    d = mujoco.MjData(m)
    d.ctrl[0] = 0 if push_only else -2*np.pi
    d.qfrc_applied[0] = -1 if push_only else -.05
    return m,d


def basic_check(out):
    """Contact-free gravity and actuator probes; not full-scene qualification."""
    xml='''<mujoco><option timestep=".0005" integrator="implicitfast" cone="elliptic" iterations="50" ls_iterations="20"/>
    <worldbody><body pos=".3 0 1"><freejoint/><geom type="box" size=".02 .03 .04" mass=".1"/></body>
    <body pos="0 0 .5"><joint name="hinge" axis="0 1 0" damping=".1" armature=".002"/>
    <geom type="capsule" fromto="0 0 0 .15 0 0" size=".01" mass=".1"/></body></worldbody>
    <actuator><position joint="hinge" kp="20" kv=".2" ctrllimited="true" ctrlrange="-1 1" forcelimited="true" forcerange="-1 1"/></actuator></mujoco>'''
    result={}
    for name,target in [('zero',0.0),('actuator',.2)]:
        m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);d.ctrl[0]=target;mujoco.mj_forward(m,d)
        p=WarpPhysics(m,d,nworld=1,nconmax=16,njmax=64)
        p.step();mujoco.mj_step(m,d);wp.synchronize_device(p.device)
        p.step(99,graph=True)
        for _ in range(99):mujoco.mj_step(m,d)
        a=p.arrays();qerr=float(np.max(abs(a['qpos'][0]-d.qpos)));verr=float(np.max(abs(a['qvel'][0]-d.qvel)))
        result[name]=dict(qpos_max_error=qerr,qvel_max_error=verr,
                          freefall_z_error_m=float(abs(a['qpos'][0,2]-d.qpos[2])),
                          overflow_bits=p.data.overflow.numpy().tolist(),
                          pass_check=qerr<1e-5 and verr<1e-4)
        p.close()
    return result


def physics_check(out, case, steps, worlds, isolate_tool=False, benchmark_controls=8):
    if case in ('thread', 'axial'):
        # Ensure native plugin registrations exist before compiling the bench.
        from assembly_policy.so101.scene import load_plugin
        load_plugin()
        m,d = thread_model(case=='axial')
    else:
        if isolate_tool:
            from assembly_policy.so101.scene import load_plugin
            load_plugin();xml=ET.fromstring(build_xml());world=xml.find('worldbody')
            for body in list(world.findall('body')):
                if body.get('name')!='driver':world.remove(body)
            xml.find('actuator').clear()
            kept={b.get('name') for b in world.iter('body')}
            for pair in list(xml.find('contact')):
                if pair.get('body1') not in kept or pair.get('body2') not in kept:
                    xml.find('contact').remove(pair)
            m=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'))
            d=mujoco.MjData(m);mujoco.mj_forward(m,d)
        else:
            m = model(); d = mujoco.MjData(m); home(m,d)
        if case=='actuator': d.ctrl[:5]+=.02
        elif case in ('button','button_release'):
            d.qfrc_applied[m.joint('button_forward').dofadr[0]]=2
        elif case=='freefall':
            for name in ('base','motor','horn','retainer','yoke'):
                d.qpos[m.joint(name+'_free').qposadr[0]+2]+=.1
    initial_qpos = d.qpos.copy()
    p = WarpPhysics(m,d,nworld=worlds)
    t = time.perf_counter(); p.step(); wp.synchronize_device(p.device)
    compile_s = time.perf_counter()-t
    print(f'First {case} step synchronized after {compile_s:.3f}s',flush=True)
    mujoco.mj_step(m,d)
    native_trace, gpu_trace = [], []
    chunk = min(80, steps-1)
    t = time.perf_counter()
    # Record both paths at every chunk; graph capture itself does not advance.
    for start in range(1, steps, chunk):
        n = min(chunk, steps-start)
        if case=='button_release' and start==401:
            d.qfrc_applied[:]=0
            p.data.qfrc_applied.zero_()
        p.step(n, graph=True)
        wp.synchronize_device(p.device)
        for _ in range(n): mujoco.mj_step(m,d)
        native_trace.append(np.r_[d.time,d.qpos,d.qvel])
        a=p.arrays();gpu_trace.append(np.r_[a['time'][0],a['qpos'][0],a['qvel'][0]])
    paired_s = time.perf_counter()-t
    native, gpu = np.array(native_trace),np.array(gpu_trace)
    err=abs(native-gpu)
    actual=p.arrays()
    report=dict(case=case,steps=steps,worlds=worlds,compile_s=compile_s,paired_s=paired_s,
                qpos_max_error=float(err[:,1:1+m.nq].max()),qvel_max_error=float(err[:,1+m.nq:].max()),
                time_max_error=float(err[:,0].max()),finite=bool(np.isfinite(gpu).all()),
                nq=m.nq,nv=m.nv,ngeom=m.ngeom,isolate_tool=isolate_tool,plugin_audit=p.plugin_audit,
                overflow_bits=p.data.overflow.numpy().tolist(),
                arm_qpos_max_error=float(err[:,1:1+min(12,m.nq)].max()))
    if case not in ('thread','axial'):
        report['parts_position_max_error_m']={}
        for name in (('driver',) if isolate_tool else ('base','motor','horn','screw','retainer','yoke','driver')):
            q=int(m.joint(name+'_free').qposadr[0])+1
            report['parts_position_max_error_m'][name]=float(np.linalg.norm(native[:,q:q+3]-gpu[:,q:q+3],axis=1).max())
    if case in ('thread','axial'):
        for key, trace in [('native',native),('gpu',gpu)]:
            index=min(len(trace)-1,round(.3/(m.opt.timestep*chunk)))
            advance=trace[-1,1]-trace[index,1];turns=(trace[-1,2]-trace[index,2])/(2*np.pi)
            report[key]=dict(advance_m=float(advance),turns=float(turns),pitch_error_m=float(advance-turns*PITCH))
    if case in ('button','button_release'):
        v=m.joint('spindle').dofadr[0]
        report['native_spindle_rad_s']=float(d.qvel[v]);report['gpu_spindle_rad_s']=float(actual['qvel'][0,v])
    # Benchmark an additional 80 native physics steps per control with fixed commands.
    report['controls_s']=None
    if benchmark_controls:
        p.step(80, graph=True);wp.synchronize_device(p.device)
        t=time.perf_counter()
        for _ in range(benchmark_controls):p.step(80,graph=True)
        wp.synchronize_device(p.device)
        report['controls_s']=worlds*benchmark_controls/(time.perf_counter()-t)
    report['benchmark_overflow_bits']=p.data.overflow.numpy().tolist()
    report['benchmark_finite']=bool(np.isfinite(p.data.qpos.numpy()).all() and np.isfinite(p.data.qvel.numpy()).all())
    report['benchmark_control_period_s']=float(80*m.opt.timestep)
    report['parameter_audit']=p.parameter_audit
    np.savez_compressed(out/(case+'_trace.npz'),native=native,gpu=gpu,initial_qpos=initial_qpos)
    p.close()
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case',choices=['sdf','basic','zero','actuator','freefall','button','button_release','thread','axial'],default='sdf')
    ap.add_argument('--steps',type=int,default=400)
    ap.add_argument('--worlds',type=int,default=1)
    ap.add_argument('--isolate-tool',action='store_true')
    ap.add_argument('--benchmark-controls',type=int,default=8)
    ap.add_argument('--out',type=Path,default=ROOT/'artifacts_local/warp_prototype')
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.steps<2 or a.worlds<1 or a.benchmark_controls<0:ap.error('Need steps>=2, worlds>=1, benchmark-controls>=0')
    wp.init()
    report=(sdf_check(a.out) if a.case=='sdf' else basic_check(a.out) if a.case=='basic'
            else physics_check(a.out,a.case,a.steps,a.worlds,a.isolate_tool,a.benchmark_controls))
    (a.out/(a.case+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
