"""Audit motor/base grid contacts against untouched Hiwonder triangles.

Read-only with respect to production assets. Outputs sampled diagnostics, not a
continuous mesh nonintersection certificate or a manufacturing-clearance claim.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

import argparse
import ctypes
import gc
import hashlib
import json
from pathlib import Path
import resource
import struct
import subprocess
import threading
import time

import igl
import mujoco
import numpy as np
import psutil
from scipy.spatial.transform import Rotation

from assembly_policy.so101.scene import ROOT, ASSETS, load_plugin, metadata
from assembly_policy.so101.fastener import origins
from assembly_policy.so101.trajectory import STATE_SPEC


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def resource_gate():
    process = psutil.Process()
    if process.memory_info().rss > 2 * 1024**3:
        raise RuntimeError('Audit exceeded the 2 GiB RSS budget')
    if psutil.virtual_memory().available < 2 * 1024**3:
        raise RuntimeError('Host has less than 2 GiB available memory')


def ray_sections(references):
    """Cross sections in the original vendor assembly frame, without pose edits."""
    origin = origins(metadata()); offset = origin['motor']-origin['base']
    rows = []
    for x in (-.010,.010):
        for y in (.005,.010):
            hits = {}
            for name, ref in references.items():
                start = np.array([x,y,-.050])-(offset if name=='motor' else 0.)
                intersections = igl.ray_mesh_intersect(start,np.array([0.,0.,1.]),ref.v,ref.f)
                hits[name] = np.unique(np.round([-.050+r[1] for r in intersections],10)).tolist()
            base_lo = max(z for z in hits['base'] if z<0)
            base_hi = min(z for z in hits['base'] if z>0)
            motor_lo, motor_hi = min(hits['motor']),max(hits['motor'])
            rows.append(dict(x_m=x,y_m=y,z_hits_in_base_frame_m=hits,
                             base_cavity_height_m=base_hi-base_lo,
                             motor_outer_height_m=motor_hi-motor_lo,
                             thickness_excess_m=(motor_hi-motor_lo)-(base_hi-base_lo),
                             bottom_overlap_m=max(0.,base_lo-motor_lo),
                             top_overlap_m=max(0.,motor_hi-base_hi)))
    return rows


def plot_section(references, output):
    """Plot an original-triangle slice, independently of collision-grid meshes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import trimesh
    origin = origins(metadata()); offset = origin['motor']-origin['base']
    fig, axes = plt.subplots(1,3,figsize=(12,4),gridspec_kw={'width_ratios':[1.3,1,1]},layout='constrained')
    for name, color in [('base','#2776b6'),('motor','#d96e19')]:
        ref = references[name]
        mesh = trimesh.Trimesh(ref.v+(offset if name=='motor' else 0),ref.f,process=False)
        lines = trimesh.intersections.mesh_plane(mesh,np.array([0.,1.,0.]),np.array([0.,.010,0.]))
        for axis in axes:
            axis.add_collection(LineCollection(lines[:,:,[0,2]]*1000,colors=color,linewidths=1.4,label=name.capitalize()))
    axes[0].set(xlim=(-14,14),ylim=(-10,38),title='Vendor assembled transform',xlabel='Base-frame X (mm)',ylabel='Base-frame Z (mm)')
    axes[0].set_aspect('equal'); axes[0].legend(loc='upper right')
    axes[1].set(xlim=(9.5,10.5),ylim=(-6.3,-5.5),title='Lower rail: actual overlap',xlabel='X (mm)',ylabel='Z (mm)')
    section = next(row for row in ray_sections(references) if row['x_m']==.010 and row['y_m']==.010)
    bottom = min(section['z_hits_in_base_frame_m']['motor'])*1000
    floor = max(z for z in section['z_hits_in_base_frame_m']['base'] if z<0)*1000
    axes[1].axhspan(bottom,floor,color='#d62728',alpha=.15)
    axes[1].annotate(f'{floor-bottom:.3f} mm',xy=(10,floor),xytext=(10,bottom),ha='center',va='top',
                     arrowprops={'arrowstyle':'<->','color':'#b21f26'},color='#b21f26')
    axes[2].set(xlim=(9.5,10.5),ylim=(25.7,26.5),title='Upper rail: almost touching',xlabel='X (mm)',ylabel='Z (mm)')
    for axis in axes: axis.grid(alpha=.18)
    fig.suptitle('Untouched CAD: 31.8 mm base cavity versus 32.0 mm motor thickness\n'
                 'Triangle slice at base Y = 10 mm; a finer SDF does not remove the source interference',fontsize=12)
    fig.savefig(output,dpi=170); plt.close(fig)


def native_reference(out):
    """Build unchanged grid_eval/get_grid source into a separate batched probe."""
    source = (ROOT/'plugin/assemblenist_sdf.cc').read_text().split('template <int N>')[0]
    source += '''
}  // original anonymous namespace
extern "C" void evaluate_grid(int id, int count, const double* p, double* out) {
  const Grid* grid=get_grid(id);
  for(int i=0;i<count;++i) out[i]=grid_eval(p+3*i,grid,nullptr);
}
'''
    cpp, library = out/'native_grid_reference.cpp', out/'native_grid_reference.so'
    cpp.write_text(source)
    mjdir = Path(mujoco.__file__).parent
    subprocess.run(['g++', '-O2', '-std=c++17', '-mf16c', '-shared', '-fPIC',
                    '-I'+str(mjdir/'include'), str(cpp), '-o', str(library),
                    str(next(mjdir.glob('libmujoco.so*'))), '-Wl,-rpath,'+str(mjdir)], check=True)
    lib = ctypes.CDLL(str(library))
    evaluate = lib.evaluate_grid
    array = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    evaluate.argtypes = [ctypes.c_int, ctypes.c_int, array, array]
    evaluate.restype = None
    def query(grid_id, points):
        points = np.ascontiguousarray(points, np.float64)
        values = np.empty(len(points))
        evaluate(grid_id, len(points), points, values)
        return values
    return lib, query, hashlib.sha256(source.encode()).hexdigest()


class Reference:
    def __init__(self, name, grid_id, query, count):
        self.name, self.grid_id, self.query = name, grid_id, query
        file = ASSETS/'parts'/f'{name}.npz'
        arrays = np.load(file)
        self.v, self.f = arrays['vertices'], arrays['faces']
        obj = ASSETS/'parts'/f'{name}.obj'
        vv, ff = [], []
        for line in obj.read_text().splitlines():
            if line.startswith('v '): vv.append([float(x) for x in line.split()[1:4]])
            if line.startswith('f '): ff.append([int(x.split('/')[0])-1 for x in line.split()[1:4]])
        discrepancy = float(np.max(np.abs(np.array(vv)-self.v)))
        if discrepancy > 5.1e-9 or not np.array_equal(np.array(ff), self.f):
            raise ValueError(f'{name}: OBJ does not match the unchanged grid-generator triangles')
        triangles = self.v[self.f]
        areas = np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0]), axis=1)/2
        rng = np.random.default_rng(20261105)
        selected = triangles[rng.choice(len(triangles), count, p=areas/areas.sum())]
        uv = rng.uniform(size=(count, 2)); uv[uv.sum(1)>1] = 1-uv[uv.sum(1)>1]
        sampled = selected[:,0]+uv[:,0,None]*(selected[:,1]-selected[:,0])+uv[:,1,None]*(selected[:,2]-selected[:,0])
        self.surface = np.concatenate([self.v, triangles.mean(1), sampled])
        grid = ASSETS/'sdf'/f'grid_{grid_id}.bin'
        with grid.open('rb') as stream:
            shape = struct.unpack('<3i', stream.read(12))
        self.receipt = dict(npz_sha256=sha(file), obj_sha256=sha(obj), grid_sha256=sha(grid),
                            vertices=len(self.v), faces=len(self.f), samples=len(self.surface),
                            obj_quantization_max_m=discrepancy, grid_shape=shape,
                            grid_bytes=grid.stat().st_size)

    def distances(self, points, exact_sign=False):
        points = np.ascontiguousarray(points, np.float64)
        # libigl's exact winding implementation can allocate a large temporary
        # for many queries; keep all reference calls within small fixed batches.
        squared, faces, closest, winds = [], [], [], []
        for start in range(0, len(points), 8 if exact_sign else 2048):
            chunk = points[start:start+(8 if exact_sign else 2048)]
            sq, face, close = igl.point_mesh_squared_distance(chunk, self.v, self.f)
            winding = (igl.winding_number if exact_sign else igl.fast_winding_number)(self.v, self.f, chunk)
            squared.append(sq); faces.append(face); closest.append(close); winds.append(winding)
            resource_gate()
        squared, faces, closest, winding = [np.concatenate(x) for x in (squared,faces,closest,winds)]
        unsigned = np.sqrt(np.maximum(0, squared))
        signed = unsigned*np.where(winding > .5, -1., 1.)
        grid = self.query(self.grid_id, points)
        return dict(signed=signed, unsigned=unsigned, winding=winding, grid=grid,
                    closest=closest, face=faces)


def stats(values):
    return {name: float(value) for name, value in zip(
        ('min', 'p01', 'median', 'p99', 'max'), np.quantile(values, [0,.01,.5,.99,1]))}


def cross_surface(source, target, source_p, source_r, target_p, target_r):
    points = (source.surface@source_r.T+source_p-target_p)@target_r
    distances = target.distances(points)
    signed, grid = distances['signed'], distances['grid']
    error = grid-signed
    # Negative error means the grid adds material relative to the signed mesh.
    relevant = np.minimum(np.abs(signed), np.abs(grid)) < .001
    false_inside = (signed > 1e-6)&(grid < -1e-6)
    inside = signed < -1e-6
    k = int(np.argmin(signed))
    critical = np.unique(np.r_[np.argsort(signed)[:16], np.argsort(error)[:16], np.argsort(error)[-16:]])
    exact = target.distances(points[critical], exact_sign=True)
    return dict(source=source.name, target=target.name, samples=len(points),
                mesh_signed_m=stats(signed), grid_m=stats(grid),
                near_samples=int(relevant.sum()),
                near_grid_minus_mesh_m=stats(error[relevant]) if relevant.any() else None,
                mesh_inside_samples=int(inside.sum()), false_grid_inside_samples=int(false_inside.sum()),
                false_grid_inside_max_m=float(-grid[false_inside].min()) if false_inside.any() else 0.,
                worst_mesh_point_target_frame=points[k].tolist(),
                worst_mesh_closest_target_frame=distances['closest'][k].tolist(),
                worst_mesh_winding=float(distances['winding'][k]),
                exact_winding_spot_count=len(critical),
                exact_winding_sign_disagreement_count=int(np.sum((exact['winding']>.5)!=(distances['winding'][critical]>.5))),
                exact_winding_spot_signed_min_m=float(exact['signed'].min()))


def audit_pose(m, d, references, label, sample_surface=True):
    mujoco.mj_forward(m, d)
    poses = {name: (d.body(name).xpos.copy(), d.body(name).xmat.reshape(3,3).copy())
             for name in references}
    positions, contacts = [], []
    for i, c in enumerate(d.contact):
        names = [m.body(int(m.geom_bodyid[g])).name for g in c.geom]
        if set(names) != {'base','motor'}: continue
        force = np.zeros(6); mujoco.mj_contactForce(m, d, i, force)
        positions.append(c.pos.copy())
        contacts.append(dict(geoms=[m.geom(int(g)).name for g in c.geom],
                             native_distance_m=float(c.dist), world_position=c.pos.tolist(),
                             world_normal=c.frame[:3].tolist(),
                             local_normal_force_N=float(force[0])))
    if positions:
        positions = np.asarray(positions)
        for name, reference in references.items():
            p, r = poses[name]; local = (positions-p)@r
            fast = reference.distances(local)
            exact = reference.distances(local, exact_sign=True)
            # Closest points are themselves on the original mesh; query their grid value.
            surface_grid = reference.query(reference.grid_id, fast['closest'])
            for index, row in enumerate(contacts):
                row[name] = dict(point_local_m=local[index].tolist(), grid_m=float(fast['grid'][index]),
                                 mesh_signed_m=float(exact['signed'][index]),
                                 mesh_unsigned_m=float(exact['unsigned'][index]),
                                 fast_winding=float(fast['winding'][index]),
                                 exact_winding=float(exact['winding'][index]),
                                 grid_minus_mesh_m=float(fast['grid'][index]-exact['signed'][index]),
                                 closest_mesh_point_grid_m=float(surface_grid[index]))
        for row in contacts:
            row['grid_sum_m'] = sum(row[name]['grid_m'] for name in references)
            row['mesh_signed_sum_m'] = sum(row[name]['mesh_signed_m'] for name in references)
            # MuJoCo stores a contact point shifted from the SDF optimizer point.
            # Its depth is the maximum of the two SDFs there, not their sum at
            # the shifted contact point nor maximum mesh penetration.
            optimizer = np.array(row['world_position'])-np.array(row['world_normal'])*row['native_distance_m']/2
            values = []
            for name, reference in references.items():
                p, r = poses[name]
                local = ((optimizer-p)@r)[None,:]
                values.append(float(reference.query(reference.grid_id, local)[0]))
            row['optimizer_world_position'] = optimizer.tolist()
            row['optimizer_grid_values_m'] = values
            row['optimizer_max_vs_native_abs_m'] = abs(max(values)-row['native_distance_m'])
    surfaces = []
    if sample_surface:
        for source, target in [('base','motor'), ('motor','base')]:
            surfaces.append(cross_surface(references[source], references[target], *poses[source], *poses[target]))
    bp, br = poses['base']; mp, mr = poses['motor']
    offset = origins(metadata())['motor']-origins(metadata())['base']
    row = dict(label=label, time_s=float(d.time),
               motor_translation_error_base_frame_m=(br.T@(mp-bp)-offset).tolist(),
               motor_relative_euler_xyz_deg=Rotation.from_matrix(br.T@mr).as_euler('xyz', degrees=True).tolist(),
               contacts=contacts, surface_comparisons=surfaces,
               max_native_penetration_m=max([max(0.,-c['native_distance_m']) for c in contacts],default=0.),
               frame_check_max_abs_m=max([c['optimizer_max_vs_native_abs_m'] for c in contacts],default=0.),
               rss_bytes=psutil.Process().memory_info().rss,
               peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    resource_gate()
    return row


def load_state(path, index):
    report = json.loads(path.with_suffix('.json').read_text())
    if sha(path.with_suffix('.mjb')) != report['model_sha256']:
        raise ValueError('Trajectory model hash mismatch')
    m = mujoco.MjModel.from_binary_path(str(path.with_suffix('.mjb')))
    d = mujoco.MjData(m)
    with np.load(path.with_suffix('.npz')) as array:
        state = array['states'][index].copy()
        control = array['controls'][index].copy()
    mujoco.mj_setState(m, d, state, STATE_SPEC); d.ctrl[:] = control
    return m, d, dict(trajectory=str(path.relative_to(ROOT)), index=index,
                      model_sha256=report['model_sha256'], trajectory_sha256=sha(path.with_suffix('.npz')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'artifacts_local/mating_geometry_audit')
    parser.add_argument('--samples', type=int, default=4000)
    args = parser.parse_args()
    process = psutil.Process(); allowed = process.cpu_affinity(); process.cpu_affinity(allowed[-2:])
    initial_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    memory = {'sampled_peak_rss_bytes':process.memory_info().rss,'minimum_host_available_bytes':psutil.virtual_memory().available}
    stopped = threading.Event()
    def monitor_memory():
        while not stopped.wait(.01):
            current = process.memory_info().rss
            available = psutil.virtual_memory().available
            memory['sampled_peak_rss_bytes'] = max(current,memory['sampled_peak_rss_bytes'])
            memory['minimum_host_available_bytes'] = min(available,memory['minimum_host_available_bytes'])
            if current>2*1024**3 or available<2*1024**3:
                print(json.dumps(dict(resource_budget_failure=True,**memory)),flush=True)
                os._exit(2)
    monitor = threading.Thread(target=monitor_memory,daemon=True); monitor.start()
    print(json.dumps(dict(initial_rusage_peak_rss_bytes=initial_peak,initial_live_rss_bytes=process.memory_info().rss)),flush=True)
    resource_gate(); started = time.monotonic(); args.out.mkdir(parents=True, exist_ok=True)
    load_plugin()
    library, query, native_sha = native_reference(args.out)
    references = {name: Reference(name, gid, query, args.samples) for name,gid in [('base',101),('motor',102)]}
    rows, inputs = [], []
    for name, index in [('table_transport_probe',22799), ('precision_nominal_entry',1199),
                         ('precision_nominal_entry',1599), ('dual_table_placement',27579)]:
        m, d, receipt = load_state(ROOT/'artifacts_local'/name,index)
        inputs.append(receipt)
        row = audit_pose(m,d,references,f'{name}:{index}')
        rows.append(row)
        print(json.dumps({k:row[k] for k in ('label','max_native_penetration_m','frame_check_max_abs_m','peak_rss_bytes')}),flush=True)
        del m,d; gc.collect()
    m, d, receipt = load_state(ROOT/'artifacts_local/prepared_holds/motor_goal_000000',2999)
    inputs.append(receipt); mujoco.mj_forward(m,d)
    bp = d.body('base').xpos.copy(); br = d.body('base').xmat.reshape(3,3).copy()
    offset = origins(metadata())['motor']-origins(metadata())['base']
    q = m.joint('motor_free').qposadr[0]
    cases = [('vendor_nominal',0.,0.,0.)]+[(f'ideal_roll{roll:g}_gap{gap:.3f}',roll,gap,.0001)
                for roll,gap in [(0.,0.)]+[(.24,gap) for gap in (0.,.001,.003,.007,.010,.014,.020,.030,.050,.070)]]
    for label,roll,gap,xshift in cases:
        d.qpos[q:q+3] = bp+br@(offset+[xshift,-gap,0.])
        quat = np.empty(4); mujoco.mju_mat2Quat(quat,(br@Rotation.from_euler('x',roll,degrees=True).as_matrix()).ravel())
        d.qpos[q+3:q+7] = quat
        row = audit_pose(m,d,references,label)
        rows.append(row)
        print(json.dumps({k:row[k] for k in ('label','max_native_penetration_m','frame_check_max_abs_m','peak_rss_bytes')}),flush=True)
    report = dict(scope='Static sampled triangle/grid comparison; not a continuous path or solid-mesh certificate',
                  source='Untouched NPZ triangles used to generate production SDFs; OBJ verified identical within export rounding',
                  sign_method='Unsigned exact float64 triangle distance, signed by fast winding > 0.5; exact winding at contacts and critical samples',
                  frame_check='Native contact distance versus maximum verbatim C++ grid_eval at optimizer point reconstructed as contact.pos - normal*distance/2, in source body frames',
                  native_reference_sha256=native_sha, source_plugin_sha256=sha(ROOT/'plugin/assemblenist_sdf.cc'),
                  inputs=inputs, assets={k:v.receipt for k,v in references.items()}, rows=rows,
                  vendor_nominal_vertical_sections=ray_sections(references),
                  elapsed_s=time.monotonic()-started, max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  initial_rusage_peak_rss_bytes=initial_peak, memory_monitor=memory,
                  cpu_affinity=process.cpu_affinity())
    unchanged = []
    for name, reference in references.items():
        unchanged.extend(sha(ASSETS/'parts'/f'{name}.{suffix}')==reference.receipt[f'{suffix}_sha256']
                         for suffix in ('obj','npz'))
        unchanged.append(sha(ASSETS/'sdf'/f'grid_{reference.grid_id}.bin')==reference.receipt['grid_sha256'])
    report['production_assets_unchanged'] = all(unchanged)
    if not report['production_assets_unchanged']:
        raise RuntimeError('Production assets changed while the audit was running')
    if max(row['frame_check_max_abs_m'] for row in rows)>1e-10:
        raise RuntimeError('Native SDF contact frame/depth convention failed verification')
    plot_section(references,args.out/'vendor_mating_cross_section.png')
    stopped.set();monitor.join()
    (args.out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.out/'audit.json'),elapsed_s=report['elapsed_s'],max_rss_bytes=report['max_rss_bytes'])),flush=True)


if __name__ == '__main__':
    main()
