"""Qualify the isolated helper SDF candidate; never change the running scene."""
import argparse
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from assembly_policy.so101.helper_collision import DIRECTORY, replace_helper_collision
from assembly_policy.so101.scene import build_xml, load_plugin, ROOT
from scripts.so101_helper_sdf import source_geometry, interpolate, distances, sha256, limits


def geometry(out):
    import igl
    import open3d as o3d
    import trimesh
    manifest = json.loads((DIRECTORY/'manifest.json').read_text())
    rows = {}
    for name, record in manifest['meshes'].items():
        mesh, surface, solid, has_sheet, _ = source_geometry(name)
        path = ROOT/record['grid']
        if sha256(path) != record['grid_sha256']:
            raise ValueError(f'Grid changed: {name}')
        cfg = record['sdf']; shape = np.asarray(cfg['shape']); origin = np.asarray(cfg['origin']); h = cfg['spacing']
        grid = np.memmap(path, mode='r', dtype='<f2', offset=44, shape=tuple(shape[::-1]))
        sample, faces = trimesh.sample.sample_surface(mesh, 20000, seed=301)
        rng = np.random.default_rng(301)
        volume = rng.uniform(mesh.bounds[0], mesh.bounds[1], (30000, 3))
        normal = mesh.face_normals[faces]
        points = np.vstack([sample, volume, sample+normal*.0003, sample-normal*.0003])
        reference = distances(points, surface, solid, has_sheet)
        candidate = interpolate(grid, (points-origin)/h)
        error = np.abs(reference-candidate)
        # A separately implemented generalized winding sum audits occupancy.
        winding = igl.fast_winding_number(mesh.vertices, mesh.faces, volume)
        volume_distance = reference[20000:50000]
        away = np.abs(volume_distance) > .00001
        sign_mismatch = ((winding > .5) != (volume_distance < 0)) & away
        # Reverse direction: source-near grid edges crossing the candidate
        # zero surface produce points evaluated against the original triangles.
        base = np.floor((sample-origin)/h).astype(int)
        crossings = []
        for axis in range(3):
            step = np.eye(3, dtype=int)[axis]
            for offset in (np.zeros(3, int), -step):
                a = base+offset; b = a+step
                va = grid[a[:, 2], a[:, 1], a[:, 0]].astype(float)
                vb = grid[b[:, 2], b[:, 1], b[:, 0]].astype(float)
                keep = va*vb < 0
                t = va[keep]/(va[keep]-vb[keep])
                crossings.append(origin + (a[keep]+t[:, None]*step)*h)
        crossings = np.vstack(crossings)
        reverse_distance = surface.compute_distance(o3d.core.Tensor(crossings.astype(np.float32)), nthreads=2).numpy()
        cavity = (volume_distance > 2*h)
        false_fill = (reference > .00001) & (candidate < 0)
        false_hole = (reference < -.00001) & (candidate > 0)
        row = dict(samples=len(points), grid_zero_crossing_samples=len(crossings),
                   source_to_grid_surface_max_m=float(np.abs(candidate[:20000]).max()),
                   grid_surface_to_source_max_m=float(reverse_distance.max()),
                   distance_error_max_m=float(error.max()), distance_error_p99_m=float(np.quantile(error, .99)),
                   theoretical_interior_bound_m=record['interpolation_bound_m'],
                   independent_winding_sign_mismatches=int(sign_mismatch.sum()),
                   sampled_cavity_points=int(cavity.sum()),
                   cavity_false_fill=int(np.sum(candidate[20000:50000][cavity] < 0)),
                   false_fill_max_source_clearance_m=float(reference[false_fill].max()) if false_fill.any() else 0.,
                   false_hole_max_source_depth_m=float(-reference[false_hole].min()) if false_hole.any() else 0.,
                   source_sha256=record['source_sha256'], grid_sha256=record['grid_sha256'])
        row['passed'] = bool(not sign_mismatch.any() and row['cavity_false_fill'] == 0 and
                         error.max() < record['interpolation_bound_m']+2e-6 and
                         reverse_distance.max() < record['interpolation_bound_m']+2e-6)
        rows[name] = row
        (out/'geometry.json').write_text(json.dumps(dict(meshes=rows, complete=manifest.get('complete', False),
            scope='90k signed-distance samples and source-near candidate zero crossings per mesh; sampled two-way surface distances, not a global Hausdorff certificate'), indent=2)+'\n')
        print(json.dumps(dict(mesh=name, **row)), flush=True)
        del grid
        limits()
    return rows


def candidate_model():
    load_plugin()
    root = ET.fromstring(build_xml())
    replacement = replace_helper_collision(root)
    mujoco.mj_setCacheCapacity(mujoco.mj_getCache(), 0)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    # The compiler asset cache is not part of the compiled MjModel or the SDF
    # plugin's immutable grids. This isolated process needs no recompile cache.
    mujoco.mj_clearCache(mujoco.mj_getCache())
    # Release native compiler temporaries before a memory-bounded physics probe.
    # This does not alter the model or any simulation state.
    import ctypes
    import gc
    gc.collect()
    trim = getattr(ctypes.CDLL(None), 'malloc_trim', None)
    if trim is not None:
        trim(0)
    return model, root, replacement


def frames(out):
    import trimesh
    model, root, replacement = candidate_model()
    data = mujoco.MjData(model)
    original = ET.fromstring(build_xml())
    source_assets = {m.get('name'): m for m in original.findall('asset/mesh')}
    manifest = json.loads((DIRECTORY/'manifest.json').read_text())
    rows = []
    source_hashes = {}
    for side in ('left', 'right'):
        for body in original.find(f'worldbody/body[@name="{side}_base"]').iter('body'):
            bodyid = model.body(body.get('name')).id
            for local, geom in enumerate(body.findall('geom')):
                if geom.get('class') != side+'_collision' or geom.get('type', 'mesh') != 'mesh':
                    continue
                gid = int(model.body_geomadr[bodyid])+local
                if int(model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_SDF):
                    raise AssertionError(f'Raw mesh remains on {body.get("name")}')
                path = Path(source_assets[geom.get('mesh')].get('file'))
                mesh = trimesh.load_mesh(path)
                aabb = model.geom_aabb[gid]
                underbound = max(0., float(np.max((aabb[:3]-aabb[3:])-mesh.bounds[0])),
                                 float(np.max(mesh.bounds[1]-(aabb[:3]+aabb[3:]))))
                source_hashes[path.name] = sha256(path)
                points, faces = trimesh.sample.sample_surface(mesh, 12, seed=77)
                points += mesh.face_normals[faces]*.0004
                quat = np.fromstring(geom.get('quat', '1 0 0 0'), sep=' '); quat /= np.linalg.norm(quat)
                rotation = np.empty(9); mujoco.mju_quat2Mat(rotation, quat); rotation = rotation.reshape(3, 3)
                offset = np.fromstring(geom.get('pos', '0 0 0'), sep=' ')
                name = path.stem
                _, surface, solid, has_sheet, _ = source_geometry(name)
                reference = distances(points, surface, solid, has_sheet)
                cfg = manifest['meshes'][name]['sdf']
                grid = np.memmap(ROOT/manifest['meshes'][name]['grid'], mode='r', dtype='<f2',
                                 offset=44, shape=tuple(cfg['shape'][::-1]))
                errs = []; frame_errors = []
                for angle in (0., .2):
                    data.qpos[:] = model.qpos0
                    for axis in ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper'):
                        data.qpos[model.joint(side+'_'+axis).qposadr[0]] = angle
                    mujoco.mj_forward(model, data)
                    world = (points@rotation.T+offset)@data.xmat[bodyid].reshape(3, 3).T+data.xpos[bodyid]
                    # Native SDF callbacks use the actual compiled geom frame.
                    # mj_geomDistance does not support SDF pairs, so a sphere
                    # query returning distmax would not validate alignment.
                    local_points = (world-data.geom_xpos[gid])@data.geom_xmat[gid].reshape(3,3)
                    frame_errors.extend(np.linalg.norm(local_points-points,axis=1).tolist())
                    actual = interpolate(grid, (local_points-np.asarray(cfg['origin']))/cfg['spacing'])
                    errs.extend(np.abs(actual-reference).tolist())
                rows.append(dict(body=body.get('name'), geom=gid, source=name, poses=2,
                                 queries=len(errs), native_world_query_error_max_m=max(errs),
                                 native_frame_alignment_error_max_m=max(frame_errors),
                                 compiled_broadphase_bounds_underfill_m=underbound))
                del grid
    raw = [i for i in range(model.ngeom) if model.body(int(model.geom_bodyid[i])).name.startswith(('left_', 'right_'))
           and (model.geom_contype[i] or model.geom_conaffinity[i]) and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH]
    # The replacement leaves every non-collision XML element byte-equivalent.
    for section in ('actuator', 'contact', 'option'):
        assert ET.tostring(root.find(section)) == ET.tostring(original.find(section))
    report = dict(replacement=replacement, remaining_helper_collision_meshes=raw, queries=rows,
                  max_world_query_error_m=max(r['native_world_query_error_max_m'] for r in rows),
                  max_frame_alignment_error_m=max(r['native_frame_alignment_error_max_m'] for r in rows),
                  max_broadphase_bounds_underfill_m=max(r['compiled_broadphase_bounds_underfill_m'] for r in rows),
                  source_hashes=source_hashes, controls_contact_filters_options_unchanged=True,
                  scope='Original source points transformed into actual compiled native SDF geom frames at two joint configurations; grid interpolation compared to triangles')
    baseline = mujoco.MjModel.from_xml_string(ET.tostring(original, encoding='unicode'))
    report['unchanged_model_fields'] = {key: bool(np.array_equal(getattr(baseline,key),getattr(model,key)))
        for key in ('body_mass','body_inertia','body_pos','body_quat','jnt_axis','jnt_range',
                    'site_pos','site_quat','actuator_gainprm','actuator_biasprm','actuator_forcerange',
                    'actuator_ctrlrange','qpos0')}
    report['passed'] = (not raw and report['max_world_query_error_m'] < .0002 and
                        report['max_frame_alignment_error_m'] < 1e-9)
    report['passed'] = report['passed'] and all(report['unchanged_model_fields'].values())
    (out/'frames.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('queries','source_hashes')}), flush=True)
    return report


def states(out, bank):
    model, _, replacement = candidate_model()
    data = mujoco.MjData(model)
    limits()
    rows = []
    for task in ('motor', 'horn', 'screw', 'retainer', 'yoke'):
        with np.load(bank/f'{task}_train.npz') as archive:
            values = archive['states']; family = archive['family']
        for f in np.unique(family):
            chosen = np.flatnonzero(family == f)
            for index in np.unique(chosen[[0, -1]]):
                mujoco.mj_resetData(model, data)
                mujoco.mj_setState(model,data,values[index],mujoco.mjtState.mjSTATE_FULLPHYSICS)
                data.ctrl[:] = data.qpos[:12]
                mujoco.mj_forward(model,data)
                depth = max([0., *(-data.contact.dist)])
                speed = float(np.abs(data.qvel[:12]).max()); start_time = data.time
                for _ in range(400):
                    mujoco.mj_step(model,data)
                    depth = max(depth, max([0., *(-data.contact.dist)]))
                    speed = max(speed, float(np.abs(data.qvel[:12]).max()))
                finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
                row = dict(task=task, family=int(f), index=int(index), max_penetration_m=float(depth),
                    max_arm_velocity_rad_s=speed, warnings=data.warning.number.tolist(), finite=finite,
                    clock_error_s=abs(float(data.time-start_time)-.2))
                row['passed'] = bool(depth<.0004 and speed<5 and finite and not any(row['warnings']) and row['clock_error_s']<1e-8)
                rows.append(row); print(json.dumps(row), flush=True); limits()
    (out/'representative_states.json').write_text(json.dumps(dict(replacement=replacement,
        source_bank=str(bank), states=rows, passed=all(r['passed'] for r in rows),
        scope='Changed helper geometry, 0.2 s fixed-joint-control native probes; these are not replacement training banks or task-success claims'),indent=2)+'\n')


def replay(out, source):
    import ctypes
    import gc
    import mmap
    proof = json.loads(source.with_suffix('.json').read_text())
    if not proof.get('passed') or any(proof['warnings']) or sha256(source.with_suffix('.mjb')) != proof['model_sha256']:
        raise ValueError('Expected a qualified, hash-matched native control archive')
    for name, digest in proof['plugin_sha256'].items():
        if sha256(ROOT/'plugin/build'/name) != digest:
            raise ValueError('Plugin differs from archived native tool demonstration')
    model, _, replacement = candidate_model()
    data = mujoco.MjData(model)
    archive = np.load(source.with_suffix('.npz'))
    initial = archive['initial_state']; controls = archive['controls']; labels = archive['labels']; stage = archive['stage']
    mujoco.mj_setState(model, data, initial, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data.ctrl[:] = controls[0]; mujoco.mj_forward(model, data)
    nstate = mujoco.mj_stateSize(model,mujoco.mjtState.mjSTATE_FULLPHYSICS)
    temporary = out/'native_tool_replay.states.incomplete.npy'
    states = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float64, shape=(len(controls),nstate))
    state_buffer = np.empty(nstate)
    depths = np.empty(len(controls)); pairs = np.empty((len(controls),2), np.int32)
    clock_error = 0.
    button = int(model.joint('button_forward').qposadr[0]); spindle = int(model.joint('spindle').dofadr[0])
    driver = int(model.joint('driver_free').qposadr[0])
    rows = []; started = time.monotonic()
    for j, label in enumerate(labels):
        indices = np.flatnonzero(stage == j); max_button = 0.; max_speed = 0.
        for i in indices:
            data.ctrl[:] = controls[i]
            before = data.time; mujoco.mj_step(model,data)
            clock_error = max(clock_error,abs(float(data.time-before)-model.opt.timestep))
            if clock_error>1e-8 or not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise FloatingPointError('Native helper replay clock reset or nonfinite state')
            mujoco.mj_getState(model,data,state_buffer,mujoco.mjtState.mjSTATE_FULLPHYSICS)
            states[i] = state_buffer
            if data.ncon:
                contact = data.contact[int(np.argmin(data.contact.dist))]
                depths[i] = max(0.,-float(contact.dist)); pairs[i] = contact.geom
            else:
                depths[i] = 0.; pairs[i] = -1
            max_button = max(max_button, float(data.qpos[button]))
            max_speed = max(max_speed, float(np.abs(data.qvel[:12]).max()))
            if i % 1000 == 0:
                states.flush()
                if hasattr(states._mmap,'madvise'): states._mmap.madvise(mmap.MADV_DONTNEED)
                limits()
        row = dict(name=str(label), button_max_m=max_button, button_m=float(data.qpos[button]),
                   spindle_rad_s=float(data.qvel[spindle]), driver_position=data.qpos[driver:driver+3].tolist(),
                   max_arm_velocity_rad_s=max_speed, max_penetration_m=float(depths[indices].max()))
        rows.append(row); print(json.dumps(row), flush=True)
    byname = {r['name']: r for r in rows}
    start = np.asarray(byname['Settle loose workcell']['driver_position'])
    lift = np.asarray(byname['Hold tool OFF above table']['driver_position'])
    on = byname['Squeeze physical forward button']; off = byname['Wait for spindle coast-down']
    end = np.asarray(byname['Withdraw empty hand']['driver_position'])
    checks = dict(lift_40mm=bool(lift[2]-start[2]>.04),
                  carry_off=byname['Hold tool OFF above table']['button_max_m']<.0006,
                  physical_on=on['button_m']>.0006 and on['spindle_rad_s']>20,
                  physical_off=off['button_m']<.0006 and abs(off['spindle_rad_s'])<.1,
                  returned=abs(end[2]-start[2])<.003,
                  penetration_below_0_3mm=depths.max()<.0003,
                  arm_speed_below_5=max(r['max_arm_velocity_rad_s'] for r in rows)<5,
                  no_warnings=not bool(np.asarray(data.warning.number).any()))
    checks = {name: bool(value) for name, value in checks.items()}
    # Avoid an additional Python serialization buffer. The native filename API
    # still allocates a temporary model buffer; memory.json records that peak.
    model_path = out/'native_tool_replay.mjb'
    mujoco.mj_saveModel(model,str(model_path))
    report = dict(source=str(source), source_model_sha256=proof['model_sha256'], replacement=replacement,
        task_stages=rows, checks=checks, passed=all(checks.values()), elapsed_seconds=time.monotonic()-started,
        scope='Original scripted controls replayed without resets in changed helper collision geometry; no trained policy or hardware claim',
        scripted=True, learned_policy=False, actuator_execution=True, state_spec=int(mujoco.mjtState.mjSTATE_FULLPHYSICS),
        physics_steps=len(controls), timestep=float(model.opt.timestep), initial_time=float(initial[0]), final_time=float(data.time),
        max_clock_step_error_s=clock_error, max_contact_penetration_m=float(depths.max()),
        warnings=data.warning.number.tolist(), model_sha256=sha256(model_path),
        plugin_sha256=proof['plugin_sha256'], model_file=model_path.name, trajectory_file='native_tool_replay.npz',
        stages=[dict(name=str(label),steps=int(np.sum(stage==j)),max_penetration_m=float(depths[stage==j].max()))
                for j,label in enumerate(labels)])
    # Release the large native model before streaming the recorded states into
    # the standard compressed trajectory format.
    del data, model
    gc.collect(); trim=getattr(ctypes.CDLL(None),'malloc_trim',None)
    if trim is not None: trim(0)
    states.flush()
    np.savez_compressed(out/'native_tool_replay.npz', states=states,controls=controls,
                        penetration=depths,deepest_geom_pair=pairs,stage=stage,labels=labels,initial_state=initial)
    del states
    temporary.unlink()
    (out/'native_tool_replay.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(checks=checks, passed=report['passed'], elapsed_seconds=report['elapsed_seconds'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['geometry', 'frames', 'states', 'replay'], required=True)
    parser.add_argument('--out', type=Path, default=ROOT/'artifacts_local/helper_sdf')
    parser.add_argument('--replay', type=Path, default=ROOT/'artifacts_local/tool_pickup/raised10_coast')
    parser.add_argument('--bank', type=Path, default=ROOT/'data/so101/banks_near_v2')
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    if args.case == 'geometry': geometry(args.out)
    elif args.case == 'frames': frames(args.out)
    elif args.case == 'states': states(args.out,args.bank)
    else: replay(args.out, args.replay)


if __name__ == '__main__':
    main()
