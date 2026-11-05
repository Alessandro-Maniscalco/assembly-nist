"""Compare pinned Menagerie jaw proxies with original CAD and native grasps.

This optional model does not modify the frozen scene. Geometry-only adoption
keeps the current contact parameters, servo settings, TCP and body inertias.
"""
import argparse
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import open3d as o3d
import trimesh

from assembly_policy.so101.menagerie_collision import replace_gripper_collision_menagerie
from scripts.check_so101_tool_gripper import make_model, run_sequence


ROOT = Path(__file__).resolve().parents[1]


def geometry_check(model):
    data = mujoco.MjData(model)
    data.qpos[model.joint('gripper').qposadr[0]] = .22
    mujoco.mj_forward(model, data)
    probe = model.geom('query_point').id
    source = ET.parse(ROOT/'assets_local/so101/so101_new_calib.xml').getroot()
    x = np.arange(-.020, .05501, .0005)
    z = np.arange(-.105, -.07499, .0005)
    y = [-.0075, 0., .0075]
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    truth = []
    proxies = []
    rows = []
    for name, meshname in [('gripper', 'wrist_roll_follower_so101_v1'),
                           ('moving_jaw_so101_v1', 'moving_jaw_so101_v1')]:
        body = model.body(name).id
        mesh = trimesh.load_mesh(ROOT/'assets_local/so101/assets'/f'{meshname}.stl')
        geom = source.find(f'.//body[@name="{name}"]/geom[@class="visual"][@mesh="{meshname}"]')
        quat = np.fromstring(geom.get('quat', '1 0 0 0'), sep=' ')
        quat /= np.linalg.norm(quat)
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quat)
        rotation = rotation.reshape(3, 3)
        offset = np.fromstring(geom.get('pos', '0 0 0'), sep=' ')
        body_points = (points-data.xpos[body])@data.xmat[body].reshape(3, 3)
        mesh_points = (body_points-offset)@rotation
        scene = o3d.t.geometry.RaycastingScene(nthreads=2)
        scene.add_triangles(o3d.core.Tensor(mesh.vertices.astype(np.float32)),
                            o3d.core.Tensor(mesh.faces.astype(np.uint32)))
        reference = scene.compute_signed_distance(
            o3d.core.Tensor(mesh_points.astype(np.float32)), nsamples=3).numpy()
        ids = [i for i in range(model.ngeom) if model.geom_bodyid[i] == body and model.geom_contype[i]]
        distance = np.full(len(points), .2)
        segment = np.empty(6)
        for index, point in enumerate(points):
            data.geom_xpos[probe] = point
            distance[index] = min(mujoco.mj_geomDistance(model, data, geom_id, probe, .2, segment)
                                  for geom_id in ids)+.000001
        over = (reference > .00001) & (distance < -.00001)
        under = (reference < -.00001) & (distance > .00001)
        rows.append(dict(body=name, samples=len(points), collision_geoms=len(ids),
                         overfill_samples=int(over.sum()), underfill_samples=int(under.sum()),
                         overfill_max_mm=float(reference[over].max()*1000) if over.any() else 0.,
                         underfill_max_mm=float(-reference[under].min()*1000) if under.any() else 0.,
                         worst_overfill_point_mm=(points[over][reference[over].argmax()]*1000).tolist()
                         if over.any() else None,
                         worst_underfill_point_mm=(points[under][reference[under].argmin()]*1000).tolist()
                         if under.any() else None))
        truth.append(reference)
        proxies.append(distance)
    reference = np.minimum(*truth).reshape(len(x), len(y), len(z))[:, 1, :]
    candidate = np.minimum(*proxies).reshape(len(x), len(y), len(z))[:, 1, :]
    return dict(jaw_angle_rad=.22, grid_step_mm=.5, y_planes_mm=[-7.5, 0, 7.5],
                scope='sampled occupancy differences in the distal finger region; not a full CAD error certificate',
                bodies=rows), (x, z, reference, candidate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    reference = make_model()
    candidate = make_model(collision_builder=replace_gripper_collision_menagerie, query_point=True)
    same = {name: bool(np.array_equal(getattr(reference, name), getattr(candidate, name)))
            for name in ['body_mass', 'body_inertia', 'jnt_axis', 'jnt_range', 'site_pos', 'site_quat',
                         'actuator_gainprm', 'actuator_biasprm', 'actuator_forcerange']}
    assert all(same.values()), same
    geometry, arrays = geometry_check(candidate)
    # Clear the author proxy's deliberate fingertip protrusion; no tool state
    # edits occur after reset. Existing SDF reset states must not be reused.
    candidate.qpos0[candidate.joint('driver_free').qposadr[0]] += .0006
    started = time.monotonic()
    sequence = run_sequence(candidate, off_tick=152)
    sequence['elapsed_s'] = time.monotonic()-started
    report = dict(unchanged=same, geometry=geometry, initial_x_correction_mm=.6,
                  sequence=sequence, status='optional approximate geometry; frozen pilot remains SDF')
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x, z, truth, proxy = arrays
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    extent = [x[0]*1000, x[-1]*1000, z[0]*1000, z[-1]*1000]
    for axis, field, title in zip(axes, [truth < 0, proxy < 0, (proxy < 0).astype(int)-(truth < 0).astype(int)],
                                   ['Original CAD', 'Pinned Menagerie proxy', 'Proxy − CAD occupancy']):
        axis.imshow(field.T, origin='lower', extent=extent, cmap='coolwarm', vmin=-1, vmax=1)
        axis.set(title=title, xlabel='gripper-frame X (mm)', ylabel='gripper-frame Z (mm)')
    fig.suptitle('Jaw opening at q=0.22 rad, Y=0; 0.5 mm sampled section')
    fig.savefig(args.out.with_suffix('.png'), dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
