"""Optional MJWarp rollout adapter, explicitly selected by training configuration.

The native SO101 environment still computes observations, costs, termination and
success at control endpoints. GPU physics retains the fine collision surfaces;
the legacy candidate excludes two analytically separated body pairs. The explicit
whole-helper SDF variant imports its supplied model directly with no additional
exclusions. A per-step domain guard applies only to the legacy separation proof.
"""
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from .env import BatchedAssemblyEnv, EnvConfig, STATE_SPEC
from . import scene
from .warp_physics import WarpPhysics

GPU_SOLVER_TOLERANCE = 1.e-6  # MJWarp's documented float32 importer floor.
GPU_SOLVER_ITERATIONS = 200
ENDPOINT_THREADS = 8


class MixedPhysics(WarpPhysics):
    """Explicit mixed SDF arithmetic, with the unchanged FP64 reference kept."""
    def __init__(self, *args, **kwargs):
        from . import warp_physics
        from scripts import so101_gpu_fp32
        original = warp_physics.user_sdf, warp_physics.user_sdf_grad
        try:
            warp_physics.user_sdf = so101_gpu_fp32.user_sdf
            warp_physics.user_sdf_grad = so101_gpu_fp32.user_sdf_grad
            super().__init__(*args, **kwargs)
        finally:
            warp_physics.user_sdf, warp_physics.user_sdf_grad = original


@wp.kernel
def _record(qpos: wp.array2d[float], qvel: wp.array2d[float], clock: wp.array[float],
            sensors: wp.array2d[float], states: wp.array3d[float], sensor_out: wp.array3d[float],
            index: int):
    world, field = wp.tid()
    if field == 0:
        states[world, index, 0] = clock[world]
    elif field <= qpos.shape[1]:
        states[world, index, field] = qpos[world, field-1]
    else:
        states[world, index, field] = qvel[world, field-1-qpos.shape[1]]
    if field < sensors.shape[1]:
        sensor_out[world, index, field] = sensors[world, field]


@wp.kernel
def _control(control: wp.array3d[float], current: wp.array2d[float], index: int):
    world, actuator = wp.tid()
    current[world, actuator] = control[world, index, actuator]


@wp.kernel
def _contact_minimum(dist: wp.array[float], world: wp.array[int], count: wp.array[int],
                     minimum: wp.array2d[float], index: int):
    contact = wp.tid()
    if contact < count[0]:
        wp.atomic_min(minimum, world[contact], index, dist[contact])


def separated_model(native, out):
    """Recompute the geometric proof from the actual model before exclusions."""
    from scripts.so101_gpu_profile import separating_slab
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    proof = separating_slab(native, mujoco.MjData(native), out)
    if min(b['separating_distance_lower_bound_m'] for s in proof['sides'] for b in s['bounds']) < .001:
        raise ValueError('Fine-jaw separation proof has insufficient margin')
    root = ET.fromstring(scene.build_xml(contact_monitor=bool(native.nsensordata)))
    contact = root.find('contact')
    if contact is None:
        contact = ET.SubElement(root, 'contact')
    for side in proof['sides']:
        ET.SubElement(contact, 'exclude', body1=side['wrist_body'], body2=side['moving_body'])
    xml = ET.tostring(root, encoding='unicode')
    (out/'gpu_scene.xml').write_text(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    # Rebuilding from the canonical XML is intentional, but must never discard
    # a caller's runtime mass/controller/layout/sensor changes silently. Compare
    # every compiled numeric array and all options, allowing only the two proved
    # exclusions and the name-table bookkeeping their addition changes.
    changed = []
    for name in dir(native):
        if name.startswith('_') or name.startswith('name_') or name in ('names_map','exclude_signature'):
            continue
        value = getattr(native,name)
        if isinstance(value,np.ndarray) and not np.array_equal(value,getattr(model,name)):
            changed.append(name)
    for name in dir(native.opt):
        if name.startswith('_'): continue
        value=getattr(native.opt,name)
        if not callable(value) and not np.array_equal(value,getattr(model.opt,name)):
            changed.append('opt.'+name)
    expected=set(map(int,native.exclude_signature))
    for side in proof['sides']:
        a,b=sorted((native.body(side['wrist_body']).id,native.body(side['moving_body']).id))
        expected.add((a<<16)+b)
    if set(map(int,model.exclude_signature)) != expected:
        changed.append('unexpected exclusions')
    # Names are semantic references used by controls, observations and rewards.
    for count,kind in (('nbody',mujoco.mjtObj.mjOBJ_BODY),('njnt',mujoco.mjtObj.mjOBJ_JOINT),
                       ('ngeom',mujoco.mjtObj.mjOBJ_GEOM),('nsite',mujoco.mjtObj.mjOBJ_SITE),
                       ('nu',mujoco.mjtObj.mjOBJ_ACTUATOR),('nsensor',mujoco.mjtObj.mjOBJ_SENSOR),
                       ('nplugin',mujoco.mjtObj.mjOBJ_PLUGIN)):
        if getattr(native,count)!=getattr(model,count) or any(
            mujoco.mj_id2name(native,kind,i)!=mujoco.mj_id2name(model,kind,i) for i in range(getattr(native,count))):
            changed.append(count+' names')
    if changed:
        raise ValueError('GPU canonical rebuild would change native model: '+', '.join(changed))
    return model, proof


class WarpRollout:
    """The FULLPHYSICS/CTRL subset of native rollout used by the SO101 adapter.

    Every control block resets GPU data and zeroes solver warm starts, matching
    native Rollout's default reset behavior. Native endpoint mj_forward is kept;
    no extra native forward call occurs inside the fine-step collection loop.

    Raw GPU time is validated against repeated float32 timestep additions at
    every fine step. Only then is the returned API clock replaced by exact
    float64 start + integer step * dt, preserving existing environment semantics.
    """
    def __init__(self, native, worlds, proof_dir=None, *, model_variant="separated_jaws"):
        if native.na or native.npluginstate or native.nmocap or native.neq:
            raise ValueError('This adapter only supports stateless plugins and no activation/mocap/equalities')
        if native.nsensordata not in (0, 1):
            raise ValueError('Only the SO101 optional minimum-contact sensor is qualified')
        if native.nsensordata == 1 and (native.nsensor != 1 or
            native.sensor(0).name != 'min_contact_distance' or
            native.sensor_type[0] != mujoco.mjtSensor.mjSENS_CONTACT or
            not np.array_equal(native.sensor_intprm[0,:2],[8,1]) or
            native.sensor_objid[0] != -1 or native.sensor_refid[0] != -1):
            raise ValueError('Expected the unfiltered global minimum-distance contact sensor')
        if native.nsensordata == 0 and native.nsensor:
            raise ValueError('Unexpected zero-dimensional sensor')
        self.native = native
        self.worlds = worlds
        self.nstate = mujoco.mj_stateSize(native, STATE_SPEC)
        if self.nstate != 1+native.nq+native.nv:
            raise ValueError('Unmapped FULLPHYSICS fields')
        if model_variant == 'whole_helper_sdf':
            active_mesh = ((native.geom_type == mujoco.mjtGeom.mjGEOM_MESH) &
                           ((native.geom_contype | native.geom_conaffinity) != 0))
            if active_mesh.any():
                raise ValueError('Whole-helper SDF variant contains active mesh colliders')
            plugin_names = [mujoco.mj_id2name(native,mujoco.mjtObj.mjOBJ_PLUGIN,i)
                            for i in range(native.nplugin)]
            helpers = [name for name in plugin_names if name.startswith('helper_sdf_')]
            if len(helpers) != 11:
                raise ValueError('Whole-helper SDF variant requires all eleven helper grids')
            self.gpu_native = native
            self.proof = dict(mode='direct whole-helper SDF import', added_exclusions=[],
                              original_exclusions=native.exclude_signature.tolist())
        elif model_variant == 'separated_jaws':
            self.gpu_native, self.proof = separated_model(native, proof_dir or scene.ROOT/'artifacts_local/warp_env/proof')
        else:
            raise ValueError('Unknown explicit GPU collision variant')
        self.model_variant = model_variant
        self.proof_dir = Path(proof_dir) if proof_dir is not None else None
        self.physics = MixedPhysics(self.gpu_native, nworld=worlds)
        # At the native float64 tolerance (1e-8), a saved policy contact cycles
        # through iterations 3..200 in float32. Use MJWarp's importer floor as
        # an explicit GPU-only numerical deviation; never suppress exhaustion.
        self.physics.model.opt.tolerance.fill_(GPU_SOLVER_TOLERANCE)
        self.physics.model.opt.iterations = GPU_SOLVER_ITERATIONS
        self.physics.model.opt.ls_iterations = 100
        # The global minimum-distance sensor must consider every allocated
        # contact per world. MJWarp defaults to only 64 matches and flags
        # truncation; using the physics contact capacity preserves its meaning.
        self.physics.model.opt.contact_sensor_maxmatch = 512
        self.options = dict(native_iterations=int(native.opt.iterations), gpu_iterations=GPU_SOLVER_ITERATIONS,
                            native_ls_iterations=int(native.opt.ls_iterations), gpu_ls_iterations=100,
                            contact_sensor_maxmatch=512,
                            native_tolerance=float(native.opt.tolerance), gpu_tolerance=GPU_SOLVER_TOLERANCE,
                            sdf_precision='mixed',
                            model_variant=model_variant, added_exclusions=len(self.proof.get('sides', [])),
                            state_precision='float32', api_clock='float64 after per-step float32 clock validation')
        self.domains = []
        for side in (('left', 'right') if model_variant == 'separated_jaws' else ()):
            joint = native.joint(side+'_gripper')
            self.domains.append((int(joint.qposadr[0]), float(joint.range[0]-.05), float(joint.range[1]+.05)))
        self.buffers = {}
        self.last_raw_clock = None
        self.last_overflow = None
        self.calls = 0
        self.warmed = False
        self.last_sensor_audit_error = None
        self.last_contact_depth = None

    def _validate_domain(self, qpos):
        for index, low, high in self.domains:
            if ((qpos[..., index] < low) | (qpos[..., index] > high)).any():
                raise FloatingPointError('GPU trajectory left the proven wrist/jaw separation domain')

    def rollout(self, model, work, initial_state, control, nstep):
        if model is not self.native:
            raise ValueError('Rollout model changed after GPU import')
        state = np.asarray(initial_state, dtype=np.float64)
        control = np.asarray(control, dtype=np.float64)
        if state.shape != (self.worlds, self.nstate) or control.shape != (self.worlds, nstep, model.nu):
            raise ValueError('Unexpected rollout shape')
        if not np.isfinite(state).all() or not np.isfinite(control).all():
            raise FloatingPointError('Nonfinite GPU rollout input')
        self._validate_domain(state[:, 1:1+model.nq])
        p = self.physics
        with wp.ScopedDevice(p.device):
            def restore():
                mjw.reset_data(p.model, p.data)
                p.data.time.assign(state[:, 0].astype(np.float32))
                p.data.qpos.assign(state[:, 1:1+model.nq].astype(np.float32))
                p.data.qvel.assign(state[:, 1+model.nq:].astype(np.float32))
            restore()
            if not self.warmed:
                # Load kernels before capture, then discard the compile-only
                # step through the same complete reset as every real block.
                p.data.ctrl.assign(control[:, 0].astype(np.float32))
                mjw.step(p.model, p.data)
                wp.synchronize_device(p.device)
                wp.load_module(module=__name__, device=p.device)
                restore()
                self.warmed = True
            if nstep not in self.buffers:
                inputs = wp.zeros((self.worlds, nstep, model.nu), dtype=float, device=p.device)
                outputs = wp.zeros((self.worlds, nstep, self.nstate), dtype=float, device=p.device)
                sensors = wp.zeros((self.worlds, nstep, model.nsensordata), dtype=float, device=p.device)
                minima = wp.full((self.worlds, nstep), 1.e10, dtype=float, device=p.device)
                with wp.ScopedCapture(device=p.device) as capture:
                    for index in range(nstep):
                        wp.launch(_control, dim=(self.worlds, model.nu), inputs=[inputs, p.data.ctrl, index], device=p.device)
                        mjw.step(p.model, p.data)
                        if model.nsensordata:
                            wp.launch(_contact_minimum, dim=p.data.naconmax,
                                      inputs=[p.data.contact.dist, p.data.contact.worldid,
                                              p.data.nacon, minima, index], device=p.device)
                        wp.launch(_record, dim=(self.worlds, self.nstate),
                                  inputs=[p.data.qpos, p.data.qvel, p.data.time, p.data.sensordata,
                                          outputs, sensors, index], device=p.device)
                self.buffers[nstep] = inputs, outputs, sensors, minima, capture.graph
            inputs, outputs, sensors, minima, graph = self.buffers[nstep]
            inputs.assign(control.astype(np.float32))
            minima.fill_(1.e10)
            wp.capture_launch(graph)
            raw = outputs.numpy()
            readings = sensors.numpy()
            overflow = p.data.overflow.numpy()
            if model.nsensordata:
                measured = minima.numpy()
                measured[measured == np.float32(1.e10)] = 0
                self.last_sensor_audit_error = float(abs(measured-readings[:, :, 0]).max())
                self.last_contact_depth = -readings[:, :, 0].min(axis=1)
        self.last_raw_clock = raw[:, :, 0].copy()
        self.last_overflow = overflow.copy()
        if overflow.any() and self.proof_dir is not None:
            import json
            self.proof_dir.mkdir(parents=True, exist_ok=True)
            path = self.proof_dir / f'failure_{self.calls:06d}'
            np.savez_compressed(path.with_suffix('.npz'), initial_state=state, control=control,
                                trajectory=raw, sensors=readings, overflow=overflow)
            path.with_suffix('.json').write_text(json.dumps(dict(options=self.options,
                completed_control_blocks=self.calls, overflow=overflow.tolist()), indent=2)+'\n')
        if model.nsensordata and self.last_sensor_audit_error > 1.e-7:
            raise FloatingPointError('GPU contact sensor disagrees with independently reduced contact distances')
        if not np.isfinite(raw).all() or not np.isfinite(readings).all():
            raise FloatingPointError('Nonfinite GPU physics trajectory or contact sensor')
        if overflow.any():
            raise FloatingPointError(f'MJWarp overflow or convergence flags: {np.unique(overflow).tolist()}')
        expected = np.empty((self.worlds, nstep), dtype=np.float32)
        clock = state[:, 0].astype(np.float32)
        for index in range(nstep):
            clock = clock+np.float32(model.opt.timestep)
            expected[:, index] = clock
        if not np.array_equal(raw[:, :, 0], expected):
            raise FloatingPointError('MJWarp raw simulation clock deviated from repeated float32 additions')
        self._validate_domain(raw[:, :, 1:1+model.nq])
        output = raw.astype(np.float64)
        output[:, :, 0] = state[:, None, 0]+np.arange(1, nstep+1)*model.opt.timestep
        self.calls += 1
        return output, readings.astype(np.float64)

    def close(self):
        self.buffers.clear()
        self.physics.close()


def archived_helper_factory(path):
    """Return an explicit archived model factory with streamed provenance checks.

    The paired JSON must come from the sensor-only candidate qualification. No
    XML recompilation or collision filtering is performed by this factory.
    """
    import hashlib
    import json
    path = Path(path).resolve()
    report = json.loads(path.with_suffix('.json').read_text())
    def digest(filename):
        with Path(filename).open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    if not report.get('passed') or not report.get('mechanical_arrays_options_names_unchanged'):
        raise ValueError('Expected qualified whole-helper sensor model metadata')
    if digest(path) != report['model_sha256']:
        raise ValueError('Archived helper model hash mismatch')
    for name, expected in report['plugin_sha256'].items():
        if digest(scene.ROOT/'plugin/build'/name) != expected:
            raise ValueError('Native physics plugin differs from archived helper model')
    manifest_path = scene.ROOT/'assets_local/so101_helper_collision/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if digest(manifest_path) != report.get('helper_manifest_sha256') or not report.get('grid_sha256'):
        raise ValueError('Helper archive is missing or differs from its frozen grid provenance')
    for filename, expected in report['grid_sha256'].items():
        if digest(scene.ROOT/filename) != expected:
            raise ValueError('Referenced collision grid differs from the qualified helper archive')
    for record in manifest['meshes'].values():
        if digest(scene.ROOT/record['grid']) != record['grid_sha256']:
            raise ValueError('Helper grid differs from its qualification manifest')
    def factory(*, contact_monitor=False):
        if not contact_monitor:
            raise ValueError('Qualified whole-helper archive includes the required contact monitor')
        scene.load_plugin()
        model = mujoco.MjModel.from_binary_path(str(path))
        if model.nsensordata != 1 or model.sensor(0).name != 'min_contact_distance':
            raise ValueError('Archived model lacks the global contact monitor')
        if model.exclude_signature.tolist() != report['exclude_signature']:
            raise ValueError('Archived model collision exclusions changed')
        return model
    factory.provenance = dict(report, model_path=str(path), helper_manifest_sha256=digest(manifest_path))
    return factory


def physics_identity(factory, endpoint_threads=ENDPOINT_THREADS):
    """Pin the optional runtime separately from the canonical reset-bank model."""
    import hashlib
    from importlib.metadata import version
    provenance=factory.provenance
    files=('assembly_policy/so101/env.py','assembly_policy/so101/warp_env.py','assembly_policy/so101/warp_physics.py',
           'scripts/so101_gpu_fp32.py')
    return dict(backend='whole_helper_sdf',model_sha256=provenance['model_sha256'],
        endpoint_threads=endpoint_threads,
        solver=dict(gpu_tolerance=GPU_SOLVER_TOLERANCE,gpu_iterations=GPU_SOLVER_ITERATIONS,
                    gpu_ls_iterations=100,contact_sensor_maxmatch=512),
        helper_manifest_sha256=provenance['helper_manifest_sha256'],
        plugin_sha256=provenance['plugin_sha256'],
        implementation={name:hashlib.sha256((scene.ROOT/name).read_bytes()).hexdigest() for name in files},
        versions={name:version(name) for name in ('mujoco','mujoco-warp','warp-lang')})


class BatchedWarpAssemblyEnv(BatchedAssemblyEnv):
    """Opt-in prototype using the unchanged native environment's public methods."""
    def __init__(self, task_names, task_of_env, ecfg=None, proof_dir=None, *,
                 model_factory=None, model_variant="separated_jaws", endpoint_threads=ENDPOINT_THREADS):
        if endpoint_threads not in (1, ENDPOINT_THREADS):
            raise ValueError('Endpoint refresh supports serial or eight workers')
        self._endpoint_pool = None
        cfg = ecfg or EnvConfig()
        if model_variant == 'whole_helper_sdf' and model_factory is None:
            raise ValueError('Whole-helper GPU variant requires an explicit qualified model factory')
        super().__init__(task_names, task_of_env, replace(cfg, nthread=1), model_factory=model_factory)
        self.engine.close()
        self.engine = None
        self.ecfg = cfg
        try:
            self.engine = WarpRollout(self.m, self.N, proof_dir=proof_dir, model_variant=model_variant)
            self.engine.options['endpoint_threads'] = endpoint_threads
            if endpoint_threads > 1:
                self._endpoint_pool = ThreadPoolExecutor(max_workers=endpoint_threads,
                                                         thread_name_prefix='so101-endpoint')
        except BaseException:
            self.close()
            raise

    def _forward_states(self):
        # Each worker owns a different MjData; the shared model is read-only.
        # Waiting for every future also propagates native errors before packing.
        if self._endpoint_pool is None:
            super()._forward_states()
        else:
            list(self._endpoint_pool.map(self._forward_world, range(self.N)))

    def close(self):
        if self._endpoint_pool is not None:
            self._endpoint_pool.shutdown(wait=True)
            self._endpoint_pool = None
        if self.engine is not None:
            self.engine.close()
