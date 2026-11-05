"""Isolated, unqualified MJWarp port of this task's native contact plugins.

Nothing imports this module from the production environment or learner. The
native equations are retained; MJWarp state/contact arithmetic uses float32.
Explicit qualification is required before using this backend for training.
"""
from pathlib import Path
import struct
import re
import numpy as np
import mujoco
import warp as wp
import mujoco_warp as mjw
from mujoco_warp._src import collision_sdf, io
from mujoco_warp._src.types import vec_pluginattr

wp.set_module_options({'enable_backward': False})


def _grid_snippet():
    # Preserve the native trilinear interpolation and gradient body verbatim.
    source = (Path(__file__).resolve().parents[2]/'plugin/assemblenist_sdf.cc').read_text()
    source = source.split('double grid_eval(')[1].split('const char* kGridNames')[0]
    body = source[source.index('{')+1:source.rfind('}')]
    body = body.replace('std::clamp', 'wp::clamp').replace('std::sqrt', 'sqrt').replace('std::floor', 'floor')
    body = body.replace('h2f(G->v[', 'float(G->v[')
    body = body.replace('return d + out;',
                        'return wp::vec_t<4, float>(float(d+out), float(grad[0]), float(grad[1]), float(grad[2]));')
    prefix = r"""
    auto bits = [&](int start) {
      uint64_t result = 0;
      for (int k=0; k<4; ++k) result |= uint64_t(a[start+k]) << (16*k);
      return result;
    };
    auto number = [&](int start) { union { uint64_t u; double d; } value; value.u=bits(start); return value.d; };
    struct Grid { int nx,ny,nz; double ox,oy,oz,h; const float* v; };
    const Grid grid{int(a[1]), int(a[2]), int(a[3]), number(12), number(16), number(20), number(24),
                    reinterpret_cast<const float*>(bits(8))};
    const Grid* G = &grid;
    double grad[3];
    """
    snippet = prefix + body
    for dtype in ('int', 'float', 'double'):
        snippet = snippet.replace(dtype+'(', 'static_cast<'+dtype+'>(')
    return snippet


@wp.func_native(_grid_snippet())
def _grid(p: wp.vec3, a: vec_pluginattr) -> wp.vec4: ...

def _analytic_snippet():
    root = Path(__file__).resolve().parents[2]
    source = (root/'plugin/assemblenist_sdf.cc').read_text()
    source = source[source.index('inline double frac('):source.index('// Grid SDF (connectors)')]
    # Attribute declarations are not needed inside the local device math class.
    start = source.index('// attribute order (bolt)')
    stop = source.index('// helix coordinates:')
    source = source[:start]+source[stop:]
    source = source.replace('kTan60', '1.7320508075688772').replace('kPi', '3.14159265358979323846')
    head = (root/'plugin/so101_driver.cc').read_text().split('double box(')[1].split('\n}\n\nmjPLUGIN_LIB_INIT')[0]
    source += '\ndouble box('+head
    source = source.replace('mjtNum', 'double')
    source = source.replace('std::max({x,y,z})', 'wp::max(wp::max(x,y),z)')
    for name in ('min', 'max', 'clamp'):
        source = source.replace('std::'+name, 'wp::'+name)
    for name in ('floor', 'sqrt', 'fabs', 'atan2', 'pow', 'hypot', 'abs'):
        source = source.replace('std::'+name, name)
    source = re.sub(r'^(?:inline )?double (\w+)\(', r'CUDA_CALLABLE static double \1(', source, flags=re.M)
    return ('struct Math {\n'+source+'\n};\n'+r"""
      double q[3] = {p[0],p[1],p[2]}, attrs[9], g[3] = {0,0,0};
      for (int i=0;i<9;++i) attrs[i]=a[i];
      double d = 0;
      if(kind==1) d=Math::bolt_eval(q,attrs,g);
      if(kind==2) d=Math::nut_eval(q,attrs,g);
      if(kind==3) {
        d=Math::head(q);
        for(int i=0;i<3;++i) {double r[3]={q[0],q[1],q[2]};r[i]+=1e-7;g[i]=(Math::head(r)-d)/1e-7;}
      }
      return wp::vec_t<4,float>(static_cast<float>(d),static_cast<float>(g[0]),static_cast<float>(g[1]),static_cast<float>(g[2]));
    """)


@wp.func_native(_analytic_snippet())
def _analytic(p: wp.vec3, a: vec_pluginattr, kind: int) -> wp.vec4: ...

@wp.func
def user_sdf(p: wp.vec3, a: vec_pluginattr, kind: int) -> float:
    if kind == 0:
        return _grid(p, a)[0]
    return _analytic(p, a, kind)[0]


@wp.func
def user_sdf_grad(p: wp.vec3, a: vec_pluginattr, kind: int) -> wp.vec3:
    v = wp.vec4()
    if kind == 0:
        v = _grid(p, a)
    else:
        v = _analytic(p, a, kind)
    return wp.vec3(v[1], v[2], v[3])


@wp.kernel
def _driver(qpos: wp.array2d[float], qvel: wp.array2d[float], passive: wp.array2d[float],
            forward_q: int, reverse_q: int, spindle_v: int):
    w = wp.tid()
    forward = qpos[w, forward_q] > .0006
    reverse = qpos[w, reverse_q] > .0006
    if forward != reverse:
        target = 26.17993878 if forward else -26.17993878
        passive[w, spindle_v] += wp.clamp(.002*(target-qvel[w, spindle_v]), -.05, .05)


@wp.kernel
def query_sdf(points: wp.array[wp.vec3], attrs: wp.array[vec_pluginattr], kinds: wp.array[int],
              output: wp.array[wp.vec4]):
    i = wp.tid()
    g = user_sdf_grad(points[i], attrs[i], kinds[i])
    output[i] = wp.vec4(user_sdf(points[i], attrs[i], kinds[i]), g[0], g[1], g[2])


class WarpPhysics:
    """Owns GPU model, immutable grids, and per-step passive driver callback.

    Hook installation is process-global in MJWarp 3.17; use an isolated process.
    """
    def __init__(self, native_model, native_data=None, nworld=1, device='cuda:0', nconmax=512, njmax=2048):
        from .scene import ASSETS
        self.native = native_model
        self.device = device
        self.grids = {}
        self.driver_indices = None
        self.plugin_audit = []
        collision_sdf.user_sdf = user_sdf
        collision_sdf.user_sdf_grad = user_sdf_grad
        # Fail closed before MJWarp drops non-geometry plugins.
        kinds = {}
        geom_instances = set(int(i) for i in native_model.geom_plugin if i >= 0)
        known_names = {'base_sdf': 0, 'male': 1, 'female': 2, 'screw_head': 3,
                       'button_motor': 'passive_driver'}
        known_slots = {}
        for instance in range(native_model.nplugin):
            name = mujoco.mj_id2name(native_model, mujoco.mjtObj.mjOBJ_PLUGIN, instance)
            if name in known_names:
                known_slots[int(native_model.plugin[instance])] = known_names[name]
        for instance in range(native_model.nplugin):
            name = mujoco.mj_id2name(native_model, mujoco.mjtObj.mjOBJ_PLUGIN, instance)
            kind = known_slots.get(int(native_model.plugin[instance]))
            if kind in (0, 1, 2, 3):
                kinds[instance] = kind
            elif kind == 'passive_driver':
                if self.driver_indices is not None:
                    raise ValueError('Multiple driver plugins require an explicit callback mapping')
                start = native_model.plugin_attradr[instance]
                end = (native_model.plugin_attradr[instance+1] if instance+1 < native_model.nplugin
                       else len(native_model.plugin_attr))
                joints = [s.decode() for s in bytes(native_model.plugin_attr[start:end]).split(b'\0') if s]
                if len(joints) != 3:
                    raise ValueError('Driver plugin must specify forward, reverse and spindle joints')
                ids = [native_model.joint(j).id for j in joints]
                self.driver_indices = (int(native_model.jnt_qposadr[ids[0]]),
                                       int(native_model.jnt_qposadr[ids[1]]),
                                       int(native_model.jnt_dofadr[ids[2]]))
            else:
                raise ValueError(f'Unsupported native plugin instance: {name!r}')
            if instance in geom_instances and instance not in kinds:
                raise ValueError(f'Non-SDF plugin attached to geom: {name}')
            self.plugin_audit.append(dict(instance=instance, name=name,
                                          kind=kinds.get(instance, 'passive_driver')))
        with wp.ScopedDevice(device):
            # MJWarp 3.17 filters plugin arrays to geom instances but retains
            # the unfiltered allocation count. Fix only these two allocations;
            # restore the library helper immediately, even if import fails.
            create_array = io._create_array
            def filtered_plugin_array(data, spec, sizes, batch_size=1):
                if 'nplugin' in spec.shape and data is not None:
                    sizes = {**sizes, 'nplugin': len(data)}
                return create_array(data, spec, sizes, batch_size)
            io._create_array = filtered_plugin_array
            try:
                self.model = mjw.put_model(native_model)
            finally:
                io._create_array = create_array
            self.model.nplugin = len(self.model.plugin)
            # put_model clamps this to 1e-6 for float32; retain the native
            # setting explicitly and report convergence limits in qualification.
            self.model.opt.tolerance.fill_(float(native_model.opt.tolerance))
            self.parameter_audit = {}
            fields = ('qpos0','qpos_spring','body_parentid','body_mass','body_inertia','body_pos','body_quat',
                      'dof_damping','dof_frictionloss','dof_armature','jnt_range','jnt_axis','jnt_limited',
                      'geom_type','geom_contype','geom_conaffinity','geom_bodyid','geom_size','geom_pos',
                      'geom_quat','geom_friction','geom_solref','geom_solimp','geom_condim',
                      'actuator_trntype','actuator_trnid','actuator_gainprm','actuator_biasprm',
                      'actuator_ctrlrange','actuator_forcerange','actuator_dyntype','actuator_dynprm',
                      'actuator_ctrllimited','actuator_forcelimited','geom_margin','geom_gap',
                      'pair_geom1','pair_geom2','pair_dim','pair_friction','pair_solref','pair_solimp',
                      'exclude_signature','eq_type','eq_obj1id','eq_obj2id','eq_active0')
            for field in fields:
                actual = getattr(self.model, field).numpy()
                expected = np.asarray(getattr(native_model, field), dtype=actual.dtype)
                same = np.array_equal(actual, np.broadcast_to(expected, actual.shape))
                self.parameter_audit[field] = same
                if not same:
                    raise ValueError(f'MJWarp changed mechanical parameter {field}')
            for field in ('timestep','tolerance','ls_tolerance','gravity','wind','density','viscosity'):
                actual = getattr(self.model.opt, field).numpy()
                expected = np.asarray(getattr(native_model.opt, field), dtype=actual.dtype)
                if not np.array_equal(actual, np.broadcast_to(expected, actual.shape)):
                    raise ValueError(f'MJWarp changed option {field}')
            for field in ('integrator','cone','solver','iterations','ls_iterations','sdf_initpoints','sdf_iterations',
                          'disableflags','enableflags'):
                if int(getattr(self.model.opt, field)) != int(getattr(native_model.opt, field)):
                    raise ValueError(f'MJWarp changed option {field}')
            attrs = self.model.plugin_attr.numpy()
            plugin_kinds = []
            for row, gid in enumerate(np.flatnonzero(native_model.geom_plugin >= 0)):
                instance = int(native_model.geom_plugin[gid])
                kind = kinds[instance]
                plugin_kinds.append(kind)
                if kind == 0:
                    grid_id = int(attrs[row, 0])
                    if grid_id not in self.grids:
                        path = ASSETS/'sdf'/f'grid_{grid_id}.bin'
                        with path.open('rb') as f:
                            n = np.fromfile(f, np.int32, 3)
                            oh = np.fromfile(f, np.float64, 4)
                            values = np.fromfile(f, np.float16).astype(np.float32)
                        if len(values) != int(np.prod(n)) or not np.isfinite(values).all():
                            raise ValueError(f'Invalid grid: {path}')
                        self.grids[grid_id] = (n, oh, wp.array(values, dtype=float, device=device))
                    n, oh, array = self.grids[grid_id]
                    chunks = [(array.ptr >> (16*k)) & 65535 for k in range(4)]
                    if sum(int(np.float32(v)) << (16*k) for k, v in enumerate(chunks)) != array.ptr:
                        raise AssertionError('Grid pointer packing lost precision')
                    attrs[row, 1:4] = n
                    attrs[row, 4:8] = oh
                    attrs[row, 8:12] = chunks
                    # Preserve original float64 grid origins/spacing at boundaries.
                    for k, value in enumerate(oh):
                        bits = struct.unpack('<Q', struct.pack('<d', value))[0]
                        attrs[row, 12+4*k:16+4*k] = [(bits >> (16*j)) & 65535 for j in range(4)]
            self.model.plugin = wp.array(plugin_kinds, dtype=int, device=device)
            self.model.plugin_attr = wp.array(attrs, dtype=vec_pluginattr, device=device)
            if self.driver_indices is not None:
                def passive(m, d):
                    wp.launch(_driver, dim=d.nworld,
                              inputs=[d.qpos, d.qvel, d.qfrc_passive, *self.driver_indices], device=device)
                self.model.callback.passive = passive
            if native_data is None:
                native_data = mujoco.MjData(native_model)
            self.data = mjw.put_data(native_model, native_data, nworld=nworld,
                                     nconmax=nconmax, njmax=njmax)
            for field in ('qpos','qvel','ctrl','qfrc_applied','xfrc_applied'):
                actual = getattr(self.data, field).numpy()
                expected = np.asarray(getattr(native_data, field), dtype=actual.dtype)
                if not np.array_equal(actual, np.broadcast_to(expected, actual.shape)):
                    raise ValueError(f'MJWarp changed initial state {field}')
        self.graphs = {}

    def step(self, steps=1, graph=False):
        with wp.ScopedDevice(self.device):
            if not graph:
                for _ in range(steps):
                    mjw.step(self.model, self.data)
            else:
                if steps not in self.graphs:
                    # Kernel compilation/allocation must happen before capture.
                    with wp.ScopedCapture(device=self.device) as capture:
                        for _ in range(steps):
                            mjw.step(self.model, self.data)
                    self.graphs[steps] = capture.graph
                wp.capture_launch(self.graphs[steps])

    def arrays(self):
        wp.synchronize_device(self.device)
        return dict(qpos=self.data.qpos.numpy(), qvel=self.data.qvel.numpy(),
                    time=self.data.time.numpy(), ctrl=self.data.ctrl.numpy())

    def close(self):
        wp.synchronize_device(self.device)
        self.graphs.clear()
        self.model.callback.passive = None
        self.data = None
        self.model = None
        self.native = None
        self.grids.clear()
