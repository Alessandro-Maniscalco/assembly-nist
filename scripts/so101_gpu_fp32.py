"""Optional mixed-precision evaluation of the SDF formulas; isolated probe only.

Decoded grid pointers and original float64 header values remain exact in transit;
the grid coordinate/boundary classification stays float64. Grid interpolation
and analytic SDF equations use float32. This avoids rounding the native upper
cell guard into an out-of-bounds index and changing its boundary-gradient branch.
The native C++/FP64 reference in warp_physics.py is never edited.
"""
import re

import warp as wp
from mujoco_warp._src.types import vec_pluginattr
from assembly_policy.so101 import warp_physics as reference


def _float_literals(code):
    code = re.sub(r'\bdouble\b', 'float', code)
    code = re.sub(r'(?<![\w.])(\d+\.\d*(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+)(?![\w.])', r'\1f', code)
    for function in ('sqrt', 'floor', 'fabs', 'atan2', 'pow', 'hypot'):
        code = re.sub(r'\b'+function+r'\(', function+'f(', code)
    return code


def grid_snippet():
    code = reference._grid_snippet()
    # Only the interpolation values/weights/gradient arithmetic change precision.
    # Header decode, coordinates, cell indices and exterior-normal math stay FP64.
    for name in ('c000', 'c010', 'c001', 'c011', 'tx', 'c00', 'c01', 'c0', 'd', 'gx', 'gy', 'gz'):
        code = code.replace('double '+name+' ', 'float '+name+' ')
    code = code.replace('return static_cast<double>(float(G->v[', 'return static_cast<float>(float(G->v[')
    return code


@wp.func_native(grid_snippet())
def grid(p: wp.vec3, a: vec_pluginattr) -> wp.vec4: ...


@wp.func_native(_float_literals(reference._analytic_snippet()))
def analytic(p: wp.vec3, a: vec_pluginattr, kind: int) -> wp.vec4: ...


@wp.func
def user_sdf(p: wp.vec3, a: vec_pluginattr, kind: int) -> float:
    if kind == 0:
        return grid(p, a)[0]
    return analytic(p, a, kind)[0]


@wp.func
def user_sdf_grad(p: wp.vec3, a: vec_pluginattr, kind: int) -> wp.vec3:
    value = wp.vec4()
    if kind == 0:
        value = grid(p, a)
    else:
        value = analytic(p, a, kind)
    return wp.vec3(value[1], value[2], value[3])


@wp.kernel
def query_sdf(points: wp.array[wp.vec3], attrs: wp.array[vec_pluginattr], kinds: wp.array[int], output: wp.array[wp.vec4]):
    i = wp.tid()
    grad = user_sdf_grad(points[i], attrs[i], kinds[i])
    output[i] = wp.vec4(user_sdf(points[i], attrs[i], kinds[i]), grad[0], grad[1], grad[2])
