"""Nominal M3 fastener supplement to Hiwonder's incomplete assembly CAD.

Hiwonder's kit manual specifies an M3 x 5 x 7 washer-head screw. Interpreted
as 3 mm diameter, 5 mm under-head length, 7 mm flange. Coarse 0.5 mm pitch
and 60 um radial simulation clearance are engineering assumptions, not measurements.
Uses AssembleNIST's existing metric helical SDF implementation.
"""
import numpy as np

PITCH = .0005
LENGTH = .005
CLEARANCE = .00006
AXIS = np.array([.00000172, -.007227, .06409296])
HORN_LIFT = .0011  # correct the supplied assembled pose, preserving the mesh


def origins(meta):
    pos = {n:np.asarray(p['origin']).copy() for n,p in meta.items()}
    pos['horn'][2] += HORN_LIFT
    pos['yoke'][2] += HORN_LIFT
    top = pos['horn'][2] + meta['horn']['bounds'][1][2]
    pos['screw'] = np.array([AXIS[0], AXIS[1], top])  # under-head bearing plane
    return pos


def attributes():
    rp=.0015-.3248*PITCH
    bolt=dict(pitch=PITCH,rp=rp,root=.00119,crest=.0015,thread_z0=0,thread_z1=LENGTH,
              head_flat=.0035,head_top=0,phase=0)
    nut=dict(pitch=PITCH,rp=rp,clearance=CLEARANCE,minor=.0012311,major=.00159,
             flat=.0020,z0=0,z1=.006,phase=0)
    return bolt,nut
