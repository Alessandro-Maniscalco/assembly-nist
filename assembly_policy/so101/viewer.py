"""Interactive native MuJoCo viewer; joint controls remain available in its UI."""
import argparse
import time
import mujoco
import mujoco.viewer
from .scene import model
from .robot import home
from .resets import reset
from .env import STATE_SPEC, TASK_NAMES


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--task',choices=TASK_NAMES,default='motor')
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--assembled',action='store_true',help='Initialize a reference-pose inspection, not an assembly demonstration')
    a=ap.parse_args();m=model(a.assembled);d=mujoco.MjData(m);home(m,d)
    if not a.assembled:
        s=reset(m,a.task,a.seed);mujoco.mj_setState(m,d,s,STATE_SPEC);d.ctrl[:]=d.qpos[:12]
    with mujoco.viewer.launch_passive(m,d) as viewer:
        viewer.cam.lookat[:]=[0,.04,.15];viewer.cam.distance=1.05;viewer.cam.azimuth=115;viewer.cam.elevation=-35
        viewer.opt.geomgroup[3]=0
        while viewer.is_running():
            start=time.perf_counter()
            mujoco.mj_step(m,d,nstep=20)
            viewer.sync()
            time.sleep(max(0,20*m.opt.timestep-(time.perf_counter()-start)))


if __name__=='__main__':main()
