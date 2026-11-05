"""Auditable native-physics trajectories for scripted feasibility demonstrations."""
from pathlib import Path
import hashlib
import json
import os
import tempfile
import mujoco
import numpy as np

STATE_SPEC = mujoco.mjtState.mjSTATE_FULLPHYSICS


class Trajectory:
    """Every call to step records one mj_step, including settling and contacts."""
    def __init__(self, model, data):
        self.m, self.d = model, data
        self.states, self.controls, self.depths, self.pairs, self.stages = [], [], [], [], []
        self.labels = []
        self.start_time = float(data.time)
        self.max_clock_error = 0.
        self.warning_start = np.array(data.warning.number).copy()
        self.initial_state = self.state()
        from .scene import ROOT
        self.plugin_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (ROOT/'plugin/build').glob('*.so')}

    def state(self):
        state = np.empty(mujoco.mj_stateSize(self.m, STATE_SPEC))
        mujoco.mj_getState(self.m, self.d, state, STATE_SPEC)
        return state

    def step(self, label):
        if label not in self.labels:
            self.labels.append(label)
        before = self.d.time
        mujoco.mj_step(self.m, self.d)
        self.max_clock_error = max(self.max_clock_error, abs(self.d.time-before-self.m.opt.timestep))
        if self.max_clock_error > 1e-8 or not np.isfinite(self.d.qpos).all() or not np.isfinite(self.d.qvel).all():
            raise RuntimeError('Simulation clock reset or nonfinite physical state')
        if self.d.ncon:
            k = int(np.argmin(self.d.contact.dist))
            c = self.d.contact[k]
            depth, pair = max(0., -float(c.dist)), [int(c.geom1), int(c.geom2)]
        else:
            depth, pair = 0., [-1,-1]
        self.states.append(self.state())
        self.controls.append(self.d.ctrl.copy())
        self.depths.append(depth)
        self.pairs.append(pair)
        self.stages.append(self.labels.index(label))

    def save(self, path, details=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        states = np.stack(self.states)
        depths = np.asarray(self.depths)
        stages = np.asarray(self.stages, np.int16)
        pairs = np.asarray(self.pairs, np.int32)
        binary = np.empty(mujoco.mj_sizeModel(self.m), np.uint8)
        mujoco.mj_saveModel(self.m, buffer=binary)
        model_path = path.with_suffix('.mjb')
        from .scene import ROOT
        model_hash=hashlib.sha256(binary).hexdigest()
        shared=ROOT/'artifacts_local/.models'/f'{model_hash}.mjb'
        shared.parent.mkdir(parents=True,exist_ok=True)
        if not shared.exists():
            fd,name=tempfile.mkstemp(prefix=f'.{model_hash}.',suffix='.tmp',dir=shared.parent)
            try:
                with os.fdopen(fd,'wb') as stream:stream.write(binary)
                try:os.link(name,shared)
                except FileExistsError:pass  # another process published the same complete model
            finally:os.unlink(name)
        if shared.stat().st_size!=binary.nbytes:raise RuntimeError('Archived model store size mismatch')
        temporary=model_path.with_suffix('.tmp.mjb')
        if temporary.exists():temporary.unlink()
        os.link(shared,temporary);os.replace(temporary,model_path)
        np.savez_compressed(path.with_suffix('.npz'), states=states, controls=np.stack(self.controls),
                            penetration=depths, deepest_geom_pair=pairs, stage=stages,
                            labels=np.asarray(self.labels), initial_state=self.initial_state)
        deepest = pairs[int(np.argmax(depths))]
        def geom_name(g):
            if g < 0: return None
            return self.m.geom(int(g)).name or f'{self.m.body(int(self.m.geom_bodyid[g])).name}/geom{g}'
        report = dict(scripted=True, learned_policy=False, actuator_execution=True,
                      state_spec=int(STATE_SPEC), physics_steps=len(states), timestep=float(self.m.opt.timestep),
                      initial_time=self.start_time, final_time=float(self.d.time),
                      max_clock_step_error_s=self.max_clock_error,
                      max_contact_penetration_m=float(depths.max()), deepest_pair=[geom_name(g) for g in deepest],
                      warnings=(np.asarray(self.d.warning.number)-self.warning_start).tolist(),
                      model_sha256=model_hash,
                      plugin_sha256=self.plugin_hashes,
                      model_file=model_path.name, trajectory_file=path.with_suffix('.npz').name,
                      stages=[dict(name=s, steps=int(np.sum(stages==i)),
                                   max_penetration_m=float(depths[stages==i].max())) for i,s in enumerate(self.labels)])
        report.update(details or {})
        path.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
        return report


def render(path, output=None, fps=25, playback_speed=1.0):
    """Render archived native states; this does not create a new physics rollout."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    from .scene import load_plugin
    if not np.isfinite(playback_speed) or playback_speed<=0:raise ValueError('Playback speed must be positive and finite')
    if fps<=0:raise ValueError('FPS must be positive')
    path=Path(path);report=json.loads(path.with_suffix('.json').read_text())
    binary=path.with_suffix('.mjb').read_bytes()
    if hashlib.sha256(binary).hexdigest()!=report['model_sha256']:
        raise ValueError('Archived model hash does not match the measured rollout')
    load_plugin();m=mujoco.MjModel.from_binary_path(str(path.with_suffix('.mjb')))
    d=mujoco.MjData(m);data=np.load(path.with_suffix('.npz'));states=data['states']
    if states.shape[1]!=mujoco.mj_stateSize(m,STATE_SPEC):raise ValueError('State/model dimension mismatch')
    if np.max(np.abs(np.diff(states[:,0])-m.opt.timestep))>1e-8:raise ValueError('State clock is discontinuous')
    out=Path(output) if output else path.with_suffix('.mp4')
    cam=mujoco.MjvCamera();cam.lookat[:]=[0,.02,.15];cam.distance=1.05;cam.azimuth=115;cam.elevation=-35
    detail=mujoco.MjvCamera();detail.distance=.30;detail.azimuth=115;detail.elevation=-30
    closeup=report.get('closeup_camera',{})
    for key in ('distance','azimuth','elevation'):
        if key in closeup:setattr(detail,key,float(closeup[key]))
    focus_offset=np.asarray(closeup.get('offset',[0,0,.025]),dtype=float)
    focus='driver' if report.get('task')=='tool_pickup' else 'motor'
    learned=bool(report.get('learned_policy',False))
    title=('NATIVE PHYSICS | TRAINED CHECKPOINT POLICY' if learned else
           'NATIVE PHYSICS | SCRIPTED FEASIBILITY, NOT A TRAINED POLICY' if report.get('scripted',True) else
           'NATIVE PHYSICS | RECORDED TRAJECTORY')
    distribution=report.get('distribution')
    opt=mujoco.MjvOption();opt.geomgroup[3]=0
    renderer=mujoco.Renderer(m,height=800,width=1280)
    writer=imageio.get_writer(out,fps=fps,codec='libx264',quality=8,macro_block_size=1)
    try:
        step_stride=round(playback_speed/(fps*m.opt.timestep))
        if step_stride<=0:raise ValueError('Playback speed is too low for this physics timestep and FPS')
        indices=np.arange(0,len(states),step_stride)
        for i in indices:
            mujoco.mj_setState(m,d,states[i],STATE_SPEC);mujoco.mj_forward(m,d)
            renderer.update_scene(d,camera=cam,scene_option=opt);im=Image.fromarray(renderer.render())
            detail.lookat[:]=d.body(focus).xpos+focus_offset
            renderer.update_scene(d,camera=detail,scene_option=opt)
            inset=Image.fromarray(renderer.render()).resize((480,300))
            im.paste(inset,(790,490));draw=ImageDraw.Draw(im)
            draw.rectangle((0,0,1280,96 if distribution else 75),fill=(21,27,36))
            label=str(data['labels'][data['stage'][i]])
            draw.text((24,12),title,font_size=22,fill='white')
            draw.text((24,44),f'{label} | t={states[i,0]:.2f}s | {m.opt.timestep*1000:g}ms physics states',font_size=17,fill=(180,213,237))
            if distribution:draw.text((24,70),'Reset distribution: '+str(distribution),font_size=15,fill=(180,213,237))
            draw.rectangle((790,463,1270,490),fill=(21,27,36))
            draw.text((800,467),f'{focus.capitalize()} close-up | physical states, {playback_speed:g}x playback',font_size=15,fill='white')
            writer.append_data(np.asarray(im))
        im.save(out.with_suffix('.png'))
    finally:
        writer.close();renderer.close()
    return out


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('path',type=Path);parser.add_argument('--out',type=Path);parser.add_argument('--speed',type=float,default=1.0)
    a=parser.parse_args();print(render(a.path,a.out,playback_speed=a.speed))
