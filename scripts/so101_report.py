"""Plot the recorded overnight lineage; distinguish sampled TRAIN and loose EVAL."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def rows(path):
    result=[]
    for line in path.read_text().splitlines():
        try:result.append(json.loads(line))
        except json.JSONDecodeError:continue  # a live writer can leave its last line incomplete
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,default=ROOT/'artifacts_local/overnight_progress')
    a=p.parse_args();a.out.parent.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    total=0;previous=None;summary=[];evaluations=[]
    colors=['#4c78a8','#f28e2b','#36a075','#b279a2']
    for phase,path in enumerate(a.runs):
        records=rows(path/'log.jsonl');cfg=json.loads((path/'config.json').read_text())
        if not records:continue
        parent=cfg.get('parent_policy')
        if previous is not None and (not parent or parent['transitions']!=previous['transitions'] or parent['iteration']!=previous['it']):
            raise ValueError('Runs do not form the specified final-checkpoint weights lineage')
        x=np.array([r['transitions'] for r in records])+total
        episodes=np.array([r['episodes'] for r in records])
        successes=np.array([round(r['episodes']*r['success']) for r in records])
        unsafe=np.array([round(r['episodes']*r['safety']) for r in records])
        def rate(count):
            return np.array([count[max(0,i-9):i+1].sum()/max(episodes[max(0,i-9):i+1].sum(),1)
                             for i in range(len(records))])
        color=colors[phase%len(colors)];label=f'Phase {phase+1}'
        axes[0,0].plot(x,100*rate(successes),color=color,label=label)
        axes[0,1].plot(x,100*rate(unsafe),color=color)
        axes[1,0].plot(x,[r['vf'] for r in records],color=color,alpha=.8)
        axes[1,1].plot(x,[r['fps'] for r in records],color=color,alpha=.8)
        if phase:
            for ax in axes.flat:ax.axvline(total,color='#777777',ls=':',lw=1)
        summary.append(dict(phase=phase+1,run=str(path.resolve()),bank=cfg['bank_dir'],
            physics_backend=cfg.get('physics_backend','native'),physics_signature=cfg.get('physics_signature'),
            environment=cfg['env'],iterations=records[-1]['it'],transitions=records[-1]['transitions'],
            completed_episodes=int(episodes.sum()),training_successes=int(successes.sum()),
            safety_terminations=int(unsafe.sum()),numerical_failures=sum(r['unstable'] for r in records),
            median_transitions_per_second=float(np.median([r['fps'] for r in records])),
            parent_policy=parent))
        for report in path.glob('reviews/*/evaluation.json'):
            r=json.loads(report.read_text())
            evaluations.append(dict(report=str(report.resolve()),mtime=report.stat().st_mtime,
                successes=sum(t['successes'] for t in r['tasks'].values()),
                trials=sum(t['trials'] for t in r['tasks'].values()),distribution=r['distribution']))
        total+=records[-1]['transitions'];previous=records[-1]
    axes[0,0].set(title='TRAIN success: sampled curriculum',ylabel='Completed episodes (%)',ylim=(0,100))
    axes[0,0].legend(loc='upper left',frameon=False)
    axes[0,1].set(title='Physical safety terminations',ylabel='Completed episodes (%)',ylim=(0,100))
    axes[1,0].set(title='Critic loss (not task reward)',ylabel='PPO value loss',yscale='log')
    axes[1,1].set(title='Measured learner throughput',ylabel='Control transitions / second')
    for ax in axes.flat:
        ax.set_xlabel('Transitions along the weights lineage');ax.grid(alpha=.18)
        ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0))
    latest=max(evaluations,key=lambda x:x['mtime']) if evaluations else None
    evaluation=(f"Latest independent loose-start check: {latest['successes']}/{latest['trials']} successes"
                if latest else 'Independent loose-start evaluation pending')
    fig.suptitle('SO101 curriculum pilot — TRAIN includes prepared held starts\n'+evaluation,fontsize=13)
    fig.savefig(a.out.with_suffix('.png'),dpi=150)
    a.out.with_suffix('.json').write_text(json.dumps(dict(phases=summary,total_lineage_transitions=total,
        independent_evaluations=evaluations,notes=['Rates use episode-weighted windows of up to ten iterations.',
        'Vertical lines mark separate phases with fresh optimizer/SGS statistics; bank/backend are recorded per phase.',
        'TRAIN rates reflect adaptive SGS sampling and do not measure loose-assembly reliability.']),indent=2)+'\n')
    print(a.out.with_suffix('.png'))


if __name__=='__main__':main()
