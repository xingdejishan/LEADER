import csv
import itertools
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0,'/root/rivermind-data/LEADER-v1-glace-independent')
from research.glace_fusion.pairwise_camera_ranking import LEVELS, pair_counts

root = Path('/root/rivermind-data/glace_nclt_corrected_20260912/pairwise_ranking_test64')
if not json.loads((root/'complete.json').read_text())['complete']:
    raise ValueError('Incomplete experiment')
records = json.loads((root/'records.json').read_text())


def aggregate(counts):
    totals = {key:int(sum(r[key] for r in counts)) for key in ['correct','wrong','ties','pairs']}
    totals['accuracy'] = totals['correct']/totals['pairs']
    totals['half_credit_accuracy'] = (totals['correct']+.5*totals['ties'])/totals['pairs']
    totals['tie_fraction'] = totals['ties']/totals['pairs']
    totals['random_strict_baseline_with_same_ties'] = .5*(1-totals['tie_fraction'])
    blocks = np.array([[r['correct'],r['ties'],r['pairs']] for r in counts]).reshape(8,8,3).sum(axis=1)
    rng = np.random.default_rng(2089)
    bootstrap = blocks[rng.integers(0,8,(10000,8))].sum(axis=1)
    totals['block_bootstrap_95pct'] = np.quantile(bootstrap[:,0]/bootstrap[:,2],[.025,.975]).tolist()
    totals['half_credit_block_bootstrap_95pct'] = np.quantile((bootstrap[:,0]+.5*bootstrap[:,1])/bootstrap[:,2],[.025,.975]).tolist()
    totals['per_frame_accuracy_median'] = float(np.median([r['accuracy'] for r in counts]))
    return totals


summary = {'frames':64,'sequence':'2012-02-12',
           'uncertainty':'descriptive block bootstrap: 8 consecutive time blocks of 8 frames, 10000 draws; candidate pairs are not treated as independent',
           'equal_true_error_candidates':'excluded','score_ties':'incorrect for strict metric; half credit separately to compare to 50% random',
           'modalities':{}}
csv_rows = []
for mode,levels in LEVELS.items():
    scored = [r['modalities'][mode]['candidates'] for r in records]
    result = {'all_pairs':aggregate([pair_counts(s) for s in scored]),
              'same_direction':aggregate([pair_counts(s,same_direction=True) for s in scored]),
              'level_pairs':{},'directions':{}}
    for low,high in itertools.combinations(levels,2):
        key=f'{low:g}_vs_{high:g}'
        result['level_pairs'][key] = {
            'all_directions':aggregate([pair_counts(s,levels=(low,high)) for s in scored]),
            'same_direction':aggregate([pair_counts(s,levels=(low,high),same_direction=True) for s in scored])}
    for direction in ['x-','x+','y-','y+','z-','z+']:
        result['directions'][direction] = aggregate([pair_counts(s,direction=direction) for s in scored])
    summary['modalities'][mode] = result
    for record,scores in zip(records,scored):
        for low,high in itertools.combinations(levels,2):
            counts = pair_counts(scores,levels=(low,high))
            csv_rows.append({'timestamp_us':record['timestamp_us'],'mode':mode,'lower_error':low,'higher_error':high,**counts})
(root/'summary.json').write_text(json.dumps(summary,indent=2))
with (root/'per_frame_pair_accuracy.csv').open('w',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)

fig,axes=plt.subplots(1,2,figsize=(10,4.8))
for ax,(mode,levels) in zip(axes,LEVELS.items()):
    values=np.full((5,5),np.nan)
    for i,j in itertools.combinations(range(5),2):
        key=f'{levels[i]:g}_vs_{levels[j]:g}'
        values[i,j]=summary['modalities'][mode]['level_pairs'][key]['all_directions']['accuracy']*100
    im=ax.imshow(values,vmin=0,vmax=100,cmap='RdYlGn')
    for i,j in itertools.combinations(range(5),2):
        ax.text(j,i,f'{values[i,j]:.1f}%',ha='center',va='center',fontsize=10)
    ax.set_xticks(range(5));ax.set_yticks(range(5));ax.set_xticklabels([f'{x:g}' for x in levels]);ax.set_yticklabels([f'{x:g}' for x in levels])
    ax.set(xlabel='Higher error candidate',ylabel='Lower error candidate',title=mode+' ('+('m' if mode=='translation' else 'deg')+')')
fig.suptitle('Pairwise ranking accuracy: 64 held-out frames, all six directions\nStrict accuracy; score ties are not correct')
fig.tight_layout();fig.savefig(root/'ranking_heatmap.png',dpi=180);plt.close(fig)
print(json.dumps({mode:{key:summary['modalities'][mode][key] for key in ['all_pairs','same_direction','directions']} for mode in LEVELS},indent=2))
