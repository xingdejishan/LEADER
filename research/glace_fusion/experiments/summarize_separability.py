import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path('/root/rivermind-data/glace_nclt_corrected_20260912/camera_separability_train64')
if not json.loads((root / 'complete.json').read_text())['complete']:
    raise ValueError('Incomplete experiment')
records = json.loads((root / 'records.json').read_text())
methods = ['GT','LEADER','wrong']


def summarize(rows):
    output = {'n':len(rows), 'methods':{}}
    for method in methods:
        output['methods'][method] = {metric:{'mean':float(np.mean([r[method][metric] for r in rows])),
                                            'median':float(np.median([r[method][metric] for r in rows]))}
                                    for metric in ['S_C','q_C','median_reprojection_px']}
    for reference in ['GT','LEADER']:
        ds = np.array([r['wrong']['S_C']-r[reference]['S_C'] for r in rows])
        dq = np.array([r[reference]['q_C']-r['wrong']['q_C'] for r in rows])
        output[reference+'_vs_wrong'] = {'score_wins':int(np.sum(ds>1e-6)), 'score_ties':int(np.sum(np.abs(ds)<=1e-6)),
                                         'score_losses':int(np.sum(ds < -1e-6)),
                                         'delta_score_mean':float(ds.mean()), 'delta_score_median':float(np.median(ds)),
                                         'delta_score_min_max':[float(ds.min()),float(ds.max())],
                                         'q_wins':int(np.sum(dq>0)), 'delta_q_mean':float(dq.mean()),
                                         'delta_q_median':float(np.median(dq))}
    return output


summary = summarize(records)
summary['sequences'] = {date:summarize([r for r in records if r['sequence']==date]) for date in sorted({r['sequence'] for r in records})}
summary['all_sync_delta_us_zero'] = all(r['sync_delta_us']==0 for r in records)
summary['old_gt_reprojection_max_difference_px'] = max(abs(r['previous_gt_median_difference_px']) for r in records)
summary['leader_translation_median_m'] = float(np.median([r['leader_error_at_camera_time']['translation_m'] for r in records]))
summary['leader_rotation_median_deg'] = float(np.median([r['leader_error_at_camera_time']['rotation_deg'] for r in records]))
summary['leader_beats_gt_score_frames'] = sum(r['LEADER']['S_C']<r['GT']['S_C'] for r in records)
(root/'summary.json').write_text(json.dumps(summary,indent=2))
flat = []
for row in records:
    item = {k:row[k] for k in ['sequence','image','N','sync_delta_us']}
    for method in methods:
        item.update({method+'_'+k:v for k,v in row[method].items()})
    item['delta_S_wrong_minus_GT'] = row['wrong']['S_C']-row['GT']['S_C']
    item['delta_q_GT_minus_wrong'] = row['GT']['q_C']-row['wrong']['q_C']
    flat.append(item)
with (root/'per_frame.csv').open('w',newline='') as handle:
    writer = csv.DictWriter(handle,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)

fig, axes = plt.subplots(1,2,figsize=(10,4.5))
for ax,other in zip(axes,['wrong','LEADER']):
    for date in sorted(summary['sequences']):
        rows = [r for r in records if r['sequence']==date]
        ax.scatter([r['GT']['S_C'] for r in rows],[r[other]['S_C'] for r in rows],label=date,s=26,alpha=.8)
    low = min(min(r[m]['S_C'] for r in records) for m in ['GT',other])-.015
    ax.plot([low,1],[low,1],'--',color='gray',linewidth=1)
    ax.set(xlim=(low,1.005),ylim=(low,1.005),xlabel='GT camera score (lower is better)',
           ylabel=other+' camera score',title='GT vs '+other)
    ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)
axes[0].legend(fontsize=8)
fig.suptitle('Frozen GLACE evidence: original 64 training images, 10 px scale')
fig.tight_layout();fig.savefig(root/'score_scatter.png',dpi=180);plt.close(fig)
print(json.dumps(summary,indent=2))
