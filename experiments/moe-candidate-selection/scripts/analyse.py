import json, numpy as np
MOE = '/home/zhangcy/projects/e2e-moe-20261020-v1'
a  = {json.loads(l)['i']: json.loads(l) for l in open('merge_probe.jsonl')}
p3 = {json.loads(l)['i']: json.loads(l) for l in open('merge_probe3.jsonl')}
adopted = [i for i, l in enumerate(open(MOE + '/results/three_route_diag.jsonl'))
           if json.loads(l).get('visual')]
print('adopted n =', len(adopted))
c5 = np.array([p3[i]['c5'] for i in range(len(p3))])
missed = [i for i in adopted if c5[i] >= 0.21]
print('adopted frames NOT flagged by C5:', missed)
for i in missed:
    print('   frame %d  A %.3f  C5 %.3f  rank of C5 among all = %d / %d'
          % (i, a[i]['a'], c5[i], int((c5 < c5[i]).sum()) + 1, len(c5)))
# how much headroom: threshold needed to catch them
for th in (0.21, 0.23, 0.25, 0.30):
    fa = np.array([a[i]['a'] < 0.21 for i in range(len(a))])
    f5 = c5 < th
    print('thr %.2f -> C5 flags %3d  agree %5.1f%%  recallA %.2f  extra %3d  adopted kept %d/%d'
          % (th, f5.sum(), 100 * (fa == f5).mean(), (f5 & fa).sum() / fa.sum(),
             int((f5 & ~fa).sum()), int(sum(f5[i] for i in adopted)), len(adopted)))
