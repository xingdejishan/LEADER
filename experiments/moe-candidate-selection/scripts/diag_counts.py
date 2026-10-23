import json
import numpy as np

P = '/home/zhangcy/projects/e2e-moe-20261020-v1/results/three_route_merged_diag.jsonl'
d = [json.loads(l) for l in open(P)]
r = np.array([x['ratio'] for x in d])
print('frames=%d  severe(ratio<0.21)=%d  visual=%d  fine=%d  rotated=%d'
      % (len(d), int((r < 0.21).sum()), sum(x['visual'] for x in d),
         sum(x['ran_fine'] for x in d), sum(x['rotated'] for x in d)))
