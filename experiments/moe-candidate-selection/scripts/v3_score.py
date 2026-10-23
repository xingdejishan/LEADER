"""Score V3 outputs against the unified scorer and against the reference table."""
import json
import os
import sys
import numpy as np

ROOT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
sys.path.insert(0, '/home/zhangcy/projects/r-unified-eval-20261018-v1')
import score_unified as S  # noqa: E402

w = S.load_world()
canon = w['hard_keys']
ref = np.load(os.path.join(ROOT, 'results/three_route_merged.npz'), allow_pickle=True)
rlook = {str(k): i for i, k in enumerate(ref['keys'])}
rpose = np.asarray(ref['poses'], np.float64)[[rlook[k] for k in canon]]

for path in sys.argv[1:]:
    z = np.load(path, allow_pickle=True)
    look = {str(k): i for i, k in enumerate(z['keys'])}
    ph = np.asarray(z['poses'], np.float64)[[look[k] for k in canon]]
    t = S.pos_err(ph, w['hard_gt'])
    r = S.rot_deg(ph[:, :3, :3], w['hard_gt'][:, :3, :3])
    d = np.abs(ph - rpose).reshape(len(ph), -1).max(1)
    print('%-20s MPE=%.4f MOE=%.4f ok1m2deg=%d medT=%.4f >10m=%d >5m=%d | vs ref npz: maxdiff=%.1e, frames!=0=%d'
          % (os.path.basename(path), t.mean(), r.mean(),
             int(np.sum((t <= 1.0) & (r <= 2.0))), float(np.median(t)),
             int(np.sum(t > 10.0)), int(np.sum(t > 5.0)), d.max(), int((d > 0).sum())))

for tag in ('v3_alg_vis',):
    rows = [json.loads(l) for l in open('/home/zhangcy/.dsh_scratch/%s.jsonl' % tag)]
    key2row = {r['key']: r for r in rows}
    refd = {json.loads(l)['key']: json.loads(l) for l in open(
        os.path.join(ROOT, 'results/three_route_merged_diag.jsonl'))}
    dg = max(abs(key2row[k]['gain'] - refd[k]['gain']) for k in refd)
    dr = max(abs(key2row[k]['ratio'] - refd[k]['ratio']) for k in refd)
    rot = sum(1 for k in refd if bool(key2row[k]['rotated']) != bool(refd[k]['rotated']))
    vis = sum(1 for k in refd if (key2row[k]['chosen'] == 'visual') != bool(refd[k]['visual']))
    print('diag: max|d gain|=%.3e max|d ratio|=%.3e rotated flips=%d visual flips=%d'
          % (dg, dr, rot, vis))
