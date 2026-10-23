"""How well can a POSELESS retrieval find the place?  Offline study, GT used only to score.

Descriptors compared, all computable without any query pose:
  cls      DINOv2 CLS cosine against the library's stored descriptors (existing index)
  ring     yaw-invariant LiDAR ring key (95th percentile height per 2 m ring, 0-60 m)
           of the query scan against the same statistic of each reference's cached map
           cloud (world points minus that reference's own body position)
  union    rank fusion of the two (sum of ranks)

Scored by: the distance from the frame's true position to the nearest selected reference
(25 m is the radius the verification crop uses).
"""
import os, sys, json, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '1'
from collections import OrderedDict
from multiprocessing import Pool
import numpy as np

PROJECT = '/home/zhangcy/projects/e2e-hypothesis-arbitration-20261023-v1'
SCORED = '/data2/zhangcy/rpge-crossvote-hard1502-20261012-v1/scored.npz'
STORE = '/data2/zhangcy/sp-lightglue-hard1502-20261013-v1'
sys.path.insert(0, PROJECT)
import visual_branch as VB
VB.SCRATCH = '/tmp/dsh-retrieval-vb'
RINGS = np.arange(0.0, 60.01, 2.0)
_G = {}


class Branch(VB.VisualBranch):
    def __init__(self, *a, **kw):
        kw.setdefault('device', 'cpu')
        super(Branch, self).__init__(*a, **kw)
        self._features = OrderedDict()
        self._projections = OrderedDict()


def ring_key(points, heights=True):
    r = np.linalg.norm(points[:, :2], axis=1)
    z = points[:, 2]
    out = np.zeros(len(RINGS) - 1)
    for a in range(len(RINGS) - 1):
        m = (r >= RINGS[a]) & (r < RINGS[a + 1])
        out[a] = np.percentile(z[m], 95) if m.sum() >= 8 else 0.0
    return out


def build_ring_index():
    path = '/tmp/dsh-ring-index.npz'
    if os.path.isfile(path):
        z = np.load(path)
        return z['keys'], z['pos']
    lib = json.load(open(os.path.join(STORE, 'library.json')))
    refs = lib['references']
    keys = np.zeros((len(refs), len(RINGS) - 1))
    pos = np.zeros((len(refs), 3))
    t = time.time()
    for i, row in enumerate(refs):
        name = '%06d-%s.npz' % (int(row['index']), row['scan_sha256'][:16])
        with np.load(os.path.join(STORE, 'map-cloud', name)) as z:
            p = z['world'] if 'world' in z.files else z['world_local']
        centre = np.asarray(row['body_position_world'], float)
        keys[i] = ring_key(np.asarray(p, float) - centre[None, :])
        pos[i] = centre
        if (i + 1) % 500 == 0:
            print('  ring index %d/%d  %.0f s' % (i + 1, len(refs), time.time() - t), flush=True)
    np.savez(path, keys=keys, pos=pos)
    return keys, pos


def init():
    _G['branch'] = Branch(scratch='/tmp/dsh-retrieval-vb', torch_threads=2)
    _G['ring'], _G['rpos'] = build_ring_index()


def work(i):
    b = _G['branch']
    key = _G['keys'][i]
    date, stamp = key.split('/')
    img = os.path.join(VB.DATA, 'NCLT', date, 'cam5', stamp + '.jpg')
    rec = _G['frames'][i]
    t = time.time()
    e = b.embed(img).astype(np.float32)
    sim = b.reference_cls @ e
    q = b.C.decode(rec['scan_path'])
    q = q[np.isfinite(q).all(1)]
    qk = ring_key(q)
    rk = _G['ring']
    d = np.abs(rk - qk[None, :]).mean(1)
    cls_rank = np.argsort(-sim)
    ring_rank = np.argsort(d)
    return dict(i=int(i), key=key, emb=e, cls_top=list(cls_rank[:50]),
                ring_top=list(ring_rank[:50]), ring_d=list(d[ring_rank[:50]]),
                secs=float(time.time() - t))


def main():
    frames = json.load(open(PROJECT + '/inputs/frames.json'))
    z = np.load(SCORED, allow_pickle=True)
    gt_keys = [str(k) for k in z['keys']]
    gt_all = np.asarray(z['reference'], float)
    lut = {k: a for a, k in enumerate(gt_keys)}
    Z = np.load(PROJECT + '/inputs/initial.npz', allow_pickle=True)
    keys = [str(k) for k in Z['keys']]
    gt = np.stack([gt_all[lut[k]] for k in keys])
    out = np.load(PROJECT + '/results/three_route_kcalib.npz', allow_pickle=True)
    e_out = np.linalg.norm(np.asarray(out['poses'], float)[:, :3, 3] - gt[:, :3, 3], axis=1)
    t3 = np.asarray(np.load(PROJECT + '/results/three_route_route3_recheck.npz',
                            allow_pickle=True)['poses'], float)
    e_geo = np.linalg.norm(t3[:, :3, 3] - gt[:, :3, 3], axis=1)
    rows = [json.loads(l) for l in open(PROJECT + '/results/three_route_kcalib_diag.jsonl')]
    gate = set(a for a, r in enumerate(rows) if r.get('ratio', 9) < 0.21)
    worst = set(np.argsort(-e_out)[:100].tolist())
    sample = sorted(gate | worst)
    _G.update(keys=keys, frames=frames, gt=gt, e_geo=e_geo)
    t = time.time()
    with Pool(6, initializer=init) as p:
        res = p.map(work, sample, chunksize=1)
    print('ran %d frames in %.0f s' % (len(res), time.time() - t))
    np.savez('/tmp/dsh-retrieval.npz', i=np.array([r['i'] for r in res]),
             emb=np.stack([r['emb'] for r in res]),
             cls_top=np.stack([r['cls_top'] for r in res]),
             ring_top=np.stack([r['ring_top'] for r in res]))
    pos = _G['rpos']
    print('\n%-28s %6s %8s %8s %8s' % ('retrieval, K', 'n', '<=25m', '<=50m', 'med_dist'))
    for name in ('cls', 'ring', 'union'):
        for K in (5, 10, 20, 30):
            hits, hits50, dist = [], 0, []
            n2 = 0
            for r in res:
                g = gt[r['i'], :3, 3]
                if name == 'cls':
                    sel = r['cls_top'][:K]
                elif name == 'ring':
                    sel = r['ring_top'][:K]
                else:
                    rc = {j: a for a, j in enumerate(r['cls_top'])}
                    rr = {j: a for a, j in enumerate(r['ring_top'])}
                    cand = sorted(set(r['cls_top'][:2 * K]) | set(r['ring_top'][:2 * K]),
                                  key=lambda j: rc.get(j, 99) + rr.get(j, 99))
                    sel = cand[:K]
                dd = np.min(np.linalg.norm(pos[np.asarray(sel)] - g[None, :], axis=1))
                dist.append(dd)
                hits.append(dd <= 25)
                hits50 += int(dd <= 50)
                n2 += int(np.sum(np.linalg.norm(pos[np.asarray(sel)] - g[None, :], axis=1) <= 25) >= 2)
            print('%-28s %6d %8d %8d %8.1f   (>=2 refs in 25 m on %d)'
                  % (name + ', K=%d' % K, len(dist), sum(hits), hits50, np.median(dist), n2))
    print('\nnote: 25 m is the verification crop radius, i.e. the tolerance that matters')


if __name__ == '__main__':
    main()
