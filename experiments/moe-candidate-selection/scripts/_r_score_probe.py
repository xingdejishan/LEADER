import json, os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[k] = '1'
sys.path.insert(0, '/home/zhangcy/projects/e2e-moe-20261020-v1')
import numpy as np
import visual_branch as VB

V = '/data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/verify-frozen_nopruneoptREF/%06d.json'
cases = [(955, 'candidate_05'), (957, 'candidate_06'), (1429, 'candidate_04'), (1428, 'candidate_10')]
b = VB.VisualBranch(device='cuda')
for idx, name in cases:
    rec = json.loads(open(V % idx).read())['poses']
    row = rec[name]
    pose = np.asarray(row['pose'])
    own = row.get('own_excluded') or []
    key = json.loads(open(V % idx).read())['key']
    date, stamp = key.split('/')
    scan = b.C.decode(os.path.join(VB.DATA, 'NCLT', date, 'velodyne_sync', stamp + '.bin'))
    body = scan[np.isfinite(scan).all(1)]
    body = body[np.linalg.norm(body, axis=1) <= b.verify_cfg['query_range_m']]
    out = []
    for voxel in (0.2, 0.4):
        q = b.C.voxel_downsample(body, voxel)
        r = b.verify(pose, q, own)
        out.append('voxel %.1f -> %.4f' % (voxel, r['geometry_score']))
    print(idx, name, 'frozen %.4f' % row['fair']['geometry_score'], '|', ' | '.join(out), flush=True)
