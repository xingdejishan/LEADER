import json, os, sys, traceback
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[k] = '1'
sys.path.insert(0, '/home/zhangcy/projects/e2e-moe-20261020-v1')
import numpy as np
import visual_branch as VB

key = '2012-05-26/1338075589789613'
Q = {q['key']: q for q in json.loads(open('/data2/zhangcy/sp-lightglue-hard1502-20261013-v1/queries-provider.json').read())['queries']}
q = Q[key]
init = np.load('/home/zhangcy/projects/e2e-moe-20261020-v1/inputs/initial.npz', allow_pickle=False)
lag = {str(k): i for i, k in enumerate(init['keys'])}
T0 = np.asarray(init['poses'], float)[lag[key]]
b = VB.VisualBranch(device='cuda')
try:
    h = b.hypothesize(key, q['image_path'], budget=8, pose_prior=T0)
    print('ok', None if h is None else (h['score'], h['n_candidates']))
except BaseException:
    traceback.print_exc()
