"""Apply the frozen geometric refiner (r-gate) to the fast-search poses."""
import os
import sys
from multiprocessing import Pool

for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '1'

import numpy as np

sys.path.insert(0, '/home/zhangcy/projects/r4-full-20261019-v1')
import r4_lib as L

W = None
POSES = None


def init():
    global W
    W = L.World()


def one(i):
    P = np.asarray(POSES[i], float)
    T, _ = L.refine_stage(P, W.raw_query(i), W.refs, W.ref_pos[:, :3, 3])
    return i, np.asarray(T, float)


if __name__ == '__main__':
    src = sys.argv[1]
    dst = sys.argv[2]
    z = np.load(src, allow_pickle=True)
    keys = [str(k) for k in z['keys']]
    POSES = np.asarray(z['poses'], float)
    init()
    with Pool(12) as p:
        res = p.map(one, range(len(keys)), chunksize=4)
    out = np.empty_like(POSES)
    for i, T in res:
        out[i] = T
    np.savez(dst, keys=np.array(keys), poses=out)
    print('wrote', dst, out.shape)
