"""Apply gate 2 to the fast-search output and write the final pose table.

    gain = cost(origin) - cost(coarse best)        (already computed by gate2b.py)
    run the fine stage only where gain > threshold

Gated-out frames keep the incoming pose (the search is skipped entirely: keeping the
coarse pose instead measured worse). Both branches are passed through the frozen
geometric refiner beforehand, so the mixed table is directly comparable.

    python apply_gate.py --threshold 0.01
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, 'results')
SCORER = '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py'


def load(name):
    z = np.load(os.path.join(RES, name), allow_pickle=True)
    return [str(k) for k in z['keys']], np.asarray(z['poses'], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threshold', type=float, default=0.01)
    a = ap.parse_args()
    keys, fast = load('fast_refined.npz')
    _, inip = load('initial_refined.npz')
    g = np.load(os.path.join(RES, 'gate2_coarse.npz'), allow_pickle=True)
    gm = {str(k): i for i, k in enumerate(g['keys'])}
    gain = np.asarray(g['gain'], float)[[gm[k] for k in keys]]
    use = gain > a.threshold
    out = np.where(use[:, None, None], fast, inip)
    np.savez(os.path.join(RES, 'final_gated.npz'), keys=np.array(keys), poses=out)

    spec = importlib.util.spec_from_file_location('score_unified', SCORER)
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    w = sc.load_world()
    m = {k: i for i, k in enumerate(keys)}
    P = np.stack([out[m[k]] for k in w['hard_keys']])
    e = sc.pos_err(P, w['hard_gt'])
    r = sc.rot_deg(P[:, :3, :3], w['hard_gt'][:, :3, :3])
    print(json.dumps(dict(threshold=a.threshold, frames=int(len(keys)),
                          run_fine=int(use.sum()), coverage=round(float(use.mean()), 3),
                          MPE=round(float(e.mean()), 4), MOE=round(float(r.mean()), 4),
                          ok=int(((e <= 1) & (r <= 2)).sum()),
                          over10=int((e > 10).sum()), over100=int((e > 100).sum())), indent=1))


if __name__ == '__main__':
    main()
