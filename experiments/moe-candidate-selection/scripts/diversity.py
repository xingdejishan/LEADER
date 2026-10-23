"""Are the 13 candidates diverse, or do they collapse onto the same pose?"""
import collections
import json

import numpy as np
from scipy.spatial.transform import Rotation

MATCH = ('/data2/zhangcy/sp-lightglue-hard1502-20261013-v1/prov/'
         'match-frozen_nopruneopt/%06d.json')


def ang(a, b):
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(a.T @ b).as_rotvec())))


def main():
    ch_counts = collections.Counter()
    per_frame = []
    for i in range(1502):
        m = json.load(open(MATCH % i))
        hyps = m.get('hypotheses') or []
        if not hyps:
            continue
        chans = [h.get('channel') for h in hyps]
        ch_counts.update(chans)
        ps = [np.asarray(h['pose'], float) for h in hyps]
        n = len(ps)
        # greedy clustering at 3 m / 5 deg
        clusters = []
        for p in ps:
            for c in clusters:
                if np.linalg.norm(p[:3, 3] - c[0][:3, 3]) <= 3.0 and ang(p[:3, :3], c[0][:3, :3]) <= 5.0:
                    c.append(p); break
            else:
                clusters.append([p])
        per_frame.append((n, len(clusters), [len(c) for c in clusters]))

    ns = np.array([p[0] for p in per_frame])
    cs = np.array([p[1] for p in per_frame])
    print('frames with hypotheses: %d' % len(per_frame))
    print('candidates per frame: median %.0f  mean %.1f' % (np.median(ns), ns.mean()))
    print('distinct pose clusters (3 m / 5 deg): median %.0f  mean %.1f' % (np.median(cs), cs.mean()))
    print('frames whose candidates all collapse into one cluster: %d (%.0f%%)'
          % (int((cs == 1).sum()), 100 * (cs == 1).mean()))
    print('frames with >=3 distinct clusters: %d (%.0f%%)'
          % (int((cs >= 3).sum()), 100 * (cs >= 3).mean()))
    print()
    print('channel distribution:', dict(ch_counts))
    big = [max(p[2]) for p in per_frame]
    print('largest cluster size: median %d   share of candidates inside it: %.0f%%'
          % (int(np.median(big)), 100 * np.mean([max(p[2]) / p[0] for p in per_frame])))


if __name__ == '__main__':
    main()
