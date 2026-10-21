"""Every statistic quoted in docs/ADAPTIVE.md, from fine_stats.npz."""
import numpy as np

STATS = '/data2/zhangcy/r5e-adaptive-20261019-v1/fine_stats.npz'
RHO = float(np.hypot(0.5, 0.4))


def load(path=STATS):
    d = np.load(path)
    cost = d['cost'].astype(np.float64)
    return dict(cost=cost,
                err=d['err'].astype(np.float64),
                centre=d['centre'].astype(np.float64),
                coarse=d['coarse'].astype(np.float64),
                dep=d['dep'], replaced=d['replaced'].astype(bool),
                N=cost.shape[0])


def report(d):
    cost, err, centre, coarse = d['cost'], d['err'], d['centre'], d['coarse']
    N, K, P = cost.shape
    keep = ~d['replaced']
    big = np.where(~np.isnan(cost), cost, np.inf)
    valid = ~np.isnan(cost)
    print('frames %d, kept %d, valid windows %.1f%%' % (N, keep.sum(), 100 * valid.mean()))

    # 1. rank of the winner
    best_c = big.reshape(N, -1).argmin(1)
    cand_c = best_c // P
    best_e = np.where(~np.isnan(err), err, np.inf).reshape(N, -1).argmin(1)
    cand_e = best_e // P
    print('\n[1] rank of the cost-optimal candidate')
    for r in range(K):
        print('    rank %d: %4d frames (%.1f%%)' % (r + 1, int((cand_c == r).sum()),
                                                    100 * (cand_c == r).mean()))
    print('    ranks 4-6 win: %.1f%%   cost-optimal == error-optimal: %.1f%%'
          % (100 * (cand_c >= 3).mean(), 100 * (best_c == best_e).mean()))

    # 2. centre vs local minimum
    cmin = np.where(valid, cost, np.nan).min(2)
    gap = cmin - centre
    print('\n[2] drop from centre to best in window: median %.4f  p90 %.4f'
          % (np.nanmedian(gap), np.nanpercentile(gap, 90)))

    # 3. location of the best pose
    loc_s, loc_d = [], []
    for n in range(N):
        a = cand_c[n]
        j = big[n, a].argmin()
        loc_s.append(j // 9); loc_d.append(j % 9)
    loc_s, loc_d = np.array(loc_s), np.array(loc_d)
    print('\n[3] along-track: edge %.0f%% / %.0f%%  interior %.0f%%'
          % (100 * (loc_s == 0).mean(), 100 * (loc_s == 4).mean(),
             100 * ((loc_s > 0) & (loc_s < 4)).mean()))
    print('    lateral   : edge %.0f%% / %.0f%%  interior %.0f%%'
          % (100 * (loc_d == 0).mean(), 100 * (loc_d == 8).mean(),
             100 * ((loc_d > 0) & (loc_d < 8)).mean()))

    # 4. candidate gaps and whether a valid bound could prune
    srt = np.sort(cmin, axis=1)
    print('\n[4] gap between best and second best candidate: median %.4f  p10 %.4f'
          % (np.median(srt[:, 1] - srt[:, 0]), np.percentile(srt[:, 1] - srt[:, 0], 10)))
    print('    frames with gap > rho (%.3f): %.0f%%' % (RHO, 100 * np.mean(srt[:, 1] - srt[:, 0] > RHO)))

    # 5. threshold sweep
    print('\n[5] adaptive expansion (centre order, gap threshold)')
    for th in (0.02, 0.05, 0.10, 0.20, 0.40, 1e9):
        E = np.empty(N); W = np.zeros(N, int)
        for n in range(N):
            order = np.argsort(centre[n])
            base = centre[n, order[0]]
            best = np.inf; chosen = None; ex = 0
            for a in order:
                if not valid[n, a].any():
                    continue
                if ex > 0 and centre[n, a] - base > th:
                    continue
                ex += 1
                j = int(np.argmin(big[n, a]))
                if big[n, a, j] < best:
                    best = big[n, a, j]; chosen = (a, j)
            W[n] = ex
            E[n] = err[n, chosen[0], chosen[1]] if chosen else np.nan
        print('    th %.2f: windows %.2f  median err %.3f  <=1m %.0f%%'
              % (th, W[keep].mean(), np.median(E[keep]), 100 * (E[keep] <= 1).mean()))


if __name__ == '__main__':
    report(load())
