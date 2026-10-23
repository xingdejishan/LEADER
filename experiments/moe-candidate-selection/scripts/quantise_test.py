"""Is LEADER's yaw error quantised?  (a property test, not a method)

LEADER's initial orientation comes from an SC2 (scan-context) match, which is a
discrete circular correlation over azimuth bins.  If the reported yaw is the bin
centre, the residual yaw error must cluster on multiples of the bin width, and the
fix would be nothing more than interpolating the correlation peak.

We do not know the bin count, so we test several candidate widths: a quantised error
shows up as a non-uniform distribution of the error modulo that width, and as peaks
in the error histogram.
"""
import numpy as np
from scipy.spatial.transform import Rotation

HIST = '/data2/zhangcy/native1502-eval-20261015-v1/historical-predictions.npz'


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()
    gt = np.asarray(w['hard_gt'], float)
    keys = [str(k) for k in w['hard_keys']]
    z = np.load(HIST, allow_pickle=True)
    hk = [str(k) for k in z['keys']]
    pos = {k: i for i, k in enumerate(hk)}
    lead = np.stack([np.asarray(z['historical_LEADER'][pos[str(k)]], float) for k in keys])

    # signed yaw error about the world vertical
    yaw_pred = np.degrees(np.arctan2(lead[:, 1, 0], lead[:, 0, 0]))
    yaw_gt = np.degrees(np.arctan2(gt[:, 1, 0], gt[:, 0, 0]))
    e = (yaw_pred - yaw_gt + 180.0) % 360.0 - 180.0
    print('signed yaw error: mean %+.2f   std %.2f   |mean| %.2f   median %.2f'
          % (e.mean(), e.std(), np.abs(e).mean(), np.median(e)))
    print('|yaw error|: median %.2f  p75 %.2f  p90 %.2f' %
          (np.median(np.abs(e)), np.percentile(np.abs(e), 75), np.percentile(np.abs(e), 90)))

    # histograms: coarse structure and modulo candidates
    h, edges = np.histogram(e, bins=72, range=(-36, 36))
    top = np.argsort(h)[-6:][::-1]
    print('\n densest 3-degree bins of the yaw error:')
    for k in top:
        print('   [%+.1f, %+.1f)  %d frames' % (edges[k], edges[k + 1], h[k]))

    print('\n uniformity of (error mod width), width in degrees:')
    for width in (1, 2, 3, 5, 6, 7.2, 10, 12):
        m = (e % width) / width
        hist, _ = np.histogram(m, bins=10, range=(0, 1))
        chi = ((hist - hist.mean()) ** 2 / max(hist.mean(), 1e-9)).sum()
        print('   width %5.1f deg   chi2(10 bins) = %7.1f   (uniform ~9)' % (width, chi))


if __name__ == '__main__':
    main()
