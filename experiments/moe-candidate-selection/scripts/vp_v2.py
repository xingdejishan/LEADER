"""Corrected protocol: no image rotation, and the camera-frame convention is fitted
on half the frames and evaluated on the other half (it is a fixed mount convention,
so this is legitimate; it cannot absorb per-frame error)."""
import json
import os

import cv2
import numpy as np
from PIL import Image

STORE = '/data2/zhangcy/sp-lightglue-hard1502-20261013-v1'
DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001/NCLT'
K = np.array([[199.716592, 0, 412.930976], [0, 199.716592, 310.584312], [0, 0, 1.0]])


def vp_of(gray, min_len=25, max_angle=35.0):
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    seg = lsd.detect(gray)[0]
    if seg is None:
        return None, 0
    seg = seg.reshape(-1, 4)
    a, b = seg[:, :2], seg[:, 2:]
    d = b - a
    length = np.linalg.norm(d, axis=1)
    ang = np.abs(np.degrees(np.arctan2(d[:, 0], -d[:, 1])))
    ang = np.minimum(ang, 180 - ang)
    keep = (length >= min_len) & (ang <= max_angle)
    a, b, length = a[keep], b[keep], length[keep]
    if len(a) < 4:
        return None, len(a)
    homo = np.cross(np.hstack([a, np.ones((len(a), 1))]), np.hstack([b, np.ones((len(b), 1))]))
    homo = homo / np.maximum(np.linalg.norm(homo[:, :2], axis=1, keepdims=True), 1e-9)
    M = (homo * length[:, None]).T @ homo
    _, vec = np.linalg.eigh(M)
    v = vec[:, 0]
    if abs(v[2]) < 1e-9:
        return None, len(a)
    return np.array([v[0] / v[2], v[1] / v[2]]), len(a)


def kabsch(p, q):
    H = p.T @ q
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ D @ U.T


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()
    gt = np.asarray(w['hard_gt'], float)
    keys = [str(k) for k in w['hard_keys']]
    T_BC = np.asarray(json.load(open(os.path.join(STORE, 'library.json')))['T_BC'], float)
    R_bc = T_BC[:3, :3]

    rng = np.random.default_rng(41)
    sample = sorted(rng.choice(len(keys), 60, replace=False).tolist())
    preds, gts, counts = [], [], []
    for i in sample:
        date, stamp = keys[i].split('/')
        path = os.path.join(DATA, date, 'cam5', stamp + '.jpg')
        if not os.path.isfile(path):
            continue
        with Image.open(path) as im:
            gray = np.asarray(im.convert('L'))
        vp, n = vp_of(gray)
        counts.append((n, vp is not None))
        if vp is None:
            continue
        ray = np.linalg.solve(K, np.array([vp[0], vp[1], 1.0]))
        ray /= np.linalg.norm(ray)
        preds.append(-ray)                              # candidate gravity direction
        gts.append(R_bc.T @ (gt[i][:3, :3].T @ np.array([0.0, 0.0, -1.0])))

    P = np.array(preds)
    G = np.array(gts)
    n = len(P)
    print('frames %d   near-vertical segments median %d' % (n, int(np.median([c[0] for c in counts]))))
    half = n // 2
    R = kabsch(P[:half], G[:half])
    err = np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i', P[half:] @ R.T, G[half:]), -1, 1)))
    err_in = np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i', P[:half] @ R.T, G[:half]), -1, 1)))
    print('convention rotation fitted on %d frames -> gravity error on the held-out %d:' % (half, n - half))
    print('   median %.2f deg   mean %.2f   p25 %.2f   p75 %.2f' %
          (np.median(err), err.mean(), np.percentile(err, 25), np.percentile(err, 75)))
    print('   (in-sample median %.2f deg, for reference)' % np.median(err_in))
    print('   share within 1 deg: %.0f%%   within 2 deg: %.0f%%' %
          (100 * (err <= 1).mean(), 100 * (err <= 2).mean()))


if __name__ == '__main__':
    main()
