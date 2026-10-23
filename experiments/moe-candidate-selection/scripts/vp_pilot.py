"""Can a classical vertical-vanishing-point estimate give usable roll/pitch on NCLT?

Pipeline per frame: LSD line segments -> keep long, near-vertical ones -> least-squares
vanishing point -> gravity direction in camera coordinates -> compare with the gravity
direction implied by the ground-truth pose.  The angle between them is the orientaion
error that the SVA/UVP-style methods would deliver on this imagery.
"""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

STORE = '/data2/zhangcy/sp-lightglue-hard1502-20261013-v1'
DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001/NCLT'
MOE = '/home/zhangcy/projects/e2e-moe-20261020-v1'
K = np.array([[199.716592, 0, 412.930976], [0, 199.716592, 310.584312], [0, 0, 1.0]])
W, H = 808, 616


def load_image(path):
    with Image.open(path) as im:
        im = im.convert('RGB').transpose(Image.Transpose.ROTATE_270)
        w, h = im.size
        if (w, h) != (W, H):
            im = im.resize((W, H), Image.Resampling.BICUBIC)
        return np.asarray(im).copy()


def vertical_vp(gray, min_len=25, max_angle=35.0):
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(gray)[0]
    if lines is None:
        return None, 0, 0
    segs = lines.reshape(-1, 4)
    a = segs[:, :2]
    b = segs[:, 2:]
    d = b - a
    length = np.linalg.norm(d, axis=1)
    # angle of the segment from the vertical image axis
    ang = np.abs(np.degrees(np.arctan2(d[:, 0], -d[:, 1])))       # 0 = vertical
    ang = np.minimum(ang, 180 - ang)
    keep = (length >= min_len) & (ang <= max_angle)
    segs, a, b = segs[keep], a[keep], b[keep]
    if len(segs) < 4:
        return None, int(keep.sum()), len(lines)
    # line coefficients, weighted by length
    homo = np.cross(np.hstack([a, np.ones((len(a), 1))]),
                    np.hstack([b, np.ones((len(b), 1))]))
    n = np.linalg.norm(homo[:, :2], axis=1, keepdims=True)
    homo = homo / np.maximum(n, 1e-9)
    wgt = np.linalg.norm(b - a, axis=1)
    M = (homo * wgt[:, None]).T @ homo
    _, vec = np.linalg.eigh(M)
    v = vec[:, 0]
    if abs(v[2]) < 1e-9:
        return None, int(keep.sum()), len(lines)
    return np.array([v[0] / v[2], v[1] / v[2]]), int(keep.sum()), len(lines)


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
    R_cb = T_BC[:3, :3]                      # body_from_camera rotation

    rng = np.random.default_rng(41)
    sample = sorted(rng.choice(len(keys), 20, replace=False).tolist())
    errs, stats = [], []
    for i in sample:
        date, stamp = keys[i].split('/')
        path = os.path.join(DATA, date, 'cam5', stamp + '.jpg')
        if not os.path.isfile(path):
            continue
        img = load_image(path)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        vp, n_vert, n_all = vertical_vp(gray)
        stats.append((n_all, n_vert))
        if vp is None:
            continue
        ray = np.linalg.solve(K, np.array([vp[0], vp[1], 1.0]))
        ray /= np.linalg.norm(ray)
        g_cam_pred = R_cb.T @ (-ray)          # camera-frame gravity (up axis)
        g_body_gt = gt[i][:3, :3].T @ np.array([0.0, 0.0, -1.0])
        g_cam_gt = R_cb.T @ g_body_gt
        c = float(np.clip(abs(np.dot(g_cam_pred, g_cam_gt)), -1, 1))
        errs.append(float(np.degrees(np.arccos(c))))

    st = np.array(stats)
    print('frames tested: %d   (segments detected median %d, near-vertical kept median %d)'
          % (len(stats), int(np.median(st[:, 0])), int(np.median(st[:, 1]))))
    e = np.array(errs)
    if len(e):
        print('usable frames: %d' % len(e))
        print('gravity-direction error:  median %.2f deg   mean %.2f   p25 %.2f   p75 %.2f'
              % (np.median(e), e.mean(), np.percentile(e, 25), np.percentile(e, 75)))
        print('share within 1 deg: %.0f%%   within 2 deg: %.0f%%'
              % (100 * (e <= 1).mean(), 100 * (e <= 2).mean()))


if __name__ == '__main__':
    main()
