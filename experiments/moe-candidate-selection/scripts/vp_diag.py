"""Diagnose the convention: image size, and several candidate mappings from the
vanishing point to the gravity direction."""
import json
import os

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

STORE = '/data2/zhangcy/sp-lightglue-hard1502-20261013-v1'
DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001/NCLT'
K = np.array([[199.716592, 0, 412.930976], [0, 199.716592, 310.584312], [0, 0, 1.0]])


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

    for i in (0, 100, 300, 700):
        date, stamp = keys[i].split('/')
        path = os.path.join(DATA, date, 'cam5', stamp + '.jpg')
        with Image.open(path) as im:
            raw = im.size
            rot = im.convert('RGB').transpose(Image.Transpose.ROTATE_270).size
        g_body = gt[i][:3, :3].T @ np.array([0.0, 0.0, -1.0])
        g_cam = R_bc.T @ g_body
        print('frame %d  raw %s  rotated %s' % (i, raw, rot))
        print('   gt gravity in camera frame: %s' % np.round(g_cam, 3))
        print('   angle of gt gravity from image +y axis: %.1f deg'
              % np.degrees(np.arccos(np.clip(g_cam[1] / max(np.linalg.norm(g_cam), 1e-9), -1, 1))))


if __name__ == '__main__':
    main()
