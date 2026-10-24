"""Regenerate `salad/reference-salad.npy`, the reference side of the SALAD channel.

    /home/zhangcy/.conda/envs/leader/bin/python salad/make_features.py --gpu 3

Writes one unit-norm 8448-dimensional descriptor per library reference, in
`library.json` order, which is the order `visual_branch.VisualBranch.references`
uses.  Row order is checked against `inputs/refs.json` before anything is written:
if the two disagree the script stops rather than producing a silently misaligned
matrix.

The feature matrix is not tracked by git (`*.npy`); the checkpoint it needs is at
`salad/weights/dino_salad.ckpt`.  Expected cost, measured on an idle card: ~105 s for
4752 images at batch 16, 160 MB on disk.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)

import salad_retrieval as SR                                        # noqa: E402

DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001'
LIBRARY = '/data2/zhangcy/sp-lightglue-hard1502-20261013-v1/library.json'
REFS = os.path.join(PROJECT, 'inputs', 'refs.json')


def image_path(key):
    date, stamp = key.split('/')
    return os.path.join(DATA, 'NCLT', date, 'cam5', stamp + '.jpg')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default='3')
    ap.add_argument('--out', default=SR.SALAD_REFERENCE)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', a.gpu)

    wanted = [str(r['key']) for r in json.load(open(REFS))]
    library = json.load(open(LIBRARY))
    got = [str(r['key']) for r in library['references']]
    if wanted != got:
        raise SystemExit('inputs/refs.json and library.json disagree on the reference'
                         ' order (%d vs %d rows); refusing to write a misaligned matrix'
                         % (len(wanted), len(got)))
    print('reference order verified: %d rows, identical to library.json' % len(wanted))

    import torch
    model = SR.build_model('cuda')
    mean = torch.tensor(SR.MEAN).view(1, 3, 1, 1)
    std = torch.tensor(SR.STD).view(1, 3, 1, 1)
    paths = [image_path(k) for k in wanted]
    if a.limit:
        paths = paths[:a.limit]
    rows = []
    started = None
    for i in range(0, len(paths), a.batch):
        chunk = [SR.model_image(p) for p in paths[i:i + a.batch]]
        x = torch.from_numpy(np.stack(chunk)).permute(0, 3, 1, 2).float().div_(255.)
        x = ((x - mean) / std).cuda()
        if started is None:
            with torch.inference_mode():
                model(x)
            torch.cuda.synchronize()
            started = time.time()
        with torch.inference_mode():
            descriptor = model(x)
        rows.append(descriptor.float().cpu().numpy())
        if (i // a.batch) % 40 == 0:
            print('  %d/%d' % (min(i + a.batch, len(paths)), len(paths)), flush=True)
    torch.cuda.synchronize()
    seconds = time.time() - started
    matrix = SR._unit(np.concatenate(rows, 0).astype(np.float32))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    partial = os.path.join('/tmp', 'reference-salad.%d.npy' % os.getpid())
    np.save(partial, matrix)
    os.replace(partial, a.out)
    print('%s  %s  %.1fs  %.2f ms/frame' % (a.out, matrix.shape, seconds,
                                            1000. * seconds / max(1, len(paths))))
    print('norm min/max %.6f / %.6f' % (np.linalg.norm(matrix, axis=1).min(),
                                        np.linalg.norm(matrix, axis=1).max()))


if __name__ == '__main__':
    main()
