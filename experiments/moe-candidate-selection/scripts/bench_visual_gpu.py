"""Time the two heavy stages of the visual branch on a GPU, using the same
construction the shipped matcher uses (SuperPoint 256 keypoints + tuned LightGlue).
Reports query extraction, per-reference extraction, and one batched match pass.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, '/home/zhangcy/projects/sp-lightglue-hard1502-20261013-v1')
from common import STORE, VENDOR, VENDOR_PY, HUB  # noqa: E402

import torch  # noqa: E402

sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(VENDOR_PY))
torch.hub.set_dir(str(HUB))

from lightglue import LightGlue, SuperPoint  # noqa: E402
from PIL import Image  # noqa: E402

DEV = 'cuda'
IMAGE_EDGE = 560
BUDGET = 256
THRESHOLD = 0.0001
CKPT = '/data2/zhangcy/sp-lightglue-hard1502-20261013-v1/lightglue-finetune/best-lightglue.pt'


def model_image(path):
    with Image.open(path) as image:
        image = image.convert('RGB').transpose(Image.Transpose.ROTATE_270)
        w, h = image.size
        scale = min(1., IMAGE_EDGE / max(w, h))
        rw, rh = round(w * scale), round(h * scale)
        image = image.resize((rw, rh), Image.Resampling.BICUBIC)
        cw, ch = rw // 14 * 14, rh // 14 * 14
        left, top = (rw - cw) // 2, (rh - ch) // 2
        image = image.crop((left, top, left + cw, top + ch))
        return np.asarray(image).copy()


extractor = SuperPoint(max_num_keypoints=BUDGET, detection_threshold=THRESHOLD).eval().to(DEV)
matcher = LightGlue(features='superpoint', flash=False,
                    depth_confidence=-1, width_confidence=-1).eval().to(DEV)
payload = torch.load(CKPT, map_location='cpu')
matcher.load_state_dict(payload['model'] if isinstance(payload, dict) and 'model' in payload else payload,
                        strict=True)


def extract(path):
    x = torch.from_numpy(model_image(path)).permute(2, 0, 1).float().div(255).to(DEV)
    with torch.inference_mode():
        return extractor.extract(x, resize=None)


library = json.load(open(STORE / 'library.json'))
queries = json.load(open(STORE / 'queries-provider.json'))['queries']
q = queries[0]
refs = [library['references'][ri] for ri in q['pair_references']]
print('frame 0 references: %d' % len(refs))
print('gpu: %s' % torch.cuda.get_device_name(0))

for _ in range(5):
    fq = extract(q['image_path'])
    fr = [extract(r['image_path']) for r in refs[:1]]
torch.cuda.synchronize()


def measure(fn, n=20):
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts)), out


t_q, fq = measure(lambda: extract(q['image_path']))
t_ref, fr = measure(lambda: extract(refs[0]['image_path']), n=10)
print('query keypoints+descriptors : %6.2f ms' % t_q)
print('one reference extraction    : %6.2f ms  (x%d per frame if not cached)' % (t_ref, len(refs)))

features = [extract(r['image_path']) for r in refs]


def batched():
    image1 = {k: torch.cat([f[k] for f in features], dim=0) for k in features[0] if k != 'image'}
    image0 = {k: fq[k].repeat(len(features), *([1] * (fq[k].dim() - 1))) for k in fq if k != 'image'}
    with torch.inference_mode():
        return matcher({'image0': image0, 'image1': image1})


t_m, out = measure(batched, n=20)
n_match = int(sum(len(m) for m in out['matches']))
print('batched match (%d pairs, %d total matches): %6.2f ms' % (len(features), n_match, t_m))
print()
print('per-frame, references NOT cached : %6.1f ms' % (t_q + t_ref * len(features) + t_m))
print('per-frame, references fully cached: %6.1f ms' % (t_q + t_m))
print('docs claim                        :   67 ms (7.7 extract + 25 match + 5 bind + 27 pnp + 2 refine)')
