import cProfile
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from collections import OrderedDict
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parent
DATA = Path('/data2/zhangcy/leader-v2-full-trajectory-20261001')
EVAL = '/home/zhangcy/projects/leader-v5-eval-throughput-20261003'
V7FULL = '/home/zhangcy/projects/v7-full-20261006'
V7PLAN = '/data2/zhangcy/v7-full-20261006/plan.json'
CKPT = '/data2/zhangcy/v7-scratch-continue-20261009-v1/checkpoints/full-epoch7.pt'
CKPT_SHA = '4df0c1c98292edec43acde531bd57d0dfbea9db2c0c4530bba9c5aff78a939d8'
SAM_CHECKPOINT = '/home/zhangcy/projects/nclt-relation-compare-20260927-a/sam_vit_l_0b3195.pth'


class CachedSAM:
    def __init__(self, root, lnm, plan):
        self.root, self.lnm, self.plan = Path(root), lnm, plan
        self.manifests = OrderedDict()

    def encode(self, rgb, spec):
        import numpy as np
        import torch
        from pathlib import PurePosixPath

        key = spec['key']
        date = key.split('/')[0]
        if date not in self.manifests:
            path = (self.root / date / 'cache_manifest.json').resolve(strict=True)
            raw = path.read_bytes()
            expected = self.plan.get('sam_manifest_sha256', {}).get(date)
            if expected and sha_file(path) != expected:
                raise ValueError('SAM manifest SHA-256 mismatch: ' + date)
            manifest = json.loads(raw)
            if (manifest.get('checkpoint_sha256') != self.lnm.SAM_CHECKPOINT_SHA256
                    or manifest.get('sam_model') != 'vit_l'
                    or manifest.get('encoder_size') != 1024
                    or manifest.get('stored_layout') != 'HWC'
                    or manifest.get('stored_dtype') != 'float16'
                    or manifest.get('stored_feature_size_hw') != [64, 64]
                    or manifest.get('feature_channels') != 256
                    or manifest.get('feature_levels') != [64, 32, 16]):
                raise ValueError('SAM preprocessing contract mismatch')
            self.manifests[date] = (path.parent, manifest)
            while len(self.manifests) > 1:
                self.manifests.popitem(last=False)
        root, manifest = self.manifests[date]
        record = manifest['frames'].get(key)
        if record is None:
            raise ValueError('SAM cache missing for ' + key + '; no CPU fallback')
        if (record.get('frame_key') != key
                or record.get('image_sha256') != sha_file(spec['image_path'])
                or record.get('checkpoint_sha256') != self.lnm.SAM_CHECKPOINT_SHA256
                or record.get('sam_model') != 'vit_l'
                or record.get('timestamp') != int(key.split('/')[1])
                or record.get('embedding_shape_hwc') != [64, 64, 256]
                or record.get('embedding_dtype') != 'float16'):
            raise ValueError('SAM cache provenance/image mismatch for ' + key)
        self.lnm.SAMFeatureStore._validate_metadata(SimpleNamespace(root=root), key, record)
        relative = PurePosixPath(record['cache_file'])
        path = root.joinpath(*relative.parts).resolve(strict=True)
        if path.stat().st_size > 4 * 2 ** 20:
            raise ValueError('oversized SAM embedding')
        array = np.load(str(path), allow_pickle=False)
        if array.shape != (64, 64, 256) or array.dtype != np.float16 or not np.isfinite(array).all():
            raise ValueError('invalid SAM embedding')
        if hashlib.sha256(array.tobytes(order='C')).hexdigest() != record['embedding_sha256']:
            raise ValueError('SAM embedding SHA-256 mismatch')
        embedding = torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous().float()
        mask = torch.from_numpy(self.lnm._decode_valid_mask(record).copy()).unsqueeze(0)
        return embedding, mask, record


def sha_file(path):
    digest = hashlib.sha256()
    with open(str(path), 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def install_loader_shims():
    from pathlib import PurePosixPath
    import data.leader_nclt_manifest as lnm

    roots = [DATA.joinpath('NCLT').resolve(), Path('/data2/dataset/NCLT').resolve()]

    def safe_child(root, date, relative_path):
        value = str(relative_path)
        relative = PurePosixPath(value)
        if ('\\' in value or ':' in value or '\x00' in value or relative.is_absolute()
                or not relative.parts or any(p in ('', '.', '..') for p in relative.parts)):
            raise ValueError('manifest paths must be clean relative paths')
        base = Path(root).resolve(strict=True)
        date_root = (base / date).resolve(strict=True)
        if date_root != base and base not in date_root.parents:
            raise ValueError('date path escapes the dataset root')
        target = date_root.joinpath(*relative.parts).resolve(strict=True)
        if not any(target == r or r in target.parents for r in [base] + roots):
            raise ValueError('manifest path escapes the dataset roots')
        return target

    lnm._safe_child = safe_child
    return lnm


class LiveSAM:
    def __init__(self, checkpoint, device):
        import torch
        from segment_anything import sam_model_registry
        from segment_anything.utils.transforms import ResizeLongestSide
        from tools.sam_cache_frozen import FROZEN_SAM_SHA256

        if sha_file(checkpoint) != FROZEN_SAM_SHA256:
            raise ValueError('SAM checkpoint SHA-256 mismatch')
        self.torch = torch
        self.device = device
        self.model = sam_model_registry['vit_l'](checkpoint=str(checkpoint))
        self.model = self.model.eval().requires_grad_(False).to(device)
        self.transform = ResizeLongestSide(1024)

    def encode(self, rgb, spec=None):
        from tools.sam_cache_frozen import make_frame_geometry, valid_token_mask

        torch = self.torch
        resized = self.transform.apply_image(rgb)
        tensor = torch.as_tensor(resized.copy(), device=self.device).permute(2, 0, 1).float()
        with torch.no_grad():
            embedding = self.model.image_encoder(self.model.preprocess(tensor)[None])[0]
        if tuple(embedding.shape) != (256, 64, 64):
            raise ValueError('unexpected SAM feature shape')
        meta = make_frame_geometry(rgb.shape[:2], resized.shape[:2])
        embedding = embedding.detach().to(device='cpu', dtype=torch.float16).float()
        mask = torch.from_numpy(valid_token_mask(meta).copy()).unsqueeze(0)
        return embedding, mask, meta


def attach_sam(sample, embedding, mask, meta, T_BC, lnm):
    import numpy as np
    import torch

    height, width = lnm.IMAGE_SIZE_HW
    rh, rw = meta['resized_size_hw']
    intrinsic = lnm.CAM5_INTRINSICS.astype(np.float32).copy()
    intrinsic[0, :] *= rw / width
    intrinsic[1, :] *= rh / height
    sample.update(sam_features=embedding, intrinsics=torch.from_numpy(intrinsic),
                  camera_from_lidar=torch.from_numpy(np.linalg.inv(T_BC).astype(np.float32)),
                  image_bounds=torch.tensor([rw, rh], dtype=torch.float32),
                  image_valid_mask=mask, sam_metadata=meta)


def build_sample(points, intensity, key, T_BC, rgb, sam, lnm, spec=None):
    import numpy as np
    import torch

    if tuple(rgb.shape[:2]) != tuple(lnm.IMAGE_SIZE_HW):
        raise ValueError('expected calibrated NCLT Cam5 image size %r, got %r'
                         % (lnm.IMAGE_SIZE_HW, rgb.shape[:2]))
    ranges = np.linalg.norm(points, axis=1)
    keep = (ranges > 1.0) & (ranges < 100.0)
    points = np.ascontiguousarray(points[keep], dtype=np.float32)
    intensity = np.asarray(intensity, dtype=np.float32)[keep]
    if not len(points):
        raise ValueError('scan contains no usable points')
    polar = lnm.cartesian_to_polar_expansion(points, lnm.DEFAULT_VOXEL_SIZE * lnm.DEFAULT_HORIZONTAL_RESOLUTION)
    features = np.column_stack((polar[:, 2], polar[:, 1], intensity)).astype(np.float32)
    coordinates, features = lnm.ME.utils.sparse_quantize(
        coordinates=polar, features=features, quantization_size=lnm.DEFAULT_VOXEL_SIZE)
    sample = dict(frame_key=key, date=key.split('/')[0], timestamp_us=int(key.split('/')[1]),
                  coords=np.asarray(coordinates, dtype=np.int32),
                  feats=np.asarray(features, dtype=np.float32), points=points,
                  T_corr=np.eye(4, dtype=np.float32), T_BC=np.asarray(T_BC, dtype=np.float32))
    embedding, mask, meta = sam.encode(rgb, spec)
    attach_sam(sample, embedding, mask, meta, T_BC, lnm)
    return sample


class OnlineFrontend:
    def __init__(self, args):
        import torch

        torch.set_num_threads(args.threads)
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction)
        free, total = torch.cuda.mem_get_info()
        if free < args.gpu_fraction * total + 512 * 2 ** 20:
            raise RuntimeError('selected GPU has insufficient free memory for the bounded runtime')
        sys.path.insert(0, EVAL)
        from runtime import Runtime
        sys.path.insert(0, V7FULL)
        import v7_runtime as vr

        c, _, config, _, _ = vr.imports()
        denied = c.deny_test(config)
        lnm = install_loader_shims()
        runtime = Runtime.__new__(Runtime)
        runtime.config = config
        runtime.denied = denied
        runtime.plan = json.loads(Path('/home/zhangcy/projects/leader-v5-leader-style-20261003/v5-training-plan.json').read_text())
        if sha_file(config['base_checkpoint']) != runtime.plan['frozen_LEADER_sha256']:
            raise ValueError('LEADER checkpoint SHA-256 mismatch')
        from tools.magic_nclt_runtime import load_pretrained_model
        runtime.leader, saved = load_pretrained_model(config['base_checkpoint'], False, torch.device('cuda:0'))
        runtime.leader.eval().requires_grad_(False)
        runtime.center = torch.as_tensor(saved['center_t'], device='cuda', dtype=torch.float32).reshape(3)
        del saved
        if sha_file(CKPT) != CKPT_SHA:
            raise ValueError('V7 checkpoint SHA-256 mismatch')
        state = torch.load(CKPT, map_location='cpu')
        self.model = vr.model(c, config, state['model'], False)
        self.parallel = None
        if args.frontend_workers > 1:
            from infer_batch_engine_v2 import ParallelFrames

            models = iter([self.model] + [vr.model(c, config, state['model'], False)
                                         for _ in range(args.frontend_workers - 1)])
            self.parallel = ParallelFrames(lambda: next(models), c.selfverify.refine,
                                          json.loads(Path(V7PLAN).read_text())['inference_method'],
                                          args.frontend_workers)
        self.warmed = False
        del state
        self.inference = json.loads(Path(V7PLAN).read_text())['inference_method']
        self.runtime, self.c, self.lnm, self.torch = runtime, c, lnm, torch
        self.sam = (CachedSAM(args.sam_cache_root, lnm, runtime.plan) if args.sam_source == 'cached'
                    else LiveSAM(args.sam_checkpoint, args.sam_device))

    def prepare(self, spec, points, intensity):
        import numpy as np
        from PIL import Image

        torch = self.torch
        started = time.monotonic()
        with Image.open(spec['image_path']) as image:
            rgb = np.asarray(image.convert('RGB')).copy()
        sample = build_sample(points, intensity, spec['key'], spec['T_BC'], rgb, self.sam, self.lnm, spec)
        loaded = dict(record=(spec['key'].split('/')[0], 0, spec['key'], spec['canonical_index']),
                      sample=sample,
                      rgb=torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0)
        return loaded, time.monotonic() - started

    def predict(self, spec, points, intensity):
        return self.predict_many([(spec, self.prepare(spec, points, intensity))])[0]

    def predict_many(self, prepared):
        torch = self.torch
        frames, timings = [], []
        with torch.no_grad():
            for spec, (loaded, sam_seconds) in prepared:
                tick = time.monotonic()
                initial = self.runtime.initial(loaded)
                geometry = self.runtime.geometry(loaded, initial[1].detach().cpu())
                frames.append(self.runtime.frame(loaded, *initial, geometry))
                torch.cuda.synchronize()
                timings.append(dict(sam=sam_seconds, leader=time.monotonic() - tick))
            tick = time.monotonic()
            if self.parallel is not None and not self.warmed:
                frame = frames[0]
                observed = self.model.observation_batch([frame['sam']], [frame['rgb']],
                                                        [frame['pixels']], [frame['hw']])[0]
                self.c.selfverify.refine(self.model, frame, observed, self.inference)
                torch.cuda.synchronize()
                self.warmed = True
            if self.parallel is not None:
                results = self.parallel.run(frames)
            else:
                results = []
                for frame in frames:
                    observed = self.model.observation_batch([frame['sam']], [frame['rgb']],
                                                            [frame['pixels']], [frame['hw']])[0]
                    results.append(self.c.selfverify.refine(self.model, frame, observed, self.inference))
            torch.cuda.synchronize()
            refine_seconds = (time.monotonic() - tick) / len(results)
            output = []
            for (refined, _, _, info), timing in zip(results, timings):
                pose = refined.detach().cpu().double().numpy().copy()
                pose[:3, 3] += self.runtime.center.detach().cpu().double().numpy()
                timing['refine'] = refine_seconds
                output.append((pose, timing, info))
        if self.runtime.denied:
            raise RuntimeError('test ground truth was accessed')
        return output

    def close(self):
        if self.parallel is not None:
            self.parallel.close()
