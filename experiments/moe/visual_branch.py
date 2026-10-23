"""Live visual hypothesis for the online driver: image -> retrieval -> SuperPoint/LightGlue
-> pose estimate -> laser verification, one frame at a time.

Reconstruction of the pipeline that produced the frozen table
    /data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/verify-frozen_nopruneoptREF
The producer is `sp-integrated-20261017-v1/verify_prov.py` (laser verification, arm
`frozen_nopruneopt` + tag `REF`) reading `prov/match-frozen_nopruneopt`, the table
`sp-integrated-20261017-v1/run_opt.sh` writes with `match_prov.py --arm frozen_noprune
--out-tag opt --batched 1 --budget 256`.  Two settings below are read off the table itself
rather than off that script, because the script's `--fast-pnp 1` does not reproduce it:
the 2000-iteration RANSAC solver matches the recorded per-candidate inlier counts exactly,
and `--score-voxel 0.4` matches the recorded `fair.geometry_score` exactly.

Nothing is read from that pose table here.  Every call re-derives the hypothesis from the
frame's own image and scan:

    retrieval        DINOv2 ViT-B/14 CLS cosine top5  union  the 5 library references
                     nearest to the pose prior (offline the prior was the frame's own
                     SC2 initial pose; online it is the driver's geometric pose T0)
    matching         SuperPoint(max_num_keypoints=256, threshold 1e-4) once, then ONE
                     batched LightGlue pass (early stop / width filter disabled) over the
                     frame's whole reference list; per-reference passes are the fallback
    2D-3D binding    every matched reference keypoint looks up its own LiDAR return in
                     model-pixel space within 3.5 px; that return carries a world point
    pose             cv2 EPnP + RANSAC (2000 iterations, 12 px, 0.999, cv2 seed 20) + LM
                     refinement, run per reference, over the whole capped candidate set,
                     and over each 25 m reference cluster; qualified solutions must agree
                     within 5 m / 10 deg and on >= 2 references.  This solver reproduces
                     the frozen table's per-candidate inlier counts exactly; the producer's
                     `--fast-pnp` variant does not, so `fast_pnp=True` is opt-in only.
    laser verify     the pose is scored against the labelled map cloud within 25 m,
                     excluding the references that produced it, as the median per-sector
                     share of returns inside 0.3 m (the frozen `fair` score), on a 0.4 m
                     query cloud so the numbers are the table's numbers

Every constant (K, T_BC, association radius, PnP gate, verification constants) is read
from the frozen plan; the heavy lifting is the frozen code (`common.model_to_raw`,
`common.model_matrix`, `common.voxel_downsample`, `common.transform`, `common.decode`)
imported from the producer's own project, so this module adds wiring, not a second geometry.

Writes: nothing outside `scratch` (an on-demand cache of reference projection tables for
references the Oct-13 run never needed).

Usage
    from visual_branch import VisualBranch
    branch = VisualBranch()                       # loads retrieval + matcher once
    hyp = branch.hypothesize(key, image_path, pose_prior=T0)
    if hyp:  hyp['pose'], hyp['score'], hyp['n_candidates'], hyp['seconds'], hyp['detail']
"""
import os
import sys
import time

import numpy as np

# --- frozen producer project (geometry, PnP, camera model, plan) ---------------
SP_PROJECT = '/home/zhangcy/projects/sp-lightglue-hard1502-20261013-v1'
SP_VENDOR = '/home/zhangcy/projects/sp-lightglue-four-20261013-v1/vendor'
SP_VENDOR_PY = '/home/zhangcy/projects/sp-lightglue-four-20261013-v1/vendor-py'
SP_HUB = '/home/zhangcy/projects/sp-lightglue-four-20261013-v1/torch_hub'
CLS_ROOT = '/home/zhangcy/projects/v7-dinov2-recall-20261012-v1'
CLS_DATA = '/data2/zhangcy/v7-dinov2-recall-20261012-v1'
REFERENCE_CLS = os.path.join(CLS_DATA, 'reference-cls.npy')
DINOV2_WEIGHTS = os.path.join(CLS_DATA, 'dinov2_vitb14_pretrain.pth')
DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001'
SCRATCH = '/home/zhangcy/.dsh_scratch/vb'
IMAGE_EDGE = 560
KEYPOINT_BUDGET = 256      # producer `--budget 256`
DETECTION_THRESHOLD = 1e-4  # producer `--threshold` (used only with the fixed budget)
PER_KEYPOINT = 3           # capped candidates per query keypoint (`--per-keypoint 3`)
ASSOC_RADIUS_PX = 3.5      # `association_radius_px`
TOP_K_REFERENCES = 5       # provider: 5 per channel
NEIGHBOURS = 16            # filtered NN escalation base (verify_prov `--neighbours`)
SCORE_VOXEL = 0.4          # scoring-time query coarsening (--score-voxel 0.4 in the REF run)
PNP_TRIALS = 8             # the producer's optional `--fast-pnp` trial count


def _model_image(path):
    from PIL import Image
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


def _unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


def _kdtree(points):
    from scipy.spatial import cKDTree
    return cKDTree(points)


class VisualBranch(object):
    """One live visual hypothesis per call.  Construct once per process."""

    def __init__(self, device='cuda', arm='frozen_noprune', torch_threads=1,
                 scratch=SCRATCH, keypoint_budget=KEYPOINT_BUDGET, fast_pnp=False, score_voxel=SCORE_VOXEL,
                 verbose=False):
        if SP_PROJECT not in sys.path:
            sys.path.insert(0, SP_PROJECT)
        for extra in (SP_VENDOR, SP_VENDOR_PY):
            if extra not in sys.path:
                sys.path.insert(0, extra)
        # the LEADER stack already owns a module named `common`, so the producer's plan and
        # camera model are loaded by path under a private name
        import importlib.util
        path = os.path.join(SP_PROJECT, 'common.py')
        spec = importlib.util.spec_from_file_location('visual_branch_common', path)
        C = importlib.util.module_from_spec(spec)
        sys.modules['visual_branch_common'] = C
        spec.loader.exec_module(C)
        self.C = C
        self.plan = C.PLAN
        self.library = C.load(os.path.join(str(C.STORE), 'library.json'))
        self.references = self.library['references']
        self.positions = np.asarray([r['body_position_world'] for r in self.references],
                                    dtype=np.float64)
        self.T_BC = np.asarray(self.library['T_BC'], dtype=np.float64)
        self.matrix = C.model_matrix()
        self.verify_cfg = self.plan['verification']
        self.pnp_cfg = self.plan['PnP']
        self.scratch = scratch
        self.projection_cache = os.path.join(scratch, 'ref-projection')
        os.makedirs(self.projection_cache, exist_ok=True)
        self.verbose = verbose
        self.arm = arm
        self.budget = int(keypoint_budget)
        self.score_voxel = score_voxel
        self.fast_pnp = bool(fast_pnp)

        self.reference_cls = _unit(np.load(REFERENCE_CLS).astype(np.float32))
        self.device = device

        import torch
        import cv2
        torch.set_num_threads(int(torch_threads))
        torch.manual_seed(20)
        torch.hub.set_dir(SP_HUB)
        cv2.setNumThreads(1)
        self.torch = torch
        self.cv2 = cv2
        from lightglue import LightGlue, SuperPoint
        if not os.path.realpath(__import__('lightglue').__file__).startswith(SP_VENDOR):
            raise RuntimeError('wrong lightglue on sys.path: %s'
                               % __import__('lightglue').__file__)
        self.extractor = SuperPoint(max_num_keypoints=self.budget,
                                    detection_threshold=DETECTION_THRESHOLD).eval()
        kwargs = dict(features='superpoint', flash=False)
        if arm in ('frozen_noprune', 'v8_noprune', 'v9_noprune', 'frozen_default'):
            # the arms that disable early stopping and the width/depth filters
            kwargs['depth_confidence'] = -1
            kwargs['width_confidence'] = -1
        self.matcher = LightGlue(**kwargs).eval()
        weights = None
        if arm == 'v8_noprune':
            weights = '/data2/zhangcy/sp-lightglue-hard1502-20261013-v1/lightglue-finetune/best-lightglue.pt'
        if weights:
            payload = torch.load(weights, map_location='cpu')
            state = payload['model'] if isinstance(payload, dict) and 'model' in payload else payload
            self.matcher.load_state_dict(state, strict=True)
        for module in (self.extractor, self.matcher):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        if device == 'cuda':
            torch.cuda.set_device(0)
            self.extractor = self.extractor.cuda()
            self.matcher = self.matcher.cuda()
        self._features = {}
        self._projections = {}
        self._cls = None

    # ------------------------------------------------------------------ models
    def _cls_model(self):
        if self._cls is None:
            torch = self.torch
            model = torch.hub.load(os.path.join(CLS_ROOT, 'dinov2-source'), 'dinov2_vitb14',
                                   source='local', pretrained=False)
            model.load_state_dict(torch.load(DINOV2_WEIGHTS, map_location='cpu'))
            model = model.eval().requires_grad_(False)
            self._cls = model.cuda() if self.device == 'cuda' else model
        return self._cls

    def embed(self, image_path):
        """DINOv2 CLS embedding, the frozen image pipeline plus the frozen normalisation."""
        torch = self.torch
        model = self._cls_model()
        x = torch.from_numpy(_model_image(image_path)).permute(2, 0, 1).float() / 255
        x = ((x - torch.tensor([.485, .456, .406])[:, None, None])
             / torch.tensor([.229, .224, .225])[:, None, None])
        with torch.inference_mode():
            g = model.forward_features(x[None].to(self.device))['x_norm_clstoken'][0]
        return _unit(g.float().cpu().numpy())

    def extract(self, image_path):
        """SuperPoint features, the producer's extractor and image pipeline."""
        torch = self.torch
        x = torch.from_numpy(_model_image(image_path)).permute(2, 0, 1).float() / 255
        if self.device == 'cuda':
            x = x.cuda()
        with torch.inference_mode():
            return self.extractor.extract(x, resize=None)

    def _cached_features(self, image_path, tag):
        if tag in self._features:
            value = self._features.pop(tag)
            self._features[tag] = value
            return value
        value = self.extract(image_path)
        self._features[tag] = value
        if len(self._features) > 64:
            self._features.popitem(last=False)
        return value

    # ------------------------------------------------------- reference tables
    def _projection(self, reference):
        """model-pixel uv and world xyz of every valid return of one reference scan."""
        index = int(reference['index'])
        if index in self._projections:
            value = self._projections.pop(index)
            self._projections[index] = value
            return value
        name = '%06d-%s.npz' % (index, reference['scan_sha256'][:16])
        local = os.path.join(self.projection_cache, name)
        shared = os.path.join(str(self.C.STORE), 'ref-projection', name)
        path = local if os.path.isfile(local) else shared
        if os.path.isfile(path):
            with np.load(path, allow_pickle=False) as z:
                value = dict(model=z['model'].astype(np.float64),
                             world=z['world'].astype(np.float64))
        else:
            value = self._build_projection(reference)
            np.savez_compressed(local, model=value['model'].astype(np.float32),
                                world=value['world'].astype(np.float32))
        value['tree'] = _kdtree(value['model'])
        self._projections[index] = value
        if len(self._projections) > 64:
            self._projections.popitem(last=False)
        return value

    def _build_projection(self, reference):
        """exactly the precompute_provider.py construction, for references it never covered."""
        C = self.C
        index = int(reference['index'])
        key = reference['key']
        scan = os.path.join(DATA, 'NCLT', key.split('/')[0], 'velodyne_sync',
                            key.split('/')[1] + '.bin')
        body = C.decode(scan)
        body = body[np.isfinite(body).all(1)]
        camera = C.transform(body, np.linalg.inv(self.T_BC))
        good = np.isfinite(camera).all(1) & (camera[:, 2] > .5) & (camera[:, 2] < 120)
        camera = camera[good]
        pixel = camera @ C.K.T
        pixel = pixel[:, :2] / pixel[:, 2:3]
        good = (pixel[:, 0] >= 0) & (pixel[:, 0] < 808) & (pixel[:, 1] >= 0) & (pixel[:, 1] < 616)
        camera, pixel = camera[good], pixel[good]
        model = (np.column_stack((pixel, np.ones(len(pixel)))) @ self.matrix.T)[:, :2]
        good = (model[:, 0] >= 0) & (model[:, 0] < 420) & (model[:, 1] >= 0) & (model[:, 1] < 560)
        camera, model = camera[good], model[good]
        world = C.transform(camera, np.asarray(reference['T_WC']))
        if self.verbose:
            print('[visual] built projection for reference %d' % index, flush=True)
        return dict(model=model.astype(np.float64), world=world.astype(np.float64))

    def _map_cloud(self, index):
        """the reference's own labelled map cloud, from the shared cache."""
        row = self.references[int(index)]
        name = '%06d-%s.npz' % (int(index), row['scan_sha256'][:16])
        path = os.path.join(str(self.C.STORE), 'map-cloud', name)
        with np.load(path, allow_pickle=False) as z:
            if 'world' in z.files:
                return z['world'].astype(np.float64)
            return z['world_local'].astype(np.float64) + np.asarray(
                row['body_position_world'], dtype=np.float64)

    # ------------------------------------------------------------- retrieval
    def retrieve(self, image_path, pose_prior=None, channels=TOP_K_REFERENCES):
        """CLS cosine top-k union position-nearest-k, the frozen provider rule."""
        similarity = self.reference_cls @ self.embed(image_path).astype(np.float32)
        visual = [int(i) for i in np.argsort(-similarity)[:channels]]
        detail = dict(visual=visual, visual_cosine=[float(similarity[i]) for i in visual],
                      position=[])
        if pose_prior is not None:
            prior = np.asarray(pose_prior, dtype=np.float64)
            distance = np.linalg.norm(self.positions - prior[:3, 3], axis=1)
            detail['position'] = [int(i) for i in np.argsort(distance)[:channels]]
            detail['position_distance_m'] = [float(distance[i]) for i in detail['position']]
        merged = list(dict.fromkeys(detail['visual'] + detail['position']))
        return merged, detail

    # ------------------------------------------------------------- matching
    def _match_frame(self, image_path, reference_ids):
        """one batched LightGlue pass over the frame's references (per-pair fallback)."""
        torch = self.torch
        qfeatures = self.extract(image_path)
        q_model = qfeatures['keypoints'][0].detach().cpu().numpy().astype(np.float64)
        frame_features = [self._cached_features(self.references[ri]['image_path'], 'r%d' % ri)
                          for ri in reference_ids]
        results = {}
        batched = False
        if len(reference_ids) > 1:
            widths = {int(f['keypoints'].shape[1]) for f in frame_features}
            widths.add(int(qfeatures['keypoints'].shape[1]))
            if len(widths) == 1:
                image1 = {key: torch.cat([f[key] for f in frame_features], dim=0)
                          for key in frame_features[0] if key != 'image'}
                image0 = {key: qfeatures[key].repeat(len(frame_features),
                                                     *([1] * (qfeatures[key].dim() - 1)))
                          for key in qfeatures if key != 'image'}
                with torch.inference_mode():
                    out = self.matcher({'image0': image0, 'image1': image1})
                for b, ri in enumerate(reference_ids):
                    results[ri] = (out['matches'][b].detach().cpu().numpy().astype(np.int64),
                                   out['scores'][b].detach().cpu().numpy().astype(np.float64))
                batched = True
        if not batched:
            for ri, features in zip(reference_ids, frame_features):
                with torch.inference_mode():
                    result = self.matcher({'image0': qfeatures, 'image1': features})
                results[ri] = (result['matches'][0].detach().cpu().numpy().astype(np.int64),
                               result['scores'][0].detach().cpu().numpy().astype(np.float64))
        candidates, per_pair = [], {}
        for ri in reference_ids:
            table = self._projection(self.references[ri])
            r_model = frame_features[reference_ids.index(ri)]['keypoints'][0].detach().cpu().numpy().astype(np.float64)
            matches, scores = results[ri]
            if not len(matches):
                per_pair[ri] = dict(matches=0, bound=0)
                continue
            distance, nearest = table['tree'].query(r_model[matches[:, 1]])
            keep = distance <= ASSOC_RADIUS_PX
            per_pair[ri] = dict(matches=int(len(matches)), bound=int(keep.sum()))
            if not keep.any():
                continue
            for qi, a, s in zip(matches[keep, 0], nearest[keep], scores[keep]):
                candidates.append((int(qi), float(s), int(ri), int(a)))
        return q_model, candidates, per_pair, batched

    def _cap(self, candidates):
        ordered = sorted(candidates, key=lambda c: (-c[1], c[0], c[2]))
        kept, seen = [], {}
        for candidate in ordered:
            count = seen.get(candidate[0], 0)
            if count >= PER_KEYPOINT:
                continue
            seen[candidate[0]] = count + 1
            kept.append(candidate)
        return kept

    # ----------------------------------------------------------- pose estimate
    def _pose_result(self, camera_from_world, residual, ids, pixels, ix):
        """identical reporting to the frozen solver, given a solved camera."""
        if len(ix) < 8 or not np.isfinite(camera_from_world).all():
            return None
        unique = np.unique(ids[ix])
        cells = np.floor(pixels[ix] / [808 / 4, 616 / 4]).astype(int).clip(0, 3)
        coverage = len(np.unique(cells[:, 1] * 4 + cells[:, 0]))
        pose = np.linalg.inv(camera_from_world) @ np.linalg.inv(self.T_BC)
        return dict(pose=pose, inlier_indices=ix, unique_query_inliers=int(len(unique)),
                    inlier_ratio=float(len(unique) / max(1, len(np.unique(ids)))),
                    median_reprojection_px=float(np.median(residual[ix])),
                    coverage_cells=int(coverage),
                    qualified=bool(len(unique) >= self.pnp_cfg['min_inliers']
                                   and coverage >= self.pnp_cfg['min_4x4_cells']
                                   and len(unique) / max(1, len(np.unique(ids)))
                                   >= self.pnp_cfg['min_inlier_ratio']))

    def _camera_of(self, rv, tv):
        camera_from_world = np.eye(4)
        camera_from_world[:3, :3] = self.cv2.Rodrigues(rv)[0]
        camera_from_world[:3, 3] = tv.ravel()
        return camera_from_world

    def _residuals(self, camera_from_world, world, pixels):
        camera = self.C.transform(world, camera_from_world)
        projected = camera @ self.C.K.T
        residual = np.linalg.norm(
            projected[:, :2] / np.maximum(projected[:, 2:3], 1e-12) - pixels, axis=1)
        return residual, camera

    def solve(self, world, pixels, query_ids):
        """the producer's `--fast-pnp 1` deterministic solver, else the frozen RANSAC."""
        if not self.fast_pnp:
            return self.C.solve_pnp(world, pixels, query_ids, self.T_BC)
        cv2 = self.cv2
        world = np.asarray(world, dtype=np.float64)
        pixels = np.asarray(pixels, dtype=np.float64)
        ids = np.asarray(query_ids, dtype=np.int32)
        if len(world) < 8:
            return None
        n = len(world)
        centre = world.mean(0)
        current = int(np.argmax(((world - centre) ** 2).sum(1)))
        distance = np.linalg.norm(world - world[current], axis=1)
        order = []
        for _ in range(min(n, 4 * PNP_TRIALS)):
            index = int(np.argmax(distance))
            order.append(index)
            distance = np.minimum(distance, np.linalg.norm(world - world[index], axis=1))
        if len(order) < 4:
            return None
        best = None
        for trial in range(PNP_TRIALS):
            pick = [order[(trial + j * PNP_TRIALS) % len(order)] for j in range(4)]
            if len(set(pick)) < 4:
                continue
            ok, rv, tv = cv2.solvePnP(world[pick], pixels[pick], self.C.K, None,
                                      flags=cv2.SOLVEPNP_EPNP)
            if not ok:
                continue
            residual, camera = self._residuals(self._camera_of(rv, tv), world, pixels)
            inliers = np.flatnonzero((residual <= 12) & (camera[:, 2] > .5))
            if best is None or len(inliers) > len(best[0]):
                best = (inliers, rv, tv)
            if len(inliers) >= 0.6 * n:
                break
        if best is None or len(best[0]) < 8:
            return None
        ix = best[0]
        rv, tv = cv2.solvePnPRefineLM(world[ix], pixels[ix], self.C.K, None, best[1], best[2])
        residual, camera = self._residuals(self._camera_of(rv, tv), world, pixels)
        ix = np.flatnonzero((residual <= 12) & (camera[:, 2] > .5))
        return self._pose_result(self._camera_of(rv, tv), residual, ids, pixels, ix)

    def _pnp(self, ids, world, reference_of, q_model):
        solution = self.solve(world, self.C.model_to_raw(self.matrix, q_model[ids]), ids)
        if solution is None:
            return None
        observed = [reference_of[j] for j in solution['inlier_indices']]
        reference_ids = sorted({r for r in observed if observed.count(r) >= 3})
        return dict(pose=np.asarray(solution['pose']).tolist(),
                    unique_query_inliers=int(solution['unique_query_inliers']),
                    inlier_ratio=float(solution['inlier_ratio']),
                    median_reprojection_px=float(solution['median_reprojection_px']),
                    coverage_cells=int(solution['coverage_cells']),
                    qualified=bool(solution['qualified']),
                    reference_ids=reference_ids)

    def _hypotheses(self, q_model, candidates, by_reference):
        """per-reference, whole-set and per-cluster PnP: the producer's three channels."""
        hypotheses = []
        calls = 0
        for ri in sorted(by_reference, key=lambda i: -len(by_reference[i][0])):
            ids, aid = by_reference[ri]
            table = self._projection(self.references[ri])
            solution = self.solve(table['world'][aid],
                                  self.C.model_to_raw(self.matrix, q_model[ids]), ids)
            calls += 1
            if solution is None:
                continue
            hypotheses.append(dict(
                channel='pair', reference_ids=[int(ri)],
                pose=np.asarray(solution['pose']).tolist(),
                unique_query_inliers=int(solution['unique_query_inliers']),
                inlier_ratio=float(solution['inlier_ratio']),
                median_reprojection_px=float(solution['median_reprojection_px']),
                coverage_cells=int(solution['coverage_cells']),
                qualified=bool(solution['qualified'])))
        capped = self._cap(candidates)
        if len(capped) >= 8:
            ids = np.array([c[0] for c in capped])
            world = np.array([self._projection(self.references[c[2]])['world'][c[3]]
                              for c in capped])
            solution = self._pnp(ids, world, [c[2] for c in capped], q_model)
            calls += 1
            if solution is not None:
                solution['channel'] = 'full'
                hypotheses.append(solution)
        clusters = []
        mu = float(self.plan['matcher']['local_region_radius_m'])
        for ri in sorted(by_reference, key=lambda i: -float(by_reference[i][1].max())):
            distances = [float(np.linalg.norm(self.positions[ri] - self.positions[group[0]]))
                         for group in clusters]
            near = int(np.argmin(distances)) if distances else None
            if near is None or distances[near] > mu:
                clusters.append([ri])
            else:
                clusters[near].append(ri)
        for cluster in clusters:
            rows = [c for c in capped if c[2] in set(cluster)]
            if len(rows) < 8:
                continue
            ids = np.array([c[0] for c in rows])
            world = np.array([self._projection(self.references[c[2]])['world'][c[3]]
                              for c in rows])
            solution = self._pnp(ids, world, [c[2] for c in rows], q_model)
            calls += 1
            if solution is None:
                continue
            solution['channel'] = 'cluster'
            hypotheses.append(solution)
        qualified = [h for h in hypotheses if h['qualified']]
        for solution in qualified:
            p = np.asarray(solution['pose'])
            support = set()
            for other in qualified:
                op = np.asarray(other['pose'])
                translation = float(np.linalg.norm(p[:3, 3] - op[:3, 3]))
                rotation = float(np.rad2deg(np.arccos(np.clip(
                    (np.trace(p[:3, :3].T @ op[:3, :3]) - 1) / 2, -1, 1))))
                if (translation <= self.pnp_cfg['consensus_translation_m']
                        and rotation <= self.pnp_cfg['consensus_rotation_deg']):
                    support.update(other['reference_ids'])
            solution['consensus_reference_count'] = len(support)
        accepted = [s for s in qualified
                    if s.get('consensus_reference_count', 0)
                    >= self.pnp_cfg['min_supporting_reference_images']]
        # the frozen table scores EVERY hypothesised pose, qualified or not, and the offline
        # rule then keeps the highest fair score; keep the same superset here and let the
        # caller's budget decide how many of them pay for verification
        for solution in hypotheses:
            solution.setdefault('consensus_reference_count', 0)
        hypotheses.sort(key=lambda s: (-s['unique_query_inliers'], -s['coverage_cells'],
                                       s['median_reprojection_px']))
        return hypotheses, accepted, calls, len(capped)

    # --------------------------------------------------- laser verification
    def _query_cloud(self, scan, score_voxel=None):
        """"""
        C = self.C
        body = np.asarray(scan, dtype=np.float64)
        body = body[np.isfinite(body).all(1)]
        body = body[np.linalg.norm(body, axis=1) <= self.verify_cfg['query_range_m']]
        voxel = float(score_voxel) if score_voxel else self.verify_cfg['query_voxel_m']
        return C.voxel_downsample(body, voxel)

    def _cluster_cloud(self, centre):
        radius = self.verify_cfg['map_radius_m']
        near = np.flatnonzero(np.linalg.norm(self.positions - centre, axis=1) <= radius)
        if not len(near):
            return None
        parts = [self._map_cloud(i) for i in near]
        labels = np.concatenate([np.full(len(p), int(i), dtype=np.int32)
                                 for p, i in zip(parts, near)])
        cloud = np.concatenate(parts)
        return cloud, labels

    def verify(self, pose, query_cloud, excluded):
        """fair per-candidate laser score: the pose sees every reference except its own."""
        cfg = self.verify_cfg
        pose = np.asarray(pose, dtype=np.float64)
        placed = self.C.transform(query_cloud, pose)
        built = self._cluster_cloud(pose[:3, 3])
        out = dict(valid=False, map_references=0, map_points=0, residual_median_m=None,
                   residual_p90_m=None, supported_bins=0, bin_overlap_median=None,
                   top_bin_share=None, geometry_score=0.0, geometry_verified=False,
                   nearest_unresolved=0, excluded=len(excluded))
        if built is None:
            out['nearest_unresolved'] = int(len(placed))
            return out
        cloud, labels = built
        tree = _kdtree(cloud)
        out['map_points'] = int(len(cloud))
        # one flag per library reference: a pose's own references must never be its own
        # support, and an excluded reference need not be inside this cluster
        available = np.ones(len(self.positions), dtype=bool)
        excluded = sorted(set(int(x) for x in excluded))
        if excluded:
            available[np.asarray(excluded, dtype=np.int64)] = False
        own = np.linalg.norm(self.positions - pose[:3, 3], axis=1)
        near = own <= cfg['map_radius_m']
        if excluded:
            near = near.copy()
            near[np.asarray(excluded, dtype=np.int64)] = False
        out['map_references'] = int(near.sum())
        disallowed = ~available[labels]
        n = len(placed)
        nearest = np.full(n, np.inf)
        done = np.zeros(n, dtype=bool)
        rows_all = np.arange(n)
        limit = len(cloud)
        for k in (NEIGHBOURS, NEIGHBOURS * 4, NEIGHBOURS * 16):
            todo = rows_all[~done]
            if not len(todo):
                break
            distance, index = tree.query(placed[todo], k=min(k, limit), workers=1)
            if distance.ndim == 1:
                distance = distance[:, None]
                index = index[:, None]
            ok = ~disallowed[index]
            any_ok = ok.any(1)
            if any_ok.any():
                first = np.argmax(ok, axis=1)
                nearest[todo[any_ok]] = distance[np.arange(len(todo)), first][any_ok]
                done[todo[any_ok]] = True
        out['nearest_unresolved'] = int((~done).sum())
        out['valid'] = True
        primary = cfg['primary_nn_threshold_m']
        for threshold in cfg['nn_thresholds_m']:
            out['overlap_%g' % threshold] = float(np.mean(nearest <= threshold))
        inside = nearest <= cfg['residual_window_m']
        if int(inside.sum()) >= 20:
            out['residual_median_m'] = float(np.median(nearest[inside]))
            out['residual_p90_m'] = float(np.percentile(nearest[inside], 90))
        azimuth = np.arctan2(query_cloud[:, 1], query_cloud[:, 0])
        bins = int(cfg['azimuth_bins'])
        bin_index = np.floor((azimuth + np.pi) / (2 * np.pi) * bins).astype(int).clip(0, bins - 1)
        overlaps, supported = [], 0
        points = np.zeros(bins)
        for b in range(bins):
            mask = bin_index == b
            if int(mask.sum()) < int(cfg['min_points_per_bin']):
                continue
            fraction = float(np.mean(nearest[mask] <= primary))
            overlaps.append(fraction)
            points[b] = int((nearest[mask] <= primary).sum())
            if fraction >= cfg['bin_support_overlap']:
                supported += 1
        total = int(points.sum())
        out['supported_bins'] = supported
        out['bin_overlap_median'] = float(np.median(overlaps)) if overlaps else None
        out['top_bin_share'] = float(points.max() / total) if total else None
        out['geometry_score'] = (out['bin_overlap_median']
                                 if out['bin_overlap_median'] is not None else 0.0)
        out['geometry_verified'] = bool(
            out['bin_overlap_median'] is not None
            and out['bin_overlap_median'] >= cfg['bin_support_overlap']
            and supported >= cfg['min_supported_bins']
            and out['residual_median_m'] is not None
            and out['residual_median_m'] <= cfg['residual_median_max_m']
            and out['residual_p90_m'] is not None
            and out['residual_p90_m'] <= cfg['residual_p90_max_m']
            and out['top_bin_share'] is not None
            and out['top_bin_share'] <= cfg['max_top_bin_share'])
        return out

    # ------------------------------------------------------------- one frame
    def hypothesize(self, key, image_path, budget=8, pose_prior=None, scan=None,
                    scan_path=None):
        """Return the best verified visual pose for this frame, or None.

        budget = how many qualified hypotheses are laser-scored (the expensive part);
        they are taken in the producer's order (most unique inliers first).
        """
        started = time.monotonic()
        key = str(key)
        if scan is None:
            if scan_path is None:
                date, stamp = key.split('/')
                scan_path = os.path.join(DATA, 'NCLT', date, 'velodyne_sync', stamp + '.bin')
            scan = self.C.decode(scan_path)
        query_cloud = self._query_cloud(scan, self.score_voxel)
        if len(query_cloud) < 500:
            return None

        references, retrieval = self.retrieve(image_path, pose_prior)
        if not references:
            return None
        t_retrieval = time.monotonic() - started
        t0 = time.monotonic()
        q_model, candidates, per_pair, batched = self._match_frame(image_path, references)
        t_match = time.monotonic() - t0
        if not candidates:
            return None
        by_reference = {}
        for candidate in candidates:
            entry = by_reference.setdefault(candidate[2], ([], []))
            entry[0].append(candidate[0])
            entry[1].append(candidate[3])
        by_reference = {ri: (np.array(ids), np.array(aid))
                        for ri, (ids, aid) in by_reference.items()}
        t0 = time.monotonic()
        hypotheses, accepted, pnp_calls, capped = self._hypotheses(q_model, candidates,
                                                                  by_reference)
        t_pnp = time.monotonic() - t0
        if not hypotheses:
            return None

        t0 = time.monotonic()
        scored = []
        for solution in hypotheses[:max(1, int(budget))]:
            row = self.verify(solution['pose'], query_cloud, solution['reference_ids'])
            scored.append(dict(pose=np.asarray(solution['pose'], dtype=np.float64),
                               score=float(row['geometry_score']),
                               verified=bool(row['geometry_verified']),
                               channel=solution['channel'],
                               reference_ids=list(solution['reference_ids']),
                               unique_query_inliers=solution['unique_query_inliers'],
                               consensus=int(solution.get('consensus_reference_count', 0)),
                               fair=row))
        t_verify = time.monotonic() - t0
        scored.sort(key=lambda r: -r['score'])
        best = scored[0]
        seconds = time.monotonic() - started
        detail = dict(key=key, arm=self.arm, batched=bool(batched),
                      retrieval=retrieval,
                      per_pair=dict((str(k), v) for k, v in per_pair.items()),
                      candidates=len(candidates), capped=capped,
                      pnp_calls=pnp_calls, hypotheses=len(hypotheses),
                      consensus_accepted=len(accepted), scored=len(scored),
                      scores=[dict(channel=r['channel'], score=round(r['score'], 6),
                                   verified=r['verified'],
                                   reference_ids=r['reference_ids'],
                                   inliers=r['unique_query_inliers'],
                                   consensus=r['consensus'],
                                   residual_median_m=r['fair']['residual_median_m'],
                                   residual_p90_m=r['fair']['residual_p90_m'],
                                   supported_bins=r['fair']['supported_bins'],
                                   top_bin_share=r['fair']['top_bin_share'])
                              for r in scored],
                      seconds=dict(retrieval=round(t_retrieval, 4),
                                   match=round(t_match, 4), pnp=round(t_pnp, 4),
                                   verify=round(t_verify, 4)),
                      total_seconds=round(seconds, 4))
        return dict(pose=best['pose'], score=best['score'], n_candidates=len(hypotheses),
                    seconds=seconds, detail=detail, verified=best['verified'],
                    channel=best['channel'], reference_ids=best['reference_ids'])