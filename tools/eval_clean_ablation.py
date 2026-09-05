import argparse
import json
import os
import sys
import time

import MinkowskiEngine as ME
import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from full_pool_robust_v1 import refine
from data.NCLTVelodyne_datagenerator_mink import NCLT_mink
from models.model_mink import LEADER
from models.sc2pcr import Matcher
from utils.pose_util import polar_expansion_to_cartesian


class IndexedSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source_index = self.indices[index]
        return source_index, self.dataset[source_index]


def collate_samples(indexed_samples):
    indices, samples = zip(*indexed_samples)
    coords, feats, points, transform, correction = zip(*samples)
    return {
        "indices": indices,
        "coords": ME.utils.batched_coordinates(coords),
        "feats": torch.from_numpy(np.concatenate(feats)).float(),
        "points": points,
        "T": torch.from_numpy(np.stack(transform)).float(),
        "T_corr": torch.from_numpy(np.stack(correction)).float(),
    }


def selected_indices(dataset, stride, offset, limit):
    indices = []
    per_sequence = {}
    for index, path in enumerate(dataset.pcs):
        sequence = os.path.basename(os.path.dirname(os.path.dirname(path)))
        position = per_sequence.get(sequence, 0)
        per_sequence[sequence] = position + 1
        if position % stride != offset % stride:
            continue
        indices.append(index)
        if limit > 0 and len(indices) >= limit:
            break
    return indices


def pose_error(centered, center_t, correction, ground_truth):
    final = centered.clone()
    final[:3, 3] += center_t
    final = final @ correction
    translation = torch.linalg.norm(final[:3, 3] - ground_truth[:3, 3]).item()
    relative = final[:3, :3].T @ ground_truth[:3, :3]
    cosine = ((torch.trace(relative) - 1.0) / 2.0).clamp(-1.0, 1.0)
    rotation = torch.rad2deg(torch.acos(cosine)).item()
    return translation, rotation


def summarize(rows):
    translation = np.asarray([row[0] for row in rows], dtype=np.float64)
    rotation = np.asarray([row[1] for row in rows], dtype=np.float64)
    return {
        "frames": int(len(rows)),
        "mean_t": float(translation.mean()),
        "mean_r": float(rotation.mean()),
        "median_t": float(np.median(translation)),
        "median_r": float(np.median(rotation)),
        "recall_0.5m_1deg": float(np.mean((translation < 0.5) & (rotation < 1.0))),
        "recall_1m_2deg": float(np.mean((translation < 1.0) & (rotation < 2.0))),
        "p95_t": float(np.quantile(translation, 0.95)),
        "p95_r": float(np.quantile(rotation, 0.95)),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-folder", default="/root/rivermind-data/datasets")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("baseline", "released_refine", "top_refine", "full_refine"),
        default=("baseline", "released_refine", "top_refine", "full_refine"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = NCLT_mink(args.dataset_folder, train=False, voxel_size=0.2, horizontal_res=1024)
    indices = selected_indices(dataset, args.stride, args.offset, args.limit)
    loader = DataLoader(
        IndexedSubset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_samples,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    model = LEADER(in_channels=3, out_channels=4, feat_channels=512, width=1024)
    model.load_state_dict(
        load_file(os.path.join(args.checkpoint, "model.safetensors"), device="cpu"),
        strict=True,
    )
    model.cuda().eval()
    with open(os.path.join(args.checkpoint, "extra.json"), "r", encoding="utf-8") as handle:
        center = torch.tensor(json.load(handle)["center_t"], device="cuda", dtype=torch.float32)
    matcher = Matcher(
        inlier_threshold=2.0,
        d_thre=2,
        num_iterations=10,
        ratio=0.15,
        nms_radius=0.1,
        max_points=3000,
        k1=30,
    )
    names = tuple(args.methods)
    results = {name: [] for name in names}
    sequences = []
    frame_indices = []
    started = time.time()
    with torch.inference_mode():
        for batch in loader:
            coordinates = batch["coords"].cuda()
            features = batch["feats"].cuda()
            encoded = model.encoder(ME.SparseTensor(features, coordinates))
            prediction = model.decoder(encoded.F).float()
            batch_index = encoded.C[:, 0].long()
            tensor_stride = torch.tensor(encoded.tensor_stride, device="cuda", dtype=torch.float32)
            centers = (encoded.C[:, 1:].float() + tensor_stride / 2) * 0.2
            local = polar_expansion_to_cartesian(centers, 204.8)
            for position, dataset_index in enumerate(batch["indices"]):
                mask = batch_index == position
                source = local[mask]
                target = prediction[mask, :3]
                reliability = prediction[mask, 3]
                keep = max(min(50, reliability.numel()), int(0.5 * reliability.numel()))
                top = torch.topk(reliability, keep).indices
                initial = matcher.estimator(source[top][None], target[top][None])[0]
                top_budget = top[: min(3000, top.numel())]
                poses = {"baseline": initial}
                if "released_refine" in names:
                    poses["released_refine"] = matcher.post_refinement(
                        initial[None], source[top_budget][None], target[top_budget][None], 20
                    )[0]
                if "top_refine" in names:
                    poses["top_refine"], _ = refine(
                        initial, source[top_budget], target[top_budget],
                        (1.2, 0.8, 0.6),
                    )
                if "full_refine" in names:
                    poses["full_refine"], _ = refine(
                        initial, source, target,
                        (2.0, 1.2, 0.8, 0.6),
                    )
                correction = batch["T_corr"][position].cuda().float()
                ground_truth = batch["T"][position].cuda().float()
                for name in names:
                    results[name].append(
                        pose_error(poses[name], center, correction, ground_truth)
                    )
                path = dataset.pcs[dataset_index]
                sequences.append(os.path.basename(os.path.dirname(os.path.dirname(path))))
                frame_indices.append(int(dataset_index))
            count = len(frame_indices)
            if count <= args.batch_size or count % 500 < args.batch_size or count == len(indices):
                print(f"evaluated {count}/{len(indices)} in {time.time() - started:.1f}s", flush=True)
    summary = {name: summarize(rows) for name, rows in results.items()}
    sequence_summary = {}
    for sequence in sorted(set(sequences)):
        mask = np.asarray(sequences) == sequence
        sequence_summary[sequence] = {
            name: summarize(np.asarray(rows)[mask]) for name, rows in results.items()
        }
    report = {
        "checkpoint": args.checkpoint,
        "align_sparse_cat": os.environ.get("LEADER_ALIGN_SPARSE_CAT") == "1",
        "periodic_all": os.environ.get("LEADER_PERIODIC_ALL") == "1",
        "frames": len(frame_indices),
        "elapsed_seconds": time.time() - started,
        "summary": summary,
        "sequence_summary": sequence_summary,
        "indices": frame_indices,
        "sequences": sequences,
        "frame_errors": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
