import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from oracle_pose_refinement import load_module


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_rows(rows, split, limit):
    accepted = ("train",) if split == "train" else ("val", "validation", "test")
    output = [row for row in rows if row["split"] in accepted]
    return output[:limit] if limit else output


def local_to_world(pose, center):
    output = pose.detach().cpu().numpy().astype(np.float64)
    output[:3, 3] += np.asarray(center, dtype=np.float64)
    return output


def stage_poses(source, prediction, center, huber, full_pool, device):
    import torch

    source = torch.as_tensor(source, dtype=torch.float32, device=device)
    prediction = torch.as_tensor(prediction, dtype=torch.float32, device=device)
    if source.ndim != 2 or source.shape[1] != 3 or prediction.ndim != 2 or prediction.shape[1] < 4:
        raise ValueError("source must be [N,3] and prediction must be [N,4+]")
    if len(source) != len(prediction) or not len(source):
        raise ValueError("source and prediction must have the same non-zero length")
    keep_count = max(min(50, len(prediction)), int(.5 * len(prediction)))
    keep = prediction[:, 3].topk(keep_count).indices
    b0_local = huber(source[keep][None], prediction[keep, :3][None], delta=.5, iters=10)[0]
    g1_output = full_pool(b0_local, source, prediction[:, :3])
    g1_local = g1_output[0] if isinstance(g1_output, (tuple, list)) else g1_output
    return local_to_world(b0_local, center), local_to_world(g1_local, center), keep.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="G1 cached two-stage refinement: IRLS-Huber top50% followed by full-pool tightening.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lidar-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluate-split", default="validation", choices=("validation", "train"))
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--irls-module", default=str(REPO.parent / "IRLS-Huber_全量结果产物" / "irls_huber.py"))
    parser.add_argument("--full-pool", default=str(REPO.parent / "glace-local" / "code" / "tools" / "full_pool_robust_v1.py"))
    args = parser.parse_args()
    if args.frames < 0:
        parser.error("frames must be non-negative")

    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rows = selected_rows(rows, args.evaluate_split, args.frames)
    if not rows:
        raise ValueError("no selected frames")
    huber_module = load_module("g1_irls_huber", Path(args.irls_module))
    pool_module = load_module("g1_full_pool", Path(args.full_pool))
    records = []
    for index, row in enumerate(rows):
        frame_id = str(row["frame_id"])
        with np.load(Path(args.lidar_cache) / (frame_id + ".npz")) as cached:
            source = np.asarray(cached["source"])
            prediction = np.asarray(cached["prediction"])
            center = np.asarray(cached["center"])
        b0, g1, keep = stage_poses(source, prediction, center, huber_module.huber_rigid_torch,
                                   pool_module.full_pool_refine, args.device)
        records.append({"frame_id": frame_id, "initial_pose": b0.tolist(),
                        "source_count": int(len(source)), "topk_count": int(len(keep)),
                        "variants": {"G1_staged": {"final_pose": g1.tolist(), "run_failed": False}}})
        print("%s %d/%d %s topk=%d" % (args.evaluate_split, index + 1, len(rows), frame_id, len(keep)), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "protocol": {
            "name": "G1 staged LiDAR refinement on cached local frames",
            "B0": "IRLS-Huber top50%, delta=0.5m, iters=10",
            "G1": "B0 followed by full-pool Tukey tightening thresholds=(1.2m, 0.6m)",
            "query_gt_in_runner": False,
            "manifest": str(args.manifest), "manifest_sha256": digest_file(args.manifest),
            "irls_module": str(args.irls_module), "irls_module_sha256": digest_file(args.irls_module),
            "full_pool": str(args.full_pool), "full_pool_sha256": digest_file(args.full_pool),
            "frames": len(records), "split": args.evaluate_split,
        }, "records": records}, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
