"""Retrieval recall harness on the frozen 1502/4752 DINOv2 benchmark.

Reproduces the published CLS/VLAD numbers from the stored top-200 tables, then scores any
descriptor matrix (queries x dim, references x dim) with the same protocol so a new compact
descriptor can be compared on identical ground truth.
"""
import json
import sys

import numpy as np

S = "/dev/shm/zhangcy-v7-dinov2-recall-20261012-v1/"
PLAN = json.load(open(S + "image-plan.json"))
RADII = (5.0, 10.0, 25.0, 50.0)
TOPK = (1, 5, 10, 50, 200)


def load_truth(keys):
    dates = {}
    out = {}
    for key in keys:
        date = key.split("/")[0]
        if date not in dates:
            dates[date] = json.load(open(
                "/data2/zhangcy/leader-v2-full-trajectory-20261001/evaluator-only-truth/%s.json" % date))
        out[key] = np.asarray(dates[date][key.split("/")[1]]["T_WB"], dtype=np.float64)
    return out


def recall_from_distances(distance, radii=RADII, topk=TOPK):
    """distance: (queries, K) distance to each ranked reference, nearest first."""
    return {str(r): {str(k): float((distance[:, :k].min(axis=1) <= r).mean()) for k in topk}
            for r in radii}


def main():
    with np.load("/dev/shm/zhangcy-v7-dinov2-recall-20261012-v1/retrieval-top200.npz") as z:
        query_keys = [str(k) for k in z["query_keys"]]
        reference_keys = [str(k) for k in z["reference_keys"]]
        positions = z["reference_positions"]
        cls_indices = z["cls_indices"]
        vlad_indices = z["vlad_indices"]
    plan_positions = {r["key"]: np.asarray(r["position_world"], dtype=np.float64)
                      for r in PLAN["references"]}
    if any(k not in plan_positions for k in reference_keys):
        print("warning: %d reference keys missing from the plan" % sum(
            1 for k in reference_keys if k not in plan_positions))
    truth = load_truth(query_keys)
    query_positions = np.stack([truth[k][:3, 3] for k in query_keys])

    print("queries %d references %d" % (len(query_keys), len(reference_keys)))
    for name, indices in (("cls", cls_indices), ("vlad", vlad_indices)):
        distance = np.linalg.norm(query_positions[:, None, :] - positions[indices], axis=-1)
        result = recall_from_distances(distance)
        print("== %s ==" % name)
        for radius in RADII:
            print("  radius %4.0fm  " % radius + "  ".join(
                "top%-3d %.3f" % (k, result[str(radius)][str(k)]) for k in TOPK))
    np.savez_compressed("/home/zhangcy/projects/nclt-cand-audit-20261023-v1/src/benchmark.npz",
                        query_keys=np.array(query_keys), reference_keys=np.array(reference_keys),
                        query_positions=query_positions, reference_positions=positions)
    print("benchmark saved")


if __name__ == "__main__":
    main()
