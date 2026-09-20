import argparse
import json
from pathlib import Path

import numpy as np


def summary(before, estimate, truth):
    delta = estimate - truth
    error = np.linalg.norm(delta, axis=1)
    before_error = np.linalg.norm(before - truth, axis=1)
    accurate = before_error < 2.
    middle = (before_error >= 2.) & (before_error < 5.)
    return {"count": int(len(error)), "median_px": float(np.median(error)), "p90_px": float(np.quantile(error, .9)),
            "lt_1px": float((error < 1.).mean()), "lt_2px": float((error < 2.).mean()),
            "lt_5px": float((error < 5.).mean()), "accurate_count": int(accurate.sum()),
            "accurate_ruined_fraction": float((accurate & (error >= 2.)).sum() / max(accurate.sum(), 1)),
            "middle_to_lt_2_fraction": float((middle & (error < 2.)).sum() / max(middle.sum(), 1)),
            "mean_du": float(delta[:, 0].mean()), "mean_dv": float(delta[:, 1].mean())}


def group_keys(data):
    return np.asarray(["|".join(map(str, row)) for row in np.column_stack((data["frame_ids"], data["camera_ids"],
                                                                              data["reference_frames"], data["reference_cameras"]))])


def shared_linear(train, validation):
    train_group = group_keys(train)
    val_group = group_keys(validation)
    features, targets, val_features = [], [], []
    for group in np.unique(train_group):
        keep = train_group == group
        features.append(np.r_[1., np.median(train["roma_pixels"][keep] - train["lidar_pixels"][keep], axis=0)])
        targets.append(np.median(train["gt_pixels"][keep] - train["lidar_pixels"][keep], axis=0))
    for group in np.unique(val_group):
        keep = val_group == group
        val_features.extend([np.r_[1., np.median(validation["roma_pixels"][keep] - validation["lidar_pixels"][keep], axis=0)]] * int(keep.sum()))
    design, target = np.asarray(features), np.asarray(targets)
    weights = np.linalg.solve(design.T @ design + 1e-3 * np.eye(design.shape[1]), design.T @ target)
    return validation["lidar_pixels"] + np.asarray(val_features) @ weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-inputs", required=True)
    parser.add_argument("--train-labels", required=True)
    parser.add_argument("--validation-inputs", required=True)
    parser.add_argument("--validation-labels", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    def load(inputs, labels):
        with np.load(inputs, allow_pickle=False) as x, np.load(labels, allow_pickle=False) as y:
            data = {key: x[key] for key in x.files if key != "metadata"}
            data.update({key: y[key] for key in y.files if key != "metadata"})
        keep = data["gt_visible"] & np.isfinite(data["gt_pixels"]).all(axis=1)
        return {key: value[keep] for key, value in data.items()}
    train, validation = load(args.train_inputs, args.train_labels), load(args.validation_inputs, args.validation_labels)
    direct = validation["lidar_pixels"]
    v2 = validation["v2_pixels"]
    shared = shared_linear(train, validation)
    train_delta = train["roma_pixels"] - train["lidar_pixels"]
    train_target = train["gt_pixels"] - train["lidar_pixels"]
    train_sigma = np.sqrt(np.maximum(np.diagonal(train["lidar_covariance"], axis1=1, axis2=2), 1e-8))
    x = np.column_stack((np.ones(len(train_delta)), train_delta, np.log(np.maximum(train_sigma, 1e-3))))
    w = np.linalg.solve(x.T @ x + 1e-3 * np.eye(x.shape[1]), x.T @ train_target)
    validation_sigma = np.sqrt(np.maximum(np.diagonal(validation["lidar_covariance"], axis1=1, axis2=2), 1e-8))
    validation_delta = validation["roma_pixels"] - validation["lidar_pixels"]
    vx = np.column_stack((np.ones(len(validation_delta)), validation_delta,
                          np.log(np.maximum(validation_sigma, 1e-3))))
    pointwise = validation["lidar_pixels"] + vx @ w
    result = {"protocol": "V2-G geometry diagnostic; train GT is used only to fit train-frame models",
              "samples": {"train": int(len(train["gt_pixels"])), "validation": int(len(validation["gt_pixels"]))},
              "comparison": {"direct_lidar": summary(direct, direct, validation["gt_pixels"]),
                             "frozen_v2": summary(direct, v2, validation["gt_pixels"]),
                             "v2g_pointwise_raw_scale": summary(direct, pointwise, validation["gt_pixels"]),
                             "v2g_shared_linear": summary(direct, shared, validation["gt_pixels"])} }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
