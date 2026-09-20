import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with np.load(args.checkpoint) as data:
        cameras = np.asarray(data["cameras"], dtype=np.int64)
        delta = np.asarray(data["delta"], dtype=np.float64)
    if delta.shape != (len(cameras), 2):
        raise ValueError("checkpoint camera/delta shape mismatch")
    bias = []
    for camera in range(6):
        values = delta[cameras == camera]
        if not len(values):
            raise ValueError("camera %d has no train correspondences" % camera)
        bias.append({"camera": camera, "count": int(len(values)),
                     "du_px": float(np.median(values[:, 0])), "dv_px": float(np.median(values[:, 1]))})
    result = {"protocol": {"name": "RoMa per-camera fixed bias fit", "split": "train only",
                            "estimator": "per-camera median of RoMa pixel delta: prediction minus GT",
                            "inference": "subtract the frozen bias from RoMa query pixels; no GT is used"},
              "camera_bias_px": bias}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
