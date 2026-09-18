"""Extract frozen PCA128 dense descriptors for the six query images."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from local_visual_refinement import DenseDescriptorExtractor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dedode-weights", required=True)
    parser.add_argument("--pca-weights", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frames", type=int, default=0)
    args = parser.parse_args()
    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if args.frames:
        rows = rows[:args.frames]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    extractor = DenseDescriptorExtractor(args.device, args.dedode_weights, args.pca_weights)
    for index, row in enumerate(rows):
        target = output / (row["frame_id"] + ".npz")
        if target.exists():
            continue
        values = {}
        with torch.no_grad():
            for view in sorted(row["views"], key=lambda item: item["camera"]):
                dense, image_hw = extractor.image(row["frame_id"], int(view["camera"]), view["image"])
                values["cam%d" % int(view["camera"])] = dense.cpu().numpy().astype(np.float16)
        np.savez_compressed(target, **values, image_hw=np.asarray(image_hw, dtype=np.int32))
        print("descriptor frame %d/%d %s shape=%s" %
              (index + 1, len(rows), row["frame_id"], values["cam0"].shape), flush=True)


if __name__ == "__main__":
    main()
