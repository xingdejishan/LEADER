import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from lscr_refinement import summarize
from xrefine_adapter import XRefineAdapter


def metric(before, after, truth):
    before_error = np.linalg.norm(before - truth, axis=1)
    after_error = np.linalg.norm(after - truth, axis=1)
    result = summarize(after_error, after - truth)
    accurate = before_error < 2.
    middle = (before_error >= 2.) & (before_error < 5.)
    result.update({"lt_1px_fraction": float((after_error < 1.).mean()),
                   "accurate_count": int(accurate.sum()),
                   "accurate_ruined_count": int((accurate & (after_error >= 2.)).sum()),
                   "accurate_ruined_fraction": float((accurate & (after_error >= 2.)).sum() / max(accurate.sum(), 1)),
                   "middle_to_lt_2_count": int((middle & (after_error < 2.)).sum()),
                   "middle_to_lt_2_fraction_of_middle": float((middle & (after_error < 2.)).sum() / max(middle.sum(), 1))})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validation-inputs", required=True)
    parser.add_argument("--validation-labels", required=True)
    parser.add_argument("--xrefine-source", required=True)
    parser.add_argument("--xrefine-weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rows_by_frame = {row["frame_id"]: row for row in rows}
    with np.load(args.validation_inputs, allow_pickle=False) as data:
        inputs = {key: data[key] for key in data.files if key != "metadata"}
        input_metadata = json.loads(str(data["metadata"].item()))
    with np.load(args.validation_labels, allow_pickle=False) as data:
        labels = {key: data[key] for key in data.files if key != "metadata"}
    visible = labels["gt_visible"] & np.isfinite(labels["gt_pixels"]).all(axis=1)
    roma = inputs["roma_pixels"].astype(np.float64)
    v2 = inputs["v2_pixels"].astype(np.float64)
    truth = labels["gt_pixels"].astype(np.float64)
    xrefine = XRefineAdapter(args.xrefine_source, args.xrefine_weights, args.device)
    predictions = {"A_roma": roma.copy(), "B_v2": v2.copy(), "C_roma_xrefine": roma.copy(), "D_v2_xrefine": v2.copy()}
    statuses = {"C_roma_xrefine": np.full(len(roma), "not_run", dtype="U32"),
                "D_v2_xrefine": np.full(len(roma), "not_run", dtype="U32")}
    started = time.time()
    for key, start_pixels in (("C_roma_xrefine", roma), ("D_v2_xrefine", v2)):
        for group in np.unique(np.column_stack((inputs["frame_ids"], inputs["camera_ids"],
                                                inputs["reference_frames"], inputs["reference_cameras"])), axis=0):
            frame, camera, reference_frame, reference_camera = [str(value) for value in group]
            selected = ((inputs["frame_ids"] == frame) & (inputs["camera_ids"] == int(camera)) &
                        (inputs["reference_frames"] == reference_frame) &
                        (inputs["reference_cameras"] == int(reference_camera)))
            row = rows_by_frame[frame]
            query_view = next(view for view in row["views"] if int(view["camera"]) == int(camera))
            reference_view = next(view for view in rows_by_frame[reference_frame]["views"]
                                  if int(view["camera"]) == int(reference_camera))
            reference_mask = np.asarray(np.load(reference_view["mask"]), dtype=bool)
            query_mask = np.asarray(np.load(query_view["mask"]), dtype=bool)
            _, final, status, _ = xrefine.refine_pair(
                reference_view["image"], query_view["image"], inputs["reference_pixels"][selected],
                start_pixels[selected], reference_mask, query_mask, args.batch_size)
            predictions[key][selected] = final
            statuses[key][selected] = status
    records = {name: metric(roma[visible], value[visible], truth[visible]) for name, value in predictions.items()}
    result = {"protocol": {"name": "XRefine frozen refinement baseline", "xrefine_commit": "5261f65ff64575a31e5330ecd5d72ee5f0ba27bb",
                            "input_cache_metadata": input_metadata, "evaluation_mask": "GT visibility labels only for metrics"},
              "samples": {"all": int(len(roma)), "visible": int(visible.sum())}, "comparison": records,
              "status_counts": {name: {str(status): int((values == status).sum()) for status in np.unique(values)}
                                for name, values in statuses.items()}, "elapsed_s": time.time() - started}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    np.savez_compressed(args.predictions, **predictions,
                        **{"status_" + name: value for name, value in statuses.items()},
                        gt_visible=visible, gt_pixels=truth)


if __name__ == "__main__":
    main()
