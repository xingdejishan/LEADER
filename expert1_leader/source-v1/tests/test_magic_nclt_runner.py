import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

import tools.evaluate_magic_nclt_dev as evaluator
from data.leader_nclt_manifest import DEV_DATES
from tools.evaluate_magic_nclt_dev import _pose_errors, _read_gt_date, _summary
from tools.magic_nclt_runtime import (
    BACKEND,
    OFFICIAL_L0_SHA256,
    canonical_json_sha256,
    model_parameter_groups,
    sha256_file,
    source_hashes,
    trr_loss,
)
from tools.train_magic_nclt import (
    _bind_l0_reference,
    _freeze_selection,
    _recover_best_transaction,
    _validate_l0_report,
    candidate_is_better,
)


class ParameterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 5)
        self.magic_fusion = nn.Linear(5, 5)
        self.decoder = nn.Linear(5, 4)


class MagicNCLTRunnerTests(unittest.TestCase):
    def test_optimizer_groups_cover_parameters_once_with_frozen_learning_rates(self):
        model = ParameterModel()
        groups = model_parameter_groups(model, 1e-5, 1e-4, 1e-4)
        self.assertEqual([group["name"] for group in groups], ["rpge", "fusion", "mmregressor"])
        self.assertEqual([group["lr"] for group in groups], [1e-5, 1e-4, 1e-4])
        parameters = [parameter for group in groups for parameter in group["params"]]
        self.assertEqual(len(parameters), len({id(parameter) for parameter in parameters}))
        self.assertEqual({id(parameter) for parameter in parameters},
                         {id(parameter) for parameter in model.parameters()})
        self.assertTrue(all(parameter.requires_grad for parameter in parameters))

    def test_candidate_selection_prioritizes_failures_then_normalized_objective(self):
        self.assertTrue(candidate_is_better(2, 1.4, None, None))
        self.assertTrue(candidate_is_better(1, 9.0, 2, 0.1))
        self.assertTrue(candidate_is_better(2, 0.9, 2, 1.0))
        self.assertFalse(candidate_is_better(2, 1.0, 2, 1.0))
        self.assertFalse(candidate_is_better(3, 0.1, 2, 1.0))

    def test_l0_reference_binds_report_hash_metrics_and_runtime_sources(self):
        project_root = Path(__file__).resolve().parents[1]
        source = source_hashes(project_root)
        cache_sha = "1" * 64
        manifests = {date: (str(index + 2) * 64) for index, date in enumerate(DEV_DATES)}
        report = {
            "role": "l0_reference_development_validation",
            "independent_test": False,
            "dev_manifest_hashes": manifests,
            "sam_cache_manifest_sha256": cache_sha,
            "provenance": {
                "checkpoint_sha256": OFFICIAL_L0_SHA256,
                "runtime_source_hashes": source,
                "predictor_sha256": source["tools/predict_magic_nclt.py"],
                "evaluator_sha256": source["tools/evaluate_magic_nclt_dev.py"],
            },
            "overall": {"frames": 1200, "success_frames": 1193,
                        "failure_frames": 7, "mpe_mean_m": 2.0,
                        "moe_mean_deg": 3.0},
            "per_date": {date: {"frames": 300} for date in DEV_DATES},
        }
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "l0.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            settings = {"dev_manifest_hashes": manifests,
                        "dev_sam_cache_sha256": cache_sha,
                        "source_code_sha256": source}
            validated, _ = _validate_l0_report(report_path, OFFICIAL_L0_SHA256, settings)
            _bind_l0_reference(settings, validated, report_path)
            self.assertEqual(settings["l0_reference_report_sha256"], sha256_file(report_path))
            self.assertEqual(settings["l0_reference_metrics"]["failure_frames"], 7)
            protocol = dict(settings)
            observed_protocol = protocol.pop("training_protocol_sha256")
            self.assertEqual(observed_protocol, canonical_json_sha256(protocol))
            report["provenance"]["runtime_source_hashes"] = {"changed": "0" * 64}
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "runtime source hashes"):
                _validate_l0_report(report_path, OFFICIAL_L0_SHA256, settings)

    def test_best_checkpoint_transaction_recovers_old_or_committed_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            best = root / "best.pt"
            backup = root / "best.previous.pt"
            best.write_bytes(b"new candidate")
            backup.write_bytes(b"previous committed best")
            _recover_best_transaction(root, {
                "best_checkpoint_sha256": hashlib.sha256(b"previous committed best").hexdigest()
            })
            self.assertEqual(best.read_bytes(), b"previous committed best")
            self.assertFalse(backup.exists())
            backup.write_bytes(b"previous committed best")
            _recover_best_transaction(root, {
                "best_checkpoint_sha256": hashlib.sha256(b"previous committed best").hexdigest()
            })
            self.assertFalse(backup.exists())
            self.assertEqual(best.read_bytes(), b"previous committed best")
            best.write_bytes(b"committed new best")
            backup.write_bytes(b"previous committed best")
            _recover_best_transaction(root, {
                "best_checkpoint_sha256": hashlib.sha256(b"committed new best").hexdigest()
            })
            self.assertEqual(best.read_bytes(), b"committed new best")
            self.assertFalse(backup.exists())
            source = root / "candidate.pt"
            source.write_bytes(best.read_bytes())
            source_sha = sha256_file(source)
            os.replace(str(source), str(root / "promoted_best.pt"))
            self.assertEqual(source_sha, sha256_file(root / "promoted_best.pt"))
            _recover_best_transaction(root, {"best_checkpoint_sha256": None})
            self.assertFalse(best.exists())

    def test_completed_run_can_idempotently_recover_selection_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            best_path = root / "best.pt"
            model_state_sha = "a" * 64
            torch.save({"model_state_sha256": model_state_sha}, best_path)
            best_sha = sha256_file(best_path)
            history = [{"epoch": index + 1} for index in range(40)]
            history[-1].update({
                "candidate_checkpoint_sha256": best_sha,
                "candidate_model_state_sha256": model_state_sha,
                "candidate_bytes_retained_as_best": True,
                "best_checkpoint_sha256": best_sha,
                "candidate_checkpoint_path": str(root / "candidate_for_validation.pt"),
                "evaluation_sha256": "b" * 64,
            })
            settings = {
                "dev_manifest_hashes": {date: "c" * 64 for date in DEV_DATES},
                "training_protocol_sha256": "d" * 64,
                "l0_reference_report_sha256": "e" * 64,
                "selection": "fewest failures then normalized objective",
            }
            first = _freeze_selection(root, settings, history, 40, 0, 1.25,
                                      best_sha, model_state_sha)
            second = _freeze_selection(root, settings, history, 40, 0, 1.25,
                                       best_sha, model_state_sha)
            self.assertEqual(first, second)
            self.assertFalse(first["selected_candidate_original_path_retained"])
            self.assertTrue(first["selected_candidate_bytes_retained_as_best"])
            self.assertEqual(first["selected_candidate_checkpoint_sha256"], best_sha)

    def test_trr_loss_matches_pointwise_leader_weighting_and_backpropagates(self):
        targets = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                                [0.0, 2.0, 0.0], [0.0, 0.0, 3.0],
                                [1.0, 1.0, 1.0]])
        predictions = torch.tensor([[0.2, 0.0, 0.0], [1.0, 0.3, 0.0],
                                    [0.0, 2.4, 0.0], [0.0, 0.0, 2.5],
                                    [1.2, 1.0, 1.0]], requires_grad=True)
        reliability = torch.tensor([0.2, -0.5, 0.7, -0.1, 1.2], requires_grad=True)
        batch_index = torch.tensor([0, 0, 1, 1, 1])
        loss, raw = trr_loss(targets, predictions, reliability, batch_index, scale=10)
        distance = torch.linalg.vector_norm(predictions - targets, dim=-1)
        factor = np.log(10) / np.pi
        cut = reliability.detach().clamp(-10 * np.pi, 10 * np.pi).atan() * factor
        full = reliability.detach().atan() * factor
        lower = torch.minimum(cut.exp(), full.exp())
        upper = torch.maximum(cut.exp(), full.exp())
        expected = []
        for index in (0, 1):
            selected = batch_index == index
            denominator = lower[selected].sum()
            weights = upper[selected] / denominator
            expected.append((distance.detach()[selected] * weights).sum())
        torch.testing.assert_close(loss.detach(), torch.stack(expected).mean())
        torch.testing.assert_close(raw.detach(), distance.detach().mean())
        loss.backward()
        self.assertTrue(torch.isfinite(predictions.grad).all())
        self.assertTrue(torch.isfinite(reliability.grad).all())

    def test_development_gt_reader_composes_camera_pose_and_checks_hashes(self):
        date = "2012-02-12"
        T_BC = np.eye(4, dtype=np.float64)
        T_BC[:3, 3] = [1.0, -2.0, 0.5]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "evaluator_only" / "test_gt"
            date_root = root / date
            gt_dir = date_root / "groundtruth_interpolated"
            gt_dir.mkdir(parents=True)
            frames = []
            gt_rows = []
            expected_poses = []
            for index in range(300):
                timestamp = 1_300_000_000_000_000 + index
                T_WB = np.eye(4, dtype=np.float64)
                T_WB[:3, 3] = [index * 0.01, 2.0, -1.0]
                T_WC = T_WB @ T_BC
                pose_path = gt_dir / "{}_T_WC.txt".format(timestamp)
                np.savetxt(pose_path, T_WC, fmt="%.17g")
                pose_sha = hashlib.sha256(pose_path.read_bytes()).hexdigest()
                relative = "groundtruth_interpolated/{}_T_WC.txt".format(timestamp)
                frames.append({"date": date, "timestamp_us": timestamp,
                               "frame_key": "{}/{}".format(date, timestamp)})
                gt_rows.append({"index": index, "timestamp_us": timestamp,
                                "ground_truth": {"pose": "T_WC", "path": relative,
                                                 "sha256": pose_sha}})
                expected_poses.append(T_WB)
            document = {"date": date, "split": "test_evaluator_only",
                        "frame_count": 300, "frames": gt_rows}
            manifest_path = date_root / "manifest.json"
            manifest_path.write_text(json.dumps(document), encoding="utf-8")
            manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            (date_root / "manifest.sha256").write_text(manifest_sha + "  manifest.json\n",
                                                         encoding="ascii")
            poses, observed_sha, _ = _read_gt_date(root, date, frames, T_BC)
            self.assertEqual(observed_sha, manifest_sha)
            for actual, expected in zip(poses, expected_poses):
                np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)

    def test_dev_metrics_score_failures_in_frame_denominator(self):
        truth = np.eye(4, dtype=np.float64)
        identity = np.eye(4, dtype=np.float64)
        translation, rotation = _pose_errors(identity, truth)
        self.assertEqual(translation, 0.0)
        self.assertEqual(rotation, 0.0)
        rows = [
            {"status": "success", "mpe_m": 1.0, "moe_deg": 2.0},
            {"status": "identity_fallback", "mpe_m": 3.0, "moe_deg": 4.0},
        ]
        result = _summary(rows)
        self.assertEqual(result["frames"], 2)
        self.assertEqual(result["success_frames"], 1)
        self.assertEqual(result["failure_frames"], 1)
        self.assertEqual(result["mpe_mean_m"], 2.0)
        self.assertEqual(result["moe_mean_deg"], 3.0)

    def test_evaluator_binds_candidate_prediction_to_full_synthetic_dev_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public_root = root / "public"
            gt_root = root / "evaluator_only" / "test_gt"
            checkpoint = root / "candidate.pt"
            checkpoint.write_bytes(b"synthetic candidate checkpoint")
            predictor = Path(evaluator.__file__).with_name("predict_magic_nclt.py").resolve()
            predictions = root / "predictions.jsonl"
            freeze_path = root / "freeze.json"
            report_path = root / "report.json"
            T_BC = np.eye(4, dtype=np.float64)
            T_BC[:3, 3] = [0.5, -0.25, 1.5]
            public_hashes = {}
            all_frames = []
            runtime = source_hashes(Path(evaluator.__file__).resolve().parents[1])
            for date_index, date in enumerate(DEV_DATES):
                timestamps = [1_320_000_000_000_000 + date_index * 1000 + i for i in range(300)]
                frames = [{"timestamp_us": timestamp, "camera_id": 5,
                           "scan": {"timestamp_us": timestamp}}
                          for timestamp in timestamps]
                public = {"date": date, "split": "test", "complete": True,
                          "frame_count": 300, "roi": {"T_BC": T_BC.tolist()},
                          "frames": frames}
                date_dir = public_root / date
                date_dir.mkdir(parents=True)
                public_path = date_dir / "manifest.json"
                public_path.write_text(json.dumps(public), encoding="utf-8")
                public_hashes[date] = sha256_file(public_path)
                gt_dir = gt_root / date
                pose_dir = gt_dir / "groundtruth_interpolated"
                pose_dir.mkdir(parents=True)
                gt_rows = []
                for index, timestamp in enumerate(timestamps):
                    pose_path = pose_dir / "{}_T_WC.txt".format(timestamp)
                    np.savetxt(pose_path, T_BC, fmt="%.17g")
                    gt_rows.append({
                        "index": index,
                        "timestamp_us": timestamp,
                        "ground_truth": {
                            "pose": "T_WC",
                            "path": "groundtruth_interpolated/{}_T_WC.txt".format(timestamp),
                            "sha256": sha256_file(pose_path),
                        },
                    })
                    all_frames.append({"date": date, "timestamp_us": timestamp,
                                       "frame_key": "{}/{}".format(date, timestamp)})
                gt_manifest = {"date": date, "split": "test_evaluator_only",
                               "frame_count": 300, "frames": gt_rows}
                gt_manifest_path = gt_dir / "manifest.json"
                gt_manifest_path.write_text(json.dumps(gt_manifest), encoding="utf-8")
                gt_sha = sha256_file(gt_manifest_path)
                (gt_dir / "manifest.sha256").write_text(gt_sha + "  manifest.json\n", encoding="ascii")
            expected_before = evaluator.EXPECTED_MANIFEST_SHA256.copy()
            evaluator.EXPECTED_MANIFEST_SHA256.update(public_hashes)
            try:
                with predictions.open("w", encoding="utf-8", newline="\n") as stream:
                    for frame in all_frames:
                        stream.write(json.dumps({
                            **frame,
                            "T_WB_pred": np.eye(4).tolist(),
                            "status": "success",
                            "valid_corr": 10,
                            "selected_corr": 5,
                            "failure_reason": None,
                        }) + "\n")
                cache_sha = "1" * 64
                l0_report_path = root / "l0.json"
                l0_report_path.write_text(json.dumps({
                    "role": "l0_reference_development_validation",
                    "independent_test": False,
                    "dev_manifest_hashes": public_hashes,
                    "sam_cache_manifest_sha256": cache_sha,
                    "provenance": {
                        "checkpoint_sha256": OFFICIAL_L0_SHA256,
                        "runtime_source_hashes": runtime,
                    },
                    "overall": {"frames": 1200, "success_frames": 1200, "failure_frames": 0,
                                "mpe_mean_m": 1.0, "moe_mean_deg": 1.0},
                    "per_date": {date: {"frames": 300} for date in DEV_DATES},
                }), encoding="utf-8")
                freeze = {
                    "schema": "leader_magic_dev_prediction_freeze_v1",
                    "role": "development_validation",
                    "independent_test": False,
                    "ground_truth_was_read_by_predictor": False,
                    "frame_count": 1200,
                    "ordered_frame_keys_sha256": canonical_json_sha256(
                        [frame["frame_key"] for frame in all_frames]),
                    "checkpoint_path": str(checkpoint.resolve()),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "predictor_path": str(predictor),
                    "predictor_sha256": sha256_file(predictor),
                    "prediction_path": str(predictions.resolve()),
                    "prediction_sha256": sha256_file(predictions),
                    "evaluator_path": str(Path(evaluator.__file__).resolve()),
                    "evaluator_sha256": sha256_file(evaluator.__file__),
                    "runtime_source_hashes": runtime,
                    "training_protocol_sha256": "2" * 64,
                    "dev_manifest_hashes": public_hashes,
                    "sam_cache_manifest_sha256": cache_sha,
                    "magic_enabled": True,
                    "backend": BACKEND,
                }
                freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
                result = evaluator.score_predictions(SimpleNamespace(
                    predictions=predictions,
                    freeze=freeze_path,
                    checkpoint=checkpoint,
                    predictor=predictor,
                    manifest_root=public_root,
                    dev_gt_root=gt_root,
                    l0_report=l0_report_path,
                    role="development_validation",
                    report=report_path,
                ))
                self.assertEqual(result["role"], "development_validation")
                self.assertEqual(result["overall"]["frames"], 1200)
                self.assertEqual(result["overall"]["failure_frames"], 0)
                self.assertEqual(result["selection"]["normalized_pose_objective"], 0.0)
            finally:
                evaluator.EXPECTED_MANIFEST_SHA256.clear()
                evaluator.EXPECTED_MANIFEST_SHA256.update(expected_before)


if __name__ == "__main__":
    unittest.main()
