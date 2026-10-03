import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from data.leader_nclt_manifest import (
    DEFAULT_HORIZONTAL_RESOLUTION,
    DEFAULT_VOXEL_SIZE,
    EXPECTED_MANIFEST_SHA256,
    SAM_CHECKPOINT_SHA256,
    TRAIN_DATES,
    TRAIN_SAM_CACHE_MANIFEST_SHA256,
    ManifestLEADERDataset,
    collate_leader_frames,
)
from tools.magic_nclt_runtime import (
    OFFICIAL_L0_SHA256,
    OFFICIAL_SOURCE_EXTRA_SHA256,
    OFFICIAL_SOURCE_WEIGHTS_SHA256,
    batch_forward,
    canonical_json_sha256,
    configure_training_mode,
    model_parameter_groups,
    safe_json_dump,
    sha256_file,
    source_hashes,
    state_dict_sha256,
    verify_official_l0,
    load_pretrained_model,
)
from tools.resource_guard import DiskGuard, GIB


EPOCHS = 40
LIDAR_LR = 1e-5
FUSION_LR = 1e-4
DECODER_LR = 1e-4
SCHEDULER_GAMMA = 0.9
SEED_DEFAULT = 20


def _paths(root, dates):
    return [Path(root) / date / "manifest.json" for date in dates]


def _atomic_torch_save(path, value, guard=None, anticipated_bytes=0):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    if guard is not None:
        guard.require(anticipated_bytes)
    torch.save(value, str(temporary))
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))
    if guard is not None:
        guard.require()


def _to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _move_optimizer_state(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                target = torch.device("cpu") if key == "step" else device
                state[key] = value.to(target)


def _move_training_state(model, optimizer, center, device):
    model.to(device)
    _move_optimizer_state(optimizer, device)
    return center.to(device)


def _make_settings(args, base_checkpoint, center, train_dataset, source_code):
    settings = {
        "schema": "leader_magic_nclt_training_v1",
        "role": "development_validation_selected_experiment",
        "independent_test": False,
        "epochs": EPOCHS,
        "seed": int(args.seed),
        "data_order": "numpy.default_rng(seed+epoch).permutation; each training frame once per epoch",
        "train_dates": list(TRAIN_DATES),
        "dev_dates": ["2012-02-12", "2012-02-19", "2012-03-31", "2012-05-26"],
        "train_frame_count": len(train_dataset),
        "dev_frame_count": 1200,
        "training_manifest_hashes": {date: train_dataset.manifest_hashes[date]
                                      for date in TRAIN_DATES},
        "dev_manifest_hashes": {
            date: EXPECTED_MANIFEST_SHA256[date]
            for date in ("2012-02-12", "2012-02-19", "2012-03-31", "2012-05-26")
        },
        "train_sam_cache_manifest_sha256": train_dataset.sam_store.manifest_sha256,
        "dev_sam_cache_sha256": args.dev_sam_cache_sha256,
        "sam_checkpoint_sha256": SAM_CHECKPOINT_SHA256,
        "base_checkpoint_sha256": sha256_file(Path(args.base_checkpoint).resolve(strict=True)),
        "base_checkpoint_source_weights_sha256": OFFICIAL_SOURCE_WEIGHTS_SHA256,
        "base_checkpoint_source_extra_sha256": OFFICIAL_SOURCE_EXTRA_SHA256,
        "base_checkpoint_center_t": np.asarray(center, dtype=np.float64).tolist(),
        "base_checkpoint_split_sha256": base_checkpoint["settings"].get("split_sha256"),
        "optimizer": "Adam",
        "learning_rates": {"rpge": LIDAR_LR, "magic_fusion": FUSION_LR,
                            "mmregressor": DECODER_LR},
        "scheduler": {"name": "StepLR", "step_size_epochs": 1,
                      "gamma": SCHEDULER_GAMMA, "steps_after_epoch": True},
        "pretrained_bn_running_stats_frozen": True,
        "pretrained_bn_affine_trainable": True,
        "new_mma_bn_running_stats_train": True,
        "voxel_size_m": DEFAULT_VOXEL_SIZE,
        "horizontal_resolution": DEFAULT_HORIZONTAL_RESOLUTION,
        "maximum_points": 0,
        "physical_batch_size": int(args.batch_size),
        "gradient_accumulation_steps": 1,
        "drop_last": False,
        "loss": "LEADER_TRR_scale10_per_batch_mean",
        "validation_every_epoch": True,
        "validation_denominator": "all 1200 frames; failures retain identity pose and count",
        "selection": "fewest failures; then MPE/MPE_L0+MOE/MOE_L0; ties choose earlier epoch",
        "pose_backend": "SC2-PCR+full_pool_refine",
        "center_t_source": "embedded official_l0.pt center_t",
        "source_code_sha256": source_code,
        "minimum_root_free_gib": float(args.min_free_gib),
        "minimum_start_free_gib": float(args.start_free_gib),
    }
    return settings


def _validate_l0_report(path, base_sha, settings):
    report_path = Path(path).resolve(strict=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    provenance = report.get("provenance", {})
    if (report.get("role") != "l0_reference_development_validation"
            or report.get("independent_test") is not False):
        raise ValueError("L0 report is not the same-code development reference")
    if provenance.get("checkpoint_sha256") != base_sha:
        raise ValueError("L0 report was not generated from this official checkpoint")
    expected_source = settings["source_code_sha256"]
    if provenance.get("runtime_source_hashes") != expected_source:
        raise ValueError("L0 report was generated with different runtime source hashes")
    if provenance.get("predictor_sha256") != expected_source.get("tools/predict_magic_nclt.py"):
        raise ValueError("L0 report predictor source hash differs from this run")
    if provenance.get("evaluator_sha256") != expected_source.get("tools/evaluate_magic_nclt_dev.py"):
        raise ValueError("L0 report evaluator source hash differs from this run")
    if report.get("dev_manifest_hashes") != settings["dev_manifest_hashes"]:
        raise ValueError("L0 report uses different development manifests")
    if report.get("sam_cache_manifest_sha256") != settings["dev_sam_cache_sha256"]:
        raise ValueError("L0 report uses different development SAM features")
    overall = report.get("overall", {})
    if (overall.get("frames") != 1200 or not isinstance(overall.get("failure_frames"), int)
            or overall["failure_frames"] < 0 or overall["failure_frames"] > 1200):
        raise ValueError("L0 report has an incomplete denominator")
    if (not isinstance(overall.get("success_frames"), int)
            or overall["success_frames"] + overall["failure_frames"] != 1200):
        raise ValueError("L0 report success/failure counts do not cover all development frames")
    try:
        mpe = float(overall["mpe_mean_m"])
        moe = float(overall["moe_mean_deg"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("L0 report has invalid reference metrics")
    if not math.isfinite(mpe) or not math.isfinite(moe) or mpe <= 0 or moe <= 0:
        raise ValueError("L0 report has invalid reference metrics")
    per_date = report.get("per_date", {})
    if set(per_date) != set(settings["dev_manifest_hashes"]):
        raise ValueError("L0 report does not contain all four development dates")
    if any(value.get("frames") != 300 for value in per_date.values()):
        raise ValueError("L0 report per-date denominator is incomplete")
    return report, report_path


def _bind_l0_reference(settings, report, report_path):
    overall = report["overall"]
    settings["l0_reference_report_sha256"] = sha256_file(report_path)
    settings["l0_reference_metrics"] = {
        "frames": int(overall["frames"]),
        "failure_frames": int(overall["failure_frames"]),
        "mpe_mean_m": float(overall["mpe_mean_m"]),
        "moe_mean_deg": float(overall["moe_mean_deg"]),
    }
    settings["training_protocol_sha256"] = canonical_json_sha256(settings)


def _new_attempt_directory(root, epoch):
    validation_root = Path(root) / "validation"
    validation_root.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while True:
        path = validation_root / "epoch_{:03d}_attempt_{:02d}".format(epoch + 1, attempt)
        try:
            path.mkdir()
            return path
        except FileExistsError:
            attempt += 1


def _run_candidate_validation(args, project_root, candidate_path, epoch, attempt_dir,
                              expected_settings):
    if sha256_file(args.l0_report) != expected_settings["l0_reference_report_sha256"]:
        raise ValueError("L0 reference report changed after the run protocol was frozen")
    predictor = project_root / "tools" / "predict_magic_nclt.py"
    evaluator = project_root / "tools" / "evaluate_magic_nclt_dev.py"
    predictions = attempt_dir / "predictions.jsonl"
    freeze = attempt_dir / "prediction_freeze.json"
    report = attempt_dir / "evaluation.json"
    environment = os.environ.copy()
    environment["LEADER_DISK_GUARD_PATH"] = str(Path(args.out).resolve().parent)
    environment["LEADER_DISK_MIN_FREE_BYTES"] = str(int(float(args.min_free_gib) * GIB))
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(project_root) + (os.pathsep + existing if existing else "")
    common = [
        "--data-root", str(args.data_root.resolve()),
        "--manifest-root", str(args.dev_manifest_root.resolve()),
        "--sam-cache-manifest", str(args.dev_sam_cache_manifest.resolve()),
        "--sam-cache-sha256", args.dev_sam_cache_sha256,
    ]
    with (attempt_dir / "predict.log").open("x", encoding="utf-8") as stream:
        subprocess.run([
            sys.executable, "-m", "tools.predict_magic_nclt", *common,
            "--checkpoint", str(candidate_path.resolve()),
            "--predictions", str(predictions), "--freeze", str(freeze),
            "--seed", str(args.seed), "--device", args.device,
        ], cwd=str(project_root), env=environment, stdout=stream, stderr=subprocess.STDOUT,
            check=True)
    if not predictions.is_file() or not freeze.is_file():
        raise RuntimeError("GT-free development predictor produced incomplete output")
    with (attempt_dir / "evaluate.log").open("x", encoding="utf-8") as stream:
        command = [
            sys.executable, "-m", "tools.evaluate_magic_nclt_dev",
            "--predictions", str(predictions), "--freeze", str(freeze),
            "--checkpoint", str(candidate_path.resolve()),
            "--predictor", str(predictor.resolve()),
            "--manifest-root", str(args.dev_manifest_root.resolve()),
            "--dev-gt-root", str(args.dev_gt_root.resolve()),
            "--role", "development_validation",
            "--l0-report", str(args.l0_report.resolve()), "--report", str(report),
        ]
        subprocess.run(command, cwd=str(project_root), env=environment,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    result = json.loads(report.read_text(encoding="utf-8"))
    if result.get("role") != "development_validation" or result["overall"]["frames"] != 1200:
        raise ValueError("candidate validation did not score the full development denominator")
    if result["provenance"]["training_protocol_sha256"] != expected_settings["training_protocol_sha256"]:
        raise ValueError("candidate prediction was not bound to the current training protocol")
    if result["selection"]["l0_reference_report_sha256"] != expected_settings["l0_reference_report_sha256"]:
        raise ValueError("candidate validation used a different L0 reference report")
    return predictions, freeze, report, result


def _write_history(path, records):
    target = Path(path)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(target))


def _read_history(path, completed_epoch):
    target = Path(path)
    if not target.exists():
        return []
    rows = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if int(value["epoch"]) <= completed_epoch:
                rows.append(value)
    rows.sort(key=lambda item: int(item["epoch"]))
    if [int(row["epoch"]) for row in rows] != list(range(completed_epoch + 1)):
        raise ValueError("training history does not cover each committed epoch exactly once")
    return rows


def _recover_best_transaction(out, committed_state):
    best_path = Path(out) / "best.pt"
    backup_path = Path(out) / "best.previous.pt"
    expected = committed_state.get("best_checkpoint_sha256")
    if expected is None:
        if backup_path.exists():
            backup_path.unlink()
        if best_path.exists():
            best_path.unlink()
        return
    if best_path.is_file() and sha256_file(best_path) == expected:
        if backup_path.exists():
            backup_path.unlink()
        return
    if backup_path.is_file() and sha256_file(backup_path) == expected:
        if best_path.exists():
            best_path.unlink()
        os.replace(str(backup_path), str(best_path))
        return
    raise ValueError("best checkpoint cannot be recovered from the committed latest state")


def candidate_is_better(failures, objective, best_failures, best_objective):
    if best_failures is None:
        return True
    return int(failures) < int(best_failures) or (
        int(failures) == int(best_failures) and float(objective) < float(best_objective)
    )


def _selection_payload(out, settings, history, best_epoch, best_failures,
                       best_objective, best_hash, best_state_hash):
    if best_epoch is None or len(history) != EPOCHS:
        raise ValueError("selection can only freeze after all 40 epochs are committed")
    selected = history[int(best_epoch) - 1]
    if (selected["candidate_checkpoint_sha256"] != best_hash
            or selected["candidate_model_state_sha256"] != best_state_hash
            or not selected["candidate_bytes_retained_as_best"]
            or selected["best_checkpoint_sha256"] != best_hash):
        raise ValueError("selected history row does not bind to the retained best checkpoint")
    best_path = Path(out) / "best.pt"
    observed_hash = sha256_file(best_path)
    if observed_hash != best_hash:
        raise ValueError("best checkpoint differs from the committed latest selection")
    checkpoint = torch.load(str(best_path), map_location="cpu")
    if checkpoint.get("model_state_sha256") != best_state_hash:
        raise ValueError("best model state hash differs from the selected candidate")
    payload = {
        "role": "development_validation_selection",
        "independent_test": False,
        "checkpoint_path": str(best_path.resolve()),
        "checkpoint_sha256": observed_hash,
        "model_state_sha256": best_state_hash,
        "selected_epoch": int(best_epoch),
        "failure_frames": int(best_failures),
        "normalized_pose_objective": float(best_objective),
        "candidate_count": EPOCHS,
        "dev_manifest_hashes": settings["dev_manifest_hashes"],
        "training_protocol_sha256": settings["training_protocol_sha256"],
        "l0_reference_report_sha256": settings["l0_reference_report_sha256"],
        "rule": settings["selection"],
        "selected_evaluation_sha256": selected["evaluation_sha256"],
        "selected_candidate_checkpoint_sha256": selected["candidate_checkpoint_sha256"],
        "selected_candidate_path_at_validation": selected["candidate_checkpoint_path"],
        "selected_candidate_model_state_sha256": selected["candidate_model_state_sha256"],
        "selected_candidate_original_path_retained": False,
        "selected_candidate_bytes_retained_as_best": (
            selected["candidate_checkpoint_sha256"] == observed_hash
        ),
    }
    if payload["selected_candidate_bytes_retained_as_best"] is not True:
        raise ValueError("selected candidate bytes are not retained under best.pt")
    return payload


def _freeze_selection(out, settings, history, best_epoch, best_failures,
                      best_objective, best_hash, best_state_hash):
    path = Path(out) / "selection_frozen.json"
    payload = _selection_payload(out, settings, history, best_epoch, best_failures,
                                 best_objective, best_hash, best_state_hash)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("existing frozen selection differs from committed training state")
    else:
        safe_json_dump(path, payload)
    return payload


def train(args):
    project_root = Path(__file__).resolve().parents[1]
    out = Path(args.out).resolve()
    guard = DiskGuard(out.parent, int(float(args.min_free_gib) * GIB))
    if guard.available_bytes() < int(float(args.start_free_gib) * GIB):
        raise RuntimeError("root filesystem has insufficient free space to start or resume training")
    base_path = Path(args.base_checkpoint).resolve(strict=True)
    base_checkpoint = torch.load(str(base_path), map_location="cpu")
    base_sha, center_np = verify_official_l0(base_path, base_checkpoint)
    if base_sha != OFFICIAL_L0_SHA256:
        raise ValueError("official L0 checkpoint differs from the frozen value")
    base_settings = dict(base_checkpoint.get("settings", {}))
    del base_checkpoint
    train_dataset = ManifestLEADERDataset(
        _paths(args.train_manifest_root, TRAIN_DATES),
        Path(args.data_root), split="train",
        sam_cache_manifest=Path(args.train_sam_cache_manifest), strict_frozen=True,
        expected_cache_sha256=TRAIN_SAM_CACHE_MANIFEST_SHA256,
        cache_memory_frames=1,
    )
    if len(train_dataset) != 1200 or train_dataset.sam_store is None:
        raise ValueError("training requires 1200 frozen full-point NCLT frames and SAM features")
    code_hashes = source_hashes(project_root)
    for name in ("tools/magic_nclt_runtime.py", "tools/train_magic_nclt.py",
                 "tools/predict_magic_nclt.py", "tools/evaluate_magic_nclt_dev.py"):
        code_hashes[name] = sha256_file(project_root / name)
    settings = _make_settings(args, {"settings": base_settings}, center_np,
                              train_dataset, code_hashes)
    l0_report, l0_report_path = _validate_l0_report(args.l0_report, base_sha, settings)
    _bind_l0_reference(settings, l0_report, l0_report_path)
    out_exists = out.exists()
    resume_state = None
    if out_exists:
        if not args.resume:
            raise FileExistsError("run directory exists; pass --resume only for the same frozen run")
        settings_path = out / "settings.json"
        latest_path = out / "latest.pt"
        if not settings_path.is_file():
            raise ValueError("resume directory has no frozen protocol")
        prior = json.loads(settings_path.read_text(encoding="utf-8"))
        if prior != settings:
            raise ValueError("resume protocol, source, data, or cache hashes differ")
        if latest_path.is_file():
            resume_state = torch.load(str(latest_path), map_location="cpu")
            if resume_state.get("settings") != settings:
                raise ValueError("latest checkpoint settings differ from the frozen run")
            committed_epoch = int(resume_state["epoch"])
            if committed_epoch >= EPOCHS:
                raise ValueError("latest checkpoint is beyond the frozen 40-epoch protocol")
            if committed_epoch == EPOCHS - 1:
                _recover_best_transaction(out, resume_state)
                completed_history = _read_history(out / "history.jsonl", EPOCHS - 1)
                frozen = _freeze_selection(
                    out, settings, completed_history,
                    resume_state["best_epoch"], resume_state["best_failures"],
                    resume_state["best_objective"], resume_state["best_checkpoint_sha256"],
                    resume_state["best_model_state_sha256"],
                )
                print(json.dumps({
                    "training_complete": True,
                    "epochs": EPOCHS,
                    "selected_epoch": frozen["selected_epoch"],
                    "best_checkpoint_sha256": frozen["checkpoint_sha256"],
                    "selection_path": str((out / "selection_frozen.json").resolve()),
                    "recovered_after_epoch_commit": True,
                }, ensure_ascii=False), flush=True)
                return
        else:
            allowed = {"settings.json", "settings.json.tmp", "provenance.json",
                       "provenance.json.tmp", "latest.pt.tmp"}
            unexpected = {path.name for path in out.iterdir()} - allowed
            if unexpected:
                raise ValueError("run directory has files despite missing latest checkpoint")
    elif args.resume:
        raise ValueError("--resume requested but the run directory does not exist")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but is unavailable")
    if device.type == "cuda" and torch.cuda.device_count() != 1:
        raise RuntimeError("formal training requires exactly one visible GPU")
    if device.type == "cpu":
        raise RuntimeError("formal 40-epoch training requires the approved GPU runtime")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    random.seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model, loaded_checkpoint = load_pretrained_model(base_path, True, device)
    del loaded_checkpoint
    center = torch.as_tensor(center_np, dtype=torch.float32, device=device)
    groups = model_parameter_groups(model, LIDAR_LR, FUSION_LR, DECODER_LR)
    optimizer = torch.optim.Adam(groups)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=SCHEDULER_GAMMA)
    if not out_exists:
        guard.require(2 * GIB)
        out.mkdir(parents=True)
        safe_json_dump(out / "settings.json", settings)
    provenance_path = out / "provenance.json"
    provenance = {
        "base_checkpoint_sha256": base_sha,
        "base_source_weights_sha256": OFFICIAL_SOURCE_WEIGHTS_SHA256,
        "base_source_extra_sha256": OFFICIAL_SOURCE_EXTRA_SHA256,
        "base_checkpoint_center_t": center_np.tolist(),
        "l0_reference_report_path": str(l0_report_path),
        "l0_reference_report_sha256": settings["l0_reference_report_sha256"],
        "training_protocol_sha256": settings["training_protocol_sha256"],
        "source_code_sha256": code_hashes,
        "role": "development_validation_only",
        "independent_test": False,
    }
    if provenance_path.exists():
        if json.loads(provenance_path.read_text(encoding="utf-8")) != provenance:
            raise ValueError("resume provenance differs from the frozen run")
    else:
        safe_json_dump(provenance_path, provenance)
    latest_path = out / "latest.pt"
    history_path = out / "history.jsonl"
    start_epoch = 0
    best_failures = None
    best_objective = None
    best_epoch = None
    best_hash = None
    history = []
    if resume_state is not None:
        last = resume_state
        if last.get("settings") != settings:
            raise ValueError("latest checkpoint settings differ from the frozen run")
        _recover_best_transaction(out, last)
        model.load_state_dict(last["model"], strict=True)
        optimizer.load_state_dict(last["optimizer"])
        scheduler.load_state_dict(last["scheduler"])
        _move_optimizer_state(optimizer, device)
        torch.set_rng_state(last["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(last["cuda_rng_all"])
        np.random.set_state(last["numpy_rng"])
        random.setstate(last["python_rng"])
        start_epoch = int(last["epoch"]) + 1
        best_failures = last["best_failures"]
        best_objective = last["best_objective"]
        best_epoch = last["best_epoch"]
        best_hash = last["best_checkpoint_sha256"]
        best_state_hash = last.get("best_model_state_sha256")
        if best_epoch is not None:
            best_path = out / "best.pt"
            if not best_path.is_file() or sha256_file(best_path) != best_hash:
                raise ValueError("best checkpoint hash differs from the committed latest state")
        history = _read_history(history_path, int(last["epoch"]))
    else:
        initial_state = {
            "model": _to_cpu(model.state_dict()),
            "optimizer": _to_cpu(optimizer.state_dict()),
            "scheduler": _to_cpu(scheduler.state_dict()),
            "epoch": -1,
            "best_failures": None,
            "best_objective": None,
            "best_epoch": None,
            "best_checkpoint_sha256": None,
            "best_model_state_sha256": None,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng_all": torch.cuda.get_rng_state_all(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "settings": settings,
        }
        _atomic_torch_save(latest_path, initial_state, guard, 1536 * 1024 ** 2)
        best_state_hash = None
    if start_epoch > EPOCHS:
        raise ValueError("latest checkpoint is beyond the frozen 40-epoch protocol")
    if start_epoch == EPOCHS:
        frozen = _freeze_selection(out, settings, history, best_epoch, best_failures,
                                   best_objective, best_hash, best_state_hash)
        print(json.dumps({"training_complete": True, "epochs": EPOCHS,
                          "selected_epoch": frozen["selected_epoch"],
                          "best_checkpoint_sha256": frozen["checkpoint_sha256"],
                          "selection_path": str((out / "selection_frozen.json").resolve()),
                          "recovered_after_epoch_commit": True}, ensure_ascii=False), flush=True)
        return
    for epoch in range(start_epoch, EPOCHS):
        guard.require()
        epoch_start = time.perf_counter()
        configure_training_mode(model)
        order = np.random.default_rng(int(args.seed) + epoch).permutation(len(train_dataset))
        order_digest = canonical_json_sha256(order.tolist())
        train_losses = []
        optimizer.zero_grad(set_to_none=True)
        for batch_index, offset in enumerate(range(0, len(order), args.batch_size)):
            guard.require()
            indices = order[offset:offset + args.batch_size]
            batch = collate_leader_frames([train_dataset[int(index)] for index in indices])
            loss, raw, details = batch_forward(model, batch, center, device, True)
            if not torch.isfinite(loss) or not torch.isfinite(raw):
                raise FloatingPointError("nonfinite loss at epoch {}, batch {}".format(epoch, batch_index))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            train_losses.append(float(loss.detach().cpu()))
            if (batch_index + 1) % int(args.log_every) == 0 or batch_index + 1 == math.ceil(len(order) / args.batch_size):
                progress = {
                    "epoch": epoch + 1,
                    "epochs": EPOCHS,
                    "batch": batch_index + 1,
                    "batches": int(math.ceil(len(order) / args.batch_size)),
                    "frame_keys": list(batch["frame_keys"]),
                    "trr_loss": train_losses[-1],
                    "sparse_outputs": int(details["predictions"].shape[0]),
                }
                safe_json_dump(out / "progress.json", progress)
                print(json.dumps(progress, ensure_ascii=False), flush=True)
            del batch, loss, raw, details
        train_mean = float(np.mean(train_losses))
        candidate = out / "candidate_for_validation.pt"
        candidate_model = _to_cpu(model.state_dict())
        candidate_model_state_sha = state_dict_sha256(candidate_model)
        candidate_payload = {
            "model": candidate_model,
            "model_state_sha256": candidate_model_state_sha,
            "center_t": center_np.tolist(),
            "settings": dict(settings, magic=True, variant="LEADER_MaGiC_candidate",
                              epoch=epoch + 1),
            "epoch": epoch + 1,
        }
        _atomic_torch_save(candidate, candidate_payload, guard, 768 * 1024 ** 2)
        candidate_sha = sha256_file(candidate)
        attempt_dir = _new_attempt_directory(out, epoch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        model.to("cpu")
        _move_optimizer_state(optimizer, torch.device("cpu"))
        center = center.detach().cpu()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        try:
            prediction_path, freeze_path, evaluation_path, evaluation = _run_candidate_validation(
                args, project_root, candidate, epoch, attempt_dir, settings)
        except Exception:
            safe_json_dump(attempt_dir / "validation_failure.json", {
                "epoch": epoch + 1,
                "checkpoint_sha256": candidate_sha,
                "attempt_directory": str(attempt_dir),
                "failure": "validation subprocess failed; run is not committed past prior epoch",
            })
            raise
        finally:
            center = _move_training_state(model, optimizer, center, device)
        objective = float(evaluation["selection"]["normalized_pose_objective"])
        failures = int(evaluation["overall"]["failure_frames"])
        if (not math.isfinite(objective) or failures < 0 or failures > 1200):
            raise ValueError("candidate evaluation returned an invalid selection metric")
        improved = candidate_is_better(failures, objective, best_failures, best_objective)
        if improved:
            best_path = out / "best.pt"
            backup_path = out / "best.previous.pt"
            if backup_path.exists():
                raise FileExistsError("unresolved best checkpoint transaction exists")
            if best_path.exists():
                os.replace(str(best_path), str(backup_path))
            os.replace(str(candidate), str(best_path))
            best_sha = sha256_file(best_path)
            if best_sha != candidate_sha:
                raise RuntimeError("selected best checkpoint is not byte-identical to its evaluated candidate")
            best_failures = failures
            best_objective = objective
            best_epoch = epoch + 1
            best_hash = best_sha
            best_state_hash = candidate_model_state_sha
        scheduler.step()
        record = {
            "epoch": epoch + 1,
            "training_frames": len(train_dataset),
            "optimizer_steps": int(math.ceil(len(train_dataset) / args.batch_size)),
            "batch_size": int(args.batch_size),
            "sample_order_sha256": order_digest,
            "train_trr_mean_of_optimizer_batches": train_mean,
            "validation_failure_frames": failures,
            "validation_mpe_mean_m": evaluation["overall"]["mpe_mean_m"],
            "validation_mpe_p90_m": evaluation["overall"]["mpe_p90_m"],
            "validation_moe_mean_deg": evaluation["overall"]["moe_mean_deg"],
            "validation_moe_p90_deg": evaluation["overall"]["moe_p90_deg"],
            "normalized_pose_objective": objective,
            "selected_as_best": bool(improved),
            "best_epoch": best_epoch,
            "best_failure_frames": best_failures,
            "best_objective": best_objective,
            "candidate_checkpoint_sha256": candidate_sha,
            "candidate_checkpoint_path": str(candidate.resolve()),
            "candidate_model_state_sha256": candidate_model_state_sha,
            "best_checkpoint_sha256": best_hash,
            "best_model_state_sha256": best_state_hash,
            "candidate_checkpoint_retained_after_epoch": False,
            "candidate_bytes_retained_as_best": bool(improved),
            "candidate_is_byte_identical_to_best": bool(improved),
            "prediction_sha256": sha256_file(prediction_path),
            "prediction_freeze_sha256": sha256_file(freeze_path),
            "evaluation_sha256": sha256_file(evaluation_path),
            "learning_rates_next_epoch": {group["name"]: group["lr"]
                                           for group in optimizer.param_groups},
            "elapsed_epoch_seconds": time.perf_counter() - epoch_start,
        }
        history.append(record)
        _write_history(history_path, history)
        del candidate_payload, candidate_model
        if candidate.exists():
            candidate.unlink()
        state = {
            "model": _to_cpu(model.state_dict()),
            "optimizer": _to_cpu(optimizer.state_dict()),
            "scheduler": _to_cpu(scheduler.state_dict()),
            "epoch": epoch,
            "best_failures": best_failures,
            "best_objective": best_objective,
            "best_epoch": best_epoch,
            "best_checkpoint_sha256": best_hash,
            "best_model_state_sha256": best_state_hash,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng_all": torch.cuda.get_rng_state_all(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "settings": settings,
        }
        latest_budget = max(1536 * 1024 ** 2, latest_path.stat().st_size + 256 * 1024 ** 2)
        _atomic_torch_save(latest_path, state, guard, latest_budget)
        backup_path = out / "best.previous.pt"
        if backup_path.exists():
            backup_path.unlink()
        progress = dict(record, phase="epoch_committed", epochs=EPOCHS)
        safe_json_dump(out / "progress.json", progress)
        print(json.dumps(progress, ensure_ascii=False), flush=True)
    frozen = _freeze_selection(out, settings, history, best_epoch, best_failures,
                               best_objective, best_hash, best_state_hash)
    print(json.dumps({"training_complete": True, "epochs": EPOCHS,
                      "selected_epoch": best_epoch,
                      "best_checkpoint_sha256": frozen["checkpoint_sha256"],
                      "selection_path": str(out / "selection_frozen.json")}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-manifest-root", type=Path, required=True)
    parser.add_argument("--dev-manifest-root", type=Path, required=True)
    parser.add_argument("--train-sam-cache-manifest", type=Path, required=True)
    parser.add_argument("--dev-sam-cache-manifest", type=Path, required=True)
    parser.add_argument("--dev-sam-cache-sha256", required=True)
    parser.add_argument("--dev-gt-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--l0-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--min-free-gib", type=float, default=4.0)
    parser.add_argument("--start-free-gib", type=float, default=7.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (args.batch_size <= 0 or args.log_every <= 0 or args.min_free_gib < 2.0
            or args.start_free_gib < args.min_free_gib + 2.0):
        raise ValueError("batch size and log interval must be positive")
    train(args)


if __name__ == "__main__":
    main()
