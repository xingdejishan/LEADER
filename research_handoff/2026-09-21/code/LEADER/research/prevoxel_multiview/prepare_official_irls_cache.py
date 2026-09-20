import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Subset

import run_mink as official
from data.NCLTVelodyne_datagenerator_mink import NCLT_mink
from models.model_mink import LEADER
from utils.pose_util import polar_expansion_to_cartesian

try:
    from irls_huber import huber_rigid_torch
except ImportError as exc:
    raise ImportError("put the official irls_huber.py on PYTHONPATH before running this script") from exc


def parse_frame_ids(path):
    payload = Path(path).read_text(encoding="utf-8")
    try:
        value = json.loads(payload)
        if isinstance(value, list):
            return [str(item["frame_id"] if isinstance(item, dict) else item) for item in value]
    except json.JSONDecodeError:
        pass
    return [line.strip() for line in payload.splitlines() if line.strip()]


def args_for_official(cli):
    return argparse.Namespace(
        dataset="NCLT", dataset_folder=cli.dataset_folder,
        batch_size=25, val_batch_size=1, max_epoch=50,
        init_learning_rate=.001, decay_epoch=1, seed=20, mode="test",
        log_dir=str(Path(cli.output).parent), num_workers=0,
        horizontal_res=1024., voxel_size=.2, max_range=100.,
        resume_model=cli.resume_model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-folder", required=True)
    parser.add_argument("--resume-model", required=True)
    parser.add_argument("--frame-ids", required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--output", required=True)
    cli = parser.parse_args()

    flags = args_for_official(cli)
    frame_ids = parse_frame_ids(cli.frame_ids)
    dataset = NCLT_mink(
        data_path=flags.dataset_folder, train=cli.split == "train",
        voxel_size=flags.voxel_size, horizontal_res=flags.horizontal_res)
    index_by_id = {Path(path).stem: index for index, path in enumerate(dataset.pcs)}
    missing = [frame_id for frame_id in frame_ids if frame_id not in index_by_id]
    if missing:
        raise ValueError("frame ids are not in official validation dataset: %s" % missing[:10])
    indices = [index_by_id[frame_id] for frame_id in frame_ids]

    full_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                             pin_memory=True)
    selected = Subset(dataset, indices)

    def collate_pair_fn(list_data):
        import MinkowskiEngine as ME

        list_data = [data for data in list_data if data is not None]
        coords, feats, _, _, corrections = zip(*list_data)
        return {
            "coords": ME.utils.batched_coordinates(coords),
            "feats": torch.from_numpy(np.concatenate(feats)).float(),
            "T_corr": torch.from_numpy(np.stack(corrections)).float(),
        }

    selected_loader = DataLoader(selected, batch_size=1, shuffle=False,
                                 collate_fn=collate_pair_fn, num_workers=0,
                                 pin_memory=True)
    accelerator = Accelerator()
    if not torch.cuda.is_available():
        raise ValueError("GPU not found")
    device = torch.device("cuda")
    model = LEADER(in_channels=3, out_channels=4, feat_channels=512,
                   width=flags.horizontal_res)
    optimizer = torch.optim.Adam(model.parameters(), flags.init_learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, flags.decay_epoch, gamma=.9)
    model, optimizer, full_loader, selected_loader, scheduler = accelerator.prepare(
        model, optimizer, full_loader, selected_loader, scheduler)
    process_info = {"epoch": -1, "train_iter": 0, "val_iter": 0}
    official.load_checkpoint(cli.resume_model, accelerator, process_info)
    model.eval()
    center_t = torch.as_tensor(dataset.get_center_t() + np.array([0, 0, -2 * flags.max_range]),
                               dtype=torch.float32, device=device)

    output = Path(cli.output)
    output.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for step, input_dict in enumerate(selected_loader):
            coords = input_dict["coords"]
            feats = input_dict["feats"]
            input_sparse = official.ME.SparseTensor(feats, coords)
            t_corr = input_dict["T_corr"]
            enc = model.encoder(input_sparse)
            stride = torch.tensor(enc.tensor_stride, device=device, dtype=torch.float32)
            batch_idx = enc.C[:, 0].long()
            voxel_centers = (enc.C[:, 1:].float() + stride / 2) * flags.voxel_size
            local = polar_expansion_to_cartesian(
                voxel_centers, flags.horizontal_res * flags.voxel_size)
            prediction = model.decoder(enc.F)
            mask = batch_idx == 0
            source = local[mask, :3].float()
            pred = prediction[mask].float()
            scores = pred[:, 3]
            keep_count = max(min(50, len(scores)), int(.5 * len(scores)))
            keep = scores.topk(keep_count).indices
            local_pose = huber_rigid_torch(
                source[keep][None], pred[keep, :3][None], delta=.5, iters=10)[0]
            local_pose[:3, 3] += center_t
            t0_official = local_pose @ t_corr[0]
            frame_id = frame_ids[step]
            np.savez_compressed(
                output / (frame_id + ".npz"),
                source=source.cpu().numpy().astype(np.float32),
                prediction=pred.cpu().numpy().astype(np.float32),
                center=center_t.cpu().numpy().astype(np.float32),
                T_corr=t_corr[0].cpu().numpy().astype(np.float32),
                topk_indices=keep.cpu().numpy().astype(np.int64),
                T0_official=t0_official.cpu().numpy().astype(np.float32),
            )
            print("%d/%d %s points=%d topk=%d" % (
                step + 1, len(frame_ids), frame_id, len(source), len(keep)), flush=True)

    manifest = {
        "concat_mode": "official",
        "t_corr_applied_in_cache": False,
        "query_gt_stored_in_cache": False,
        "source_order": "official_voxel_centers_l",
        "target_order": "official_decoder_output",
        "prediction_dtype": "float32",
        "topk_rule": "max(min(50,N),int(0.5*N))",
        "baseline_pose_stored": True,
        "checkpoint": str(cli.resume_model),
        "split": cli.split,
        "preprocessing": {"dataset": "NCLT", "voxel_size": .2,
                           "horizontal_res": 1024., "max_range": 100.},
        "frame_ids": frame_ids,
    }
    (output / "cache_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
