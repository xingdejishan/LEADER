"""Does the linear pose-impact predictor track the pose the refiner actually returns?

For a sample of evaluation frames: refine with SAM features, refine with the compressed
features, and compare the actual pose difference with the linear prediction
dxi = -(H + lambda I)^-1 dg evaluated at the SAM pose.
"""
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = "/home/zhangcy/projects/e2e-moe-20261020-v1"
SAMPLES = Path("/home/zhangcy/projects/nclt-full-v7-int8-20261023-v1/results/samples.json")


def main():
    head_path = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    lam = 1e-3
    eps = 0.05

    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")
    sys.path.insert(0, PROJECT)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    import torch.nn.functional as F
    import e2e_online as online
    from encoder import LinearHead
    from scipy.spatial.transform import Rotation

    state = torch.load(head_path, map_location="cuda")
    head = LinearHead(state["mean"], state["weight"]).cuda()
    head.weight.data = state["weight"].cuda()
    head.eval()

    boot = online.boot_online()
    model = boot["model"]
    rt = boot["runtime"]
    selfverify = boot["c"].selfverify
    settings = boot["inference_method"]
    lookup = {record[2]: record for record in rt.records}
    rows = json.loads(SAMPLES.read_text())["samples"]
    keys = [row["key"] for row in rows if row["key"] in lookup][:limit]

    def pose_error(a, b):
        dt = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
        dr = float(np.degrees(Rotation.from_matrix(
            (a[:3, :3].T @ b[:3, :3]).T if False else a[:3, :3] @ b[:3, :3].T).magnitude()))
        return dt, dr

    pred_r, pred_t, act_r, act_t = [], [], [], []
    with torch.no_grad():
        for key in keys:
            boot["memo"]["velo"].clear()
            boot["memo"]["sha256"].clear()
            loaded = rt.load(lookup[key])
            initial = rt.initial(loaded)
            geometry = rt.geometry(loaded, initial[1].detach().cpu())
            frame = rt.frame(loaded, *initial, geometry)
            sam = frame["sam"]
            code = head(sam)
            quantized = torch.clamp(torch.round(code / (code.abs().amax(dim=(2, 3), keepdim=True).clamp_min(1e-8) / 7.0)), -7, 7) * (code.abs().amax(dim=(2, 3), keepdim=True).clamp_min(1e-8) / 7.0)
            compressed = head.expand(quantized)[0].float()
            try:
                frame["sam"] = sam
                obs_t = model.observation_batch([sam], [frame["rgb"]], [frame["pixels"]], [frame["hw"]])[0]
                refined_t, _, _, _ = selfverify.refine(model, frame, obs_t, boot["inference_method"])
                frame["sam"] = compressed
                obs_s = model.observation_batch([compressed], [frame["rgb"]], [frame["pixels"]], [frame["hw"]])[0]
                refined_s, _, _, _ = selfverify.refine(model, frame, obs_s, boot["inference_method"])
            except RuntimeError:
                continue
            T_t = refined_t.detach().double().cpu().numpy()
            T_s = refined_s.detach().double().cpu().numpy()
            dt, dr = pose_error(T_t, T_s)
            act_t.append(dt); act_r.append(dr)

            with torch.enable_grad():
                pose_ref = refined_t.detach().double()
                warm, _ = model.warm_start(frame["initial"], frame["source"], frame["target"])
                ctx_t = selfverify.frame_context(model, frame, obs_t, warm.double(), settings)
                ctx_s = selfverify.frame_context(model, frame, obs_s, warm.double(), settings)
                scales = pose_ref.new_tensor([settings["tau_R"]] * 3 + [settings["tau_t"]] * 3)
                g_t = selfverify.value_gradient(model, pose_ref, ctx_t, scales, False)[1].detach()
                g_s = selfverify.value_gradient(model, pose_ref, ctx_s, scales, False)[1].detach()
                cols = []
                for k in range(6):
                    vals = []
                    for sign in (1.0, -1.0):
                        z = torch.zeros(6, dtype=pose_ref.dtype, device=pose_ref.device)
                        z[k] = sign * eps
                        local = selfverify.move(pose_ref, z * scales)
                        vals.append(selfverify.value_gradient(model, local, ctx_t, scales, False)[1].detach())
                    cols.append((vals[0] - vals[1]) / (2 * eps))
                H = torch.stack(cols, dim=1)
                dxi = -torch.linalg.solve(H + lam * torch.eye(6, dtype=H.dtype, device=H.device),
                                          (g_s - g_t).reshape(6, 1)).reshape(6)
                phys = (dxi * scales.double()).cpu().numpy()
            pred_r.append(abs(float(np.degrees(phys[:3]).__abs__().max())))
            pred_t.append(float(np.linalg.norm(phys[3:])))
            print('%d %s pred %.2fdeg/%.3fm act %.2fdeg/%.3fm' % (
                len(act_r), key, pred_r[-1], pred_t[-1], dr, dt), flush=True)

    pr = np.array(pred_r); pt = np.array(pred_t)
    ar = np.array(act_r); at = np.array(act_t)
    print('\nn=%d' % len(pr))
    for name, p, a in (('rotation_deg', pr, ar), ('translation_m', pt, at)):
        corr = float(np.corrcoef(p, a)[0, 1]) if len(p) > 2 else float('nan')
        print('%-14s pred mean %.3f  actual mean %.3f  corr %.3f  |ratio median %.2f' % (
            name, p.mean(), a.mean(), corr, float(np.median(a / np.maximum(p, 1e-9)))))
    json.dump(dict(pred_rot=pr.tolist(), pred_trans=pt.tolist(),
                   act_rot=ar.tolist(), act_trans=at.tolist()),
              open('/home/zhangcy/projects/nclt-compact-encoder-20261023-v1/pose_predict_check.json', 'w'))


if __name__ == '__main__':
    main()
