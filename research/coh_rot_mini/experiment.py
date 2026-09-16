"""Minimal mechanism test: can the network LEARN L_coh-rot and move MOE with it?

Protocol (frozen, 2026-09-16):
  * 64 training frames + 32 fixed validation frames, drawn date-stratified from the
    original 578-frame fit pool (seed 4242).
  * RPGE frozen (features come from cache); only the original MMRegressor is trained.
  * Arm A: TRR only.  Arm B: TRR + lambda * L_coh-rot.
    Same init, same frame order, same optimiser, same LR schedule. lambda ramps
    0 -> 0.1 over the first 5 epochs then stays.
  * Read-outs: (1) Kabsch rotation bias R_e on validation frames with FIXED frozen-model
    weights; (2) end-to-end MOE / MPE through the original SC2-PCR Matcher; (3) MPE must
    not degrade. 30-50 epochs is enough for a mechanism screen.
"""
import argparse
import ast
import collections
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

LEADER_REPO = Path('/mnt/c/Users/zhang/Documents/ChatGPT/LEADER/LEADER')
WORKSPACE = Path('/mnt/c/Users/zhang/Documents/ChatGPT/LEADER')
CACHE = Path('/home/zhang/crossframe-visual-probe')
PROTO = Path('/home/zhang/anchored-contrastive-fusion/protocol.json')
CHECKPOINT = WORKSPACE / 'research/image_gate_checkpoint'
OUT = Path('/home/zhang/coh-rot-scr')

sys.path.insert(0, str(LEADER_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from coh_rot import coherent_rotation_loss, kabsch_forward, weights_from_scores  # noqa: E402

N_TRAIN, N_VAL, EPOCHS, BATCH, SEED = 64, 32, 40, 8, 4242
LAMBDA_MAX, THETA0_DEG, GATE_DEG = 0.1, 1.0, 20.0
LR, WD, WARMUP = 1e-5, 1e-4, 5


def official_trr():
    tree = ast.parse((LEADER_REPO / 'run_mink.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TRR')
    scope = dict(np=np, torch=torch, Tensor=torch.Tensor)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(LEADER_REPO / 'run_mink.py'), 'exec'), scope)
    return scope['TRR'](scale=10.)


def pick_subset(rows):
    fit = [r for r in rows if r['role'] == 'fit']
    rng = np.random.default_rng(SEED)
    by_date = collections.defaultdict(list)
    for r in fit:
        by_date[r['session_id']].append(r)
    train, val = [], []
    total = len(fit)
    for date in sorted(by_date):
        group = sorted(by_date[date], key=lambda r: int(r['frame_id']))
        k = len(group)
        nt = max(1, round(k * N_TRAIN / total))
        nv = max(1, round(k * N_VAL / total))
        perm = rng.permutation(k)
        train += [group[perm[i]] for i in range(nt)]
        val += [group[perm[nt + i]] for i in range(nv)]
    return train[:N_TRAIN], val[:N_VAL]


def load_model_and_frames(train_rows, val_rows):
    from models.model_mink import MMRegressor
    from safetensors.torch import load_file
    dec = MMRegressor(feat_channels=512, out_channels=4).cuda().eval()
    state = load_file(str(CHECKPOINT / 'model.safetensors'))
    dec.load_state_dict({k[len('decoder.'):]: v for k, v in state.items() if k.startswith('decoder.')})
    center_np = np.asarray(json.loads((CHECKPOINT / 'extra.json').read_text())['center_t'], np.float32)
    center = torch.tensor(center_np, device='cuda')

    def load(rows):
        out = []
        for row in rows:
            fid = row['frame_id']
            with np.load(CACHE / 'lidar' / (fid + '.npz')) as a:
                f = torch.tensor(a['features'], device='cuda')
                src = np.asarray(a['source'], np.float32)
                GT = np.asarray(a['GT'], np.float32)
                target = torch.tensor(src @ GT[:3, :3].T + GT[:3, 3] - center_np, device='cuda')
            out.append(dict(row=row, f=f, source=torch.tensor(src, device='cuda'),
                            GT=torch.tensor(GT, device='cuda'), target=target))
        return out

    return dec, center, load(train_rows), load(val_rows)


@torch.no_grad()
def reference_weights(dec, frames):
    """Frozen model reliability -> fixed weights (un-normalised), one tensor per frame."""
    ws = []
    for d in frames:
        pred = dec(d['f'])
        ws.append(weights_from_scores(pred[:, 3]).detach())
    return ws


@torch.no_grad()
def evaluate(dec, frames, refw, solver):
    from pose_replay_min import selected_indices
    dec.eval()
    records = []
    for d, w in zip(frames, refw):
        pred = dec(d['f'])
        idx = selected_indices(pred)
        T = solver.estimator(d['source'][idx][None], pred[idx, :3][None])[0]
        trans = (T[:3, 3] + d['center'] - d['GT'][:3, 3]).norm().item()
        cos = ((T[:3, :3].T @ d['GT'][:3, :3]).trace() - 1) / 2
        rot = torch.rad2deg(cos.clamp(-1, 1).acos()).item()
        # Kabsch rotation bias with FIXED reference weights (normalised de-centering!)
        wn = w / w.sum().clamp_min(1e-12)
        X = pred[:, :3] - (wn[:, None] * pred[:, :3]).sum(0)
        Y = d['target'] - (wn[:, None] * d['target']).sum(0)
        Q, _ = kabsch_forward(X, Y, wn)
        cosq = ((Q.diagonal().sum() - 1) / 2).clamp(-1, 1)
        re = torch.rad2deg(torch.acos(cosq)).item()
        records.append(dict(frame_id=d['row']['frame_id'], date=d['row']['session_id'],
                            trans=trans, rot=rot, re=re))
    return records


def summarize(records):
    t = np.array([r['trans'] for r in records]); r = np.array([r['rot'] for r in records])
    e = np.array([r['re'] for r in records])
    return dict(MPE=float(t.mean()), MOE=float(r.mean()), Re_mean=float(e.mean()),
                Re_median=float(np.median(e)), rot95=float(np.percentile(r, 95)),
                trans95=float(np.percentile(t, 95)),
                fail=float(((t > 1) | (r > 2)).mean()))


def run_arm(name, dec, initial, train, val, refw_train, refw_val, trr, solver, p_out):
    from pose_replay_min import selected_indices  # noqa: F401
    torch.manual_seed(SEED)
    dec.load_state_dict(initial)
    dec.requires_grad_(True)
    dec.train()
    opt = torch.optim.AdamW([dict(params=dec.parameters(), lr=LR)], weight_decay=WD)
    rng = np.random.default_rng(SEED)
    folder = p_out / name
    folder.mkdir(parents=True, exist_ok=True)
    logs, evals = [], {}
    t0 = time.perf_counter()
    for epoch in range(0, EPOCHS + 1):
        if epoch % 5 == 0:
            rec = evaluate(dec, val, refw_val, solver)
            evals[epoch] = summarize(rec)
            json.dump(rec, open(folder / f'eval_{epoch}.json', 'w'))
            print(f'  [{name}] epoch {epoch}: ' + ', '.join(f'{k}={v:.4f}' for k, v in evals[epoch].items()), flush=True)
        if epoch == EPOCHS:
            break
        factor = epoch / WARMUP if epoch < WARMUP else .1 + .9 * (1 + math.cos(math.pi * (epoch - WARMUP) / max(1, EPOCHS - WARMUP))) / 2
        for g in opt.param_groups:
            g['lr'] = LR * factor
        lam = LAMBDA_MAX * min(1.0, (epoch + 1) / WARMUP)
        dec.train()
        order = rng.permutation(len(train))
        tot_trr, tot_rot, used, sk_g, sk_d, angs = [], [], 0, 0, 0, []
        for off in range(0, len(train), BATCH):
            batch = [train[i] for i in order[off:off + BATCH]]
            opt.zero_grad(set_to_none=True)
            pred = dec(torch.cat([d['f'] for d in batch]))
            ids = torch.cat([torch.full((len(d['f']),), i, device='cuda', dtype=torch.long)
                             for i, d in enumerate(batch)])
            regression = trr(torch.cat([d['target'] for d in batch]), pred[:, :3], pred[:, 3], ids)[0].mean()
            loss = regression / 1.0
            rotv = 0.0
            if name == 'B':
                rl, st = coherent_rotation_loss(pred[:, :3], torch.cat([d['target'] for d in batch]),
                                                torch.cat([refw_train[i] for i in order[off:off + BATCH]]),
                                                ids, theta0_deg=THETA0_DEG, gate_deg=GATE_DEG)
                loss = loss + lam * rl
                rotv = float(rl)
                used += st['valid']; sk_g += st['skipped_gate']; sk_d += st['skipped_degenerate']
                if st['angle_mean'] is not None:
                    angs.append(st['angle_mean'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
            opt.step()
            tot_trr.append(float(regression)); tot_rot.append(rotv)
        log = dict(epoch=epoch + 1, trr=float(np.mean(tot_trr)), rot=float(np.mean(tot_rot)), lam=lam,
                   rot_used=used, rot_skipped_gate=sk_g, rot_skipped_degen=sk_d,
                   rot_angle_mean=(float(np.mean(angs)) if angs else None))
        logs.append(log)
        if (epoch + 1) % 5 == 0:
            print(f'  [{name}] epoch {epoch+1}: trr={log["trr"]:.4f} rot={log["rot"]:.5f} '
                  f'used={used} gate={sk_g} degen={sk_d} angle={log["rot_angle_mean"]}', flush=True)
    json.dump(dict(logs=logs, evals=evals, seconds=time.perf_counter() - t0), open(folder / 'training.json', 'w'))
    return evals, logs


def main():
    global EPOCHS
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    args = ap.parse_args()
    EPOCHS = args.epochs
    OUT.mkdir(exist_ok=True)
    rows = json.loads(PROTO.read_text())['rows']
    train_rows, val_rows = pick_subset(rows)
    print(f'train frames: {len(train_rows)}  val frames: {len(val_rows)}')
    print('train by date:', dict(collections.Counter(r['session_id'] for r in train_rows)))
    print('val by date:', dict(collections.Counter(r['session_id'] for r in val_rows)))
    json.dump(dict(train=[r['frame_id'] for r in train_rows], val=[r['frame_id'] for r in val_rows],
                   seed=SEED, script_sha=__file__), open(OUT / 'subset.json', 'w'))

    from models.sc2pcr import Matcher
    dec, center, train, val = load_model_and_frames(train_rows, val_rows)
    for d in train + val:
        d['center'] = center
    trr = official_trr()
    solver = Matcher(inlier_threshold=2., d_thre=2, num_iterations=10, ratio=.15,
                     nms_radius=.1, max_points=3000, k1=30)
    refw_train = reference_weights(dec, train)
    refw_val = reference_weights(dec, val)
    initial = copy.deepcopy(dec.state_dict())

    results = {}
    for arm in ['A', 'B']:
        print(f'=== arm {arm} ===', flush=True)
        evals, logs = run_arm(arm, dec, initial, train, val, refw_train, refw_val, trr, solver, OUT)
        results[arm] = evals
    json.dump(results, open(OUT / 'summary.json', 'w'), indent=1)

    print('\n===== A vs B on the 32 validation frames =====')
    print(f'{"epoch":>6} | {"A: MPE":>8} {"A: MOE":>8} {"A: Re":>8} | {"B: MPE":>8} {"B: MOE":>8} {"B: Re":>8}')
    for ep in sorted(results['A']):
        a, b = results['A'][ep], results['B'][ep]
        print(f'{ep:>6} | {a["MPE"]:8.4f} {a["MOE"]:8.4f} {a["Re_mean"]:8.4f} | '
              f'{b["MPE"]:8.4f} {b["MOE"]:8.4f} {b["Re_mean"]:8.4f}')
    last = max(results['A'])
    a, b = results['A'][last], results['B'][last]
    print(f'\nfinal (epoch {last}):')
    print(f'  MOE: A {a["MOE"]:.4f} -> B {b["MOE"]:.4f}  ({100*(b["MOE"]-a["MOE"])/a["MOE"]:+.2f}%)')
    print(f'  MPE: A {a["MPE"]:.4f} -> B {b["MPE"]:.4f}  ({100*(b["MPE"]-a["MPE"])/a["MPE"]:+.2f}%)')
    print(f'  Re : A {a["Re_mean"]:.4f} -> B {b["Re_mean"]:.4f}  ({100*(b["Re_mean"]-a["Re_mean"])/a["Re_mean"]:+.2f}%)')


if __name__ == '__main__':
    main()
