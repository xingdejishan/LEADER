import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'image_gate'))
import run
from model import BenefitModulation, auxiliary_losses, reliability_rank
from models.sc2pcr import Matcher

OUT = Path('/home/zhang/benefit-channel-modulation')
ARGS = SimpleNamespace(output=Path('/home/zhang/leader-image-gate-raw'),
                       checkpoint=run.WORKSPACE / 'research/image_gate_checkpoint')


def selected(pred):
    return pred[:, 3].topk(max(min(50, len(pred)), len(pred) // 2)).indices


def pose_error(item, pred, matcher, fixed=False):
    torch.manual_seed(2089)
    indices = item['indices'] if fixed else selected(pred)
    pose = matcher.estimator(item['source'][indices][None], pred[indices, :3][None])[0]
    pose[:3, 3] += item['center']
    translation = (pose[:3, 3] - item['GT'][:3, 3]).norm().item()
    cosine = ((pose[:3, :3].T @ item['GT'][:3, :3]).trace() - 1) / 2
    return [translation, torch.rad2deg(cosine.clamp(-1, 1).acos()).item()]


def forward(head, decoder, item, arm, seed, warmup=False):
    visual = item['image'] if arm == 'aligned' else item['shuffled'][seed]
    output = head(item['features'], visual, item['rank'], item['valid'], warmup)
    return decoder(output['fused']), output


@torch.no_grad()
def evaluate(head, decoder, data, matcher, arm, seed, detailed=False):
    records = []
    for item in data:
        pred, output = forward(head, decoder, item, arm, seed)
        record = dict(frame_id=item['frame_id'], standard=pose_error(item, pred, matcher))
        if detailed:
            record['fixed'] = pose_error(item, pred, matcher, True)
            error = (pred[:, :3] - item['target']).norm(dim=-1)
            delta = error - item['error']
            mask = item['valid'] & item['selected']
            record['protected'] = dict(count=int(mask.sum()), delta_sum=float(delta[mask].sum()),
                                      harmed=int((delta[mask] > .01).sum()), helped=int((delta[mask] < -.01).sum()))
            attempt = decoder(output['attempt'])
            benefit = item['error'] - (attempt[:, :3] - item['target']).norm(dim=-1)
            record['gate_bins'] = []
            for b in range(5):
                mask = item['valid'] & (item['rank'] >= b / 5) & (item['rank'] < (b + 1) / 5 + (1e-6 if b == 4 else 0))
                g, v = output['gate'][mask].cpu().numpy(), benefit[mask].cpu().numpy()
                record['gate_bins'].append(dict(gate=g.tolist(), benefit=v.tolist()))
        records.append(record)
    return records


def main():
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True)
    rows = json.loads((ARGS.output / 'manifest.json').read_text())
    train_rows = sorted([r for r in rows if r['split'] == 'train'], key=lambda r: r['frame_id'])
    val_rows = sorted([r for r in rows if r['split'] == 'val'], key=lambda r: r['frame_id'])
    protocol = dict(fit=[r['frame_id'] for r in train_rows[:51]], internal=[r['frame_id'] for r in train_rows[51:]],
                    development=[r['frame_id'] for r in val_rows], seeds=[2089, 2090, 2091], epochs=100,
                    batch_frames=4, optimizer='AdamW', lr=1e-3, final_lr=1e-5, weight_decay=1e-4,
                    warmup_epochs=5, fixed_gate_epochs=10, checkpoint_epochs=list(range(10, 101, 10)),
                    selection='Minimum internal mean final translation; earliest epoch wins ties; epoch 0 reference only; no refit',
                    shuffle='Fixed per-frame permutation of valid descriptors, independently seeded; same at train and evaluation',
                    loss='original TRR + selected-visible ReLU coordinate degradation + 0.01 balanced gate BCE after epoch 10',
                    scope='51 fit / 13 internal / 32 previously touched local development frames; not full NCLT',
                    hypothesis='Aligned must beat baseline and same-seed shuffled in all three seeds, without mean rotation or p95 translation degradation; fixed-candidate diagnostic must not degrade.')
    run.save_json(OUT / 'protocol.json', protocol)
    decoder = run.load_leader(ARGS).decoder
    matcher = Matcher(inlier_threshold=2., d_thre=2, num_iterations=10, ratio=.15, nms_radius=.1, max_points=3000, k1=30)
    data = []
    for row in train_rows + val_rows:
        item = run.frame(ARGS, row)
        item['frame_id'] = row['frame_id']
        item['indices'] = selected(item['prediction'])
        item['selected'] = torch.zeros(len(item['features']), device='cuda', dtype=torch.bool)
        item['selected'][item['indices']] = True
        item['rank'] = reliability_rank(item['prediction'][:, 3])
        item['error'] = (item['prediction'][:, :3] - item['target']).norm(dim=-1)
        item['shuffled'] = {}
        for seed in protocol['seeds']:
            rng = np.random.default_rng(seed + int(row['frame_id']) % 1000000007)
            indices = torch.where(item['valid'])[0]
            visual = item['image'].clone()
            visual[indices] = visual[indices[torch.as_tensor(rng.permutation(len(indices)), device='cuda')]]
            item['shuffled'][seed] = visual
        data.append(item)
    head = BenefitModulation().cuda()
    parity = 0.
    with torch.no_grad():
        for item in data:
            pred, output = forward(head, decoder, item, 'aligned', 2089)
            assert torch.equal(output['fused'], item['features'])
            parity = max(parity, (pred - item['prediction']).abs().max().item())
    assert parity < 1e-4, parity
    run.save_json(OUT / 'preflight.json', dict(identity=True, cached_prediction_max_error=parity,
                                            parameters=sum(p.numel() for p in head.parameters())))
    baseline = []
    with torch.no_grad():
        for item in data[64:]:
            baseline.append(dict(frame_id=item['frame_id'], standard=pose_error(item, item['prediction'], matcher)))
    run.save_json(OUT / 'baseline.json', baseline)
    trr = run.official_trr()
    for seed in protocol['seeds']:
        for arm in ['aligned', 'shuffled']:
            destination = OUT / f'{arm}_{seed}'
            destination.mkdir(exist_ok=True)
            if (destination / 'development.json').exists():
                continue
            torch.manual_seed(seed)
            head = BenefitModulation().cuda()
            optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
            rng = np.random.default_rng(seed)
            logs, best = [], float('inf')
            for epoch in range(1, 101):
                lr = .001 * epoch / 5 if epoch <= 5 else 1e-5 + (.001 - 1e-5) * (1 + math.cos(math.pi * (epoch - 5) / 95)) / 2
                optimizer.param_groups[0]['lr'] = lr
                order = rng.permutation(51)
                totals = []
                for offset in range(0, 51, 4):
                    batch = order[offset:offset + 4]
                    optimizer.zero_grad(set_to_none=True)
                    for index in batch:
                        item = data[index]
                        pred, output = forward(head, decoder, item, arm, seed, epoch <= 10)
                        with torch.no_grad():
                            attempt = decoder(output['attempt'])
                        keep, gate, _ = auxiliary_losses(pred, attempt, item['error'], item['target'], output['logits'], item['valid'], item['selected'])
                        loss_trr = trr(item['target'], pred[:, :3], pred[:, 3], torch.zeros(len(pred), dtype=torch.long, device='cuda'))[0].mean()
                        loss = loss_trr + keep + (.01 * gate if epoch > 10 else 0)
                        assert torch.isfinite(loss)
                        (loss / len(batch)).backward()
                        totals.append([loss.item(), keep.item(), gate.item()])
                    assert all(p.grad is None for p in decoder.parameters())
                    optimizer.step()
                log = dict(epoch=epoch, lr=lr, loss=np.mean(totals, axis=0).tolist())
                if epoch % 10 == 0:
                    records = evaluate(head, decoder, data[51:64], matcher, arm, seed)
                    score = np.mean([r['standard'][0] for r in records])
                    log['internal'] = run.metrics([r['standard'] for r in records])
                    if score < best:
                        best = score
                        torch.save(head.state_dict(), destination / 'best.pt')
                        run.save_json(destination / 'selection.json', dict(epoch=epoch, mean_translation=float(score)))
                    run.save_json(destination / f'internal_{epoch}.json', records)
                    print(f'{arm} seed={seed} epoch={epoch} loss={log["loss"][0]:.6f} internal={score:.6f}', flush=True)
                logs.append(log)
                run.save_json(destination / 'training.json', logs)
            torch.save(head.state_dict(), destination / 'last.pt')
            head.load_state_dict(torch.load(destination / 'best.pt'))
            run.save_json(destination / 'development.json', evaluate(head, decoder, data[64:], matcher, arm, seed, True))
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
