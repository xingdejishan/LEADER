import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'image_gate'))
import run
from model import AnchoredFusion, contrastive
from models.sc2pcr import Matcher

CACHE = Path('/home/zhang/crossframe-visual-probe')
OUT = Path('/home/zhang/anchored-contrastive-fusion')
SEEDS = [2089, 2090, 2091]


def top(prediction):
    n = len(prediction)
    return prediction[:, 3].topk(max(min(50, n), n // 2)).indices


def make_protocol():
    rows = json.loads((CACHE / 'manifest.json').read_text())
    for date in sorted({r['session_id'] for r in rows}):
        subset = [r for r in rows if r['session_id'] == date and r['split'] == 'reference']
        boundary = int(.8 * len(subset))
        for i, r in enumerate(subset):
            r['role'] = 'fit' if i < boundary else 'internal'
    for r in rows:
        if r['split'] == 'query':
            r['role'] = 'development'
    protocol = dict(rows=rows, seeds=SEEDS, epochs=100, batch_frames=8, learning_rate=.001, warmup=5,
        final_lr=.00001, weight_decay=.0001, rho=.05, contrastive_weight=.01, temperature=.1,
        candidates=16, positive_m=.5, negative_m=2., checkpoint_epochs=list(range(0, 101, 10)),
        checkpoint_selection='Internal mean final translation, earliest checkpoint wins ties; zero identity eligible; no refit',
        anchors='Only fit-frame image-valid original LiDAR features, frozen. Same fixed library for fit/internal/development.',
        contrastive='Negative log summed positive probability over positives and explicit negatives; 0.5m<distance<2m ignored; mean per eligible frame; no GT positive insertion; correct and incorrect original rankings both included',
        protection='First floor(K/2) in original descending top-K; residual zero for protected and image-invalid voxels',
        exclusion='Same frame, same-date time gap <10s, or camera-center distance<5m AND relative rotation<15deg',
        mechanism='Original LiDAR top16 from FIT library fixed before training; ambiguity has positive and explicit negative and original top1-top2 cosine<=.02; also record original Matcher/protected intersections',
        shuffle='Independent fixed valid-descriptor permutation per frame/seed for training and evaluation; anchors remain original LiDAR',
        pass_rule='Aligned selected models beat baseline and same-seed shuffled mean translation in all three seeds and each date; seed-mean trajectory-block paired CI upper bound<0 for both comparisons; no mean rotation or translation-p95 degradation in all three seeds; contrastive ranking improves versus original and shuffled',
        bootstrap='10000 resamples, seed271828; contiguous blocks of up to10 chronological frames, split at date or >10s gaps; resample blocks within each date; seed-mean paired per-frame differences',
        scope='578 fit /145 internal /182 previously probed development; no old32 or test data; no final NCLT improvement claim',
        source_cache_manifest_sha256=run.digest(CACHE/'manifest.json'))
    run.save_json(OUT/'protocol.json', protocol)
    print({role:sum(r['role']==role for r in rows) for role in ['fit','internal','development']}, flush=True)
    return protocol


def load_data(decoder, rows):
    center = np.asarray(json.loads((run.WORKSPACE/'research/image_gate_checkpoint/extra.json').read_text())['center_t'], np.float32)
    data = []
    for r in rows:
        with np.load(CACHE/'lidar'/(r['frame_id']+'.npz')) as l, np.load(CACHE/'visual_raw'/(r['frame_id']+'.npz')) as v:
            item = dict(row=r, f=torch.tensor(l['features'], device='cuda'), image=torch.tensor(v['image'], device='cuda'),
                        valid=torch.tensor(v['valid'], device='cuda'), source=torch.tensor(l['source'], device='cuda'),
                        target=torch.tensor(l['source']@l['GT'][:3,:3].T+l['GT'][:3,3]-center, device='cuda', dtype=torch.float32),
                        gt=torch.tensor(l['GT'], device='cuda', dtype=torch.float32), center=torch.tensor(center, device='cuda'),
                        xyz=l['representative_world'], pose=l['camera_pose'])
        with torch.no_grad():
            item['base'] = decoder(item['f'])
        item['indices'] = top(item['base'])
        item['protected'] = torch.zeros(len(item['f']), device='cuda', dtype=torch.bool)
        item['protected'][item['indices'][:len(item['indices'])//2]] = True
        item['editable'] = item['valid'] & ~item['protected']
        item['shuffle'] = {}
        valid = torch.where(item['valid'])[0]
        for s in SEEDS:
            rng = np.random.default_rng(s+int(r['frame_id'])%1000000007)
            order = torch.arange(len(item['f']),device='cuda')
            order[valid] = valid[torch.as_tensor(rng.permutation(len(valid)),device='cuda')]
            item['shuffle'][s] = order
        data.append(item)
    return data


def mine(data):
    fit = [d for d in data if d['row']['role']=='fit']
    anchors = F.normalize(torch.cat([d['f'][d['valid']] for d in fit]), dim=-1)
    xyz = np.concatenate([d['xyz'][d['valid'].cpu().numpy()] for d in fit])
    owners = torch.cat([torch.full((int(d['valid'].sum()),),i,device='cuda',dtype=torch.long) for i,d in enumerate(fit)])
    poses = np.stack([d['pose'] for d in fit])
    folder = OUT/'candidates'
    folder.mkdir(exist_ok=True)
    stats = []
    for i,d in enumerate(data):
        file = folder/(d['row']['frame_id']+'.npz')
        if not file.exists():
            pose = d['pose']
            distance = np.linalg.norm(poses[:,:3,3]-pose[:3,3],axis=1)
            rotation = np.rad2deg(np.arccos(np.clip((np.einsum('nij,ij->n',poses[:,:3,:3],pose[:3,:3])-1)/2,-1,1)))
            recent = np.array([f['row']['session_id']==d['row']['session_id'] and abs(int(f['row']['frame_id'])-int(d['row']['frame_id']))<10000000 for f in fit])
            eligible = torch.tensor(~(recent|((distance<5)&(rotation<15))),device='cuda')[owners]
            assert int(eligible.sum())>=16
            indices = torch.where(d['valid'])[0]
            blocks, gaps = [], []
            for start in range(0,len(indices),64):
                sim = F.normalize(d['f'][indices[start:start+64]],dim=-1)@anchors.T
                sim[:,~eligible] = -float('inf')
                scores, candidates = sim.topk(16,dim=-1)
                blocks.append(candidates.cpu().numpy())
                gaps.extend((scores[:,0]-scores[:,1]).cpu().tolist())
            candidates = np.concatenate(blocks)
            spatial = np.linalg.norm(xyz[candidates]-d['xyz'][indices.cpu().numpy(),None],axis=-1)
            np.savez_compressed(file,indices=indices.cpu().numpy(),candidates=candidates,positive=spatial<=.5,negative=spatial>=2,gap=np.array(gaps))
        with np.load(file) as a:
            d['query'] = torch.tensor(a['indices'],device='cuda')
            d['candidates'] = torch.tensor(a['candidates'],device='cuda')
            d['positive'] = torch.tensor(a['positive'],device='cuda')
            d['negative'] = torch.tensor(a['negative'],device='cuda')
            has_both = d['positive'].any(1)&d['negative'].any(1)
            d['train_mask'] = has_both & d['editable'][d['query']]
            d['ambiguous'] = has_both & torch.tensor(a['gap']<=.02,device='cuda')
        correct = d['positive'][:,0]
        original_selected = torch.zeros_like(d['valid'])
        original_selected[d['indices']] = True
        stats.append(dict(frame_id=d['row']['frame_id'],role=d['row']['role'],date=d['row']['session_id'],
                          train_pairs=int(d['train_mask'].sum()),original_correct_train=int((correct&d['train_mask']).sum()),
                          original_wrong_train=int((~correct&d['train_mask']).sum()),ambiguous=int(d['ambiguous'].sum()),
                          ambiguous_matcher=int((d['ambiguous']&original_selected[d['query']]).sum()),
                          ambiguous_protected=int((d['ambiguous']&d['protected'][d['query']]).sum())))
        if i%100==0:
            print('candidates',i+1,len(data),flush=True)
    run.save_json(OUT/'candidate_stats.json',stats)
    return anchors


def visual(d, arm, seed):
    return d['image'] if arm=='aligned' else d['image'][d['shuffle'][seed]]


def pose(d,pred,matcher,fixed=False):
    torch.manual_seed(2089)
    indices = d['indices'] if fixed else top(pred)
    transform = matcher.estimator(d['source'][indices][None],pred[indices,:3][None])[0]
    transform[:3,3] += d['center']
    translation = (transform[:3,3]-d['gt'][:3,3]).norm().item()
    cosine = ((transform[:3,:3].T@d['gt'][:3,:3]).trace()-1)/2
    return [translation,torch.rad2deg(cosine.clamp(-1,1).acos()).item()]


@torch.no_grad()
def evaluate(head,decoder,data,anchors,matcher,arm,seed,detail=False):
    records = []
    for d in data:
        fused = head(d['f'],visual(d,arm,seed),d['editable']) if head is not None else d['f']
        pred = decoder(fused) if head is not None else d['base']
        assert torch.equal(fused[~d['editable']],d['f'][~d['editable']])
        assert torch.equal(pred[~d['editable']],d['base'][~d['editable']])
        r = dict(frame_id=d['row']['frame_id'],date=d['row']['session_id'],standard=pose(d,pred,matcher))
        if detail:
            r['fixed'] = pose(d,pred,matcher,True)
            final = torch.zeros_like(d['valid'])
            final[top(pred)] = True
            r['protected_count'] = int(d['protected'].sum())
            r['protected_retained'] = int((final&d['protected']).sum())
            mask = d['ambiguous']
            sim = (F.normalize(fused[d['query']],dim=-1)[:,None]*anchors[d['candidates']]).sum(-1)
            choice = sim.argmax(1)
            hits = d['positive'].gather(1,choice[:,None])[:,0]
            original = d['positive'][:,0]
            r['mechanism'] = dict(count=int(mask.sum()),correct=int((hits&mask).sum()),
                                 rescue=int((hits&~original&mask).sum()),damage=int((~hits&original&mask).sum()))
            selected = torch.zeros_like(d['valid'])
            selected[d['indices']] = True
            visible_selected = selected&d['valid']
            error_delta = (pred[:,:3]-d['target']).norm(dim=-1)-(d['base'][:,:3]-d['target']).norm(dim=-1)
            r['selected_visible'] = dict(count=int(visible_selected.sum()),delta_sum=float(error_delta[visible_selected].sum()),
                                        harmed=int((error_delta[visible_selected]>.01).sum()))
        records.append(r)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['prepare','train','evaluate'],default='train')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    protocol = json.loads((OUT/'protocol.json').read_text()) if (OUT/'protocol.json').exists() else make_protocol()
    decoder = run.load_leader(SimpleNamespace(checkpoint=run.WORKSPACE/'research/image_gate_checkpoint')).decoder
    data = load_data(decoder,protocol['rows'])
    anchors = mine(data)
    fit = [d for d in data if d['row']['role']=='fit']
    internal = [d for d in data if d['row']['role']=='internal']
    dev = [d for d in data if d['row']['role']=='development']
    matcher = Matcher(inlier_threshold=2.,d_thre=2,num_iterations=10,ratio=.15,nms_radius=.1,max_points=3000,k1=30)
    if args.stage=='prepare':
        head = AnchoredFusion().cuda()
        assert sum(p.numel() for p in head.parameters())==37408
        for d in data:
            assert torch.equal(head(d['f'],d['image'],d['editable']),d['f'])
        run.save_json(OUT/'preflight.json',dict(identity_frames=len(data),parameters=37408,all_original_predictions_recomputed=True))
        return
    if args.stage=='evaluate':
        run.save_json(OUT/'baseline_development.json',evaluate(None,decoder,dev,anchors,matcher,'aligned',2089,True))
        from fusion import ImageGate
        old = ImageGate().cuda()
        old.load_state_dict(torch.load('/home/zhang/leader-image-gate-raw/aligned.pt'))
        records = []
        with torch.no_grad():
            for d in dev:
                pred = decoder(old(d['f'],d['image'],d['valid']))
                records.append(dict(frame_id=d['row']['frame_id'],date=d['row']['session_id'],standard=pose(d,pred,matcher),fixed=pose(d,pred,matcher,True)))
        run.save_json(OUT/'line1_development.json',records)
        for seed in SEEDS:
            for arm in ['aligned','shuffled']:
                folder = OUT/f'{arm}_{seed}'
                head = AnchoredFusion().cuda()
                head.load_state_dict(torch.load(folder/'best.pt'))
                run.save_json(folder/'development.json',evaluate(head,decoder,dev,anchors,matcher,arm,seed,True))
                print('evaluated',arm,seed,flush=True)
        return
    trr = run.official_trr()
    baseline = evaluate(None,decoder,internal,anchors,matcher,'aligned',2089)
    run.save_json(OUT/'baseline_internal.json',baseline)
    base_score = np.mean([r['standard'][0] for r in baseline])
    for seed in SEEDS:
        for arm in ['aligned','shuffled']:
            folder = OUT/f'{arm}_{seed}'
            folder.mkdir(exist_ok=True)
            if (folder/'complete.json').exists():
                continue
            torch.manual_seed(seed)
            head = AnchoredFusion().cuda()
            optimizer = torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001)
            rng = np.random.default_rng(seed)
            best, logs = base_score, []
            torch.save(head.state_dict(),folder/'best.pt')
            torch.save(head.state_dict(),folder/'epoch0.pt')
            run.save_json(folder/'selection.json',dict(epoch=0,mean_translation=float(best)))
            for epoch in range(1,101):
                start_time = time.perf_counter()
                lr = .001*epoch/5 if epoch<=5 else .00001+(.001-.00001)*(1+math.cos(math.pi*(epoch-5)/95))/2
                optimizer.param_groups[0]['lr'] = lr
                order = rng.permutation(len(fit))
                totals = []
                for offset in range(0,len(fit),8):
                    batch = [fit[i] for i in order[offset:offset+8]]
                    optimizer.zero_grad(set_to_none=True)
                    features = torch.cat([d['f'] for d in batch])
                    images = torch.cat([visual(d,arm,seed) for d in batch])
                    editable = torch.cat([d['editable'] for d in batch])
                    fused = head(features,images,editable)
                    pred = decoder(fused)
                    batch_idx = torch.cat([torch.full((len(d['f']),),i,device='cuda',dtype=torch.long) for i,d in enumerate(batch)])
                    regression = trr(torch.cat([d['target'] for d in batch]),pred[:,:3],pred[:,3],batch_idx)[0].mean()
                    losses = []
                    point_offset = 0
                    for d in batch:
                        use = d['train_mask']
                        if use.any():
                            query = fused[point_offset+d['query'][use]]
                            losses.append(contrastive(query,anchors[d['candidates'][use]],d['positive'][use],d['negative'][use]).mean())
                        point_offset += len(d['f'])
                    contrast = torch.stack(losses).mean() if losses else fused.sum()*0
                    loss = regression+.01*contrast
                    assert torch.isfinite(loss)
                    loss.backward()
                    assert all(p.grad is None for p in decoder.parameters())
                    optimizer.step()
                    totals.append([loss.item(),regression.item(),contrast.item()])
                log = dict(epoch=epoch,lr=lr,loss=np.mean(totals,axis=0).tolist(),seconds=time.perf_counter()-start_time)
                if epoch%10==0:
                    records = evaluate(head,decoder,internal,anchors,matcher,arm,seed)
                    score = np.mean([r['standard'][0] for r in records])
                    log['internal'] = run.metrics([r['standard'] for r in records])
                    if score<best:
                        best = score
                        torch.save(head.state_dict(),folder/'best.pt')
                        run.save_json(folder/'selection.json',dict(epoch=epoch,mean_translation=float(best)))
                    run.save_json(folder/f'internal_{epoch}.json',records)
                    print(arm,seed,epoch,'internal',score,'epoch_seconds',log['seconds'],flush=True)
                elif epoch==1:
                    print(arm,seed,epoch,'epoch_seconds',log['seconds'],flush=True)
                logs.append(log)
                run.save_json(folder/'training.json',logs)
            torch.save(head.state_dict(),folder/'last.pt')
            run.save_json(folder/'complete.json',dict(epochs=100,updates=100*math.ceil(len(fit)/8)))
    print('TRAINING COMPLETE',flush=True)


if __name__=='__main__':
    main()
