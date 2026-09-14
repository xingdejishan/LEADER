import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.stats import rankdata, spearmanr
from torch.nn import functional as F
import run
from fusion import ImageGate
from context_experiment import ContextGate
import lpgrf_experiment as lp


ROOTS = dict(old='/home/zhang/leader-image-gate', line1='/home/zhang/leader-image-gate-raw',
             line3='/home/zhang/leader-image-gate-context', line4='/home/zhang/leader-image-gate-geometry',
             line2='/home/zhang/leader-image-gate-lpgrf')
OUTPUT = run.HERE/'results/fusion_diagnostic'


def selection(prediction):
    n = len(prediction)
    mask = torch.zeros(n, dtype=torch.bool, device='cuda')
    mask[prediction[:, 3].topk(max(min(50, n), int(.5*n))).indices] = True
    return mask


def summary(data):
    if not len(data):
        return dict(count=0)
    before, after, gate, relative, shift, selected, new_selected, confidence, wrong, constant = data.T
    gain = before-after
    helpful = gain > .01
    harmful = gain < -.01
    auc = None
    eligible = helpful | harmful
    labels = helpful[eligible]
    if labels.any() and (~labels).any():
        ranks = rankdata(gate[eligible])
        auc = float((ranks[labels].sum()-labels.sum()*(labels.sum()+1)/2)/(labels.sum()*(~labels).sum()))
    rho = float(spearmanr(gate, gain).statistic) if np.ptp(gate)>0 and np.ptp(gain)>0 else None
    return dict(count=len(data), mean_error_before=float(before.mean()), mean_error_after=float(after.mean()),
        mean_error_delta_m=float((after-before).mean()), improved_over_1cm=float(helpful.mean()), harmed_over_1cm=float(harmful.mean()),
        mean_improvement_m=float(gain.clip(0).sum()/len(gain)), mean_harm_m=float((-gain).clip(0).sum()/len(gain)),
        gate_mean=float(gate.mean()), gate_helpful_mean=float(gate[helpful].mean()) if helpful.any() else None,
        gate_harmful_mean=float(gate[harmful].mean()) if harmful.any() else None, gate_helpfulness_auc=auc, gate_gain_spearman=rho,
        relative_feature_delta_median=float(np.median(relative)), coordinate_shift_median_m=float(np.median(shift)),
        confidence_delta_mean=float(confidence.mean()), wrong_random_mean_error=float(wrong.mean()), constant_image_mean_error=float(constant.mean()))


def main():
    torch.set_num_threads(4)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = json.loads((Path(ROOTS['line1'])/'manifest.json').read_text())
    run.save_json(OUTPUT/'protocol.json', dict(scope='Frozen-weight diagnostic, same 64 train/32 development frames; no training or threshold search',
        questions=['Does broader coverage expose harmful residuals?', 'Do gates discriminate useful from harmful corrections?', 'Do confidence changes affect selection?', 'Does image content matter beyond constant residuals?'],
        threshold_m=.01, selected_inlier_threshold_m=2., seed=2089,
        constant='Per-method mean of all valid training descriptors; for line2 computed with its finetuned image block',
        reference='Original frozen decoder for old/1/3/4; same finetuned decoder with no image for line2',
        oracle='GT selects per-voxel better coordinate prediction; diagnostic only, never deployable or a method result'))
    from models.sc2pcr import Matcher
    matcher = Matcher(inlier_threshold=2., d_thre=2, num_iterations=10, ratio=.15, nms_radius=.1, max_points=3000, k1=30)
    all_results = {}
    for name, root in ROOTS.items():
        args = SimpleNamespace(output=Path(root), checkpoint=run.WORKSPACE/'research/image_gate_checkpoint')
        if name == 'line2':
            model, image_model, gate, _, pca = lp.components(args)
            saved = torch.load(args.output/'aligned.pt', map_location='cuda')
            model.decoder.load_state_dict(saved['decoder'])
            image_model.layer3.load_state_dict(saved['layer3'])
            gate.load_state_dict(saved['fusion'])
        else:
            model = run.load_leader(args)
            gate = (ContextGate() if name in ['line3', 'line4'] else ImageGate()).cuda().eval()
            gate.load_state_dict(torch.load(args.output/'aligned.pt', map_location='cuda'))
        cached = []
        with torch.inference_mode():
            for row in rows:
                item = lp.load_frame(args, row) if name=='line2' else run.frame(args, row)
                visual = lp.image_features(args, row, item, image_model, pca) if name=='line2' else item['image']
                cached.append((row, {k:v.cpu() for k,v in item.items()}, visual.cpu()))
        training = torch.cat([visual[item['valid']] for row,item,visual in cached if row['split']=='train'])
        mean_image = training.mean(0).cuda()
        grouped = {split:{k:[] for k in ['valid', 'selected_before', 'selected_after', 'newly_visible']} for split in ['train','val']}
        records = []
        poses = {k:[] for k in ['reference', 'fused', 'coordinates_only', 'confidence_only', 'wrong_random', 'constant_image', 'oracle_coordinate_choice']}
        checks = dict(invalid_prediction_max_error=0., cached_baseline_prediction_max_error=0.)
        with torch.inference_mode():
            for frame_index, (row, cpu_item, cpu_visual) in enumerate(cached):
                item = {k:v.cuda() for k,v in cpu_item.items()}
                visual = cpu_visual.cuda()
                features, valid, target = item['features'], item['valid'], item['target']
                def fuse(value):
                    return gate(features, value, valid, item['distance']) if name=='line2' else gate(features, value, valid)
                fused_features = fuse(visual)
                reference = model.decoder(features)
                prediction = model.decoder(fused_features)
                if name != 'line2':
                    error = float((reference-item['prediction']).abs().max())
                    assert error<=1e-4
                    checks['cached_baseline_prediction_max_error'] = max(checks['cached_baseline_prediction_max_error'], error)
                if (~valid).any():
                    error = float((prediction[~valid]-reference[~valid]).abs().max())
                    assert error<=1e-4
                    checks['invalid_prediction_max_error'] = max(checks['invalid_prediction_max_error'], error)
                if name=='line2':
                    local = gate.image_norm(torch.where(valid[:,None], visual, torch.zeros_like(visual)))
                    ql = F.leaky_relu(gate.lidar_gate_proj(gate.lidar_norm(features)), .01)
                    qi = F.leaky_relu(gate.image_gate_proj(local), .01)
                    weights = gate.gate(torch.cat([ql,qi,(item['distance'][:,None]/100).clamp(0,1),valid[:,None].float()],-1))[:,0]
                else:
                    inner = gate.gate if isinstance(gate, ContextGate) else gate
                    value = torch.where(valid[:,None],visual,torch.zeros_like(visual))
                    if isinstance(gate, ContextGate):
                        value = value[:,:128]+gate.adapter(value[:,128:])
                    local = inner.image_norm(torch.where(valid[:,None],value,torch.zeros_like(value)))
                    weights = inner.gate(torch.cat([inner.lidar_norm(features),local],-1)).sigmoid()[:,0]
                indices = torch.where(valid)[0]
                wrong = visual.clone()
                generator = torch.Generator(device='cuda').manual_seed(2089+frame_index)
                wrong[indices] = visual[indices[torch.randperm(len(indices), generator=generator, device='cuda')]]
                wrong_pred = model.decoder(fuse(wrong))
                constant_pred = model.decoder(fuse(mean_image[None].expand_as(visual)))
                e0 = (reference[:,:3]-target).norm(dim=-1)
                e1 = (prediction[:,:3]-target).norm(dim=-1)
                s0,s1 = selection(reference),selection(prediction)
                shift = (prediction[:,:3]-reference[:,:3]).norm(dim=-1)
                relative = (fused_features-features).norm(dim=-1)/features.norm(dim=-1).clamp_min(1e-8)
                values = torch.stack([e0,e1,weights,relative,shift,s0.float(),s1.float(),prediction[:,3]-reference[:,3],
                    (wrong_pred[:,:3]-target).norm(dim=-1),(constant_pred[:,:3]-target).norm(dim=-1)],dim=1).cpu().numpy()
                split = row['split']
                masks = dict(valid=valid, selected_before=valid&s0, selected_after=valid&s1)
                with np.load(Path(ROOTS['old'])/'visual'/(row['frame_id']+'.npz')) as old:
                    masks['newly_visible'] = valid&~torch.tensor(old['valid'],device='cuda')
                for key, mask in masks.items():
                    grouped[split][key].append(values[mask.cpu().numpy()])
                record = dict(frame_id=row['frame_id'],split=split,total=len(valid),valid=int(valid.sum()),
                    selected_valid_before=int((valid&s0).sum()),selected_valid_after=int((valid&s1).sum()),
                    selected_count=int(s0.sum()),selected_inliers_before=int(((e0<2)&s0).sum()),selected_inliers_after=int(((e1<2)&s1).sum()),
                    entered_selection=int((s1&~s0).sum()),left_selection=int((s0&~s1).sum()),
                    selected_mean_error_before=float(e0[s0].mean()),selected_mean_error_after=float(e1[s1].mean()),
                    coordinate_shift_on_constant_vs_actual=float((constant_pred[valid,:3]-prediction[valid,:3]).norm(dim=-1).mean()) if valid.any() else None)
                if split=='val':
                    coordinate_only = torch.cat([prediction[:,:3],reference[:,3:]],-1)
                    confidence_only = torch.cat([reference[:,:3],prediction[:,3:]],-1)
                    oracle = torch.cat([torch.where((e1<e0)[:,None],prediction[:,:3],reference[:,:3]),reference[:,3:]],-1)
                    for key,pred in [('reference',reference),('fused',prediction),('coordinates_only',coordinate_only),('confidence_only',confidence_only),
                        ('wrong_random',wrong_pred),('constant_image',constant_pred),('oracle_coordinate_choice',oracle)]:
                        error = lp.pose_error(matcher,item,pred)
                        poses[key].append(error)
                    record['pose_errors'] = {key:value[-1] for key,value in poses.items()}
                records.append(record)
        results = {split:{key:summary(np.concatenate(parts)) for key,parts in groups.items()} for split,groups in grouped.items()}
        results['pose'] = {key:run.metrics(value) for key,value in poses.items()}
        results['checks'] = checks
        for split in ['train','val']:
            subset = [r for r in records if r['split']==split]
            results[split]['selection'] = {key:sum(r[key] for r in subset) for key in ['selected_count','selected_valid_before','selected_valid_after','selected_inliers_before','selected_inliers_after','entered_selection','left_selection']}
            results[split]['frames_with_valid'] = sum(len(part)>0 for part in grouped[split]['valid'])
            results[split]['frame_mean_coordinate_delta_m'] = float(np.mean([float((part[:,1]-part[:,0]).mean()) for part in grouped[split]['valid'] if len(part)]))
        original_records = json.loads((args.output/'validation_frames.json').read_text())
        expected = np.array([r['errors']['aligned'] for r in original_records])
        assert np.allclose(expected,poses['fused'],atol=1e-5)
        run.save_json(OUTPUT/(name+'_frames.json'), records)
        run.save_json(OUTPUT/(name+'.json'),results)
        all_results[name]=results
        print(json.dumps(dict(method=name,valid=results['val']['valid'],selected=results['val']['selected_before'],poses=results['pose']),indent=2),flush=True)
        del model,gate,cached,training
    run.save_json(OUTPUT/'summary.json',all_results)


if __name__=='__main__':
    main()
