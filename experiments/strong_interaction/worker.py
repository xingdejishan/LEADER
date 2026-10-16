import os
import sys
import json
import time
import hashlib
import traceback
import resource
import shutil
from pathlib import Path
from collections import OrderedDict, Counter

import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT/'plan.json').read_text())
STORE = Path(PLAN['store'])
OLD = Path(PLAN['v8_store'])
V9 = Path('/data2/zhangcy/lightglue-v9-hardneg-20261016-v1')
sys.path.insert(0, PLAN['v8_project'])
from common import K, model_matrix
sys.path.insert(0, PLAN['vendor'])
sys.path.insert(0, PLAN['vendor_py'])
torch.hub.set_dir(PLAN['hub'])
from lightglue import LightGlue, SuperPoint
from lightglue.lightglue import filter_matches
from collaborative import CollaborativeV8

torch.set_num_threads(1)
torch.manual_seed(20)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
BEGIN = time.monotonic()
LIB = json.loads((OLD/'library.json').read_text())['references']
KM = model_matrix() @ K
LRU = OrderedDict()
ANCHORS = OrderedDict()
EXTRACTOR = None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def save(name, value):
    p = STORE/name
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value, allow_nan=False))
    os.replace(temp, p)


def guard():
    assert os.getuid() == 1003
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 < 8*2**30
    free = int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
    assert free > 24*2**30
    assert shutil.disk_usage(STORE).free > 12*2**30
    assert time.monotonic()-BEGIN < 6*3600
    if torch.cuda.is_initialized():
        apps = os.popen('nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader,nounits').read()
        for line in apps.splitlines():
            pid, uuid = line.split(', ')
            if uuid == PLAN['gpu_uuid'] and (Path('/proc')/pid).exists():
                assert (Path('/proc')/pid).stat().st_uid == 1003, 'foreign GPU process; stop own worker'


def progress(stage, **kw):
    guard()
    value = dict(stage=stage, elapsed=time.monotonic()-BEGIN, **kw)
    save('progress.json', value)
    print(json.dumps(value), flush=True)


def features(index):
    if index in LRU:
        LRU.move_to_end(index)
        return LRU[index]
    prior = V9/'features'/('%06d.pt'%index)
    if not prior.exists():
        prior=Path('/data2/zhangcy/comatcher-v8-20261016-v3/features')/('%06d.pt'%index)
    own = STORE/'features'/('%06d.pt'%index)
    if prior.exists() or own.exists():
        value = torch.load(prior if prior.exists() else own, map_location='cpu')
    else:
        row = LIB[index]
        assert sha(row['image_path']) == row['image_sha256']
        with Image.open(row['image_path']) as im:
            im = im.convert('RGB').transpose(Image.Transpose.ROTATE_270)
            w, h = im.size
            scale = min(1., 560/max(w, h))
            rw, rh = round(w*scale), round(h*scale)
            im = im.resize((rw, rh), Image.Resampling.BICUBIC)
            cw, ch = rw//14*14, rh//14*14
            left, top = (rw-cw)//2, (rh-ch)//2
            array = np.asarray(im.crop((left, top, left+cw, top+ch))).copy()
        x = torch.from_numpy(array).permute(2, 0, 1).float()/255
        with torch.no_grad():
            value = {k:v.cpu() for k,v in EXTRACTOR.extract(x, resize=None).items()}
        own.parent.mkdir(parents=True, exist_ok=True)
        torch.save(value, own)
    prior_kp = OLD/'label-dataset-features'/('%06d.pt'%index)
    if prior_kp.exists():
        assert torch.equal(value['keypoints'][0].double(), torch.load(prior_kp).double())
    assert 0 < value['keypoints'].shape[1] <= 1024
    LRU[index] = value
    if len(LRU) > 16:
        LRU.popitem(last=False)
    return value


def anchors(index):
    if index in ANCHORS:
        ANCHORS.move_to_end(index)
        return ANCHORS[index]
    path = OLD/'label-dataset-geometry'/('%06d.npz'%index)
    assert path.exists(), path
    with np.load(path) as z:
        pixel, camera = z['model'].astype(float), z['camera'].astype(float)
    kp = features(index)['keypoints'][0].numpy()
    distance, nearest = cKDTree(pixel).query(kp)
    own_camera = camera[nearest]
    valid = distance <= 3.5
    pose = np.asarray(LIB[index]['T_WC'])
    world = own_camera @ pose[:3,:3].T + pose[:3,3]
    result = dict(pixel=kp, camera=own_camera, world=world, valid=valid, distance=distance)
    ANCHORS[index] = result
    if len(ANCHORS)>16:
        ANCHORS.popitem(last=False)
    return result


def project(world, index):
    pose = np.asarray(LIB[index]['T_WC'])
    xyz = (world-pose[:3,3]) @ pose[:3,:3]
    uvh = xyz @ KM.T
    return uvh[:,:2]/np.maximum(uvh[:,2:3], 1e-9), xyz[:,2]


def geometry(refs):
    count = len(refs)
    sizes=[len(anchors(r)['pixel']) for r in refs]
    n=max(sizes)
    projection = np.zeros((count,count,n,2),np.float32)
    known = np.zeros((count,count,n),bool)
    tracks = np.full((count,count,n),-1,np.int64)
    valid_points=np.arange(n)[None,:]<np.array(sizes)[:,None]
    for i, ri in enumerate(refs):
        a = anchors(ri)
        ni=sizes[i]
        for j, rj in enumerate(refs):
            b = anchors(rj)
            if i == j:
                projection[i,j,:ni] = (a['pixel']-np.array([210,280]))/280
                continue
            uv, depth = project(a['world'], rj)
            distance, ids = cKDTree(b['pixel']).query(uv)
            inside = (depth>.5)&(depth<120)&(uv[:,0]>=0)&(uv[:,0]<420)&(uv[:,1]>=0)&(uv[:,1]<560)
            valid = a['valid'] & inside & b['valid'][ids] & (distance<=3.5)
            valid &= np.abs(b['camera'][ids,2]-depth)<=.6
            projection[i,j,:ni] = (uv-np.array([210,280]))/280
            known[i,j,:ni] = valid
            tracks[i,j,:ni][valid] = ids[valid]
    return dict(projection=projection, known=known, tracks=tracks,valid=valid_points)


def labels(q, r):
    a, b = anchors(q), anchors(r)
    uv, depth = project(a['world'], r)
    distance, ids = cKDTree(b['pixel']).query(uv)
    valid = a['valid'] & b['valid'][ids] & (depth>.5)&(depth<120)&(distance<=3.5)
    valid &= np.abs(b['camera'][ids,2]-depth)<=.6
    return np.column_stack([np.flatnonzero(valid), ids[valid]]).tolist()


def manifest():
    old = json.loads((V9/'manifest.json').read_text())
    dates = PLAN['train_dates']
    positions = np.array([r['T_WC'] for r in LIB])[:,:3,3]
    date = [r['key'].split('/')[0] for r in LIB]
    directions = np.array([r['T_WC'] for r in LIB])[:,:3,2]
    def group(entry):
        q, anchor = entry['q'], entry['r']
        eligible = np.array([d in dates and d!=date[q] for d in date])
        eligible &= directions @ directions[anchor] > .5
        dist = np.linalg.norm(positions-positions[anchor],axis=1)
        candidates = np.flatnonzero(eligible)
        ordered = candidates[np.lexsort((candidates,dist[candidates]))]
        assert len(ordered)>=10, ('insufficient refs',q,len(ordered))
        chosen=ordered[:10]
        return dict(q=q, refs=chosen.tolist(), anchor=anchor,reference_anchor_distances_m=dist[chosen].tolist())
    rng = np.random.RandomState(20)
    bydate = {}
    for entry in old['training']:
        bydate.setdefault(date[entry['q']],{})[entry['q']] = entry
    for d in dates:
        entries=list(bydate[d].values())
        rng.shuffle(entries)
        bydate[d]=entries
    training = []
    for i in range(PLAN['training_groups']//3):
        for d in dates:
            entries=bydate[d]
            training.append(group(entries[i]))
    testold=old['test']
    test=[group(testold[i]) for i in np.linspace(0,len(testold)-1,PLAN['test_queries'],dtype=int)]
    assert set(g['q'] for g in training).isdisjoint(g['q'] for g in test)
    assert all(date[r] in dates for g in training+test for r in g['refs'])
    value = dict(training=training,test=test,selection='anchor from existing V8 geometry-labelled pair; remaining references nearest to anchor, heading dot>.5; no hard radius because some anchors have <10 nearby views, distances reported; no query pose used for expansion; controlled anchor-assisted group, not end-to-end retrieval',
                 prior_manifest_sha=sha(V9/'manifest.json'),library_sha=sha(OLD/'library.json'))
    save('manifest.json',value)
    return value


def tensors(q, refs, geo):
    query={k:v.cuda() for k,v in features(q).items()}
    references=[{k:v.cuda() for k,v in features(r).items()} for r in refs]
    return query,references,{k:torch.from_numpy(v).cuda() for k,v in geo.items()}


def make_model():
    base=LightGlue(features='superpoint',flash=False,depth_confidence=-1,width_confidence=-1)
    base.load_state_dict(torch.load(PLAN['v8_checkpoint'],map_location='cpu')['model'],strict=True)
    return CollaborativeV8(base).cuda()


def independent_scores(m, query, references):
    captured={}
    def capture(module, args, output):
        captured['scores']=output[0]
    handle=m.base.log_assignment[-1].register_forward_hook(capture)
    width=max(r['descriptors'].shape[1] for r in references)
    results=[]
    try:
        for reference in references:
            m.base(dict(image0=query,image1=reference))
            score=captured['scores']
            size=score.shape[-1]-1
            if size<width:
                score=torch.cat([score[:,:,:size],score.new_full((1,score.shape[1],width-size),-1e9),score[:,:,-1:]],-1)
            results.append(score)
    finally:
        handle.remove()
    return torch.cat(results)


def positive_loss(scores, gt):
    terms=[]
    for j, pairs in enumerate(gt):
        if pairs:
            idx=torch.tensor(pairs,device=scores.device)
            terms.append(-scores[j,idx[:,0],idx[:,1]].mean())
    assert terms, 'no supervised references'
    return torch.stack(terms).mean()


def train(m, data, mode):
    m.train()
    base_parameters=list(m.base.parameters())
    gate_parameters=[p for n,p in m.named_parameters() if 'gate' in n]
    extra_parameters=[p for n,p in m.named_parameters() if not n.startswith('base.') and 'gate' not in n]
    optimizer=torch.optim.Adam([dict(params=base_parameters,lr=PLAN['lr'],multiplier=1.),
                                dict(params=extra_parameters,lr=PLAN['lr']*10,multiplier=10.),
                                dict(params=gate_parameters,lr=PLAN['lr']*1000,multiplier=1000.)])
    records=[]
    for step,g in enumerate(data,1):
        count=3 if step%2 else 10
        refs=g['refs'][:count]
        geo=geometry(refs)
        x=tensors(g['q'],refs,geo)
        gt=g['labels'][:count]
        optimizer.zero_grad(set_to_none=True)
        scores,conf,intermediate=m(*x,enabled=mode=='collaborative')
        loss_pos=positive_loss(scores,gt)
        loss_conf=scores.new_zeros(())
        if mode=='collaborative':
            final=filter_matches(scores.detach(),.1)[1]
            for logits, current in zip(conf[:-1],intermediate[:-1]):
                perpoint=torch.nn.functional.binary_cross_entropy_with_logits(logits,(current==final).float(),reduction='none')
                loss_conf=loss_conf+(perpoint*x[2]['valid']).sum()/x[2]['valid'].sum()
            loss_conf=loss_conf/8
        loss=loss_pos+.5*loss_conf
        assert torch.isfinite(loss)
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
        assert torch.isfinite(norm)
        rate=PLAN['lr']*min(1.,step/5)*(.1+.9*(1+np.cos(np.pi*step/len(data)))/2)
        for item in optimizer.param_groups:
            item['lr']=rate*item['multiplier']
        optimizer.step()
        records.append(dict(step=step,query=g['q'],references=count,positive=float(loss_pos),confidence=float(loss_conf),gradient=float(norm)))
        del scores,loss,loss_pos,loss_conf,x
        progress('training',arm=mode,step=step,total=len(data),loss=records[-1])
    m.eval()
    path=STORE/(mode+'.pt')
    torch.save(dict(model=m.state_dict(),steps=len(data),mode=mode),path)
    save(mode+'-training.json',dict(complete=True,records=records,checkpoint_sha=sha(path)))
    return path


def assess(q,r,pairs,confidence):
    a,b=anchors(q),anchors(r)
    if len(pairs)==0:
        return dict(matches=0,evaluable=0,correct=0,correct_queries=[],predictions=[])
    qi,ri=pairs[:,0],pairs[:,1]
    uv,depth=project(b['world'][ri],q)
    error=np.linalg.norm(uv-a['pixel'][qi],axis=1)
    measured=a['camera'][qi,2]
    valid=b['valid'][ri]&a['valid'][qi]&(depth>.5)&(depth<120)
    correct=valid&(error<=5)&(np.abs(measured-depth)<=.6)
    predictions=[dict(qi=int(i),ri=int(j),confidence=float(c),evaluable=bool(v),correct=bool(t),error_px=float(e),depth_error_m=float(abs(z-d)))
                 for i,j,c,v,t,e,z,d in zip(qi,ri,confidence,valid,correct,error,measured,depth)]
    return dict(matches=len(pairs),evaluable=int(valid.sum()),correct=int(correct.sum()),correct_queries=qi[correct].tolist(),predictions=predictions)


@torch.no_grad()
def evaluate(m, groups, arm, count, enabled, ablate=None):
    m.eval()
    records=[]
    for index,g in enumerate(groups):
        refs=g['refs'][:count]
        x=tensors(g['q'],refs,geometry(refs))
        torch.cuda.synchronize()
        begun=time.monotonic()
        if enabled:
            scores,_,_=m(*x,enabled=True,ablate=ablate)
        else:
            scores=independent_scores(m,x[0],x[1])
        torch.cuda.synchronize()
        duration=time.monotonic()-begun
        m0,_,confidence,_=filter_matches(scores,.1)
        pair_records=[]
        for i,r in enumerate(refs):
            qi=torch.where(m0[i]>=0)[0]
            pairs=torch.stack([qi,m0[i,qi]],-1).cpu().numpy()
            conf=confidence[i,qi].cpu().numpy()
            pair_records.append(dict(reference=r,**assess(g['q'],r,pairs,conf)))
        matches=sum(p['matches'] for p in pair_records)
        evaluable=sum(p['evaluable'] for p in pair_records)
        correct=sum(p['correct'] for p in pair_records)
        unique=sorted(set(v for p in pair_records for v in p['correct_queries']))
        kpts=features(g['q'])['keypoints'][0].numpy()
        grid=set(tuple(v) for v in np.floor(kpts[unique]/np.array([420/8,560/8])).astype(int))
        records.append(dict(q=g['q'],refs=refs,matches=matches,evaluable=evaluable,correct=correct,
                            precision=correct/max(1,evaluable),confirmed_fraction_all=correct/max(1,matches),
                            unique_correct=len(unique),covered_cells=len(grid),matching_seconds=duration,pairs=pair_records))
        del scores,x
        if (index+1)%4==0:
            progress('evaluation',arm=arm,references=count,queries=index+1,total=len(groups))
    total={k:sum(x[k] for x in records) for k in ['matches','evaluable','correct','unique_correct']}
    summary=dict(**total,precision=total['correct']/max(1,total['evaluable']),
                 confirmed_fraction_all=total['correct']/max(1,total['matches']),
                 mean_correct=total['correct']/len(records),mean_unique_correct=total['unique_correct']/len(records),
                 macro_precision=float(np.mean([x['precision'] for x in records])),
                 mean_covered_cells=float(np.mean([x['covered_cells'] for x in records])),
                 mean_matching_seconds=float(np.mean([x['matching_seconds'] for x in records])))
    result=dict(arm=arm,references=count,queries=len(records),summary=summary,records=records)
    save('%s-%d.json'%(arm,count),result)
    return result


def gates(m, group):
    x=tensors(group['q'],group['refs'][:3],geometry(group['refs'][:3]))
    m.eval()
    with torch.no_grad():
        independent,_,_=m(*x,enabled=False)
        direct=independent_scores(m,x[0],x[1])
        gap=float((independent.exp()-direct.exp()).abs().max())
        assert gap<5e-5
        assert torch.equal(filter_matches(independent,.1)[0],filter_matches(direct,.1)[0])
        full,_,_=m(*x)
        effects={}
        for component in ['source','target','propagation']:
            ablated,_,_=m(*x,ablate=component)
            effects[component]=float((full-ablated).abs().max())
        assert all(v>1e-6 for v in effects.values()),effects
    m.train()
    scores,_,_=m(*x)
    loss=positive_loss(scores,group['labels'][:3]);loss.backward()
    grads={n:float(p.grad.abs().sum()) for n,p in m.named_parameters() if p.grad is not None and not n.startswith('base.')}
    assert all(np.isfinite(v) for v in grads.values())
    assert sum(v for n,v in grads.items() if '.source.' in n)>0
    assert sum(v for n,v in grads.items() if '.target.' in n)>0
    assert not any('gate' in n for n,p in m.named_parameters())
    assert all(float(v)==1. for n,v in m.named_buffers() if 'gate' in n)
    m.zero_grad(set_to_none=True)
    save('mechanism-gates.json',dict(complete=True,probability_gap=gap,component_ablation_effect=effects,new_parameter_gradients=grads,fixed_gates=1,tracks=int((x[2]['tracks']>=0).sum())))


def main():
    global EXTRACTOR
    assert sha(PLAN['v8_checkpoint'])==PLAN['v8_sha']
    EXTRACTOR=SuperPoint(max_num_keypoints=1024).eval()
    for p in EXTRACTOR.parameters():
        p.requires_grad_(False)
    prior=Path(PLAN['prior_store'])
    assert json.loads((prior/'completed.json').read_text())['complete']
    prior_audit=json.loads((prior/'final-audit.json').read_text())
    reused_names=['manifest.json','feature-bindings.json','positive_control-training.json',
                  'evaluation-inputs.npz','evaluation-input-bindings.json','v8-3.json','v8-10.json',
                  'positive_control-3.json','positive_control-10.json','collaborative-3.json','collaborative-10.json']
    reused={}
    for name in reused_names:
        digest=sha(prior/name)
        if name in prior_audit['reports']:
            assert digest==prior_audit['reports'][name],name
        reused[name]=digest
    assert reused['evaluation-inputs.npz']==json.loads((prior/'evaluation-input-bindings.json').read_text())['inputs_sha']
    save('reused-bindings.json',dict(prior_store=str(prior),files=reused,prior_audit_sha=sha(prior/'final-audit.json')))
    data=json.loads((prior/'manifest.json').read_text())
    save('manifest.json',data)
    save('feature-bindings.json',json.loads((prior/'feature-bindings.json').read_text()))
    del EXTRACTOR
    torch.cuda.set_per_process_memory_fraction(.32)
    m=make_model()
    gates(m,data['training'][0])
    progress('mechanisms_verified',peak_MiB=torch.cuda.max_memory_reserved()/2**20)
    del m
    torch.cuda.empty_cache()
    checkpoints={}
    for mode in ['collaborative']:
        torch.manual_seed(20)
        m=make_model()
        checkpoints[mode]=train(m,data['training'],mode)
        del m
        torch.cuda.empty_cache()
    save('checkpoint-freeze.json',{k:dict(path=str(v),sha=sha(v)) for k,v in checkpoints.items()})
    results=[]
    for arm in ['v8','positive_control']:
        for count in [3,10]:
            reused=json.loads((prior/('%s-%d.json'%(arm,count))).read_text())
            save('%s-%d.json'%(arm,count),reused)
            results.append(reused)
    for arm in ['collaborative']:
        m=make_model()
        if arm!='v8':
            m.load_state_dict(torch.load(checkpoints[arm],map_location='cpu')['model'],strict=True)
        for count in [3,10]:
            results.append(evaluate(m,data['test'],arm,count,arm=='collaborative'))
        for component in ['source','target','propagation']:
            evaluate(m,data['test'],'ablate_'+component,3,True,ablate=component)
        del m
        torch.cuda.empty_cache()
    report=dict(complete=True,results=[{k:v for k,v in x.items() if k!='records'} for x in results],
                elapsed=time.monotonic()-BEGIN,checkpoint_freeze=json.loads((STORE/'checkpoint-freeze.json').read_text()),
                evaluation='reference keypoint bound to own LiDAR <=3.5px, query keypoint bound <=3.5px; known GT transforms offline scoring; correct iff reference 3D reprojects <=5px from query keypoint and query depth agrees <=0.6m; precision denominator only evaluable predictions; unbound predictions unknown, report confirmed/all separately; geometrical correctness proxy not human identity truth',
                grouping=data['selection'],peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20)
    save('report.json',report)
    assert sha(PLAN['v8_checkpoint'])==PLAN['v8_sha']
    save('completed.json',dict(complete=True,report_sha=sha(STORE/'report.json'),elapsed=time.monotonic()-BEGIN))
    progress('completed',report_sha=sha(STORE/'report.json'))


try:
    main()
except BaseException:
    save('failure.json',dict(traceback=traceback.format_exc(),elapsed=time.monotonic()-BEGIN))
    raise
