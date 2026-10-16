"""Step 5: fine-tune LightGlue on the LiDAR-reprojection supervision.

SuperPoint is frozen: LightGlue consumes keypoints and descriptors, so SuperPoint
never enters the graph and is used exactly as the frozen pipeline uses it.

Supervision. Only keypoints that carry a label are trained, as a cross entropy
over reference keypoints (plus LightGlue's dustbin column) of the log-assignment
row for that query keypoint. Unlabelled keypoints are IGNORED, not treated as
"no match": 94 percent of keypoints are unlabelled and most of them do have a
true correspondence that this construction simply could not resolve, so calling
them negatives would teach the model to suppress correct matches.

The log-assignment matrix is not returned by the vendored LightGlue, so
filter_matches is wrapped to capture it. Point pruning is disabled so assignment
indices stay aligned with the input keypoint order.
"""
from common import *
import argparse
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as functional
import sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
sys.path.insert(0,str(VENDOR))
sys.path.insert(0,str(VENDOR_PY))
torch.hub.set_dir(str(HUB))
import lightglue.lightglue as lgmodule
from lightglue import LightGlue,SuperPoint

parser=argparse.ArgumentParser()
parser.add_argument('--steps',type=int,default=0)
parser.add_argument('--limit',type=int,default=0)
parser.add_argument('--frames',type=int,default=0)
argument=parser.parse_args()

denied=deny_GT()
T=PLAN['lightglue_finetune']
OUT=STORE/T['output_dir']
OUT.mkdir(parents=True,exist_ok=True)
DATASET=STORE/PLAN['lightglue_finetune'].get('dataset_dir',PLAN['label_dataset']['output_dir'])
library=load(STORE/'library.json')
references=library['references']

torch.manual_seed(int(T['seed']))
device='cuda' if torch.cuda.is_available() else 'cpu'

captured={}
original=lgmodule.filter_matches


def capturing(scores,threshold):
    captured['scores']=scores
    return original(scores,threshold)


lgmodule.filter_matches=capturing

extractor=SuperPoint(max_num_keypoints=int(T['max_num_keypoints'])).eval()
for parameter in extractor.parameters():
    parameter.requires_grad_(False)
matcher=LightGlue(features='superpoint',flash=False,
                  depth_confidence=-1,width_confidence=-1).to(device)
matcher.train()
for parameter in matcher.parameters():
    parameter.requires_grad_(True)
optimiser=torch.optim.Adam([p for p in matcher.parameters() if p.requires_grad],lr=float(T['learning_rate']))
warmup=max(1,int(T['warmup_steps']))
total_steps=argument.steps or int(T['steps'])


def set_rate(step):
    if step<warmup:
        scale=(step+1)/warmup
    elif T.get('cosine_decay'):
        import math
        progress=(step-warmup)/max(1,total_steps-warmup)
        scale=0.5*(1.0+math.cos(math.pi*min(1.0,progress)))
    else:
        scale=1.0
    for group in optimiser.param_groups:
        group['lr']=float(T['learning_rate'])*scale


def model_image(path):
    with Image.open(path) as image:
        image=image.convert('RGB')
        image=image.transpose(Image.Transpose.ROTATE_270)
        w,h=image.size
        scale=min(1.,float(T['image_edge'])/max(w,h))
        rw,rh=round(w*scale),round(h*scale)
        image=image.resize((rw,rh),Image.Resampling.BICUBIC)
        cw,ch=rw//14*14,rh//14*14
        left,top=(rw-cw)//2,(rh-ch)//2
        image=image.crop((left,top,left+cw,top+ch))
        return np.asarray(image).copy()


def features(image_path):
    """extracted on CPU with the same configuration the labels were built with,
    so keypoint indices and their order are identical, then moved to the device"""
    array=model_image(image_path)
    x=torch.from_numpy(array).permute(2,0,1).float().div(255)
    with torch.no_grad():
        value=extractor.extract(x,resize=None)
    return {k:v.to(device) for k,v in value.items()}


def build_index():
    """entries are (query image path, reference image path, label array) so the
    same trainer serves both the library dataset and the hard-frame dataset,
    whose frame indices live in different pools"""
    hard='hard' in DATASET.name
    query_images={}
    if hard:
        for q in load(STORE/'queries.json')['queries']:
            query_images[int(q['index'])]=q['image_path']
    entries=[]
    per_date={}
    cap=int(T.get('frames_per_date') or 0)
    for path in sorted(DATASET.glob('*.json')):
        record=json.loads(path.read_text())
        index=int(record['index'])
        date_of=record['key'].split('/')[0]
        if cap:
            used=per_date.get(date_of,0)
            if used>=cap:
                continue
        query_image=query_images[index] if hard else references[index]['image_path']
        query_date=date_of
        counted=False
        for pair in record.get('pairs') or []:
            labels=pair['labels']
            if len(labels)<int(T['min_labels_per_pair']):
                continue
            reference_row=references[int(pair['reference'])]
            reference_date=reference_row['key'].split('/')[0]
            entries.append((query_image,reference_row['image_path'],
                            np.asarray(labels,dtype=np.int64),query_date,reference_date))
            counted=True
        if cap and counted:
            per_date[date_of]=per_date.get(date_of,0)+1
    return entries


entries=build_index()
if argument.frames:
    entries=entries[:argument.frames]
import zlib


def held_out(image_path):
    return zlib.crc32(image_path.encode())%int(T['validation_modulus'])==int(T['validation_residue'])


# Protocol: the held-out dates are excluded from BOTH sides of every training
# pair, not just from the query side, otherwise a held-out image can still enter
# training as a reference and leak. If no dates are configured the split falls
# back to a per-image hash, which does NOT control for scene overlap.
train_dates=set(T.get('train_dates') or [])
validation_dates=set(T.get('validation_dates') or [])
if train_dates and validation_dates:
    validation=[e for e in entries if e[3] in validation_dates]
    training=[e for e in entries if e[3] in train_dates and e[4] in train_dates]
else:
    validation=[e for e in entries if held_out(e[0])]
    training=[e for e in entries if not held_out(e[0])]
print('training pairs %d  validation pairs %d'%(len(training),len(validation)),flush=True)


@torch.no_grad()
def agreement(subset):
    """precision and recall of the matcher's matches against the geometric
    labels on held-out pairs; this, not the training loss, is the viability
    signal"""
    matcher.eval()
    precisions=[];recalls=[]
    for query_index,reference_index,labels,_,_ in subset:
        output=matcher({'image0':features(query_index),'image1':features(reference_index)})
        matches=output['matches'][0].cpu().numpy()
        truth=set((int(a),int(b)) for a,b in labels)
        if len(matches)==0:
            precisions.append(0.0);recalls.append(0.0)
            continue
        hits=sum(1 for a,b in matches if (int(a),int(b)) in truth)
        precisions.append(hits/len(matches))
        recalls.append(hits/max(1,len(truth)))
    matcher.train()
    return float(np.mean(precisions)),float(np.mean(recalls))


validation_sample=validation[:int(T['validation_pairs'])]
if validation_sample:
    precision,recall=agreement(validation_sample)
    print(json.dumps(dict(stage='frozen',precision=round(precision,4),recall=round(recall,4),
                          pairs=len(validation_sample))),flush=True)

total_steps=argument.steps or int(T['steps'])
history=[]
begun=time.monotonic()
step=0
pending=0
best_score=-1.0
accumulate=max(1,int(T['accumulate']))
representative=0.0
optimiser.zero_grad(set_to_none=True)
while step<total_steps:
    for query_index,reference_index,labels,_,_ in training:
        if step>=total_steps:
            break
        guard()
        set_rate(step)
        output=matcher({'image0':features(query_index),'image1':features(reference_index)})
        scores=captured['scores']
        # scores is [1, M+1, N+1] and already a normalised log assignment from
        # LightGlue's sigmoid_log_double_softmax, so the loss is the negative mean
        # of the assignment at the ground-truth entries. Applying another softmax
        # here was what diverged: it re-normalises an already calibrated matrix.
        rows=torch.from_numpy(labels[:,0]).to(device)
        columns=torch.from_numpy(labels[:,1]).to(device)
        valid=(rows<scores.shape[1]-1)&(columns<scores.shape[2]-1)
        if int(valid.sum())==0:
            continue
        loss=-scores[0][rows[valid],columns[valid]].mean()
        representative=float(loss)
        (loss/accumulate).backward()
        pending+=1
        if pending<accumulate:
            continue
        torch.nn.utils.clip_grad_norm_([p for p in matcher.parameters() if p.requires_grad],float(T['gradient_clip']))
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)
        pending=0
        step+=1
        if step%int(T['report_every'])==0:
            entry=dict(step=step,loss=representative,matched=int(output['matches'][0].shape[0]),
                       labels=int(valid.sum()),rate=optimiser.param_groups[0]['lr'],seen=step*accumulate)
            if validation_sample and step%int(T['validate_every'])==0:
                entry['validation_precision'],entry['validation_recall']=agreement(validation_sample)
                score=entry['validation_precision']+entry['validation_recall']
                if score>best_score:
                    best_score=score
                    torch.save(dict(model=matcher.state_dict(),step=step,
                                    precision=entry['validation_precision'],
                                    recall=entry['validation_recall'],
                                    config=dict(depth_confidence=-1,width_confidence=-1,features='superpoint',flash=False)),
                               OUT/'best-lightglue.pt')
                    entry['saved_best']=True
            history.append(entry)
            print(json.dumps({k:(round(v,5) if isinstance(v,float) else v) for k,v in entry.items()}),flush=True)
            save(OUT/'progress.json',dict(step=step,total=total_steps,history=history,complete=False))
    if not training:
        break

torch.save(dict(model=matcher.state_dict(),
                config=dict(depth_confidence=-1,width_confidence=-1,features='superpoint',flash=False)),
           OUT/'finetuned-lightglue.pt')
save(OUT/'training.json',dict(complete=True,steps=step,pairs=len(entries),history=history,
                             device=device,seconds=time.monotonic()-begun,
                             superpoint_frozen=True,unlabelled_ignored=True,
                             min_labels_per_pair=int(T['min_labels_per_pair']),
                             learning_rate=float(T['learning_rate'])))
print(json.dumps(dict(complete=True,steps=step,seconds=round(time.monotonic()-begun,1))))
