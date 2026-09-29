import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .io import load_arrays
from .mapping import read_records,camera_rays,sha256,TRAIN_DATES
from .training_runtime import atomic_json


def prepare(base,output):
    base,output=Path(base),Path(output)
    settings=json.loads((base/'settings.json').read_text())
    prepared=json.loads((base/'prepared.json').read_text())
    records=read_records(settings['data_root'],[Path(settings['data_root'])/d/'manifest.json' for d in TRAIN_DATES],mapping=True)
    labels=json.loads((base/'training_labels.json').read_text())
    features={}
    cv2.setNumThreads(1)
    def feature(i):
        if i not in features:
            path=base/'features'/(records[i]['image_sha']+'.npz')
            meta=json.loads(path.with_suffix('.json').read_text())
            if sha256(path)!=meta['sha256']:
                raise ValueError('reference feature cache changed')
            with np.load(path,allow_pickle=False) as data:
                features[i]=(data['pixels'].copy(),data['descriptors'].astype(np.float32))
        return features[i]
    by_key={r['key']:i for i,r in enumerate(records)}
    centers=np.stack([r['T_WC'][:3,3] for r in records])
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    result={}
    for name,parity in (('odd',0),('even',1)):
        arrays,meta=load_arrays(prepared['maps'][name])
        mean=arrays['mean'].numpy()
        lookup={}
        references=[by_key[key] for key in meta['reference_frames']]
        for track,observations in enumerate(meta['observations']):
            for observation in observations:
                index=by_key[observation['frame']]
                pixels,_=feature(index)
                distance=np.linalg.norm(pixels-np.asarray(observation['pixel']),axis=1)
                feature_id=int(distance.argmin())
                if distance[feature_id]>1e-6:
                    raise ValueError('reference observation absent from frozen features')
                lookup[(index,feature_id)]=track
        for index in range(parity,len(records),2):
            pixels,query=feature(index)
            count=min(64,len(query))
            votes=[{} for _ in range(count)]
            distances=np.linalg.norm(centers[references]-centers[index],axis=1)
            nearest=[references[k] for k in np.argsort(distances) if .25<distances[k]<15.][:8]
            for reference in nearest:
                _,desc=feature(reference)
                if min(len(query),len(desc))<2:
                    continue
                reverse={pair[0].queryIdx:pair[0].trainIdx for pair in matcher.knnMatch(desc,query,k=2)
                         if len(pair)==2 and pair[0].distance<.8*pair[1].distance}
                for pair in matcher.knnMatch(query,desc,k=2):
                    if len(pair)!=2 or pair[0].distance>=.8*pair[1].distance:
                        continue
                    match=pair[0]
                    if match.queryIdx>=count or reverse.get(match.trainIdx)!=match.queryIdx:
                        continue
                    track=lookup.get((reference,match.trainIdx))
                    if track is not None:
                        votes[match.queryIdx].setdefault(track,set()).add(reference)
            truth=np.asarray(labels[records[index]['key']]['pose'])
            directions=camera_rays(pixels[:count]) @ records[index]['T_BC'][:3,:3].T
            origin=records[index]['T_BC'][:3,3]
            identities=[-1]*count
            for row,choices in enumerate(votes):
                confirmed=[]
                for track,support in choices.items():
                    if len(support)<2:
                        continue
                    delta=(mean[track]-truth[:3,3]) @ truth[:3,:3]-origin
                    cosine=np.dot(delta,directions[row])/max(np.linalg.norm(delta),1e-12)
                    if cosine>np.cos(.015):
                        confirmed.append(track)
                if len(confirmed)==1:
                    identities[row]=confirmed[0]
            result[records[index]['key']]=dict(map=name,identities=identities)
            if len(result)%100==0:
                print(json.dumps(dict(phase='identity-labels',frames=len(result),total=1200)),flush=True)
    if len(result)!=1200:
        raise ValueError('incomplete identity labels')
    atomic_json(output,result)
    summary=dict(frames=len(result),confirmed=sum(sum(i>=0 for i in row['identities']) for row in result.values()),
        labelled_frames=sum(any(i>=0 for i in row['identities']) for row in result.values()),
        sha256=sha256(output),maps={k:sha256(v) for k,v in prepared['maps'].items()},
        rule='train-only; opposite split; mutual RootSIFT ratio .8 in >=2 of nearest 8 reference views within 15m; unique track; GT bearing <=.015rad',
        supervision='conservative inferred multi-view identities, not manually labelled correspondences',time=time.time())
    atomic_json(output.with_suffix('.summary.json'),summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    prepare(args.base,args.output)
