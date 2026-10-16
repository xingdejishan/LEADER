import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from lightglue import LightGlue
from lightglue.lightglue import filter_matches


ROOT=Path(__file__).resolve().parent


def load_model(checkpoint, device='cpu'):
    metadata=json.loads((ROOT/'checkpoint.json').read_text())
    path=Path(checkpoint)
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):
            digest.update(chunk)
    if digest.hexdigest()!=metadata['sha256']:
        raise ValueError('Checkpoint SHA-256 does not match this version')
    base=LightGlue(features=None,flash=False,depth_confidence=-1,width_confidence=-1)
    if metadata['kind']=='strong-interaction':
        from collaborative import CollaborativeV8
        model=CollaborativeV8(base)
    else:
        model=base
    payload=torch.load(path,map_location='cpu')
    model.load_state_dict(payload['model'],strict=True)
    return model.to(device).eval()


def load_features(path, device='cpu'):
    with np.load(path,allow_pickle=False) as source:
        values={key:np.array(source[key],dtype=np.float32) for key in ['keypoints','descriptors','image_size']}
    for key in ['keypoints','descriptors']:
        if values[key].ndim==2:
            values[key]=values[key][None]
    if values['image_size'].ndim==1:
        values['image_size']=values['image_size'][None]
    n=values['keypoints'].shape[1]
    if n<1 or values['keypoints'].shape!=(1,n,2) or values['descriptors'].shape!=(1,n,256):
        raise ValueError('Expected one image with [N,2] keypoints and [N,256] SuperPoint descriptors')
    if values['image_size'].shape!=(1,2) or not (values['image_size']>0).all():
        raise ValueError('image_size must contain positive width and height')
    if not all(np.isfinite(value).all() for value in values.values()):
        raise ValueError('Features must be finite')
    return {key:torch.from_numpy(value).to(device) for key,value in values.items()}


def load_geometry(path, references, device='cpu'):
    count=len(references)
    sizes=[r['keypoints'].shape[1] for r in references]
    width=max(sizes)
    with np.load(path,allow_pickle=False) as source:
        projection=np.array(source['projection'],dtype=np.float32)
        known=np.array(source['known'],dtype=bool)
        tracks=np.array(source['tracks'],dtype=np.int64)
        valid=np.array(source['valid'],dtype=bool)
    expected=np.arange(width)[None,:]<np.array(sizes)[:,None]
    if projection.shape!=(count,count,width,2) or known.shape!=(count,count,width) or tracks.shape!=known.shape:
        raise ValueError('Geometry dimensions do not match reference features')
    if valid.shape!=(count,width) or not np.array_equal(valid,expected):
        raise ValueError('valid must mark exactly the real, unpadded reference points')
    if not np.isfinite(projection).all() or (tracks< -1).any():
        raise ValueError('Invalid projection or track index')
    for i in range(count):
        for j in range(count):
            if (tracks[i,j]>=sizes[j]).any() or (tracks[i,j,~valid[i]]>=0).any():
                raise ValueError('Track index refers to padding or an unavailable point')
    return {key:torch.from_numpy(value).to(device) for key,value in
            dict(projection=projection,known=known,tracks=tracks,valid=valid).items()}


@torch.no_grad()
def match_features(model, query, references, geometry=None, threshold=.1):
    if not references or not 0<=threshold<=1:
        raise ValueError('Provide reference images and a threshold between 0 and 1')
    model.eval()
    if hasattr(model,'interactions'):
        if geometry is None:
            raise ValueError('The collaborative model requires reference-only geometry')
        matrices=model(query,references,geometry)[0]
    else:
        matrices=[]
        captured={}
        def capture(module,args,output):
            captured['score']=output[0]
        handle=model.log_assignment[-1].register_forward_hook(capture)
        try:
            for reference in references:
                model(dict(image0=query,image1=reference))
                matrices.append(captured['score'][0])
        finally:
            handle.remove()
    results=[]
    for i,matrix in enumerate(matrices):
        matches,_,confidence,_=filter_matches(matrix[None],threshold)
        query_index=torch.where(matches[0]>=0)[0]
        results.append(dict(query_index=query_index.cpu().numpy(),
                            reference_index=matches[0,query_index].cpu().numpy(),
                            confidence=confidence[0,query_index].cpu().numpy()))
    return results
