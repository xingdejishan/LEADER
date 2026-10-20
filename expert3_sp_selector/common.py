import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
STORE=Path('/data2/zhangcy/sp-selector-20261014-v1')
DATA=Path('/data2/zhangcy/leader-v2-full-trajectory-20261001')
VENDOR=Path('/home/zhangcy/projects/sp-lightglue-four-20261013-v1/vendor')
VENDOR_PY=Path('/home/zhangcy/projects/sp-lightglue-four-20261013-v1/vendor-py')
HUB=Path('/home/zhangcy/projects/sp-lightglue-four-20261013-v1/torch_hub')
HUB_CHECKPOINTS=HUB/'checkpoints'
PLAN=json.loads((ROOT/'plan.json').read_text())
K=np.array([[199.716592,0,412.930976],[0,199.716592,310.584312],[0,0,1.]],dtype=np.float64)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,allow_nan=False))
    os.replace(temporary,path)


def decode(path):
    dtype=np.dtype([('x','<u2'),('y','<u2'),('z','<u2'),('i','u1'),('l','u1')])
    scan=np.fromfile(path,dtype=dtype)
    return np.column_stack([scan[n] for n in ['x','y','z']]).astype(np.float64)*.005-100


def transform(points,matrix):
    return np.asarray(points)@matrix[:3,:3].T+matrix[:3,3]


def voxel_downsample(points,size):
    """One representative point per voxel cell.

    Uses a packed 1-D key instead of np.unique(axis=0): same result set, but
    fast enough to voxelise a 25 m multi-scan map on every pose.
    """
    points=np.asarray(points,dtype=np.float64)
    if len(points)==0:
        return points.reshape(0,3)
    keys=np.floor(points/size).astype(np.int64)+(1<<19)
    assert keys.min()>=0 and keys.max()<(1<<20),(keys.min(),keys.max())
    packed=(keys[:,0]<<40)|(keys[:,1]<<20)|keys[:,2]
    _,index=np.unique(packed,return_index=True)
    return points[np.sort(index)]


def model_matrix(raw_size=(808,616)):
    rw0,rh0=raw_size
    w,h=rh0,rw0
    scale=min(1.,560/max(w,h))
    rw,rh=round(w*scale),round(h*scale)
    cw,ch=rw//14*14,rh//14*14
    left,top=(rw-cw)//2,(rh-ch)//2
    sx,sy=rw/w,rh/h
    return np.array([[0,-sx,sx*(rh0-1)+.5*sx-.5-left],[sy,0,.5*sy-.5-top],[0,0,1.]],dtype=np.float64)


def model_to_raw(matrix,points):
    points=np.asarray(points,dtype=np.float64)
    return (np.column_stack((points,np.ones(len(points))))@np.linalg.inv(matrix).T)[:,:2]


def patch_of(model_uv):
    cell=np.floor((np.asarray(model_uv,dtype=np.float64)+.5)/14).astype(np.int64)
    return cell,cell[:,1]*30+cell[:,0]


def rotation_in_place(pose,angle):
    c,s=np.cos(angle),np.sin(angle)
    out=np.array(pose,dtype=np.float64,copy=True)
    rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    out[:3,:3]=rotation@np.asarray(pose,dtype=np.float64)[:3,:3]
    return out


def offground_mask(query,index,params):
    z=query[:,2]
    bins=params['azimuth_bins']
    percentile=params['ground_percentile']
    band=params['ground_band_m']
    level=np.full(bins,-np.inf)
    for b in range(bins):
        mask=index==b
        if int(mask.sum())>=20:
            level[b]=np.percentile(z[mask],percentile)
    return z>level[index]+band


def solve_pnp(world, pixels, query_ids, body_from_camera, seed=20):
    """Verbatim copy of the frozen four-frame run's geometry.solve_pnp."""
    import cv2
    world=np.asarray(world,dtype=np.float64);pixels=np.asarray(pixels,dtype=np.float64)
    ids=np.asarray(query_ids,dtype=np.int32)
    if len(world)<8:
        return None
    cv2.setRNGSeed(seed)
    ok,rv,tv,inliers=cv2.solvePnPRansac(world,pixels,K,None,iterationsCount=2000,reprojectionError=12.,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers)<8:
        return None
    ix=inliers.ravel()
    rv,tv=cv2.solvePnPRefineLM(world[ix],pixels[ix],K,None,rv,tv)
    camera_from_world=np.eye(4);camera_from_world[:3,:3]=cv2.Rodrigues(rv)[0];camera_from_world[:3,3]=tv.ravel()
    camera=transform(world,camera_from_world)
    projected=camera@K.T
    residual=np.linalg.norm(projected[:,:2]/np.maximum(projected[:,2:3],1e-12)-pixels,axis=1)
    ix=np.flatnonzero((residual<=12)&(camera[:,2]>.5))
    if len(ix)<8 or not np.isfinite(camera_from_world).all():
        return None
    unique=np.unique(ids[ix])
    cells=np.floor(pixels[ix]/[808/4,616/4]).astype(int).clip(0,3)
    coverage=len(np.unique(cells[:,1]*4+cells[:,0]))
    pose=np.linalg.inv(camera_from_world)@np.linalg.inv(body_from_camera)
    return dict(pose=pose,inlier_indices=ix,unique_query_inliers=len(unique),inlier_ratio=len(unique)/max(1,len(np.unique(ids))),median_reprojection_px=float(np.median(residual[ix])),coverage_cells=coverage,qualified=len(unique)>=15 and coverage>=6 and len(unique)/max(1,len(np.unique(ids)))>=.2)


def guard(max_rss_gib=14.0,min_available_gib=24.0):
    import resource
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<max_rss_gib*2**30
    assert int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024>min_available_gib*2**30


def deny_GT():
    denied=[]
    def audit(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            p=os.path.abspath(os.fsdecode(args[0]))
            if 'evaluator-only-truth' in p or p.endswith('scored.npz') or p.endswith('errors.npz') or 'diagnostic-reference' in p:
                denied.append(p)
                raise PermissionError('GT prohibited in this stage')
    sys.addaudithook(audit)
    return denied
