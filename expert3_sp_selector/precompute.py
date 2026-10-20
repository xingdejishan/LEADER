"""Stage 2: build per-reference caches in parallel shards.

  map-cloud/<index>-<sha16>.npz        world points of one training reference scan
  ref-projection/<index>-<sha16>.npz   model-space pixel + world point of every
                                       valid return, for keypoint association

Both are pure functions of the frozen reference scans and poses.
"""
from common import *
import argparse
import time

parser=argparse.ArgumentParser()
parser.add_argument('--shard',type=int,default=0)
parser.add_argument('--of',type=int,default=1)
parser.add_argument('--limit',type=int,default=0)
argument=parser.parse_args()

denied=deny_GT()
(STORE/'map-cloud').mkdir(parents=True,exist_ok=True)
(STORE/'ref-projection').mkdir(parents=True,exist_ok=True)
library=load(STORE/'library.json')
references=library['references']
match_indices=set(load(STORE/'match-references.json')['indices'])
matrix=model_matrix()
started=time.monotonic()
done=0
for row in references:
    index=row['index']
    if index%argument.of!=argument.shard:
        continue
    if argument.limit and done>=argument.limit:
        break
    guard()
    scan=str(DATA/'NCLT'/row['key'].split('/')[0]/'velodyne_sync'/(row['key'].split('/')[1]+'.bin'))
    cloud_path=STORE/'map-cloud'/('%06d-%s.npz'%(index,row['scan_sha256'][:16]))
    projection_path=STORE/'ref-projection'/('%06d-%s.npz'%(index,row['scan_sha256'][:16]))
    if cloud_path.is_file() and (index not in match_indices or projection_path.is_file()):
        done+=1
        continue
    assert sha(scan)==row['scan_sha256'],(index,scan)
    body=decode(scan)
    body=body[np.isfinite(body).all(1)]
    world_from_body=np.asarray(row['T_WC'])@np.linalg.inv(np.asarray(library['T_BC']))
    world=transform(body,world_from_body)
    if not cloud_path.is_file():
        cloud=voxel_downsample(world,PLAN['verification']['map_voxel_m'])
        np.savez_compressed(cloud_path,world=cloud.astype(np.float32))
    if index in match_indices and not projection_path.is_file():
        camera=transform(body,np.linalg.inv(np.asarray(library['T_BC'])))
        good=np.isfinite(camera).all(1)&(camera[:,2]>.5)&(camera[:,2]<120)
        camera=camera[good]
        pixel=camera@K.T
        pixel=pixel[:,:2]/pixel[:,2:3]
        good=(pixel[:,0]>=0)&(pixel[:,0]<808)&(pixel[:,1]>=0)&(pixel[:,1]<616)
        camera,pixel=camera[good],pixel[good]
        model=(np.column_stack((pixel,np.ones(len(pixel))))@matrix.T)[:,:2]
        good=(model[:,0]>=0)&(model[:,0]<420)&(model[:,1]>=0)&(model[:,1]<560)
        camera,pixel,model=camera[good],pixel[good],model[good]
        # camera = body -> camera; world = T_WC @ camera. Applying the
        # body->camera transform twice here was a real bug caught by comparing
        # this table against the frozen four-frame run (world off by up to 80 m).
        np.savez_compressed(projection_path,model=model.astype(np.float32),pixel=pixel.astype(np.float32),world=transform(camera,np.asarray(row['T_WC'])).astype(np.float32))
    done+=1
    if done%50==0:
        print(json.dumps(dict(shard=argument.shard,processed=done,seconds=time.monotonic()-started)),flush=True)
assert not denied
print(json.dumps(dict(shard=argument.shard,complete=True,processed=done,seconds=time.monotonic()-started)))
