import json
import os
from pathlib import Path
import time

from common import ROOT, STORE, PLAN, atomic, guard, sha, gpu_setup
device=gpu_setup()
import numpy as np
import requests
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from tools.sam_cache_frozen import build_sam_cache

six=Path(PLAN['six_root']);data=Path(PLAN['data_root'])
cal=six/'calibration';meta=json.loads((cal/'manifest.json').read_text())
assert sha(cal/'manifest.json')==PLAN['calibration_manifest_sha256']
def transform(camera):
    p=cal/('x_lb3_c%d.csv'%camera);assert sha(p)==meta['files'][p.name]
    v=np.loadtxt(p,delimiter=',');t=np.eye(4)
    t[:3,:3]=Rotation.from_euler('xyz',v[3:],degrees=True).as_matrix();t[:3,3]=v[:3]
    return t

datasets={}
for split,dates,indices in [('train',PLAN['train_dates'],PLAN['train_indices']),('dev',PLAN['dev_dates'],PLAN['dev_indices'])]:
    items=[]
    for date in dates:
        manifest=data/date/'manifest.json';assert sha(manifest)==PLAN['manifest_sha256'][date]
        d=json.loads(manifest.read_text());images=json.loads((STORE/'images'/date/'manifest.json').read_text())
        tbc5=np.array(d['roi']['T_BC']);tlc5=transform(5)
        for i in indices:
            frame=d['frames'][i];ts=frame['timestamp_us'];key=f'{date}/{ts}'
            cams=[]
            for camera in PLAN['cameras']:
                p=cal/('K_cam%d.csv'%camera);assert sha(p)==meta['files'][p.name]
                k=np.loadtxt(p,delimiter=',');k[:2]*=.5
                tbc=tbc5@np.linalg.inv(tlc5)@transform(camera)
                if camera==5:
                    assert np.max(np.abs(tbc-tbc5))<1e-12
                    assert np.max(np.abs(k-np.array([[199.716592,0,413.180976],[0,199.716592,310.834312],[0,0,1]])))<1e-8
                    path=data/date/frame['image']['path'];digest=frame['image']['sha256']
                else:
                    r=images['files'][f'Cam{camera}/{ts}'];path=Path(r['absolute_path']);digest=r['sha256']
                assert sha(path)==digest
                assert Image.open(path).size==(808,616)
                sam_key=f'{date}/Cam{camera}/{ts}'
                cache=None;record=None
                if camera!=5:
                    p=six/'sam'/date/f'Cam{camera}'/'cache_manifest.json'
                    if p.exists():
                        assert sha(p)==PLAN['sam_existing_shards'][str(p)]
                        record=json.loads(p.read_text())['frames'].get(sam_key)
                        if record is not None:cache=p.parent/record['cache_file']
                else:
                    p=Path('/dev/shm/zhangcy-leader-deepfusion-20260927/dev-sam/cache_manifest.json') if split=='dev' else Path('/home/zhangcy/projects/magic-paper-repro-20260927/training-sam-cache-v1/cache_manifest.json')
                    if not p.exists():
                        options=list(Path('/home/zhangcy/projects/nclt-relation-compare-20260927-a/sam').glob('**/cache_manifest.json'))
                        p=next(q for q in options if key in json.loads(q.read_text()).get('frames',{}))
                    record=json.loads(p.read_text())['frames'][key];cache=p.parent/record['cache_file']
                if cache is None:
                    p=Path(PLAN['prior_small_store'])/'sam/cache_manifest.json'
                    if p.exists():
                        assert sha(p)==PLAN['prior_small_sam_manifest_sha256']
                        record=json.loads(p.read_text())['frames'].get(sam_key)
                        if record is not None:cache=p.parent/record['cache_file']
                cams.append({'camera':camera,'image':str(path),'image_sha256':digest,'sam_key':sam_key,'cache':None if cache is None else str(cache),'record':record,'K':k.tolist(),'T_CB':np.linalg.inv(tbc).tolist()})
            items.append({'key':key,'date':date,'timestamp':ts,'dataset_index':dates.index(date)*300+i,'cameras':cams})
    datasets[split]=items
pending=[c for frames in datasets.values() for f in frames for c in f['cameras'] if c['cache'] is None]
atomic(STORE/'selected-inputs-before-encoding.json',datasets)
if pending:
    frames=[{'frame_key':c['sam_key'],'timestamp':int(c['sam_key'].split('/')[-1]),'image_path':c['image'],'image_sha256':c['image_sha256']} for c in pending]
    manifest=STORE/'pending-sam.json';atomic(manifest,{'frames':frames})
    image_root=Path(os.path.commonpath([str(Path(c['image']).resolve().parent) for c in pending]))
    cache=build_sam_cache(manifest,image_root,Path('/home/zhangcy/projects/nclt-relation-compare-20260927-a/sam_vit_l_0b3195.pth'),STORE/'sam',device=str(device),batch_size=1,minimum_free_bytes=40*1024**3)
    records=json.loads(Path(cache).read_text())['frames']
    for c in pending:
        c['record']=records[c['sam_key']];c['cache']=str(Path(cache).parent/c['record']['cache_file'])
for frames in datasets.values():
    for f in frames:
        for c in f['cameras']:
            r=c['record'];assert r['image_sha256']==c['image_sha256'] and r['checkpoint_sha256']=='3adcc4315b642a4d2101128f611684e8734c41232a17c648ed1693702a49a622'
            arr=np.load(c['cache'],allow_pickle=False);assert arr.shape==(64,64,256) and arr.dtype==np.float16
            assert np.isfinite(arr).all() and __import__('hashlib').sha256(arr.tobytes()).hexdigest()==r['embedding_sha256']
            c['cache_file_sha256']=sha(c['cache'])
        guard()
weights=STORE/'resnet18-f37072fd.pth'
prior_weights=Path(PLAN['prior_small_store'])/weights.name
if not weights.exists() and prior_weights.exists():
    assert sha(prior_weights)=='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
    os.link(prior_weights,weights)
if not weights.exists():
    session=requests.Session();session.trust_env=False
    with session.get('https://download.pytorch.org/models/resnet18-f37072fd.pth',stream=True,timeout=(15,30)) as response:
        response.raise_for_status()
        with weights.with_suffix('.part').open('xb') as stream:
            size=0
            for block in response.iter_content(1024**2):
                size+=len(block);assert size<50*1024**2;stream.write(block);guard()
    weights.with_suffix('.part').rename(weights)
assert sha(weights).startswith('f37072fd')
atomic(STORE/'inputs.json',datasets)
atomic(STORE/'assets-complete.json',{'new_sam_embeddings':len(pending),'all_camera_inputs':sum(len(f['cameras']) for frames in datasets.values() for f in frames),'input_sha256':sha(STORE/'inputs.json'),'resnet_sha256':sha(weights)})
print(json.dumps({'event':'assets_complete','new_sam_embeddings':len(pending)}),flush=True)
