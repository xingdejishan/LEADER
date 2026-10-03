import hashlib,json,os,resource,shutil,sys,time,traceback
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
from v5_paths import BASE,DATA,CACHE as STORE,SOURCE,RUNTIME
sys.path[:0]=[str(SOURCE),str(RUNTIME)]

from data.leader_nclt_manifest import TRAIN_DATES,_safe_child
from data.robotcar_sdk.python.velodyne import get_velo
from tools.far_covolume_runtime import visible_pool
from tools.far_runtime import verify_sources
from geometry_cache import encode,decode

assert os.name=='posix'
os.nice(15)
torch.set_num_threads(1)
config=json.loads((BASE/'config-v2.json').read_text())
verify_sources(config)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
source_hashes={n:sha((RUNTIME if n=='geometry_cache.py' else ROOT)/n) for n in ('build_geometry_cache.py','geometry_cache.py')}

def save(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False))
    os.replace(tmp,path)

def budget():
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<4*2**30
    assert shutil.disk_usage(STORE).free>40*2**30
    available=int(next(s for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')).split()[1])*1024
    assert available>32*2**30

try:
    started=time.monotonic()
    references=torch.load(STORE/'frames32.pt',map_location='cpu') if (STORE/'frames32.pt').exists() else []
    reference_by_key={r['key']:r for r in references}
    total=0
    completed=[]
    for date in TRAIN_DATES:
        directory=STORE/'geometry'/date
        directory.mkdir(parents=True,exist_ok=True)
        assert directory.stat().st_uid==os.getuid()
        manifest_path=DATA/'NCLT'/date/'manifest.json'
        initial_path=DATA/'initial'/date/'initial_cache.json'
        manifest=json.loads(manifest_path.read_text())
        initial=json.loads(initial_path.read_text())
        manifest_sha,initial_sha=sha(manifest_path),sha(initial_path)
        assert manifest['split']=='train' and initial['ground_truth_used_for_prediction'] is False and not initial.get('failures')
        assert initial['backend']=='raw_SC2-PCR_no_full_pool_refine' and initial['manifest_sha256']=={date:manifest_sha}
        assert initial['base_checkpoint_sha256']==config['base_checkpoint_sha256']
        assert all(config['source_sha256'][n]==h for n,h in initial['producer_source_sha256'].items())
        camera=torch.from_numpy(np.linalg.inv(np.asarray(manifest['roi']['T_BC'],dtype=np.float64)).astype(np.float32))
        rows=manifest['frames']
        expected={date+'/'+str(row['timestamp_us']) for row in rows}
        assert expected==set(initial['frames'])
        path=directory/'partial.json'
        if (directory/'completed.json').exists():
            value=json.loads((directory/'completed.json').read_text())
            assert value['manifest_sha256']==manifest_sha and value['initial_manifest_sha256']==initial_sha and value['producer']==source_hashes
            assert set(value['frames'])==expected and sha(directory/'geometry.bin')==value['binary_sha256']
            total+=len(rows)
            completed.append(date)
            continue
        if path.exists():
            value=json.loads(path.read_text())
            assert value['manifest_sha256']==manifest_sha and value['initial_manifest_sha256']==initial_sha and value['producer']==source_hashes
            assert list(value['frames'])==[date+'/'+str(r['timestamp_us']) for r in rows[:len(value['frames'])]]
        else:
            value={'date':date,'complete':False,'manifest_sha256':manifest_sha,'initial_manifest_sha256':initial_sha,'producer':source_hashes,'parent_source_sha256':config['source_sha256'],'range_m':[1.,100.],'max_points':2048,'ground_truth_used':False,'frames':{},'bytes':0,'representative_exact_checks':0}
        binary=directory/'geometry.bin'
        mode='r+b' if binary.exists() else 'w+b'
        with binary.open(mode) as stream:
            assert binary.resolve().parent==directory.resolve() and binary.stat().st_uid==os.getuid()
            assert os.fstat(stream.fileno()).st_size>=value['bytes']
            stream.truncate(value['bytes'])
            stream.seek(value['bytes'])
            for index,row in enumerate(rows):
                key=date+'/'+str(row['timestamp_us'])
                if key in value['frames']:continue
                budget()
                assert row['camera_id']==5 and row['scan']['timestamp_us']==row['timestamp_us']
                scan_path=_safe_child(DATA/'NCLT',date,row['scan']['path'])
                assert sha(scan_path)==row['scan']['sha256']
                item=initial['frames'][key]
                assert item['scan_sha256']==row['scan']['sha256'] and item['image_sha256']==row['image']['sha256']
                p=initial_path.parent/item['file']
                assert sha(p)==item['sha256']
                with np.load(p,allow_pickle=False) as archive:source=torch.from_numpy(archive['source'].copy())
                raw,_=get_velo(str(scan_path))
                raw=np.asarray(raw,dtype=np.float32)
                ranges=np.linalg.norm(raw,axis=1)
                raw=np.ascontiguousarray(raw[(ranges>1.)&(ranges<100.)])
                points,pixels,visible,structure=visible_pool({'points':raw,'camera_from_lidar':camera},source,2048)
                tensors={'points':points,'pixels':pixels,'visible_ids':visible,'structure':structure}
                data,pcount,scount=encode(tensors)
                assert scount==len(source)
                if key in reference_by_key:
                    ref=reference_by_key[key]
                    decoded=decode(data,pcount,scount)
                    for name,v in decoded.items():assert torch.equal(v,ref[name]),(key,name)
                    value['representative_exact_checks']+=1
                record={'offset':value['bytes'],'bytes':len(data),'points':pcount,'source_points':scount,'sha256':hashlib.sha256(data).hexdigest(),'scan_sha256':row['scan']['sha256'],'initial_sha256':item['sha256']}
                stream.write(data)
                value['frames'][key]=record
                value['bytes']+=len(data)
                if (index+1)%256==0 or index+1==len(rows):
                    stream.flush()
                    os.fsync(stream.fileno())
                    save(path,value)
                    save(STORE/'geometry-status.json',{'date':date,'date_frames':len(value['frames']),'total_completed':total+len(value['frames']),'target':107339,'seconds':time.monotonic()-started,'pid':os.getpid(),'GPU_used':False})
            assert set(value['frames'])==expected
        value.update(complete=True,binary_sha256=sha(binary),completed_time=time.time())
        save(directory/'completed.json',value)
        total+=len(rows)
        completed.append(date)
    assert total==107339
    verify_sources(config)
    save(STORE/'geometry-completed.json',{'complete':True,'frames':total,'dates':completed,'seconds':time.monotonic()-started,'producer':source_hashes,'GPU_used':False,'ground_truth_used':False,'parent_weights_modified':False})
except BaseException as error:
    save(STORE/'geometry-failed.json',{'error':repr(error),'traceback':traceback.format_exc()})
    raise
