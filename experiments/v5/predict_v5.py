import argparse,hashlib,json,os,resource,shutil,sys,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parent
from v5_paths import BASE,DATA,STORE,SOURCE,RUNTIME,RAW_GT
sys.path[:0]=[str(SOURCE),str(RUNTIME)]
from data.leader_nclt_manifest import ManifestLEADERDataset,DEV_DATES
from tools.far_covolume_runtime import visible_pool
from tools.far_runtime import verify_sources
from tools.far_v3_runtime import sc2_initial
from tools.magic_nclt_runtime import load_pretrained_model,state_dict_sha256,PoseEstimationFailure
from models.far_covolume_standard import CoVolumeStandardFAR
from field_index_select import enable as reference_enable
from fused_field import enable as fused_enable
from batched_solver import enable as batch_enable

assert os.name=='posix'
os.nice(15)
torch.set_num_threads(1)
torch.cuda.set_per_process_memory_fraction(.65)
parser=argparse.ArgumentParser()
parser.add_argument('--epoch',type=int,required=True)
parser.add_argument('--smoke',action='store_true')
args=parser.parse_args()
assert args.epoch in range(5,51,5) or (args.smoke and args.epoch==0)
OUTPUT=STORE/'evaluation-smoke' if args.smoke else STORE/'evaluations'/('epoch{:02d}'.format(args.epoch))
expected_frames=8 if args.smoke else 104732
OUTPUT.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(path,value):
    p=Path(path)
    q=p.with_suffix('.tmp')
    q.write_text(json.dumps(value,allow_nan=False))
    os.replace(q,p)

def budget():
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<10*2**30
    assert shutil.disk_usage(STORE).free>40*2**30
    available=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
    assert available>32*2**30

try:
    config=json.loads((BASE/'config-v2.json').read_text())
    verify_sources(config)
    plan=json.loads((ROOT/'v5-training-plan.json').read_text())
    for name,h in plan['sources'].items():assert sha(ROOT/name)==h,name
    assert sha(DATA/'catalog.json')==plan['catalog_sha256']
    denied=[]
    forbidden=[os.path.abspath(config['gt_root']),str(DATA/'evaluator-only-truth'),str(RAW_GT)]
    def deny(event,values):
        if event=='open' and values and isinstance(values[0],(str,bytes,os.PathLike)):
            p=os.path.abspath(os.fsdecode(values[0]))
            if any(p==root or p.startswith(root+os.sep) for root in forbidden):
                denied.append(p)
                raise PermissionError('development GT prohibited online')
    sys.addaudithook(deny)
    datasets={}
    records=[]
    for date in DEV_DATES:
        assert sha(DATA/'NCLT'/date/'manifest.json')==plan['public_manifest_sha256'][date]
        assert sha(DATA/'sam'/date/'cache_manifest.json')==plan['sam_manifest_sha256'][date]
        ds=ManifestLEADERDataset([DATA/'NCLT'/date/'manifest.json'],DATA/'NCLT','dev',strict_frozen=False,sam_cache_manifest=DATA/'sam'/date/'cache_manifest.json',cache_memory_frames=0)
        assert all(r.T_WB is None for r in ds.records)
        datasets[date]=ds
        records.extend((date,i,r.frame_key) for i,r in enumerate(ds.records))
    assert len(records)==len({r[2] for r in records})==104732
    if args.smoke:records=records[:8]
    leader,saved_leader=load_pretrained_model(config['base_checkpoint'],False,torch.device('cuda:0'))
    leader.eval().requires_grad_(False)
    center=torch.as_tensor(saved_leader['center_t'],device='cuda',dtype=torch.float32).reshape(3)
    del saved_leader
    checkpoints={'V5':STORE/'smoke-model.pt' if args.smoke else STORE/'training/checkpoints'/('{:02d}.pt'.format(args.epoch))}
    assert sha(checkpoints['V5'])==json.loads(checkpoints['V5'].with_suffix('.json').read_text())['sha256']
    checkpoint_hashes={n:sha(p) for n,p in checkpoints.items()}
    models={}
    initial_states={}
    for name,path in checkpoints.items():
        saved=torch.load(path,map_location='cpu')
        model=CoVolumeStandardFAR(**config['far_covolume']).cuda().eval().requires_grad_(False)
        model.load_state_dict(saved['model'])
        assert saved['epoch']==args.epoch and saved['plan_sha256']==sha(ROOT/'v5-training-plan.json')
        fused_enable(model)
        batch_enable(model)
        model.point_chunk=4096
        models[name]=model
        initial_states[name]=state_dict_sha256(model.state_dict())
        del saved
    binding={'plan_sha256':sha(ROOT/'v5-training-plan.json'),'checkpoint_sha256':checkpoint_hashes,'model_state_sha256':initial_states,'frames':expected_frames,'sources':plan['sources'],'systems':['LEADER+V5'],'epoch':args.epoch,'smoke_only':args.smoke,'GT_read_online':False}
    (OUTPUT/'chunks').mkdir(exist_ok=True)
    if (OUTPUT/'prediction-binding.json').exists():assert json.loads((OUTPUT/'prediction-binding.json').read_text())==binding
    else:save(OUTPUT/'prediction-binding.json',binding)
    started=time.monotonic()
    failures=0
    reused=0
    with torch.no_grad():
        for start in range(0,len(records),8):
            budget()
            stop=min(start+8,len(records))
            keys=[r[2] for r in records[start:stop]]
            destination=OUTPUT/'chunks'/('{:06d}.json'.format(start))
            if destination.exists():
                chunk=json.loads(destination.read_text())
                assert chunk['binding']==binding and chunk['keys']==keys and chunk['forbidden_GT_attempts']==0
                assert set(chunk['predictions'])==set(models)
                for rows in chunk['predictions'].values():assert [r['frame_key'] for r in rows]==keys
                failures+=chunk['failures']
                reused+=stop-start
                continue
            frames=[]
            outputs={n:{} for n in models}
            for global_index in range(start,stop):
                date,index,key=records[global_index]
                ds=datasets[date]
                sample=ds[index]
                assert 'T_WB' not in sample
                try:
                    initial,source,target,total=sc2_initial(leader,sample,torch.device('cuda:0'),config['seed']+global_index)
                    points,pixels,visible,structure=visible_pool(sample,source,config['point_max'])
                    with Image.open(ds.records[index].image_path) as image:
                        rgb=torch.from_numpy(np.asarray(image.convert('RGB')).copy()).permute(2,0,1).float()/255
                    frame={'key':key,'date':date,'timestamp_us':int(sample['timestamp_us']),'initial':initial,'source':source,'target':target,'points':points.cuda(),'pixels':pixels.cuda(),'visible_ids':visible.cuda(),'structure':structure.cuda(),'rgb':rgb.cuda(),'sam':sample['sam_features'].to(device='cuda',dtype=torch.float32),'hw':sample['sam_metadata']['resized_size_hw'],'valid_corr':total,'selected_corr':len(source)}
                    frames.append(frame)
                except PoseEstimationFailure as error:
                    failures+=1
                    row={'frame_key':key,'date':date,'timestamp_us':int(sample['timestamp_us']),'T_WB_pred':torch.eye(4).double().tolist(),'status':'identity_fallback','failure_reason':str(error),'valid_corr':error.valid_corr,'selected_corr':error.selected_corr}
                    for rows in outputs.values():rows[key]=row
            timings={}
            if frames:
                values=lambda name:[f[name] for f in frames]
                for name in (list(models) if (start//8)%2==0 else list(models)[::-1]):
                    model=models[name]
                    torch.cuda.synchronize()
                    tick=time.monotonic()
                    observed=model.observation_batch(values('sam'),values('rgb'),values('pixels'),values('hw'))
                    poses,_,_=model.forward_batch(values('points'),observed,values('initial'),values('source'),values('target'),values('structure'),values('visible_ids'))
                    torch.cuda.synchronize()
                    timings[name]=time.monotonic()-tick
                    for frame,pose in zip(frames,poses):
                        world=pose.clone()
                        world[:3,3]+=center
                        outputs[name][frame['key']]={'frame_key':frame['key'],'date':frame['date'],'timestamp_us':frame['timestamp_us'],'T_WB_pred':world.cpu().double().tolist(),'status':'success','visible_points':len(frame['points']),'valid_corr':frame['valid_corr'],'selected_corr':frame['selected_corr']}
            save(destination,{'binding':binding,'keys':keys,'predictions':{n:[rows[k] for k in keys] for n,rows in outputs.items()},'failures':sum(outputs['V5'][k]['status']!='success' for k in keys),'backend_seconds':timings,'forbidden_GT_attempts':len(denied)})
            if start%128==0 or stop==len(records):
                save(OUTPUT/'progress.json',{'frames':stop,'target':expected_frames,'failures':failures,'cached_reused_frames':reused,'process_seconds':time.monotonic()-started,'pid':os.getpid(),'prediction_freeze_pending':True})
            del frames,outputs
    for name,model in models.items():assert state_dict_sha256(model.state_dict())==initial_states[name]
    chunks=[]
    directories={n:OUTPUT/n for n in models}
    for p in directories.values():p.mkdir(exist_ok=True)
    streams={n:(p/'predictions.tmp').open('w') for n,p in directories.items()}
    count=0
    for start in range(0,len(records),8):
        path=OUTPUT/'chunks'/('{:06d}.json'.format(start))
        chunk=json.loads(path.read_text())
        assert chunk['binding']==binding and chunk['keys']==[r[2] for r in records[start:start+8]]
        chunks.append({'file':path.name,'sha256':sha(path),'frames':len(chunk['keys'])})
        for name,rows in chunk['predictions'].items():
            for row in rows:streams[name].write(json.dumps(row,allow_nan=False)+'\n')
        count+=len(chunk['keys'])
    assert count==expected_frames and not denied
    for name,stream in streams.items():
        stream.close()
        p=directories[name]
        os.replace(p/'predictions.tmp',p/'predictions.jsonl')
        save(p/'prediction_freeze.json',dict(binding,arm=name,prediction_sha256=sha(p/'predictions.jsonl'),failures=failures,forbidden_GT_attempts=0,public_manifest_sha256=plan['public_manifest_sha256'],sam_manifest_sha256=plan['sam_manifest_sha256'],GT_protocol='official nearest full trajectory',independent_test=False))
    save(OUTPUT/'chunks-freeze.json',{'chunks':chunks,'frames':count,'binding':binding})
    verify_sources(config)
    save(OUTPUT/'completed.json',{'complete':True,'frames':count,'failures':failures,'binding':binding,'frozen_before_GT_evaluation':True,'process_seconds':time.monotonic()-started,'cached_reused_frames':reused,'standalone_LEADER_evaluation':False})
except BaseException as error:
    save(OUTPUT/'failed.json',{'error':repr(error),'traceback':traceback.format_exc(),'time':time.time()})
    raise
