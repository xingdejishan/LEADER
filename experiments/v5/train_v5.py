import argparse,concurrent.futures,json,os,resource,shutil,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
from v5_paths import BASE,DATA,STORE,SOURCE,RUNTIME,RAW_GT
sys.path[:0]=[str(SOURCE),str(RUNTIME)]
from models.far_covolume_standard import CoVolumeStandardFAR,registration_loss_batch
from tools.far_runtime import verify_sources
from tools.magic_nclt_runtime import state_dict_sha256
from fused_field import enable as fused_enable
from batched_solver import enable as batch_enable
from full_loader import FullLoader,sha

assert os.name=='posix'
os.nice(15)
torch.set_num_threads(1)
torch.cuda.set_per_process_memory_fraction(.85)
parser=argparse.ArgumentParser()
parser.add_argument('--resume',action='store_true')
args=parser.parse_args()
OUTPUT=STORE/'training'
OUTPUT.mkdir(exist_ok=args.resume)
(OUTPUT/'checkpoints').mkdir(exist_ok=True)

def save(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False))
    os.replace(temporary,path)

def budget():
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<10*2**30
    assert shutil.disk_usage(STORE).free>40*2**30
    available=int(next(line for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')).split()[1])*1024
    assert available>32*2**30

def evaluate(epoch):
    report=STORE/'evaluations'/('epoch{:02d}'.format(epoch))/'report.json'
    if report.exists():
        existing=json.loads(report.read_text())
        assert existing['complete'] and existing['epoch']==epoch and existing['plan_sha256']==plan_hash
        return
    torch.cuda.empty_cache()
    save(OUTPUT/'progress.json',dict(phase='evaluating_LEADER_plus_V5',epoch=epoch,epochs=50,frames=107339,pid=os.getpid(),standalone_LEADER_evaluation=False))
    for script,device in [('predict_v5.py',os.environ['CUDA_VISIBLE_DEVICES']),('score_v5.py','-1')]:
        with (STORE/('epoch{:02d}-{}.log'.format(epoch,script))).open('a') as log:
            process=subprocess.run([sys.executable,'-B',str(ROOT/script),'--epoch',str(epoch)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=device),stdout=log,stderr=subprocess.STDOUT)
        assert process.returncode==0,(script,epoch,process.returncode)
    assert report.exists()
    value=json.loads(report.read_text())
    save(OUTPUT/('evaluation-{:02d}.json'.format(epoch)),dict(epoch=epoch,report_sha256=sha(report),statistics=value['statistics'],system='LEADER+V5'))

try:
    startup=time.monotonic()
    config=json.loads((BASE/'config-v2.json').read_text())
    plan=json.loads((ROOT/'v5-training-plan.json').read_text())
    plan_hash=sha(ROOT/'v5-training-plan.json')
    for name,digest in plan['sources'].items():assert sha(ROOT/name)==digest
    for path,digest in plan['dependencies'].items():assert sha(path)==digest
    verify_sources(config)
    assert sha(BASE/'config-v2.json')==plan['parent_config_sha256']
    assert sha(config['base_checkpoint'])==plan['frozen_LEADER_sha256']
    np.random.seed(20)
    torch.manual_seed(20)
    torch.cuda.manual_seed_all(20)
    rng=np.random.RandomState(20)
    loader=FullLoader(config)
    model=CoVolumeStandardFAR(**config['far_covolume']).cuda().train()
    fused_enable(model)
    batch_enable(model)
    model.point_chunk=4096
    assert model.iterations==3 and tuple(model.search_spacings)==(.4,.1)
    assert tuple(model.tukey_thresholds)==(1.2,.6) and tuple(model.template.shape)==(125,3)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=.9)
    initial_hash=state_dict_sha256(model.state_dict())
    resume=None
    if args.resume:
        durable=json.loads((OUTPUT/'latest-state.json').read_text())
        assert sha(OUTPUT/durable['file'])==durable['sha256']
        resume=torch.load(OUTPUT/durable['file'],map_location='cpu')
        assert resume['plan_sha256']==plan_hash
        model.load_state_dict(resume['model'])
        optimizer.load_state_dict(resume['optimizer'])
        scheduler.load_state_dict(resume['scheduler'])
        rng.set_state(resume['numpy_rng'])
        torch.set_rng_state(resume['torch_rng'])
        torch.cuda.set_rng_state_all(resume['cuda_rng'])
        global_updates=resume['global_updates']
        checkpoint_serial=resume['checkpoint_serial']
        starting_epoch=resume['epoch']+int(resume['epoch_complete'])
        completed_epochs=resume['epoch'] if resume['epoch_complete'] else resume['epoch']-1
        for finished_epoch in range(5,completed_epochs+1,5):evaluate(finished_epoch)
    else:
        global_updates=0
        checkpoint_serial=0
        starting_epoch=1
        assert initial_hash==plan['scratch_initial_weights_sha256']
        save(OUTPUT/'initialization.json',dict(from_scratch=True,initial_weights_sha256=initial_hash,seed=20,optimizer='Adam',learning_rate=.001,lr_gamma_per_epoch=.9,batch=25,last_batch=14,epochs=50,updates_per_epoch=4294,frames=107339,manifests=loader.manifests,plan_sha256=plan_hash,startup_seconds=time.monotonic()-startup,frozen_LEADER_sha256=plan['frozen_LEADER_sha256'],standalone_LEADER_evaluation=False))

    for epoch in range(starting_epoch,51):
        if resume is not None and epoch==resume['epoch'] and not resume['epoch_complete']:
            indices=resume['indices']
            offset=resume['next_offset']
            total_loss=resume['weighted_loss_sum']
            elapsed_before=resume['active_epoch_seconds']
            epoch_started=resume['epoch_started_at']
        else:
            indices=rng.permutation(len(loader.records)).tolist()
            offset=0
            total_loss=0.
            elapsed_before=0.
            epoch_started=time.time()
        assert len(indices)==len(set(indices))==107339 and min(indices)==0 and max(indices)==107338
        order_hash=__import__('hashlib').sha256(np.asarray(indices,dtype='<i8').tobytes()).hexdigest()
        begin=time.monotonic()
        wait_total=transfer_total=compute_total=checkpoint_total=0.
        learning_rate=optimizer.param_groups[0]['lr']
        assert abs(learning_rate-.001*.9**(epoch-1))<1e-12

        def checkpoint(epoch_complete=False):
            global checkpoint_serial
            tick=time.monotonic()
            torch.cuda.synchronize()
            checkpoint_serial+=1
            state=dict(model={name:value.detach().cpu().clone() for name,value in model.state_dict().items()},optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),numpy_rng=rng.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),indices=indices,next_offset=offset,weighted_loss_sum=total_loss,active_epoch_seconds=elapsed_before+time.monotonic()-begin,epoch_started_at=epoch_started,epoch=epoch,epoch_complete=epoch_complete,global_updates=global_updates,checkpoint_serial=checkpoint_serial,plan_sha256=plan_hash,config=config)
            path=OUTPUT/('latest-slot{}.pt'.format(checkpoint_serial%2))
            temporary=path.with_suffix('.tmp')
            torch.save(state,temporary)
            os.replace(temporary,path)
            save(OUTPUT/'latest-state.json',dict(file=path.name,sha256=sha(path),epoch=epoch,epoch_complete=epoch_complete,next_offset=offset,global_updates=global_updates,plan_sha256=plan_hash))
            return time.monotonic()-tick

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pending=pool.submit(loader.batch,indices[offset:offset+25],True)
            while offset<len(indices):
                budget()
                tick=time.monotonic()
                frames=pending.result()
                wait_total+=time.monotonic()-tick
                end=min(offset+25,len(indices))
                if end<len(indices):pending=pool.submit(loader.batch,indices[end:end+25],True)
                tick=time.monotonic()
                group=[{name:value.to('cuda:0',non_blocking=True) if torch.is_tensor(value) else value for name,value in frame.items()} for frame in frames]
                torch.cuda.synchronize()
                transfer_total+=time.monotonic()-tick
                values=lambda name:[frame[name] for frame in group]
                optimizer.zero_grad(set_to_none=True)
                tick=time.monotonic()
                observed=model.observation_batch(values('sam'),values('rgb'),values('pixels'),values('hw'))
                loss,metrics=registration_loss_batch(model,values('points'),values('raw'),observed,values('truth'),values('initial'),values('source'),values('target'),values('structure'),values('visible_ids'))
                assert torch.isfinite(loss),'nonfinite loss'
                loss.backward()
                gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
                assert torch.isfinite(gradient),'nonfinite gradient'
                optimizer.step()
                torch.cuda.synchronize()
                compute_total+=time.monotonic()-tick
                total_loss+=float(loss)*len(group)
                offset=end
                global_updates+=1
                if global_updates%128==0:checkpoint_total+=checkpoint()
                if global_updates%16==0 or offset==len(indices):
                    active=elapsed_before+time.monotonic()-begin
                    save(OUTPUT/'progress.json',dict(phase='training',epoch=epoch,epochs=50,frames=offset,target=107339,global_updates=global_updates,learning_rate=learning_rate,loss_mean=total_loss/offset,active_epoch_seconds=active,fps=offset/active,estimated_epoch_hours=107339/(offset/active)/3600,estimate_not_complete_epoch=True,load_wait_seconds=wait_total,transfer_seconds=transfer_total,compute_seconds=compute_total,checkpoint_seconds=checkpoint_total,peak_reserved_bytes=torch.cuda.max_memory_reserved(),pid=os.getpid()))
                del observed,loss,group,frames
        assert offset==107339
        scheduler.step()
        checkpoint_total+=checkpoint(True)
        epoch_path=OUTPUT/'checkpoints'/('{:02d}.pt'.format(epoch))
        temporary=epoch_path.with_suffix('.tmp')
        torch.save(dict(model={name:value.detach().cpu().clone() for name,value in model.state_dict().items()},epoch=epoch,plan_sha256=plan_hash,config=config),temporary)
        os.replace(temporary,epoch_path)
        save(epoch_path.with_suffix('.json'),dict(epoch=epoch,sha256=sha(epoch_path),model_state_sha256=state_dict_sha256(model.state_dict()),plan_sha256=plan_hash))
        save(OUTPUT/('epoch{:02d}-completed.json'.format(epoch)),dict(epoch=epoch,frames=107339,global_updates=global_updates,learning_rate=learning_rate,next_learning_rate=optimizer.param_groups[0]['lr'],frame_order_sha256=order_hash,wall_seconds=time.time()-epoch_started,active_seconds=elapsed_before+time.monotonic()-begin,loss_mean=total_loss/107339,last_batch=14,checkpoint_sha256=sha(epoch_path),complete=True))
        resume=None
        if epoch%5==0:evaluate(epoch)
    loader.close()
    verify_sources(config)
    save(OUTPUT/'completed.json',dict(complete=True,epochs=50,frames_per_epoch=107339,global_updates=global_updates,all_ten_combination_evaluations_completed=True,stage2_LEADER_finetune_pending=True,standalone_LEADER_evaluation=False))
except BaseException as error:
    save(OUTPUT/'failed.json',dict(error=repr(error),traceback=traceback.format_exc(),time=time.time()))
    raise
