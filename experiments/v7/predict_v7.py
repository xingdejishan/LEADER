import argparse
import concurrent.futures
import json
import time
import portable_runtime as r


def run(checkpoint,parallel_requested):
    plan=r.validate()
    c,_,config,_,_=r.imports()
    denied=c.deny_test(config)
    c.torch.cuda.set_per_process_memory_fraction(.88)
    c.torch.manual_seed(20)
    c.torch.cuda.manual_seed_all(20)
    checkpoint_sha=r.sha(checkpoint)
    state=c.torch.load(checkpoint,map_location='cpu')
    model=r.model(c,config,state['model'],False)
    body=c.state_dict_sha256(model.state_dict())
    loader=r.FrameLoader(config,'test')
    center=c.torch.as_tensor(loader.manifest['center_t'],dtype=c.torch.float64).numpy()
    settings=plan['inference_method']
    def serial(frames):
        observed=[model.observation_batch([frame['sam']],[frame['rgb']],[frame['pixels']],[frame['hw']])[0] for frame in frames]
        return [c.selfverify.refine(model,frame,observation,settings) for frame,observation in zip(frames,observed)]
    with c.torch.no_grad():
        warm=r.cuda([loader.frame(0)])
        serial(warm)
        c.torch.cuda.synchronize()
        del warm
        parallel=None
        if parallel_requested:
            from parallel_inference import ParallelFrames
            parallel=ParallelFrames(lambda:r.model(c,config,state['model'],False),c.selfverify.refine,settings,4)
            indices=[]
            for date in plan['test_dates']:
                selected=[index for index,row in enumerate(loader.records) if row[0]==date]
                assert selected
                indices.extend(selected[int((j+.5)*len(selected)/8)] for j in range(8))
            exact=True
            for start in range(0,len(indices),4):
                frames=r.cuda([loader.frame(index) for index in indices[start:start+4]])
                reference=serial(frames)
                optimized=parallel.run(frames)
                exact=exact and all(c.torch.equal(a[0],b[0]) and a[3]==b[3] for a,b in zip(reference,optimized))
            r.save('checkpoint-parity-'+checkpoint_sha[:12]+'.json',dict(frames=32,exact_parity=exact,checkpoint_sha256=checkpoint_sha,GT_online_attempts=len(denied)))
            if not exact:
                parallel.close()
                parallel=None
            del frames,reference,optimized
        del state
        binding=dict(checkpoint_sha256=checkpoint_sha,body_sha256=body,plan_sha256=r.sha(r.STORE/'plan.json'),manifest_sha256=plan['test_manifest_sha256'],frames=len(loader.records))
        prefix='evaluation-'+checkpoint_sha[:12]
        directory=r.STORE/prefix
        directory.mkdir(exist_ok=True)
        assert not (directory/'prediction-freeze.json').exists()
        chunks=[]
        start=0
        while (directory/('chunk-'+str(start)+'.json')).exists():
            name='chunk-'+str(start)+'.json'
            old=r.read(directory/name)
            assert old['binding']==binding and old['GT_online_attempts']==0
            assert [row['frame_key'] for row in old['rows']]==[row[2] for row in loader.records[start:start+len(old['rows'])]]
            chunks.append(dict(file=name,sha256=r.sha(directory/name)))
            start+=len(old['rows'])
        rows=[]
        chunk_start=start
        batch=4 if parallel else 1
        started=time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            def load(offset):
                return [loader.frame(index) for index in range(offset,min(len(loader.records),offset+batch))]
            pending=pool.submit(load,start) if start<len(loader.records) else None
            for offset in range(start,len(loader.records),batch):
                r.budget()
                frames=r.cuda(pending.result())
                if offset+batch<len(loader.records):
                    pending=pool.submit(load,offset+batch)
                output=parallel.run(frames) if parallel else serial(frames)
                c.torch.cuda.synchronize()
                for frame,(pose,_,_,info) in zip(frames,output):
                    assert c.torch.isfinite(pose).all() and 1<=info['rounds']<=12
                    world=pose.detach().cpu().double().numpy().copy()
                    world[:3,3]+=center
                    rows.append(dict(frame_key=frame['key'],T_WB_pred=world.tolist(),shared_input_sha256=frame['input_sha256'],rounds=info['rounds'],accepted=info['accepted'],stop=info['status']))
                stop=min(len(loader.records),offset+batch)
                if len(rows)==64 or stop==len(loader.records):
                    name='chunk-'+str(chunk_start)+'.json'
                    r.save(prefix+'/'+name,dict(binding=binding,rows=rows,GT_online_attempts=0))
                    chunks.append(dict(file=name,sha256=r.sha(directory/name)))
                    rows=[]
                    chunk_start=stop
                    r.save('prediction-progress.json',dict(frames=stop,target=len(loader.records),checkpoint_sha256=checkpoint_sha,mode='parallel4' if parallel else 'serial',seconds=time.monotonic()-started))
                del frames,output
        assert not denied and c.state_dict_sha256(model.state_dict())==body
        r.save(prefix+'/prediction-freeze.json',dict(complete=True,binding=binding,chunks=chunks,GT_online_attempts=0,frozen_before_scoring=True,mode='parallel4' if parallel else 'serial',wall_seconds_this_attempt=time.monotonic()-started,scope='Prepared frozen frontend inputs; excludes model startup, serial initialization, parity guard, and frontend preparation'))
        if parallel:
            parallel.close()
        print(json.dumps(dict(prediction_complete=True,directory=str(directory))))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--serial',action='store_true')
    args=parser.parse_args()
    run(args.checkpoint,not args.serial)
