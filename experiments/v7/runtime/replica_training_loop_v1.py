import concurrent.futures,time
from collections import Counter


def batches(runtime,c,replicas,model,optimizer,loader,indices,perturbations,settings,start,end,checkpoint,progress):
    timings=Counter()
    processed=0
    begun=time.monotonic()
    def load(a,b):
        tick=time.monotonic()
        frames=[loader.frame(int(indices[i]),True) for i in range(a,b)]
        return frames,time.monotonic()-tick
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        pending=pool.submit(load,start,min(start+25,end))
        for offset in range(start,end,25):
            runtime.budget()
            stop=min(offset+25,end)
            tick=time.monotonic()
            cpu_frames,load_seconds=pending.result()
            timings['load_wait_seconds']+=time.monotonic()-tick
            timings['CPU_prepare_background_seconds']+=load_seconds
            if stop<end:pending=pool.submit(load,stop,min(stop+25,end))
            tick=time.monotonic()
            frames=c.cuda(cpu_frames)
            for local,frame in enumerate(frames):
                frame['perturbation']=c.torch.as_tensor(perturbations[indices[offset+local]],device='cuda',dtype=c.torch.float32)
            c.torch.cuda.synchronize()
            timings['H2D_seconds']+=time.monotonic()-tick
            tick=time.monotonic()
            result,parts=replicas.backward(model,frames,settings,True)
            c.torch.cuda.synchronize()
            timings['observation_forward_backward_reduce_seconds']+=time.monotonic()-tick
            tick=time.monotonic()
            gradient=c.torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            assert c.torch.isfinite(gradient)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            c.torch.cuda.synchronize()
            timings['optimizer_seconds']+=time.monotonic()-tick
            processed+=len(frames)
            checkpoint_seconds=checkpoint(stop,result,processed)
            timings['checkpoint_seconds']+=checkpoint_seconds
            tick=time.monotonic()
            progress(stop,result,processed,dict(timings),time.monotonic()-begun)
            timings['progress_save_seconds']+=time.monotonic()-tick
            del frames,cpu_frames,result,parts,frame
    return dict(processed=processed,seconds=time.monotonic()-begun,timings=dict(timings))
