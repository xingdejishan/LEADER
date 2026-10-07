import argparse
import concurrent.futures
import gc
import hashlib
import json
import os
import time
from collections import Counter
import portable_runtime as r

try:
    import resource
except ImportError:
    resource=None

def setup():
    p = r.validate()
    c, parent, config, solver, Loader = r.imports()
    denied = c.deny_test(config)
    c.torch.cuda.set_per_process_memory_fraction(.88)
    c.torch.manual_seed(20)
    c.torch.cuda.manual_seed_all(20)
    return p, c, parent, config, solver, Loader, denied


def train(epoch):
    p, c, parent, config, solver, Loader, denied = setup()
    loader = Loader(config)
    count = len(loader.records)
    assert count == p['train_frames']
    latest = r.STORE / 'latest-checkpoint.json'
    if latest.exists():
        receipt = r.read(latest)
        assert r.sha(r.STORE / receipt['file']) == receipt['sha256']
        state = c.torch.load(r.STORE / receipt['file'], map_location='cpu')
        assert state['plan_sha256'] == r.sha(r.STORE / 'plan.json')
        assert (state['epoch'] == epoch and not state['epoch_complete']) or (state['epoch'] == epoch - 1 and state['epoch_complete'])
    else:
        assert epoch == 1
        state = c.torch.load(p['initial_checkpoint'], map_location='cpu')
    m = r.model(c, config, state['model'], True)
    opt = c.torch.optim.Adam(m.parameters(), lr=p['learning_rate'])
    sched = c.torch.optim.lr_scheduler.StepLR(opt, 1, gamma=p['lr_gamma'])
    continuing = 'plan_sha256' in state and state['plan_sha256'] == r.sha(r.STORE / 'plan.json')
    updates = 0
    offset = 0
    totals = Counter()
    prior_seconds = 0.
    if continuing:
        opt.load_state_dict(state['optimizer'])
        sched.load_state_dict(state['scheduler'])
        for parameter, x in opt.state.items():
            x['step'] = x['step'].cpu()
            assert x['exp_avg'].device == parameter.device == x['exp_avg_sq'].device
        updates = state['updates']
        if state['epoch'] == epoch:
            offset = state['offset']
            totals = Counter(state['totals'])
            prior_seconds = state['active_seconds']
        c.torch.set_rng_state(state['torch_rng'])
        c.torch.cuda.set_rng_state(state['cuda_rng'])
    else:
        assert c.state_dict_sha256(m.state_dict()) == p['initial_body_sha256']
    del state
    assert abs(opt.param_groups[0]['lr'] - p['learning_rate'] * p['lr_gamma']**(epoch - 1)) < 1e-12
    rng = c.np.random.RandomState(20)
    for i in range(1, epoch + 1):
        indices = rng.permutation(count)
        perturbations = rng.normal(size=(count, 6)) * .5
    assert len(set(indices)) == count
    order_sha = hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest()
    assert offset == count or offset % 25 == 0
    c.np.savez_compressed(r.STORE / ('order-epoch' + str(epoch) + '.npz'), indices=indices)
    begun = time.monotonic()
    seconds = 0.
    opt.zero_grad(set_to_none=True)

    def checkpoint(complete=False):
        slot = (updates // 10) % 2
        target = r.STORE / ('state-slot' + str(slot) + '.pt')
        payload = dict(model={n: x.detach().cpu().clone() for n, x in m.state_dict().items()}, optimizer=opt.state_dict(), scheduler=sched.state_dict(), epoch=epoch, epoch_complete=complete, offset=offset, updates=updates, totals=dict(totals), active_seconds=prior_seconds + time.monotonic() - begun, torch_rng=c.torch.get_rng_state(), cuda_rng=c.torch.cuda.get_rng_state(), plan_sha256=r.sha(r.STORE / 'plan.json'), body_sha256=c.state_dict_sha256(m.state_dict()), order_sha256=order_sha, gradients_cleared=True,execution_amendment_sha256=r.sha(r.STORE/'derivative-execution-amendment-v6.json'))
        temporary = target.with_suffix('.tmp')
        c.torch.save(payload, temporary)
        os.replace(temporary, target)
        r.save('latest-checkpoint.json', dict(file=target.name, sha256=r.sha(target), epoch=epoch, epoch_complete=complete, offset=offset, updates=updates, body_sha256=payload['body_sha256']))
        return payload

    import complete_mixed_backward_v1 as mixed
    import scheduled_replica_solver_v1 as execution
    import replica_training_loop_v1 as training
    amendment=r.read(r.STORE/'derivative-execution-amendment-v6.json')
    assert amendment['scientific_plan_sha256']==r.sha(r.STORE/'plan.json')
    assert all(r.sha(r.ROOT/name)==digest for name,digest in amendment['sources'].items())
    assert r.read(r.STORE/'batch-full-step-scheduled7-v1/report.json')['parity_passed']
    mixed.enable(m,c.mixed_field)
    def factory():
        replica=r.model(c,config,m.state_dict(),True)
        mixed.enable(replica,c.mixed_field)
        return replica
    if offset<count:
        cpu_rng=c.torch.get_rng_state();gpu_rng=c.torch.cuda.get_rng_state()
        warm=c.cuda([loader.frame(int(indices[offset]),True)])[0]
        warm['perturbation']=c.torch.as_tensor(perturbations[indices[offset]],device='cuda',dtype=c.torch.float32)
        observed=m.observation_batch([warm['sam']],[warm['rgb']],[warm['pixels']],[warm['hw']])[0]
        value,_=execution.solver.loss_batch(m,[warm],[observed],p['training_method'],True)[0]
        value.backward();c.torch.cuda.synchronize();m.zero_grad(set_to_none=True)
        c.torch.set_rng_state(cpu_rng);c.torch.cuda.set_rng_state(gpu_rng)
        del warm,observed,value,cpu_rng,gpu_rng
    replicas=execution.ReplicaFrames(factory,7)
    resume_offset=offset
    def batch_checkpoint(stop,result,processed):
        nonlocal offset,updates
        offset=stop;updates+=1
        for value,info in result:
            assert c.np.isfinite(value)
            totals['loss']+=value
            for key in ['correlation','pose','shape','rounds','accepted']:totals[key]+=info[key]
        if updates%10 and offset!=count:return 0.
        tick=time.monotonic();checkpoint();return time.monotonic()-tick
    def batch_progress(stop,result,processed,timings,elapsed):
        r.save('progress.json',dict(phase='full_V7_training_replica24',epoch=epoch,epochs=5,frames=offset,target=count,updates=updates,learning_rate=opt.param_groups[0]['lr'],mean_loss=totals['loss']/offset,active_seconds=prior_seconds+time.monotonic()-begun,engine_estimated_epoch_hours=elapsed/processed*count/3600,engine_timing_frames=processed,resume_offset=resume_offset,single_GPU=True,replicas=7,effective_batch=25,timings=timings,background_prepare_overlaps_compute=True,execution_amendment_sha256=r.sha(r.STORE/'derivative-execution-amendment-v6.json'),estimate_not_complete=True,pid=os.getpid()))
    execution_result=training.batches(r,c,replicas,m,opt,loader,indices,perturbations,p['training_method'],offset,count,batch_checkpoint,batch_progress) if offset<count else dict(processed=0,seconds=0.,timings={})
    replicas.close()
    seconds=execution_result['timings'].get('observation_forward_backward_reduce_seconds',0.)
    assert offset == count and updates == epoch * p['updates_per_epoch'] and not denied
    sched.step()
    payload = checkpoint(True)
    target = r.STORE / ('epoch' + str(epoch) + '.pt')
    assert not target.exists()
    c.torch.save(payload, target)
    record = dict(execution_amendment_sha256=r.sha(r.STORE/'derivative-execution-amendment-v6.json'),execution_result=execution_result,complete=True, epoch=epoch, frames=count, updates=updates, updates_this_epoch=p['updates_per_epoch'], order_sha256=order_sha, checkpoint_sha256=r.sha(target), body_sha256=payload['body_sha256'], means={k: v / count for k, v in totals.items()}, active_seconds=payload['active_seconds'], compute_seconds_this_attempt=seconds, inherited_small_updates=480, fresh_full_optimizer=True, GT_online_attempts=0, peak_reserved_bytes=c.torch.cuda.max_memory_reserved(), peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    r.save('epoch' + str(epoch) + '-completed.json', record)
    loader.close()
    print(json.dumps(record), flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--epoch',type=int,required=True,choices=range(1,6))
    args=parser.parse_args()
    train(args.epoch)
