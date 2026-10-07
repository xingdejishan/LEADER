import concurrent.futures,time,queue
import torch
import target_derivative_engine_v2 as solver


class ReplicaFrames:
    def __init__(self,factory,workers=4):
        with torch.random.fork_rng(devices=[0]):
            self.models=[factory() for _ in range(workers)]
        self.streams=[torch.cuda.Stream() for _ in range(workers)]
        self.pool=concurrent.futures.ThreadPoolExecutor(max_workers=workers)

    def backward(self,model,frames,settings,shape_enabled):
        parent=torch.cuda.current_stream()
        current=model.state_dict()
        for replica,stream in zip(self.models,self.streams):
            replica.load_state_dict(current)
            replica.zero_grad(set_to_none=True)
            stream.wait_stream(parent)
        pending=queue.Queue()
        for index in range(len(frames)):pending.put(index)
        def run(worker):
            torch.cuda.set_device(0)
            values=[];forward=0.;backward=0.;observation=0.
            replica=self.models[worker]
            with torch.cuda.stream(self.streams[worker]),torch.enable_grad():
                while True:
                    try:index=pending.get_nowait()
                    except queue.Empty:break
                    frame=frames[index]
                    tick=time.monotonic()
                    observed=replica.observation_batch([frame['sam']],[frame['rgb']],[frame['pixels']],[frame['hw']])[0]
                    observation+=time.monotonic()-tick
                    tick=time.monotonic()
                    value,info=solver.loss_batch(replica,[frame],[observed],settings,shape_enabled,query_chunk=256000)[0]
                    forward+=time.monotonic()-tick
                    tick=time.monotonic()
                    (value/len(frames)).backward()
                    self.streams[worker].synchronize()
                    backward+=time.monotonic()-tick
                    values.append((index,float(value.detach()),info))
                    del value,observed,info
            self.streams[worker].synchronize()
            return values,dict(observation_seconds=observation,forward_seconds=forward,backward_seconds=backward)
        futures=[self.pool.submit(run,worker) for worker in range(min(len(frames),len(self.models)))]
        returned=[future.result() for future in futures]
        for stream in self.streams:parent.wait_stream(stream)
        replica_parameters=[dict(replica.named_parameters()) for replica in self.models]
        for name,parameter in model.named_parameters():
            gradients=[items[name].grad for items in replica_parameters if items[name].grad is not None]
            parameter.grad=sum(gradients) if gradients else None
        ordered=sorted((row for values,_ in returned for row in values),key=lambda row:row[0])
        assert [row[0] for row in ordered]==list(range(len(frames)))
        return [(value,info) for _,value,info in ordered],[parts for _,parts in returned]

    def close(self):
        self.pool.shutdown(wait=True)
