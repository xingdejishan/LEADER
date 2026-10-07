import concurrent.futures
import torch

class ParallelFrames:
    def __init__(self,factory,refine,settings,workers=4):
        self.models=[factory() for _ in range(workers)]
        self.streams=[torch.cuda.Stream() for _ in range(workers)]
        self.pool=concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        self.refine=refine
        self.settings=settings

    def run(self,frames):
        assert 1<=len(frames)<=len(self.models)
        parent=torch.cuda.current_stream()
        for stream in self.streams:
            stream.wait_stream(parent)
        def one(index):
            torch.cuda.set_device(0)
            with torch.cuda.stream(self.streams[index]),torch.no_grad():
                model=self.models[index]
                frame=frames[index]
                observed=model.observation_batch([frame['sam']],[frame['rgb']],[frame['pixels']],[frame['hw']])[0]
                result=self.refine(model,frame,observed,self.settings)
            self.streams[index].synchronize()
            return result
        futures=[self.pool.submit(one,i) for i in range(len(frames))]
        results=[future.result() for future in futures]
        for stream in self.streams:
            parent.wait_stream(stream)
        return results

    def close(self):
        self.pool.shutdown(wait=True)
