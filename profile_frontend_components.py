import argparse
import inspect
import json
import time
from pathlib import Path
from types import SimpleNamespace

import e2e_online as E
from online_frontend import OnlineFrontend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--keys-file', required=True)
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    config = E.parser().parse_args(['--keys-file', args.keys_file, '--gpu', args.gpu,
                                  '--frontend-workers', '1', '--gpu-fraction', '.08'])
    config.shard_index, config.shard_count = 0, 1
    specs = E.select_inputs(config)
    if not 24 <= len(specs) <= 384:
        raise ValueError('expected bounded existing key pool')
    specs = [specs[round(i*(len(specs)-1)/23)] for i in range(24)]
    E.configure(config)
    guard, front = E.ResourceGuard(config), None
    try:
        import numpy as np
        front = OnlineFrontend(config)
        torch = front.torch
        timers = {}

        def timed(function, name):
            def call(*values, **kwargs):
                torch.cuda.synchronize()
                started = time.monotonic()
                result = function(*values, **kwargs)
                torch.cuda.synchronize()
                timers[name] = timers.get(name,0.)+time.monotonic()-started
                return result
            return call

        front.runtime.leader.encoder.forward = timed(front.runtime.leader.encoder.forward,'leader_encoder')
        front.runtime.leader.decoder.forward = timed(front.runtime.leader.decoder.forward,'leader_decoder')
        initial_function = front.runtime.initial.__func__.__globals__['sc2_initial']
        factory = initial_function.__globals__['make_matcher']

        def matcher(*values, **kwargs):
            result = factory(*values, **kwargs)
            return SimpleNamespace(estimator=timed(result.estimator,'sc2_estimator'))

        initial_function.__globals__['make_matcher'] = matcher
        for name in ('initial','geometry','frame'):
            setattr(front.runtime,name,timed(getattr(front.runtime,name),name))

        def load(spec):
            for name in ('scan','image'):
                if E.sha_file(spec[name+'_path']) != spec[name+'_sha256']:
                    raise ValueError('input SHA mismatch')
            _, xyz, intensity = E.read_scan(spec['scan_path'])
            return spec, front.prepare(spec,xyz,intensity)

        for _ in range(2):
            front.predict_many([load(specs[0])])
        rows = []
        for spec in specs:
            guard.check()
            timers.clear()
            started = time.monotonic()
            pose, stages, info = front.predict_many([load(spec)])[0]
            rows.append(dict(key=spec['key'],T=pose.tolist(),components_seconds=dict(timers),
                             stage_seconds=stages,total_seconds=time.monotonic()-started))
        report = dict(frames=24,scope='instrumented single-frame synchronized components, no experts/no GT',
                      components_mean_ms={k:1000*float(np.mean([r['components_seconds'][k] for r in rows]))
                                          for k in rows[0]['components_seconds']},
                      stages_mean_ms={k:1000*float(np.mean([r['stage_seconds'][k] for r in rows]))
                                      for k in rows[0]['stage_seconds']},
                      source=inspect.getsourcefile(initial_function),rows=rows)
        with E.output_path(args.out).open('x') as stream:
            json.dump(report,stream,indent=2)
        print(json.dumps({k:v for k,v in report.items() if k != 'rows'}))
    finally:
        if front:
            front.close()
        guard.close()


if __name__ == '__main__':
    main()
