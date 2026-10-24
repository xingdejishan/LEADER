import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from benchmark_acceleration import available_ram, query_gpus, stop_owned


ROOT = Path(__file__).resolve().parent
PYTHON = '/home/zhangcy/.conda/envs/leader/bin/python'
FIELDS = ('initial', 'source', 'target', 'points', 'pixels', 'visible_ids', 'structure')


def save(path, value):
    pending = path.with_suffix(path.suffix+'.pending')
    pending.write_text(json.dumps(value, allow_nan=False))
    os.replace(pending, path)


def wait_for(path, guard=None, timeout=800):
    started = time.monotonic()
    while not path.exists():
        if time.monotonic()-started > timeout:
            raise TimeoutError('benchmark barrier timeout: '+str(path))
        if guard:
            guard.check()
        time.sleep(.5)


def cached_frontend(args):
    from types import SimpleNamespace
    import torch
    from online_frontend import CachedSAM, CKPT, CKPT_SHA, EVAL, V7FULL, V7PLAN, install_loader_shims
    import e2e_online as E

    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(args.gpu_fraction)
    sys.path.insert(0, EVAL)
    sys.path.insert(0, V7FULL)
    import v7_runtime as vr
    c, _, config, _, _ = vr.imports()
    from infer_batch_engine_v2 import ParallelFrames
    denied = c.deny_test(config)
    lnm = install_loader_shims()
    plan = json.loads(Path('/home/zhangcy/projects/leader-v5-leader-style-20261003/v5-training-plan.json').read_text())
    if E.sha_file(CKPT) != CKPT_SHA or E.sha_file(config['base_checkpoint']) != plan['frozen_LEADER_sha256']:
        raise ValueError('checkpoint SHA mismatch')
    base = torch.load(config['base_checkpoint'], map_location='cpu')
    center = torch.as_tensor(base['center_t'], dtype=torch.float32).reshape(3).cuda()
    del base
    state = torch.load(CKPT, map_location='cpu')
    inference = json.loads(Path(V7PLAN).read_text())['inference_method']
    parallel = ParallelFrames(lambda:vr.model(c, config, state['model'], False), c.selfverify.refine, inference, 4)
    del state
    return SimpleNamespace(torch=torch, c=c, model=parallel.models[0], parallel=parallel,
                           inference=inference, sam=CachedSAM(args.sam_cache_root, lnm, plan),
                           runtime=SimpleNamespace(plan=plan, denied=denied, center=center),
                           close=parallel.close)


def worker(args):
    import e2e_online as E
    from online_frontend import OnlineFrontend

    config = E.parser().parse_args(['--keys-file', str(args.directory/'keys.json'),
                                  '--gpu', str(args.gpu), '--gpu-fraction', '.08'])
    config.shard_index, config.shard_count = args.shard, args.shards
    specs = E.select_inputs(config)
    runtime_dir = E.configure(config)
    guard = E.ResourceGuard(config)
    front = None
    try:
        import numpy as np
        from PIL import Image

        init = time.monotonic()
        front = cached_frontend(config) if args.cached_only else OnlineFrontend(config)
        initialization = time.monotonic()-init
        torch = front.torch
        cache = args.cache_directory or runtime_dir/('frontend-alignment-'+args.directory.name)
        if not E.within(cache, runtime_dir):
            raise ValueError('benchmark cache must stay in project runtime')
        cache.mkdir(exist_ok=True)
        files, timings = {}, {}

        for name in ('initial', 'geometry', 'frame'):
            if args.cached_only:
                break
            original = getattr(front.runtime, name)

            def timed(*values, _original=original, _name=name):
                tick = time.monotonic()
                result = _original(*values)
                torch.cuda.synchronize()
                timings[_name] = timings.get(_name, 0.)+time.monotonic()-tick
                return result

            setattr(front.runtime, name, timed)

        def load(spec, cached=False):
            guard.check()
            for name in ('scan', 'image'):
                if E.sha_file(spec[name+'_path']) != spec[name+'_sha256']:
                    raise ValueError('input SHA mismatch')
            if not cached:
                _, xyz, intensity = E.read_scan(spec['scan_path'])
                return spec, front.prepare(spec, xyz, intensity)
            with Image.open(spec['image_path']) as image:
                rgb = np.asarray(image.convert('RGB')).copy()
            sam, _, metadata = front.sam.encode(rgb, spec)
            path, expected_hash = files[spec['key']]
            if path.stat().st_size > 8*2**20 or E.sha_file(path) != expected_hash:
                raise ValueError('benchmark frame cache changed or oversized')
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name:archive[name].copy() for name in FIELDS}
                details = json.loads(str(archive['metadata']))
            if details['key'] != spec['key'] or details['global_index'] != spec['canonical_index']:
                raise ValueError('benchmark frame cache identity mismatch')
            result = dict(details)
            result.update({name:torch.from_numpy(value).pin_memory() for name, value in arrays.items()})
            result.update(rgb=torch.from_numpy(rgb).permute(2, 0, 1).float()/255,
                          sam=sam, hw=metadata['resized_size_hw'])
            return spec, result

        for spec in specs:
            guard.check()
            path = cache/('%06d.npz' % spec['canonical_index'])
            if path.exists():
                with np.load(path, allow_pickle=False) as archive:
                    details = json.loads(str(archive['metadata']))
                if details['key'] != spec['key']:
                    raise ValueError('existing benchmark cache identity mismatch')
                files[spec['key']] = (path, E.sha_file(path))
                continue
            if args.cached_only:
                raise ValueError('cached-only diagnostic requires existing benchmark cache')
            _, (loaded, _) = load(spec)
            with torch.no_grad():
                initial = front.runtime.initial(loaded)
                geometry = front.runtime.geometry(loaded, initial[1].detach().cpu())
                frame = front.runtime.frame(loaded, *initial, geometry)
            arrays = {name:frame[name].detach().cpu().numpy().copy() for name in FIELDS}
            details = {k:v for k,v in frame.items() if k not in FIELDS+('rgb', 'sam')}
            path = cache/('%06d.npz' % spec['canonical_index'])
            with path.open('xb') as stream:
                np.savez(stream, metadata=np.asarray(json.dumps(details)), **arrays)
            if path.stat().st_size > 8*2**20:
                raise ValueError('oversized benchmark frame cache')
            files[spec['key']] = (path, E.sha_file(path))
            del frame, arrays, loaded, initial, geometry
        timings.clear()
        if not args.cached_only:
            warm = [load(spec) for spec in specs[:4]]
            front.predict_many(warm)
            del warm
        save(args.directory/('ready-%d.json' % args.shard),
             dict(initialization_seconds=initialization, cache_frames=len(files),
                  cache_bytes=sum(p.stat().st_size for p,_ in files.values())))
        batches = [specs[i:i+4] for i in range(0, len(specs), 4)]
        for mode in (('cached',) if args.cached_only else ('live', 'cached')):
            if mode == 'cached':
                warm = [load(spec, True) for spec in specs[:4]]
                frames = [{k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in f.items()}
                          for _,f in warm]
                with torch.no_grad():
                    if args.cached_only:
                        f = frames[0]
                        observed = front.model.observation_batch([f['sam']], [f['rgb']], [f['pixels']], [f['hw']])[0]
                        front.c.selfverify.refine(front.model, f, observed, front.inference)
                    front.parallel.run(frames)
                torch.cuda.synchronize()
                del warm, frames
            save(args.directory/('%s-ready-%d.json' % (mode,args.shard)), {'ready':True})
            wait_for(args.directory/(mode+'-go.json'), guard)
            timings.clear()
            rows = []
            started = time.monotonic()
            with ThreadPoolExecutor(max_workers=1) as prefetch:
                def load_batch(batch):
                    tick = time.monotonic()
                    result = [load(spec, mode == 'cached') for spec in batch]
                    return result, time.monotonic()-tick

                pending = prefetch.submit(load_batch, batches[0])
                for index, batch in enumerate(batches):
                    guard.check()
                    inputs, load_seconds = pending.result()
                    if index+1 < len(batches):
                        pending = prefetch.submit(load_batch, batches[index+1])
                    tick = time.monotonic()
                    if mode == 'live':
                        results = front.predict_many(inputs)
                        poses = [result[0].tolist() for result in results]
                    else:
                        frames = [{k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in f.items()}
                                  for _,f in inputs]
                        with torch.no_grad():
                            predictions = front.parallel.run(frames)
                        poses = []
                        for result in predictions:
                            pose = result[0].detach().cpu().double().numpy().copy()
                            pose[:3, 3] += front.runtime.center.detach().cpu().double().numpy()
                            poses.append(pose.tolist())
                        del frames, predictions
                    torch.cuda.synchronize()
                    elapsed = time.monotonic()-tick
                    finished = time.monotonic()
                    for spec, pose in zip(batch, poses):
                        rows.append(dict(key=spec['key'], T=pose, load_seconds_amortized=load_seconds/len(batch),
                                         compute_seconds_amortized=elapsed/len(batch),
                                         batch_started_at=tick, batch_finished_at=finished))
            save(args.directory/('%s-%d.json' % (mode,args.shard)),
                 dict(rows=rows, started_at=started, finished_at=time.monotonic(),
                      component_seconds=timings, GT_online_attempts=len(front.runtime.denied),
                      peak_torch_allocated_bytes=torch.cuda.max_memory_allocated()))
            del rows, inputs
        if front.runtime.denied:
            raise RuntimeError('GT access detected')
    finally:
        if front:
            front.close()
        guard.close()


def launch(args):
    if os.path.realpath(sys.prefix) != os.path.realpath(Path(PYTHON).parent.parent):
        raise RuntimeError('existing leader environment required')
    device = query_gpus()[args.gpu]
    need = args.shards*(.08*device['total']+768)
    if device['util'] > 5 or need > .75*device['free'] or device['free']-need < 4096:
        raise RuntimeError('GPU insufficient idle/free global headroom for bounded workers')
    if available_ram() < 192*2**30:
        raise MemoryError('insufficient host RAM headroom')
    directory = ROOT/'results'/('frontend-alignment-20261024-'+str(time.time_ns()))
    directory.mkdir(mode=0o700)
    keys, canonical = [], 0
    for date in ('2012-02-12','2012-02-19','2012-03-31','2012-05-26'):
        manifest = json.loads(Path('/data2/zhangcy/leader-v2-full-trajectory-20261001/NCLT',date,'manifest.json').read_text())
        keys.extend('%s/%s' % (date,row['timestamp_us']) for row in manifest['frames'])
    if len(keys) != 104732:
        raise ValueError('unexpected canonical frame count')
    selected = [keys[round(i*(len(keys)-1)/383)] for i in range(384)]
    save(directory/'keys.json', selected)
    print(json.dumps({'directory':str(directory),'frames':384,'gpu':args.gpu}),flush=True)
    processes, logs = [], []
    started = time.monotonic()
    try:
        for shard in range(args.shards):
            log = (directory/('worker-%d.log' % shard)).open('x')
            logs.append(log)
            processes.append(subprocess.Popen([PYTHON,'-B',str(Path(__file__).resolve()),'worker',
                    '--directory',str(directory),'--gpu',str(args.gpu),'--shard',str(shard),
                    '--shards',str(args.shards),
                    *(['--cache-directory', str(args.cache_directory)] if args.cache_directory else []),
                    *(['--cached-only'] if args.cached_only else [])],
                    cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    env=dict(os.environ,PYTHONUTF8='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                             MKL_NUM_THREADS='1')))
        for mode in (('cached',) if args.cached_only else ('live','cached')):
            while not all((directory/('%s-ready-%d.json' % (mode,i))).exists() for i in range(args.shards)):
                monitor(processes,args.gpu,started)
                time.sleep(2)
            save(directory/(mode+'-go.json'),{'time':time.monotonic()})
            while not all((directory/('%s-%d.json' % (mode,i))).exists() for i in range(args.shards)):
                monitor(processes,args.gpu,started)
                time.sleep(2)
        for process in processes:
            if process.wait(timeout=30) != 0:
                raise RuntimeError('frontend worker failed')
    finally:
        stop_owned(processes)
        for log in logs:
            log.close()
    report = {'gpu':args.gpu,'shards':args.shards,'streams_per_shard':4,'frames':384,
              'weights':'current original V7 epoch7; historical six-shard full run used epoch10',
              'cache':'original input-cache-v8 is empty; finite benchmark-only cache rebuilt from live input',
              'scope':'no experts, no GT, no production changes; explicit warmup and cross-shard phase barrier'}
    all_rows = {}
    for mode in (('cached',) if args.cached_only else ('live','cached')):
        shards = [json.loads((directory/('%s-%d.json' % (mode,i))).read_text()) for i in range(args.shards)]
        rows = [r for shard in shards for r in shard['rows']]
        if len(rows) != 384 or len({r['key'] for r in rows}) != 384:
            raise RuntimeError('frontend benchmark coverage mismatch')
        span = max(s['finished_at'] for s in shards)-min(s['started_at'] for s in shards)
        report[mode] = dict(processing_span_seconds=span,aggregate_ms_per_frame=1000*span/len(rows),
                fps=len(rows)/span,mean_compute_work_ms=1000*sum(r['compute_seconds_amortized'] for r in rows)/len(rows),
                mean_load_work_ms=1000*sum(r['load_seconds_amortized'] for r in rows)/len(rows),
                component_work_ms_per_frame={name:1000*sum(s['component_seconds'].get(name,0) for s in shards)/len(rows)
                                             for name in ('initial','geometry','frame')},
                peak_torch_allocated_bytes=max(s['peak_torch_allocated_bytes'] for s in shards))
        all_rows[mode] = {r['key']:r for r in rows}
    import numpy as np
    if args.cached_only:
        save(directory/'summary.json', report)
        print(json.dumps(report),flush=True)
        return
    report['cached_vs_live_max_translation_difference_m'] = max(float(np.linalg.norm(
        np.asarray(all_rows['cached'][k]['T'])[:3,3]-np.asarray(all_rows['live'][k]['T'])[:3,3])) for k in selected)
    save(directory/'summary.json', report)
    print(json.dumps(report),flush=True)


def monitor(processes, gpu, started):
    if time.monotonic()-started > 1200:
        raise TimeoutError('finite frontend benchmark timeout')
    if any(p.poll() not in (None,0) for p in processes):
        raise RuntimeError('worker failed; inspect isolated benchmark logs')
    if query_gpus()[gpu]['free'] < 4096 or available_ram() < 32*2**30:
        raise MemoryError('frontend benchmark safety floor reached')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=('launch','worker'))
    parser.add_argument('--gpu',type=int,default=4)
    parser.add_argument('--directory',type=Path)
    parser.add_argument('--shard',type=int)
    parser.add_argument('--shards',type=int,choices=(3,6),default=3)
    parser.add_argument('--cache-directory',type=Path)
    parser.add_argument('--cached-only',action='store_true')
    args = parser.parse_args()
    if args.mode == 'worker':
        if args.directory is None or not 0 <= args.shard < args.shards or ROOT/'results' not in args.directory.resolve().parents:
            parser.error('worker requires project-owned benchmark directory and shard 0..5')
        worker(args)
    else:
        launch(args)
