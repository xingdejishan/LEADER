import argparse
import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import e2e_online as E
import online_frontend

SOURCE_FILE = Path(__file__).resolve()

def stats(values):
    import numpy as np
    a = np.asarray(values)*1000
    return dict(n=len(a),mean_ms=float(a.mean()),p50_ms=float(np.median(a)),
                p95_ms=float(np.percentile(a,95)),maximum_ms=float(a.max()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--keys-file',required=True)
    parser.add_argument('--gpu',default='1')
    parser.add_argument('--threads',type=int,choices=(1,4),default=1)
    parser.add_argument('--batch',type=int,choices=(1,4),default=1)
    parser.add_argument('--out',required=True)
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--scratch-root',type=Path)
    args = parser.parse_args()
    config = E.parser().parse_args(['--keys-file',args.keys_file,'--gpu',args.gpu,
                                  '--threads',str(args.threads)])
    config.shard_index,config.shard_count = 0,1
    specs = E.select_inputs(config)
    if not 32 <= len(specs) <= 384:
        raise ValueError('expected bounded existing input pool')
    specs = [specs[round(i*(len(specs)-1)/31)] for i in range(32)]
    if args.scratch_root is not None:
        parent = Path('/data2/zhangcy').resolve(strict=True)
        target = args.scratch_root
        if (not target.is_absolute() or target.parent.resolve(strict=True) != parent
                or not target.name.startswith('e2e-leader-audit-')
                or parent.stat().st_uid != os.getuid() or target.exists() or target.is_symlink()):
            raise ValueError('scratch must be a new isolated directory under owned /data2/zhangcy')
        target.mkdir(mode=0o700)
        (target/'results').mkdir(mode=0o700)
        E.PROJECT = target.resolve(strict=True)
    E.configure(config)
    guard = E.ResourceGuard(config)
    try:
        def gpu_snapshot():
            result = subprocess.check_output(['nvidia-smi','-i',args.gpu,
                    '--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            processes = subprocess.check_output(['nvidia-smi','-i',args.gpu,
                    '--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
            return dict(device=result.strip(),processes=processes.strip())

        before_gpu = gpu_snapshot()
        if float(before_gpu['device'].split(',')[-1]) > 5:
            raise RuntimeError('selected GPU is not idle; no benchmark model loaded')
        import numpy as np
        import torch
        import MinkowskiEngine as ME
        torch.set_num_threads(args.threads)
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(.15)
        snapshot_root = SOURCE_FILE.parent/'official_leader_audit'
        snapshot = json.loads((snapshot_root/'snapshot.json').read_text())
        for name,digest in snapshot['sources_sha256'].items():
            if E.sha_file(snapshot_root/name) != digest:
                raise ValueError('official source snapshot changed')
        spec = importlib.util.spec_from_file_location('isolated_official_leader',snapshot_root/'models/model_mink.py')
        official_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(official_module)
        names = {'cartesian_to_polar_expansion','polar_expansion_to_cartesian'}
        tree = ast.parse((snapshot_root/'utils/pose_util.py').read_text())
        functions = ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
        namespace = dict(np=np,torch=torch)
        exec(compile(functions,'official-polar-functions','exec'),namespace)
        sys.path.insert(0,'/home/zhangcy/projects/leader-far-covolume-standard-20261001/source-v1')
        from models.model_mink import LEADER as Current
        from models.model_mink import MinkowskiSparseTensorCat as aligned_cat
        from models.sc2pcr import Matcher
        base = json.loads(Path('/home/zhangcy/projects/leader-far-covolume-standard-20261001/config-v2.json').read_text())
        if E.sha_file(base['base_checkpoint']) != base['base_checkpoint_sha256']:
            raise ValueError('frozen LEADER checkpoint changed')
        state = torch.load(base['base_checkpoint'],map_location='cpu')
        center = torch.as_tensor(state['center_t'],device='cuda',dtype=torch.float32).reshape(3)
        models = {}
        for name,constructor in [('official',official_module.LEADER),('current',Current)]:
            model = constructor(in_channels=3,out_channels=4,feat_channels=512,width=1024)
            model.load_state_dict(state['model'],strict=True)
            models[name] = model.cuda().eval().requires_grad_(False)
        del state
        original_cat = official_module.MinkowskiSparseTensorCat
        models['official_aligned'] = models['official']
        matcher = Matcher(inlier_threshold=2.,d_thre=2,num_iterations=10,ratio=.15,
                          nms_radius=.1,max_points=3000,k1=30)
        polar = namespace['cartesian_to_polar_expansion']
        cartesian = namespace['polar_expansion_to_cartesian']

        def prepare(batch):
            coordinates,features,counts = [],[],[]
            for record in batch:
                _,points,intensity = E.read_scan(record['scan_path'])
                radius = np.linalg.norm(points,axis=1)
                mask = (radius>1.) & (radius<100.)
                points,intensity = points[mask],intensity[mask]
                projected = polar(points,1024*.2)
                values = np.column_stack((projected[:,2],projected[:,1],intensity)).astype(np.float32)
                co,fe = ME.utils.sparse_quantize(coordinates=projected,features=values,quantization_size=.2)
                coordinates.append(np.asarray(co,dtype=np.int32));features.append(np.asarray(fe,dtype=np.float32))
                counts.append(dict(raw_points=len(radius),kept_points=len(points),input_voxels=len(co)))
            co,fe = ME.utils.sparse_collate(coordinates,features)
            return co.cuda(),fe.cuda(),counts

        def solve(name,batch,retain=False):
            official_module.MinkowskiSparseTensorCat = aligned_cat if name == 'official_aligned' else original_cat
            wall_start = time.monotonic()
            co,fe,counts = prepare(batch)
            torch.cuda.synchronize()
            tick = time.monotonic()
            sparse = ME.SparseTensor(features=fe,coordinates=co)
            with torch.no_grad():
                encoded = models[name].encoder(sparse)
                stride = torch.tensor(encoded.tensor_stride,device='cuda',dtype=torch.float32)
                positions = cartesian((encoded.C[:,1:].float()+stride/2)*.2,1024*.2)
                prediction = models[name].decoder(encoded.F)
                batch_ids = encoded.C[:,0].long()
                outputs = []
                for i,record in enumerate(batch):
                    valid = batch_ids == i
                    p,source = prediction[valid].float(),positions[valid].float()
                    count = max(min(50,len(p)),int(.5*len(p)))
                    ids = p[:,3].topk(count).indices
                    torch.manual_seed(base['seed']+record['canonical_index'])
                    torch.cuda.manual_seed_all(base['seed']+record['canonical_index'])
                    pose = matcher.estimator(source[ids][None],p[ids,:3][None])[0]
                    pose = pose.clone()
                    pose[:3,3] += center
                    outputs.append(pose)
                script_stop = time.monotonic()
                torch.cuda.synchronize()
                sync_stop = time.monotonic()
                poses = [p.cpu().double().tolist() for p in outputs]
                output_stop = time.monotonic()
            row = dict(keys=[r['key'] for r in batch],T=poses,batch_size=len(batch),input_counts=counts,
                       official_code_boundary_seconds=script_stop-tick,
                       synchronized_compute_seconds=sync_stop-tick,
                       input_to_T_seconds=output_stop-wall_start,
                       preparation_and_transfer_seconds=tick-wall_start)
            arrays = (encoded.C.cpu().numpy().copy(),prediction.detach().cpu().numpy().copy()) if retain else None
            return row,arrays

        for name in models:
            for _ in range(2):
                solve(name,specs[:1])
        torch.cuda.synchronize()
        resident = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        solve('official',specs[:1])
        peak_single = torch.cuda.max_memory_allocated()
        dynamic = max(1,peak_single-resident)
        estimated = resident+args.batch*dynamic*1.75+128*2**20
        cap = torch.cuda.get_device_properties(0).total_memory*.15
        if args.batch>1 and (estimated>cap*.9 or torch.cuda.mem_get_info()[0]-estimated<2**30):
            raise MemoryError('batch preflight rejects estimated memory; no large batch allocated')
        if args.batch>1:
            for name in models:
                solve(name,specs[:args.batch])
        torch.cuda.reset_peak_memory_stats()
        rows = {'official':[],'current':[],'official_aligned':[]}
        feature_differences,pose_differences = [],[]
        aligned_differences,aligned_pose_differences = [],[]
        for start in range(0,len(specs),args.batch):
            guard.check()
            batch = specs[start:start+args.batch]
            results = {}
            order = ['official','current','official_aligned']
            shift = (start//args.batch)%3
            order = order[shift:]+order[:shift]
            for name in order:
                row,arrays = solve(name,batch,True)
                rows[name].append(row);results[name] = arrays
            a,b = results['official'],results['current']
            ia,ib = np.lexsort(a[0].T[::-1]),np.lexsort(b[0].T[::-1])
            if not np.array_equal(a[0][ia],b[0][ib]):
                raise ValueError('official/current output coordinate mismatch')
            feature_differences.append(float(np.max(np.abs(a[1][ia]-b[1][ib]))))
            pose_differences.append(float(np.max(np.abs(np.asarray(rows['official'][-1]['T'])-
                                                          np.asarray(rows['current'][-1]['T'])))))
            aligned = results['official_aligned']
            ic = np.lexsort(aligned[0].T[::-1])
            if not np.array_equal(aligned[0][ic],b[0][ib]):
                raise ValueError('aligned official/current output coordinate mismatch')
            aligned_differences.append(float(np.max(np.abs(aligned[1][ic]-b[1][ib]))))
            aligned_pose_differences.append(float(np.max(np.abs(np.asarray(rows['official_aligned'][-1]['T'])-
                                                                  np.asarray(rows['current'][-1]['T'])))))
        profiler_table = None
        if args.profile:
            with torch.autograd.profiler.profile(use_cuda=True,record_shapes=False,profile_memory=False) as prof:
                solve('official',specs[:1])
            profiler_table = prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=20)
        summary = dict(scope='official-network single process; no V7/SAM/GT; strict FP32; raw input each solve',
                       pid=os.getpid(),
                       official_commit=snapshot['commit'],gpu=args.gpu,cpu_threads=args.threads,batch_size=args.batch,
                       checkpoint=base['base_checkpoint'],checkpoint_sha256=base['base_checkpoint_sha256'],
                       frames=len(specs),memory_preflight_estimated_bytes=int(estimated),
                       benchmark_source_sha256=E.sha_file(SOURCE_FILE),gpu_before=before_gpu,gpu_after=gpu_snapshot(),
                       peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
                       max_official_current_prediction_difference=max(feature_differences),
                       max_official_current_pose_element_difference=max(pose_differences),
                       max_aligned_official_current_prediction_difference=max(aligned_differences),
                       max_aligned_official_current_pose_element_difference=max(aligned_pose_differences),
                       model_parameters=sum(p.numel() for p in models['official'].parameters()))
        for name,data in rows.items():
            summary[name] = dict(
                synchronized_compute=stats([r['synchronized_compute_seconds'] for r in data]),
                synchronized_compute_amortized=stats([r['synchronized_compute_seconds']/r['batch_size'] for r in data]),
                official_code_boundary_amortized=stats([r['official_code_boundary_seconds']/r['batch_size'] for r in data]),
                input_to_batch_T=stats([r['input_to_T_seconds'] for r in data]),
                preparation_and_transfer_amortized=stats([r['preparation_and_transfer_seconds']/r['batch_size'] for r in data]))
        result = dict(summary=summary,rows=rows,profiler=profiler_table)
        with E.output_path(args.out).open('x') as stream:
            json.dump(result,stream,indent=2)
        print(json.dumps(summary),flush=True)
    finally:
        guard.close()


if __name__ == '__main__':
    main()
