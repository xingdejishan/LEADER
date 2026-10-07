import argparse
import json
from pathlib import Path
import portable_runtime as r

parser=argparse.ArgumentParser()
parser.add_argument('--train-manifest',type=Path,required=True)
parser.add_argument('--test-manifest',type=Path,required=True)
parser.add_argument('--initial-checkpoint',type=Path,required=True)
parser.add_argument('--evaluator-gt',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
protocol=r.read(r.ROOT/'protocol.json')
assert r.sha(args.initial_checkpoint)==protocol['initial_checkpoint_sha256']
for split,count in [('train',107339),('test',104732)]:
    manifest=r.read(getattr(args,split+'_manifest'))
    assert manifest['split']==split and len(manifest['records'])==count
    assert manifest['frozen_LEADER_sha256']==r.read(r.ROOT/'model.json')['base_checkpoint_sha256']
required=['hash_fused.so','mixed_gradient_terms.so','mixed_backward_complete_v1.so','target_mixed_backward_v1.so']
assert (r.V5/'runtime'/required[0]).is_file(),'Compile inherited V5 hash kernel first'
assert all((r.RUNTIME/name).is_file() for name in required[1:]),'Compile V7 CUDA kernels first'
settings=dict(train_manifest=str(args.train_manifest.resolve()),test_manifest=str(args.test_manifest.resolve()),initial_checkpoint=str(args.initial_checkpoint.resolve()),evaluator_gt=str(args.evaluator_gt.resolve()),output=str(args.output.resolve()))
local=r.ROOT/'local'
local.mkdir(exist_ok=True)
assert not (local/'paths.json').exists() and not (args.output/'plan.json').exists()
args.output.mkdir(parents=True,exist_ok=True)
sources={str(path.relative_to(r.ROOT)).replace('\\','/'):r.sha(path) for path in r.ROOT.rglob('*') if path.is_file() and path.suffix in ['.py','.cu','.json'] and 'local' not in path.parts and path.name!='release_manifest.json'}
protocol.update(initial_checkpoint=str(args.initial_checkpoint.resolve()),sources=sources,train_manifest_sha256=r.sha(args.train_manifest),test_manifest_sha256=r.sha(args.test_manifest),portable_publication=True)
(local/'paths.json').write_text(json.dumps(settings,indent=2),encoding='utf-8')
r.STORE=args.output.resolve()
r.save('plan.json',protocol)
r.save('derivative-execution-amendment-v6.json',dict(scientific_plan_sha256=r.sha(r.STORE/'plan.json'),sources=sources,portable_publication=True))
proof=r.read(r.ROOT/'results/training-step-parity.json')
assert proof['parity_passed']
r.save('batch-full-step-scheduled7-v1/report.json',proof)
print(json.dumps(dict(configured=True,training_started=False,portable_GPU_validation_complete=False)))
