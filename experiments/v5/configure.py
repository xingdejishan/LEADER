import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--data',type=Path,required=True)
parser.add_argument('--leader',type=Path,required=True)
parser.add_argument('--geometry-cache',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--raw-gt',type=Path,required=True)
args=parser.parse_args()

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*2**20),b''):digest.update(block)
    return digest.hexdigest()

def save(path,value):
    text=json.dumps(value,indent=2)+'\n'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8'))==value,'Existing local configuration differs; use a separate checkout/output for another run'
    else:path.write_text(text,encoding='utf-8')

data=args.data.resolve()
leader=args.leader.resolve()
cache=args.geometry_cache.resolve()
output=args.output.resolve()
protocol=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
assert sha(data/'catalog.json')==protocol['catalog_sha256']
assert sha(leader)==protocol['frozen_LEADER_sha256']
config=json.loads((ROOT/'config.template.json').read_text(encoding='utf-8'))
for name,digest in config['source_sha256'].items():assert sha(ROOT/'source'/name)==digest,name
config.update(base_checkpoint=str(leader),data_root=str(data/'NCLT'),gt_root=str(data/'evaluator-only-truth'))
local=ROOT/'local'
local.mkdir(exist_ok=True)
save(local/'paths.json',dict(data=str(data),geometry_cache=str(cache),output=str(output),raw_gt=str(args.raw_gt.resolve())))
save(local/'config-v2.json',config)
runtime=ROOT/'runtime'
assert (runtime/'hash_fused.so').is_file(),'Run compile_cuda.py first'
dependencies={str(runtime/name):sha(runtime/name) for name in ('fused_field.py','batched_solver.py','hash_fused.cu','hash_fused.so','full_loader.py','geometry_cache.py','field_index_select.py','backend_runtime.py')}
for date,digest in protocol['public_manifest_sha256'].items():
    assert sha(data/'NCLT'/date/'manifest.json')==digest
    assert sha(data/'sam'/date/'cache_manifest.json')==protocol['sam_manifest_sha256'][date]
plan=dict(protocol,parent_config_sha256=sha(local/'config-v2.json'),dependencies=dependencies,
    sources={name:sha(ROOT/name) for name in ('train_v5.py','predict_v5.py','score_v5.py')},portable_publication=True)
save(ROOT/'v5-training-plan.json',plan)
output.mkdir(parents=True,exist_ok=True)
cache.mkdir(parents=True,exist_ok=True)
print(json.dumps(dict(configured=True,plan_sha256=sha(ROOT/'v5-training-plan.json'),training_started=False)))
