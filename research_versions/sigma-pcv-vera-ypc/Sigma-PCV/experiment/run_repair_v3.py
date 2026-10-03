import hashlib
import json
from pathlib import Path
import sys

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'source-v2'))
assert sys.prefix=='/home/zhangcy/.conda/envs/leader'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
policy_path=root/'execution-repair-v3.json'
policy=json.loads(policy_path.read_text())
assert all(sha(root/name)==digest for name,digest in policy['execution_sha256'].items())
assert sha(root/'config-v2.json')==policy['config_sha256']
config=json.loads((root/'config-v2.json').read_text())
store=Path(config['store'])
policy_sha=sha(policy_path)
if '--phase' in sys.argv:
    import tools.run_sigma as run
    original_dataset=run.make_dataset
    def dataset_with_canonical_keys(*args,**kwargs):
        dataset=original_dataset(*args,**kwargs)
        before=dataset.keys
        dataset.keys=list(before)
        assert tuple(dataset.keys)==tuple(before)
        return dataset
    run.make_dataset=dataset_with_canonical_keys
    original_save=run.save
    def save_with_execution(*args,**kwargs):
        return original_save(*args,**kwargs,execution_repair_sha256=policy_sha)
    run.save=save_with_execution
    phase=sys.argv[sys.argv.index('--phase')+1]
    run.main()
    if phase=='predict':
        path=store/'prediction_freeze.json'
        record=json.loads(path.read_text())
        record['execution_repair_sha256']=policy_sha
        record['execution_source_sha256']=policy['execution_sha256']
        path.write_text(json.dumps(record,indent=2))
else:
    import tools.score_sigma as score
    freeze=json.loads((store/'prediction_freeze.json').read_text())
    assert freeze['execution_repair_sha256']==policy_sha
    score.main()
    path=store/'evaluation.json'
    record=json.loads(path.read_text())
    record['execution_repair_sha256']=policy_sha
    path.write_text(json.dumps(record,indent=2))
