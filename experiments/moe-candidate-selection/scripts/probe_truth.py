import json, os, glob
import numpy as np

DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001'
print(os.listdir(DATA)[:20])
tr = os.path.join(DATA, 'evaluator-only-truth')
print('truth dir exists', os.path.isdir(tr), len(glob.glob(os.path.join(tr, '*.json'))))
files = sorted(glob.glob(os.path.join(tr, '*.json')))
print([os.path.basename(f) for f in files])
if files:
    obj = json.load(open(files[0]))
    print(type(obj), list(obj)[:10] if isinstance(obj, dict) else len(obj))
    if isinstance(obj, dict):
        for k, v in list(obj.items())[:3]:
            print('  ', k, type(v), (len(v) if hasattr(v, '__len__') else v))
        # print a sample
        k0 = list(obj)[0]
        print(json.dumps({k0: obj[k0]}, default=str)[:800])

P = '/home/zhangcy/projects/e2e-moe-20261020-v1'
with open(os.path.join(P, 'results/fullb-shard1.jsonl')) as fh:
    line = fh.readline()
    print('\njsonl keys:', list(json.loads(line).keys()))
    print(line[:600])
