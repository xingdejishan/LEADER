import json, os, glob
P = '/home/zhangcy/projects/e2e-three-route-20261020-v1/results'
for f in sorted(glob.glob(os.path.join(P, '*.json*')))[:6]:
    print(os.path.basename(f), os.path.getsize(f))
    try:
        if f.endswith('.json'):
            d = json.load(open(f))
            print('  type', type(d).__name__, 'keys', list(d)[:8] if isinstance(d, dict) else len(d))
        else:
            with open(f) as fh:
                line = fh.readline()
            r = json.loads(line)
            print('  record keys:', list(r.keys()))
            for k in ('leader', 'refine', 'visual'):
                if k in r and isinstance(r[k], dict):
                    print('   ', k, list(r[k].keys())[:8])
    except Exception as e:
        print('  err', e)
