import json, os, sys
from pathlib import Path
S = Path('/data2/zhangcy/sp-lightglue-hard1502-20261013-v1')
lib = json.loads((S/'library.json').read_text())
print('library keys:', list(lib.keys()))
print('n_refs:', len(lib['references']))
print('ref0 keys:', list(lib['references'][0].keys()))
r = lib['references'][0]
for k,v in r.items():
    if not isinstance(v,(list,dict)):
        print('  ', k, '=', v)
    else:
        print('  ', k, type(v).__name__, (len(v) if hasattr(v,'__len__') else ''))
print('T_BC:', lib['T_BC'])
for name in os.listdir(S):
    if name.endswith('.npy') or name.endswith('.npz') or name.endswith('.json'):
        print('FILE', name, os.path.getsize(S/name))
for d in ('ref-projection','map-cloud'):
    p = S/d
    if p.is_dir():
        fs = sorted(os.listdir(p))
        print(d, len(fs), fs[:2], os.path.getsize(p/fs[0]) if fs else None)
    else:
        print(d, 'MISSING')
q = json.loads((S/'queries-provider.json').read_text())['queries']
print('queries:', len(q), 'q0 keys:', list(q[0].keys()))
print('refs per query sample:', [len(x['pair_references']) for x in q[:5]])
