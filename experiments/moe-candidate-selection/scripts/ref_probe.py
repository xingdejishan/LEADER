import json
refs = json.load(open('/home/zhangcy/projects/e2e-moe-20261020-v1/inputs/refs.json'))
print('refs:', len(refs))
r = refs[0]
print('fields:', list(r.keys()))
for k, v in r.items():
    print('  %-20s %s' % (k, str(v)[:80]))
