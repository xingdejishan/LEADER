import json
P = '/home/zhangcy/projects/e2e-moe-20261020-v1/results/vb_ctrl-no-rotation-shard0.jsonl'
for line in open(P):
    r = json.loads(line)
    print(r['key'], 'ratio', r.get('ratio'), 'chosen', r.get('chosen'), 'visual', r.get('visual'),
          'gated', r.get('visual_gated'), 'vsec', r.get('visual_seconds'),
          'same_pose', r.get('pose') == r.get('route3_pose'))
Q = '/home/zhangcy/projects/e2e-moe-20261020-v1/results/vb_v2-no-rotation-shard0.jsonl'
print('--- v2 rows')
for line in open(Q):
    r = json.loads(line)
    print(r['key'], 'ratio', round(r.get('ratio') or -1, 4), 'chosen', r.get('chosen'),
          'vsec', r.get('visual_seconds'), 'score', r.get('visual_score'),
          'c3', None if r.get('c_route3') is None else round(r['c_route3'], 4),
          'cv', None if r.get('c_visual') is None else round(r['c_visual'], 4),
          'dcost', None if r.get('dcost') is None else round(r['dcost'], 4),
          'changed', r.get('pose') != r.get('route3_pose'))
