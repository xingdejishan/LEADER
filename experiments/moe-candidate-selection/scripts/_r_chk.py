import json
V = '/data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/verify-frozen_nopruneoptREF/%06d.json'
for idx, name in ((1429, 'candidate_04'), (957, 'candidate_06'), (955, 'candidate_05'), (1428, 'candidate_10')):
    rec = json.loads(open(V % idx).read())['poses']
    row = rec[name]
    print(idx, name, 'own_excluded', row.get('own_excluded'), 'provenance', row.get('provenance'),
          'channel', row.get('channel'), 'score', row['fair']['geometry_score'],
          'excluded', row['fair'].get('excluded'), 'map_refs', row['fair'].get('map_references'))
    for other, orow in rec.items():
        if other.startswith('candidate_') or other == 'baseline_V7':
            s = (orow.get('fair') or {}).get('geometry_score')
            print('    ', other, 'score', s, 'own', orow.get('own_excluded'),
                  'channel', orow.get('channel'), orow.get('provenance'))
