import json

r = json.load(open('/data2/zhangcy/v7-alternating-e7-full-20261018-v6/report.json'))
print('top-level keys:', sorted(r.keys()))
for key in ('frames', 'delta_alternating_minus_V7', 'limitations', 'reference',
            'all_shared_inputs_equal', 'per_date'):
    if key in r:
        print('\n%s:' % key)
        print(json.dumps(r[key], ensure_ascii=False)[:900])
st = r.get('statistics') or {}
for name, v in st.items():
    print('\n[%s] %s' % (name, json.dumps(v, ensure_ascii=False)[:500]))
