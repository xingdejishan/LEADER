import json
p = '/home/zhangcy/projects/epi-factor-clean-20261015-v2/report.json'
d = json.load(open(p))
def walk(o, prefix=''):
    if isinstance(o, dict):
        for k, v in o.items():
            walk(v, prefix + '/' + str(k))
    elif isinstance(o, list):
        print(prefix, 'list len', len(o))
    else:
        s = str(o)
        if len(s) < 120:
            print('%-58s %s' % (prefix, s))
walk(d)
