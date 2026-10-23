import sys
P = '/home/zhangcy/projects/e2e-moe-20261020-v1/stages/fast_search'
for name in ('fast_lib.py', 'pipeline.py', 'adaptive.py'):
    print('=' * 30, name, '=' * 30)
    print(open(P + '/' + name).read())
