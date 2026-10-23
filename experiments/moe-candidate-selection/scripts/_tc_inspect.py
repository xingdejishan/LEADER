import os
P = '/home/zhangcy/projects/e2e-moe-20261020-v1'
for root, dirs, files in os.walk(os.path.join(P, 'stages')):
    dirs[:] = [d for d in dirs if not d.startswith('__')]
    for f in sorted(files):
        if f.endswith('.py'):
            p = os.path.join(root, f)
            print(p, os.path.getsize(p))
print('---- inputs ----')
print(sorted(os.listdir(os.path.join(P, 'inputs')))[:40])
print('---- results ----')
print(sorted(os.listdir(os.path.join(P, 'results')))[:60])
