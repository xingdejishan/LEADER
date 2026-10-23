import numpy as np

Z = '/data2/zhangcy/LEADER-v7-scratch-eval1-fast-20261008-v2/errors.npz'
OURS = {
    '2012-02-12/1329068220036076': '4a3f2df7b3e26571f9e302dd5e8f4a00083beee4ccb7624dc56b54af566764d4',
    '2012-02-12/1329068221836354': '2f09d46d7f4d7186baa68469a359c36cc7ef0b06c5279048405908a470863c0e',
}
with np.load(Z, allow_pickle=False) as z:
    print('fields =', z.files)
    keys = [str(x) for x in z['frame_keys']]
    if 'shared_input_sha256' in z.files:
        digests = [str(x) for x in z['shared_input_sha256']]
    else:
        digests = None
print('n_baseline =', len(keys), 'n_sha =', None if digests is None else len(digests))
for key, ours in OURS.items():
    if key not in keys:
        print(key, '-> not in frozen baseline table')
        continue
    i = keys.index(key)
    base = digests[i] if digests is not None else '<no sha field>'
    print(key, 'index', i)
    print('   baseline =', base)
    print('   ours     =', ours)
    print('   MATCH    =', base == ours)
