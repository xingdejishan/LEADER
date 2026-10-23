import numpy as np

BASE = '/data2/zhangcy/LEADER-v7-scratch-eval1-fast-20261008-v2'
OURS = {
    '2012-02-12/1329068220036076': '4a3f2df7b3e26571f9e302dd5e8f4a00083beee4ccb7624dc56b54af566764d4',
    '2012-02-12/1329068221836354': '2f09d46d7f4d7186baa68469a359c36cc7ef0b06c5279048405908a470863c0e',
}
P = BASE + '/predictions-000000.npz'
with np.load(P, allow_pickle=True) as z:
    print('chunk fields =', z.files)
    for name in z.files:
        print('  ', name, z[name].shape if hasattr(z[name], 'shape') else type(z[name]))
    keys = [str(x) for x in z['frame_keys']] if 'frame_keys' in z.files else None
    shas = [str(x) for x in z['shared_input_sha256']] if 'shared_input_sha256' in z.files else None
print('chunk n =', None if keys is None else len(keys))
if keys is not None and shas is not None:
    for key, ours in OURS.items():
        if key in keys:
            i = keys.index(key)
            print(key, 'baseline', shas[i], 'MATCH', shas[i] == ours)
        else:
            print(key, '-> absent from chunk')
