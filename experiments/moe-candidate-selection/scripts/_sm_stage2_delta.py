import json
import numpy as np

BASE = '/data2/zhangcy/LEADER-v7-scratch-eval1-fast-20261008-v2'
OURS = {
    '2012-02-12/1329068220036076': np.array([-54.904846, -190.958023, 3.485886]),
    '2012-02-12/1329068221836354': np.array([-52.858322, -192.565994, 3.654770]),
}
with np.load(BASE + '/predictions-000000.npz', allow_pickle=True) as z:
    keys = [str(x) for x in z['frame_keys']]
    poses = np.asarray(z['poses'], float)
    rounds = np.asarray(z['rounds'], int)
with np.load('/home/zhangcy/projects/e2e-moe-20261020-v1/inputs/initial.npz', allow_pickle=False) as a:
    ikeys = [str(x) for x in a['keys']]
    iposes = np.asarray(a['poses'], float)

print('baseline chunk poses[998][:3,3] =', poses[keys.index('2012-02-12/1329068220036076')][:3, 3])
print('initial.npz   poses[998][:3,3] =', iposes[0][:3, 3])
print('initial.npz == baseline chunk (bitwise, all frames):',
      bool(np.array_equal(iposes, poses[[keys.index(k) for k in ikeys]])))
for key, ours in OURS.items():
    i = keys.index(key)
    b = poses[i]
    r = iposes[ikeys.index(key)]
    print(key)
    print('   baseline_t =', np.round(b[:3, 3], 6), 'rounds', int(rounds[i]))
    print('   our_t      =', np.round(ours, 6))
    print('   |ours-baseline| = %.6f m   |ours-initial.npz| = %.6f m'
          % (np.linalg.norm(ours - b[:3, 3]), np.linalg.norm(ours - r[:3, 3])))
