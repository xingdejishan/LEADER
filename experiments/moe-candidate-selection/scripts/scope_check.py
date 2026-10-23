import json, numpy as np
from scipy.spatial.transform import Rotation
import importlib.util

spec = importlib.util.spec_from_file_location(
    'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
su = importlib.util.module_from_spec(spec); spec.loader.exec_module(su)
w = su.load_world()
gt = np.asarray(w['hard_gt'], float); base = np.asarray(w['hard_base'], float)
Rd = np.swapaxes(base[:, :3, :3], -1, -2) @ gt[:, :3, :3]
rv = np.degrees(Rotation.from_matrix(Rd).as_rotvec())
tail = set(int(i) for i in np.flatnonzero(np.abs(rv[:, 2]) > 8))

diag = [json.loads(l) for l in open('/home/zhangcy/projects/e2e-moe-20261020-v1/results/three_route_diag.jsonl')]
adopted = set(i for i, d in enumerate(diag) if d.get('visual'))
gated = set(i for i, d in enumerate(diag) if np.isfinite(d.get('ratio', np.nan)) and d['ratio'] < 0.21)

reach = json.load(open('/home/zhangcy/.dsh_scratch/yaw_reach.json'))
reachable = set(r['i'] for r in reach if r['tag'] == 'TAIL' and r['in_window'] and r['improves'] and r['local_ident'])

print('yaw-tail frames (|yaw|>8, along-track output): %d' % len(tail))
print('  of which already gated (ratio<0.21)        : %d  %s' % (len(tail & gated), sorted(tail & gated)))
print('  of which the visual branch already adopted : %d  %s' % (len(tail & adopted), sorted(tail & adopted)))
print('  left to the along-track branch             : %d  %s' % (len(tail - adopted), sorted(tail - adopted)))
print()
print('reachable by a +-15 yaw sweep (from earlier): %d  %s' % (len(reachable), sorted(reachable)))
print('  ... of which NOT adopted by visual        : %d  %s' % (len(reachable - adopted), sorted(reachable - adopted)))
print()
print('tail frames neither gated nor adopted       : %d  %s' % (len(tail - gated - adopted), sorted(tail - gated - adopted)))
