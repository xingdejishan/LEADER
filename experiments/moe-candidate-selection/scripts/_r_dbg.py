import json, os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[k] = '1'
sys.path.insert(0, '/home/zhangcy/projects/e2e-moe-20261020-v1')
import numpy as np
import visual_branch as VB

Q = {q['key']: q for q in json.loads(open('/data2/zhangcy/sp-lightglue-hard1502-20261013-v1/queries-provider.json').read())['queries']}
key = '2012-03-31/1333219399630286'
q = Q[key]
frozen = json.loads(open('/data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/verify-frozen_nopruneoptREF/%06d.json' % 955).read())
match = json.loads(open('/data2/zhangcy/sp-lightglue-hard1502-20261013-v1/prov/match-frozen_nopruneopt/000955.json').read())
print('frozen pair_references', match['pair_references'], 'query provider refs', q['pair_references'])
print('frozen stats: full_candidates', match['full_candidates'], 'capped', match['capped_candidates'],
      'hypotheses', [(h['channel'], h['unique_query_inliers'], h['qualified']) for h in match['hypotheses']])
print('frozen per_pair', match['per_pair'])

b = VB.VisualBranch(device='cuda', verbose=True, fast_pnp=True)
refs, retr = b.retrieve(q['image_path'], pose_prior=np.asarray(frozen['poses']['baseline_V7']['pose']))
print('live retrieval', refs)
print('live retrieval detail', retr)
q_model, cands, per_pair, batched = b._match_frame(q['image_path'], refs)
print('batched', batched, 'candidates', len(cands), 'per_pair', per_pair)
print('q keypoints', len(q_model))
print('capped', len(b._cap(cands)))
by_ref = {}
for c in cands:
    e = by_ref.setdefault(c[2], ([], []))
    e[0].append(c[0]); e[1].append(c[3])
by_ref = {ri: (np.array(a), np.array(d)) for ri, (a, d) in by_ref.items()}
acc, calls, capped = b._hypotheses(q_model, cands, by_ref)
print('pnp calls', calls, 'accepted', len(acc))
for h in acc:
    print('  ', h['channel'], h['unique_query_inliers'], h['coverage_cells'],
          h['inlier_ratio'], h['qualified'], h.get('consensus_reference_count'), h['reference_ids'],
          np.round(np.asarray(h['pose'])[:3, 3], 2))
# also full hypothesis diagnostics: recompute without the qualified filter
print('per-ref bound detail:')
for ri, (ids, aid) in by_ref.items():
    table = b._projection(b.references[ri])
    sol = b.solve(table['world'][aid], b.C.model_to_raw(b.matrix, q_model[ids]), ids)
    print('   ref', ri, 'bound', len(ids), '->', None if sol is None else
          (sol['unique_query_inliers'], sol['coverage_cells'], round(sol['inlier_ratio'], 3), sol['qualified']))
