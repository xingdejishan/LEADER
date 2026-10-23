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
match = json.loads(open('/data2/zhangcy/sp-lightglue-hard1502-20261013-v1/prov/match-frozen_nopruneopt/000955.json').read())
print('frozen hypotheses:', [(h['channel'], h['unique_query_inliers'], h['coverage_cells'],
                             round(h['median_reprojection_px'], 2), h['qualified'],
                             h.get('consensus_reference_count')) for h in match['hypotheses']])

b = VB.VisualBranch(device='cuda', fast_pnp=False)
q_model, cands, per_pair, batched = b._match_frame(q['image_path'], match['pair_references'])
by_ref = {}
for c in cands:
    e = by_ref.setdefault(c[2], ([], []))
    e[0].append(c[0]); e[1].append(c[3])
by_ref = {ri: (np.array(a), np.array(d)) for ri, (a, d) in by_ref.items()}
print('RANSAC path per-reference solutions:')
for ri in sorted(by_ref, key=lambda i: -len(by_ref[i][0])):
    ids, aid = by_ref[ri]
    table = b._projection(b.references[ri])
    sol = b.solve(table['world'][aid], b.C.model_to_raw(b.matrix, q_model[ids]), ids)
    print('   ref', ri, 'bound', len(ids), '->', None if sol is None else
          (sol['unique_query_inliers'], sol['coverage_cells'], round(sol['median_reprojection_px'], 2), sol['qualified']))
acc, calls, capped = b._hypotheses(q_model, cands, by_ref)
print('RANSAC accepted', len(acc), [(h['channel'], h['unique_query_inliers'], h.get('consensus_reference_count')) for h in acc])
