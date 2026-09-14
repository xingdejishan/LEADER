import json
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import torch
from types import SimpleNamespace
import experiment as e

OUT = Path(__file__).resolve().parent / 'results/reachability'
TAU = 1e-7
PAD = 1e-10


def norm(a):
    a = np.asarray(a, dtype=np.float64)
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def bounds(x, a, delta, w, rho):
    assert np.isfinite(delta).all() and np.isfinite(w).all()
    assert np.linalg.norm(delta) <= rho + 1e-15
    assert w.min() >= 0 and abs(w.sum()-1) < 1e-14
    avg = w @ a
    return float(np.min(a @ (x+delta))-PAD), float(x @ avg+rho*np.linalg.norm(avg)+PAD)


def solve(x, a, rho):
    b = a @ x
    w = np.eye(len(a))[b.argmin()]
    delta = np.zeros(512)
    metadata = {'status': 'locked', 'primal_violation': 0., 'dual_violation': 0.}
    if rho:
        q, _ = np.linalg.qr(a.T, mode='reduced')
        h = a @ q
        n = q.shape[1]
        start = np.r_[np.zeros(n), b.min()]
        result = minimize(lambda z: -z[-1], start, jac=lambda z: np.r_[np.zeros(n), -1.],
            constraints=[{'type':'ineq', 'fun':lambda z:b+h@z[:-1]-z[-1], 'jac':lambda z:np.c_[h,-np.ones(len(a))]},
                         {'type':'ineq', 'fun':lambda z:rho*rho-z[:-1]@z[:-1], 'jac':lambda z:np.r_[-2*z[:-1],0.]}],
            method='SLSQP', options={'ftol':1e-12,'maxiter':500})
        raw = q @ result.x[:-1]
        delta = raw * min(1., rho*(1-1e-12)/max(np.linalg.norm(raw),1e-300))
        def objective(v):
            av = v @ a
            return b @ v + rho*np.linalg.norm(av)
        def gradient(v):
            av = v @ a
            return b + rho*(a@av)/max(np.linalg.norm(av),1e-300)
        dual = minimize(objective, np.full(len(a),1/len(a)), jac=gradient,
            bounds=[(0,1)]*len(a), constraints=[{'type':'eq','fun':lambda v:v.sum()-1,'jac':lambda v:np.ones(len(a))}],
            method='SLSQP', options={'ftol':1e-12,'maxiter':500})
        w = np.maximum(dual.x,0)
        w /= w.sum()
        metadata = dict(status=str(result.message), dual_status=str(dual.message), iterations=int(result.nit), dual_iterations=int(dual.nit),
            primal_violation=float(max(0,np.linalg.norm(raw)-rho,-np.min(b+a@raw-result.x[-1]))),
            dual_violation=float(max(0,-dual.x.min(),abs(dual.x.sum()-1))),
            raw_primal_objective=float(-result.fun),raw_dual_objective=float(dual.fun),raw_gap=float(dual.fun+result.fun))
    lo, hi = bounds(x,a,delta,w,rho)
    assert lo <= hi+1e-12
    return delta,w,lo,hi,metadata


def classify(ln,un,lf,uf):
    if lf > TAU:
        return 'reachable'
    if un < -TAU:
        return 'negative_blocked'
    if ln > TAU and uf < -TAU:
        return 'gray_blocked'
    return 'unresolved'


def main():
    OUT.mkdir(exist_ok=True)
    protocol = dict(rho=.05,tau=TAU,bound_padding=PAD,precision='float64; unit vectors reconstructed from cached features',
        labels='P<=0.5m,N>=2m,G otherwise; existing masks unchanged',
        population='All original ambiguous development queries retained; solve visual-P and LiDAR-N or LiDAR-G separately',
        ties='Scores within tau of maximum: earliest fixed candidate index; original unchanged query retains candidate zero; margin within tau recorded separately',
        solver='Independent primal and dual SLSQP, ftol1e-12 maxiter500; repaired feasible witnesses determine classification, never status',
        correction='Strict positive-vs-all-nonpositive margin >tau required for certified actual correction; gray and ties separate',
        references='Existing 578 fit anchors and fixed16 candidate indices, no retrieval',source_protocol_sha256=e.run.digest(e.OUT/'protocol.json'))
    e.run.save_json(OUT/'protocol.json',protocol)
    torch.set_num_threads(4)
    rows = json.loads((e.OUT/'protocol.json').read_text())['rows']
    anchors, images = [], []
    for r in rows:
        if r['role'] != 'fit':
            continue
        with np.load(e.CACHE/'lidar'/(r['frame_id']+'.npz')) as l, np.load(e.CACHE/'visual_raw'/(r['frame_id']+'.npz')) as v:
            anchors.append(l['features'][v['valid']]); images.append(v['image'][v['valid']])
    anchors,images = norm(np.concatenate(anchors)), norm(np.concatenate(images))
    decoder = e.run.load_leader(SimpleNamespace(checkpoint=e.run.WORKSPACE/'research/image_gate_checkpoint')).decoder
    heads = {}
    for seed in e.SEEDS:
        for arm in ['aligned','shuffled']:
            name = f'{arm}_{seed}'
            head = e.AnchoredFusion().cuda().eval()
            head.load_state_dict(torch.load(e.OUT/name/'best.pt'))
            heads[name] = (head,arm,seed)
    records, certificates, witnesses = [], [], {}
    for frame,r in enumerate([r for r in rows if r['role']=='development']):
        d = e.load_data(decoder,[r])[0]
        path = e.OUT/'candidates'/(r['frame_id']+'.npz')
        with np.load(path) as c:
            indices,candidates,p,n,gap = [c[k] for k in ['indices','candidates','positive','negative','gap']]
        ambiguous = p.any(1)&n.any(1)&(gap<=.02)
        x = norm(d['f'].cpu().numpy()[indices])
        vi = norm(d['image'].cpu().numpy()[indices])
        visual_scores = np.einsum('nd,nkd->nk',vi,images[candidates])
        visual_choice = (visual_scores >= visual_scores.max(1,keepdims=True)-TAU).argmax(1)
        actual = {}
        with torch.no_grad():
            for name,(head,arm,seed) in heads.items():
                fused = head(d['f'],e.visual(d,arm,seed),d['editable']).cpu().numpy()[indices]
                scores = np.einsum('nd,nkd->nk',norm(fused),anchors[candidates])
                actual[name] = scores
        for i in np.flatnonzero(ambiguous):
            labels = np.where(p[i],'P',np.where(n[i],'N','G'))
            rec = dict(frame_id=r['frame_id'],date=r['session_id'],voxel=int(indices[i]),candidate_row=int(i),
                original=str(labels[0]),visual=str(labels[visual_choice[i]]),protected=bool(d['protected'][indices[i]]),models={})
            for name,scores in actual.items():
                s = scores[i]
                choice = int(np.flatnonzero(s>=s.max()-TAU)[0])
                margin = float(s[p[i]].max()-s[~p[i]].max())
                rec['models'][name] = dict(choice=choice,label=str(labels[choice]),positive_margin=margin,strict_correct=margin>TAU,
                    boundary=abs(margin)<=TAU)
            if rec['original'] in ['N','G'] and rec['visual']=='P':
                k = anchors[candidates[i]]
                rho = 0. if rec['protected'] else .05
                levels = {}
                for level,comp in [('N',n[i]),('NG',~p[i])]:
                    lows, highs = [], []
                    for pos in np.flatnonzero(p[i]):
                        a = k[pos]-k[comp]
                        delta,w,lo,hi,meta = solve(x[i],a,rho)
                        key = str(len(certificates))
                        witnesses['d'+key]=delta; witnesses['w'+key]=w
                        certificates.append(dict(key=key,record=len(records),positive=int(pos),competitors=np.flatnonzero(comp).tolist(),level=level,
                            lower=lo,upper=hi,rho=rho,**meta))
                        lows.append(lo); highs.append(hi)
                    levels[level]=[max(lows),max(highs)]
                rec['bounds']=levels
                rec['classification']=classify(*levels['N'],*levels['NG'])
                witnesses['x'+str(len(records))]=x[i]
                witnesses['k'+str(len(records))]=k
            records.append(rec)
        if frame%20==0:
            print('frames',frame+1,'queries',len(records),'certificates',len(certificates),flush=True)
    e.run.save_json(OUT/'records.json',records)
    e.run.save_json(OUT/'certificates.json',certificates)
    np.savez_compressed(OUT/'witnesses.npz',**witnesses)
    e.run.save_json(OUT/'inputs.json',dict(weights={name:e.run.digest(e.OUT/name/'best.pt') for name in heads},
        candidates={r['frame_id']:e.run.digest(e.OUT/'candidates'/(r['frame_id']+'.npz')) for r in rows if r['role']=='development'}))
    print('COMPLETE',flush=True)


if __name__=='__main__':
    main()
