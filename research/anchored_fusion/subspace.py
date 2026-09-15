import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import torch
import experiment as e
from reachability import TAU, PAD, classify

HERE = Path(__file__).resolve().parent
SOURCE = HERE/'results/reachability'
OUT = HERE/'results/subspace'


def solve(x, a, q, rho):
    b, h = a@x, a@q
    c = np.zeros(q.shape[1])
    w = np.eye(len(a))[b.argmin()]
    meta = dict(status='locked',dual_status='locked',primal_violation=0.,dual_violation=0.,raw_gap=0.)
    if rho:
        n = len(c)
        result = minimize(lambda z:-z[-1],np.r_[c,b.min()],jac=lambda z:np.r_[np.zeros(n),-1.],
            constraints=[{'type':'ineq','fun':lambda z:b+h@z[:-1]-z[-1],'jac':lambda z:np.c_[h,-np.ones(len(a))]},
                         {'type':'ineq','fun':lambda z:rho*rho-z[:-1]@z[:-1],'jac':lambda z:np.r_[-2*z[:-1],0.]}],
            method='SLSQP',options={'ftol':1e-12,'maxiter':500})
        raw = result.x[:-1]
        c = raw*min(1.,rho*(1-1e-12)/max(np.linalg.norm(q@raw),np.linalg.norm(raw),1e-300))
        def fun(v):
            return b@v+rho*np.linalg.norm(v@h)
        def jac(v):
            hv=v@h
            return b+rho*(h@hv)/max(np.linalg.norm(hv),1e-300)
        dual=minimize(fun,np.full(len(a),1/len(a)),jac=jac,bounds=[(0,1)]*len(a),
            constraints=[{'type':'eq','fun':lambda v:v.sum()-1,'jac':lambda v:np.ones(len(a))}],
            method='SLSQP',options={'ftol':1e-12,'maxiter':500})
        w=np.maximum(dual.x,0); w/=w.sum()
        meta=dict(status=str(result.message),dual_status=str(dual.message),iterations=int(result.nit),dual_iterations=int(dual.nit),
            primal_violation=float(max(0,np.linalg.norm(raw)-rho,-np.min(b+h@raw-result.x[-1]))),
            dual_violation=float(max(0,-dual.x.min(),abs(dual.x.sum()-1))),raw_primal=float(-result.fun),raw_dual=float(dual.fun),raw_gap=float(dual.fun+result.fun))
    avg=w@a
    lo=float(np.min(a@(x+q@c))-PAD)
    hi=float(x@avg+rho*np.linalg.norm(q.T@avg)+PAD)
    assert lo<=hi+1e-12
    return c,w,lo,hi,meta


def main():
    OUT.mkdir(exist_ok=True)
    protocol=dict(rho=.05,tau=TAU,pad=PAD,source_commit='f6b63e9',
        source_hashes={f:e.run.digest(SOURCE/f) for f in ['protocol.json','records.json','candidate_labels.json','witnesses.npz','inputs.json']},
        structure='Linear640x32,ReLU,Linear32x512 with bias, scalar norm clipping, additive residual; no channel multiplication after output',
        basis='float64 reduced QR of [W2,b2], retain all 33 columns, no PCA truncation',
        basis_acceptance='min singular value >1e-12*max; spectral orthogonality and relative reconstruction errors <1e-12; otherwise unresolved',
        solver='primal and dual SLSQP independently, ftol1e-12 maxiter500, status does not classify',
        population='All 454 explicit LiDAR-N visual-P and 666 LiDAR-G visual-P; primary nested subset is original161 full-space reachable',
        labels_and_ties='Exact prior candidate labels, fixed order, tau1e-7; G stays uncertain; N and N+G optimized separately',
        actual_check='Recompute deployed float32 head, normalize fused-minus-original by original norm, project into Q and repair radius; all prior strict corrections must remain positive-margin legal witnesses',
        actual_roundoff='Float32 fused-minus-original may have tiny out-of-span roundoff; record it, require norm excess and span residual <=1e-6, then use feasible projected witness',
        scope='No training, no new model, no pose evaluation; GT-free coefficients and ReLU feasibility are not imposed; subspace relaxation only')
    e.run.save_json(OUT/'protocol.json',protocol)
    torch.set_num_threads(4)
    records=json.loads((SOURCE/'records.json').read_text())
    labels=json.loads((SOURCE/'candidate_labels.json').read_text())
    expected=json.loads((SOURCE/'inputs.json').read_text())['weights']
    selected=[(i,r) for i,r in enumerate(records) if 'bounds' in r]
    with np.load(SOURCE/'witnesses.npz') as z:
        vectors={i:(z['x'+str(i)],z['k'+str(i)]) for i,r in selected}
    for name,weight_hash in expected.items():
        folder=OUT/name; folder.mkdir(exist_ok=True)
        if (folder/'records.json').exists():
            continue
        weight=e.OUT/name/'best.pt'
        assert e.run.digest(weight)==weight_hash
        state=torch.load(weight,map_location='cpu')
        matrix=np.column_stack([state['net.2.weight'].numpy(),state['net.2.bias'].numpy()]).astype(np.float64)
        q,_=np.linalg.qr(matrix,mode='reduced')
        sv=np.linalg.svd(matrix,compute_uv=False)
        orth=float(np.linalg.norm(q.T@q-np.eye(33),2))
        reconstruction=float(np.linalg.norm(matrix-q@(q.T@matrix),2)/np.linalg.norm(matrix,2))
        basis_ok=bool(sv[-1]>1e-12*sv[0] and orth<1e-12 and reconstruction<1e-12)
        e.run.save_json(folder/'basis.json',dict(weight_sha256=weight_hash,rank=33,sv=sv.tolist(),orthogonality=orth,reconstruction=reconstruction,accepted=basis_ok))
        np.savez_compressed(folder/'basis.npz',Q=q,matrix=matrix)
        head=e.AnchoredFusion().cuda().eval(); head.load_state_dict(state)
        arm,seed=name.split('_'); seed=int(seed)
        certs,results,saved=[],[],{}
        current=None
        for offset,(rid,r) in enumerate(selected):
            if current!=r['frame_id']:
                current=r['frame_id']
                with np.load(e.CACHE/'lidar'/(current+'.npz')) as l,np.load(e.CACHE/'visual_raw'/(current+'.npz')) as v:
                    f=torch.tensor(l['features'],device='cuda')
                    image=torch.tensor(v['image'],device='cuda')
                    valid=torch.tensor(v['valid'],device='cuda')
                if arm=='shuffled':
                    ids=torch.where(valid)[0]
                    order=torch.arange(len(f),device='cuda')
                    rng=np.random.default_rng(seed+int(current)%1000000007)
                    order[ids]=ids[torch.as_tensor(rng.permutation(len(ids)),device='cuda')]
                    image=image[order]
                with torch.no_grad():
                    fused=head(f,image,valid)
                f=f.cpu().numpy(); fused=fused.cpu().numpy()
            x,k=vectors[rid]
            lab=np.array(labels[current][r['candidate_row']])
            rho=0. if r['protected'] else .05
            voxel=r['voxel']
            original=f[voxel].astype(float)
            actual_f=original if r['protected'] else fused[voxel].astype(float)
            actual=(actual_f-original)/np.linalg.norm(original)
            scores=k@(actual_f/np.linalg.norm(actual_f))
            margin=float(scores[lab=='P'].max()-scores[lab!='P'].max())
            assert (margin>TAU)==r['models'][name]['strict_correct']
            assert abs(margin-r['models'][name]['positive_margin'])<1e-10
            c_actual=q.T@actual
            span_error=float(np.linalg.norm(actual-q@c_actual))
            excess=float(max(0,np.linalg.norm(actual)-rho))
            assert span_error<=1e-6 and excess<=1e-6
            c_actual*=min(1.,rho*(1-1e-12)/max(np.linalg.norm(c_actual),np.linalg.norm(q@c_actual),1e-300))
            witness_scores=k@(x+q@c_actual)
            witness_margin=float(witness_scores[lab=='P'].max()-witness_scores[lab!='P'].max()-PAD)
            if margin>TAU:
                assert witness_margin>TAU
            saved['actual'+str(rid)]=actual
            saved['feasible'+str(rid)]=c_actual
            levels={}
            for level,comp in [('N',lab=='N'),('NG',lab!='P')]:
                lows,highs=[],[]
                for p in np.flatnonzero(lab=='P'):
                    a=k[p]-k[comp]
                    c,w,lo,hi,meta=solve(x,a,q,rho)
                    actual_lo=float(np.min(a@(x+q@c_actual))-PAD)
                    if actual_lo>lo:
                        c,lo=c_actual.copy(),actual_lo
                    key=str(len(certs)); saved['c'+key]=c; saved['w'+key]=w
                    certs.append(dict(key=key,record=rid,positive=int(p),competitors=np.flatnonzero(comp).tolist(),level=level,rho=rho,lower=lo,upper=hi,**meta))
                    lows.append(lo); highs.append(hi)
                levels[level]=[max(lows),max(highs)]
            result=dict(record=rid,frame_id=current,date=r['date'],original=r['original'],protected=r['protected'],full_classification=r['classification'],
                bounds=levels,classification=classify(*levels['N'],*levels['NG']) if basis_ok else 'basis_unresolved',
                actual_correct=margin>TAU,actual_margin=margin,actual_span_error=span_error,actual_norm_excess=excess,feasible_actual_margin=witness_margin)
            if result['actual_correct'] and basis_ok:
                assert result['classification']=='reachable'
            results.append(result)
            if offset%250==0:
                print(name,offset,len(selected),flush=True)
        np.savez_compressed(folder/'witnesses.npz',**saved)
        e.run.save_json(folder/'certificates.json',certs)
        e.run.save_json(folder/'records.json',results)
        print(name,'COMPLETE',flush=True)


if __name__=='__main__':
    main()
