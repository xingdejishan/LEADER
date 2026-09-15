import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json
import argparse
from pathlib import Path
import hashlib
import numpy as np

OUT=Path('/home/zhang/certificate-supervised-fusion')


def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    gamma,pad=protocol['gamma'],protocol['teacher_pad']
    fit={r['frame_id'] for r in protocol['rows'] if r['role']=='fit'}
    files=sorted((OUT/'teacher').glob('*.npz'))
    assert {f.stem for f in files}==fit and len(fit)==578
    hashes={}; total=0; margins=[]; norms=[]
    for frame,path in enumerate(files):
        meta=json.loads(path.with_suffix('.json').read_text())
        with np.load(path) as f:
            arrays={k:f[k] for k in f.files}
        if 'reference' in arrays:
            for check in meta['checks']:
                v=check['voxel']; i=check['candidate_row']
                arrays['x'+str(v)]=arrays['x'][v]
                arrays['k'+str(v)]=arrays['reference'][i]
            arrays.pop('x'); arrays.pop('reference')
            temporary=path.with_suffix('.compact.npz')
            np.savez_compressed(temporary,**arrays); temporary.replace(path)
        kind=arrays['kind']; target=arrays['targets']; p=arrays['positive']; n=arrays['negative']
        assert not ((kind>0)&~arrays['editable']).any()
        assert np.array_equal(kind[arrays['indices']]==1,p[:,0]&arrays['editable'][arrays['indices']])
        assert not target[kind==1].any()
        by_voxel={}
        for check in meta['checks']:
            v=check['voxel']; i=check['candidate_row']; pos=check['positive']; key=check['key']
            assert n[i,0] and p[i,pos] and arrays['editable'][v]
            x,k=arrays['x'+str(v)],arrays['k'+str(v)]
            delta,w=arrays['d'+key],arrays['w'+key]
            a=k[pos]-k[~p[i]]
            assert np.isfinite(delta).all() and np.isfinite(w).all() and w.min()>=0
            lower=float((gamma-a@x)@w-.5*np.linalg.norm(w@a)**2-pad)
            upper=float(.5*delta@delta+pad)
            margin=float(np.min(a@(x+delta))-pad)
            assert abs(lower-check['lower'])<1e-12 and abs(upper-check['upper'])<1e-12 and abs(margin-check['margin'])<1e-12
            if check['status']=='certified':
                assert margin>=gamma and np.linalg.norm(delta)<=.05 and upper-lower<=1e-8
            elif check['status']=='radius_blocked':
                assert lower>.5*.05**2+pad
            by_voxel.setdefault(v,[]).append(check)
            total+=1
        for v,checks in by_voxel.items():
            i=checks[0]['candidate_row']
            assert sorted(c['positive'] for c in checks)==np.flatnonzero(p[i]).tolist()
            valid=[c for c in checks if c['status']=='certified']
            reliable=bool(valid) and all(c['status']!='uncertain' for c in checks)
            if reliable:
                best=min(valid,key=lambda c:(round(float(arrays['d'+c['key']]@arrays['d'+c['key']]),12),c['positive']))
                assert np.array_equal(target[v],arrays['d'+best['key']].astype(np.float32))
                rejected=arrays['selected'][v] and meta['coordinate_deltas'][str(v)]>0
                assert kind[v]==(0 if rejected else 2)
                if kind[v]==2:
                    x,k=arrays['x'+str(v)],arrays['k'+str(v)]
                    margin=float(np.min((k[best['positive']]-k[~p[i]])@(x+target[v].astype(float)))-pad)
                    assert margin>=gamma and np.linalg.norm(target[v].astype(float))<=.05
                    margins.append(margin); norms.append(float(np.linalg.norm(target[v])))
            else:
                assert kind[v]==0
        assert int((kind==2).sum())==meta['kept']
        hashes[path.stem]=hashlib.sha256(path.read_bytes()).hexdigest()
        if frame%100==0:
            print('verified',frame+1,flush=True)
    result=dict(frames=578,certificates=total,kept=len(norms),minimum_teacher_margin=min(margins),maximum_teacher_norm=max(norms),
        mean_teacher_norm=float(np.mean(norms)),teacher_sha256=hashes,protocol_sha256=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest())
    (OUT/'teacher_verified.json').write_text(json.dumps(result,indent=2))
    print({k:v for k,v in result.items() if k!='teacher_sha256'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--folder',type=Path,default=OUT)
    OUT=parser.parse_args().folder
    main()
