from common import *
from refinement_core import refine
import argparse,time
from scipy.spatial import cKDTree
from collections import OrderedDict

parser=argparse.ArgumentParser();parser.add_argument('--shard',type=int,default=0);parser.add_argument('--of',type=int,default=8);args=parser.parse_args();denied=deny_GT();started=time.monotonic()
while not (STORE/'prediction-freeze.json').exists() or not (STORE/'control-verify-completed.json').exists():time.sleep(2)
assert load(STORE/'control-verify-completed.json')['complete'];assert sha(STORE/'predictions.json')==load(STORE/'prediction-freeze.json')['predictions_sha'];rows=load(STORE/'predictions.json')['rows'];queries={q['index']:q for q in load(STORE/'queries.json')['queries']};lib=load(STORE/'library.json')['references'];positions=np.array([r['body_position_world'] for r in lib]);tree=cKDTree(positions);cache=OrderedDict();output=[];SP=Path('/data2/zhangcy/sp-lightglue-hard1502-20261013-v1')
def cloud(i):
 if i in cache:a=cache.pop(i);cache[i]=a;return a
 r=lib[i];a=np.load(STORE/'map-cloud'/('%06d-%s.npz'%(i,r['scan_sha256'][:16])))['world'].astype(np.float64);cache[i]=a
 if len(cache)>48:cache.popitem(last=False)
 return a
for row in rows:
 if row['index']%args.of!=args.shard:continue
 guard(max_rss_gib=12,min_available_gib=32);q=queries[row['index']];base=np.asarray(q['baseline_V7']);p=SP/'verify'/('%06d.json'%q['original_index']) if q['group']=='hard' else STORE/'verify_controls'/('%06d.json'%q['index']);record=load(p)['poses'];c=[x for n,x in record.items() if n.startswith('candidate_')];best=max(c,key=lambda x:x['geometry_score'] or 0.) if c else None;bs=record['baseline_V7']['geometry_score'] or 0.;methods=dict(selector=np.asarray(row['pose']),B=np.asarray(best['pose']) if best and (best['geometry_score'] or 0.)>bs else base,C=np.asarray(best['pose']) if best and (best['geometry_score'] or 0.)>bs+.1 else base);points=None;solved={};result={}
 for name,T in methods.items():
  if np.array_equal(T,base):result[name]=dict(pose=T.tolist(),initial=T.tolist(),detail=dict(status='baseline_unchanged'));continue
  key=T.tobytes()
  if key not in solved:
   if points is None:
    assert sha(q['scan_path'])==q['scan_sha256'];a=decode(q['scan_path']);radius=np.linalg.norm(a,axis=1);a=voxel_downsample(a[(radius>=1)&(radius<=50)],.4);points=a[np.linspace(0,len(a)-1,min(len(a),2048)).round().astype(int)]
   ids=tree.query_ball_point(T[:3,3],25.);ids=[i for i in ids if i not in q['pair_references']];ids=sorted(ids,key=lambda i:(np.linalg.norm(positions[i]-T[:3,3]),i))[:24]
   if ids:target=voxel_downsample(np.concatenate([cloud(i) for i in ids]),.5);new,detail=refine(points,target,T)
   else:new,detail=T,dict(status='no_validation_map')
   solved[key]=(new,detail)
  new,detail=solved[key];result[name]=dict(pose=new.tolist(),initial=T.tolist(),detail=detail)
 output.append(dict(key=row['key'],index=row['index'],group=row['group'],methods=result))
 if len(output)%50==0:print(json.dumps(dict(shard=args.shard,frames=len(output),seconds=time.monotonic()-started)),flush=True)
assert not denied;save(STORE/('refinement-shard-%02d.json'%args.shard),dict(rows=output,GT_online_attempts=len(denied)));print(json.dumps(dict(complete=True,shard=args.shard,frames=len(output),seconds=time.monotonic()-started)))
