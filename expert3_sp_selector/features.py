from common import *
from collections import OrderedDict
import argparse,time
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

parser=argparse.ArgumentParser();parser.add_argument('--shard',type=int,default=0);parser.add_argument('--of',type=int,default=1);parser.add_argument('--group',default='all',choices=['all','hard','extra']);args=parser.parse_args();denied=deny_GT();params=load(STORE/'params.json');qs=load(STORE/'queries.json')['queries'];lib=load(STORE/'library.json')['references'];positions=np.asarray([r['body_position_world'] for r in lib]);position_tree=cKDTree(positions);cache=OrderedDict();map_hashes={};started=time.monotonic();done=0;train_baselines={r['key']:r['baseline_variants'] for r in load(STORE/'train-baselines.json')['rows']}
if args.group=='all' and (STORE/'hard-feature-launch.json').exists():
 while not (STORE/'hard-feature-completed.json').exists():time.sleep(2)
 assert load(STORE/'hard-feature-completed.json')['complete']
if args.group=='all' and (STORE/'extra-feature-launch.json').exists():
 while not (STORE/'extra-feature-completed.json').exists():time.sleep(2)
 assert load(STORE/'extra-feature-completed.json')['complete']
def cloud(i):
 if i in cache:v=cache.pop(i);cache[i]=v;return v
 r=lib[i];f=STORE/'map-cloud'/('%06d-%s.npz'%(i,r['scan_sha256'][:16]))
 if str(f) not in map_hashes:map_hashes[str(f)]=sha(f)
 z=np.load(f);v=z['world'].astype(np.float64);cache[i]=v
 if len(cache)>64:cache.popitem(last=False)
 return v
def summarize(points,T,excluded,date):
 placed=transform(points,T);ids=position_tree.query_ball_point(T[:3,3],params['map_radius']);ids=[i for i in ids if i not in excluded and (date is None or not lib[i]['key'].startswith(date+'/'))];ids=sorted(ids,key=lambda i:(np.linalg.norm(positions[i]-T[:3,3]),i))[:24]
 if not ids:return np.zeros(27),dict(valid=False,refs=[],points=0)
 target=voxel_downsample(np.concatenate([cloud(i) for i in ids]),params['map_voxel']);tree=cKDTree(target);d,nearest=tree.query(placed,workers=1);az=(np.floor((np.arctan2(points[:,1],points[:,0])+np.pi)/(2*np.pi)*36).astype(int)).clip(0,35);cells=np.floor(points).astype(np.int64);fold=((cells[:,0]+3*cells[:,1]+5*cells[:,2])&1);usable=[b for b in range(36) if (az==b).sum()>=8];values=[]
 for f in [0,1]:
  for scale in params['scales']:
   vals=[np.exp(-.5*(d[(az==b)&(fold==f)]/scale)**2).mean() for b in usable if ((az==b)&(fold==f)).sum()>=4];values.append(float(np.mean(vals)) if vals else 0.)
 clip=np.minimum(d,3.);binvalues=[np.exp(-.5*(d[az==b]/.6)**2).mean() for b in usable];support=d<=.6;count=int(support.sum());eig=np.zeros(6);normal_eig=np.zeros(3)
 if count>=30 and len(target)>=12:
  chosen=np.flatnonzero(support)[::max(1,count//512)][:512];_,nn=tree.query(target[nearest[chosen]],k=12,workers=1);neighbors=target[nn];centered=neighbors-neighbors.mean(1,keepdims=True);cov=np.einsum('nki,nkj->nij',centered,centered)/12;ee,vv=np.linalg.eigh(cov);good=(ee[:,0]/np.maximum(ee.sum(1),1e-9))<.15;n=vv[good,:,0];lever=(placed[chosen[good]]-T[:3,3])/20.
  if len(n)>=20:
   J=np.column_stack([n,np.cross(lever,n)]);eig=np.linalg.eigvalsh(J.T@J/len(J));normal_eig=np.linalg.eigvalsh(n.T@n/len(n))
 idx=np.flatnonzero(fold==1);idx=idx[::max(1,len(idx)//512)][:512];probe=[];base=float(np.exp(-.5*(d[idx]/.6)**2).mean()) if len(idx) else 0.
 for axis in range(3):
  for sign in [-1,1]:
   translation=np.zeros(3);translation[axis]=sign*.2;pd=tree.query(placed[idx]+translation,workers=1)[0];probe.append(float(np.exp(-.5*(pd/.6)**2).mean()))
   rv=np.zeros(3);rv[axis]=sign*np.deg2rad(.5);R=Rotation.from_rotvec(rv).as_matrix();rotated=(placed[idx]-T[:3,3])@R.T+T[:3,3];pd=tree.query(rotated,workers=1)[0];probe.append(float(np.exp(-.5*(pd/.6)**2).mean()))
 extra=[float(np.median(clip)),float(np.quantile(clip,.9)),float(np.mean(clip)),len(usable)/36.,float(np.quantile(binvalues,.25)) if binvalues else 0.,float(np.std(binvalues)) if binvalues else 0.,float(support.mean()),count/len(points),float(eig[0]),float(eig[1]),float(eig[2]),float(eig[-1]),float(normal_eig[0]),float(normal_eig[1]),float(normal_eig[2]),base-max(probe) if probe else 0.,float(np.mean(probe)) if probe else 0.,len(ids)/24.,len(target)/100000.]
 feature=np.asarray(values+extra);assert feature.shape==(27,) and np.isfinite(feature).all();return feature,dict(valid=len(usable)>=4 and count>=30,refs=ids,points=len(points),support=count,spatial_folds=[int((fold==f).sum()) for f in [0,1]],bin_count=len(usable))
for q in qs:
 if args.group=='hard' and q['group']!='hard':continue
 if args.group=='extra' and q['group']=='hard':continue
 if q['index']%args.of!=args.shard:continue
 out=STORE/'features'/('%06d.json'%q['index'])
 if out.exists():continue
 guard(max_rss_gib=12,min_available_gib=32);assert sha(q['scan_path'])==q['scan_sha256'];points=decode(q['scan_path']);radius=np.linalg.norm(points,axis=1);points=voxel_downsample(points[(radius>=1)&(radius<=50)],.4);points=points[np.linspace(0,len(points)-1,min(len(points),2048)).round().astype(int)];excluded=set(q['pair_references']);date=q['key'].split('/')[0] if q['group']=='train' else None
 matchpath=STORE/'match'/('%06d.json'%q['index'])
 while not matchpath.exists():time.sleep(2)
 match=load(matchpath);candidates=match['hypotheses'];entries=[dict(name='baseline',pose=q['baseline_V7'],reference_ids=[])]+[dict(c,name='candidate_%d'%i) for i,c in enumerate(candidates)]
 if q['group']=='train':entries.extend(dict(name='train_baseline_%d'%i,pose=p,reference_ids=[]) for i,p in enumerate(train_baselines[q['key']]))
 result=[]
 for e in entries:
  T=np.asarray(e['pose']);f,diagnostic=summarize(points,T,excluded,date);support=set()
  for c in candidates:
   if c['channel']!='pair':continue
   t=np.asarray(c['pose']);angle=np.rad2deg(np.arccos(np.clip((np.trace(T[:3,:3].T@t[:3,:3])-1)/2,-1,1)))
   if np.linalg.norm(T[:3,3]-t[:3,3])<1 and angle<3:support.update(c['reference_ids'])
  f=np.r_[f,len(support)/5.];result.append(dict(name=e['name'],pose=e['pose'],feature=f.tolist(),evidence=diagnostic,reference_ids=e['reference_ids']))
 save(out,dict(key=q['key'],index=q['index'],group=q['group'],entries=result,match_sha256=sha(matchpath),query_scan_sha=q['scan_sha256'],GT_online_attempts=len(denied)));done+=1
 if done%25==0:print(json.dumps(dict(shard=args.shard,frames=done,seconds=time.monotonic()-started)),flush=True)
assert not denied
save(STORE/('feature-map-bindings-%02d-%s.json'%(args.shard,args.group)),dict(files=map_hashes))
print(json.dumps(dict(complete=True,shard=args.shard,frames=done,seconds=time.monotonic()-started)))
