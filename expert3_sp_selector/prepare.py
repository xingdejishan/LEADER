import hashlib,json,os,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parent
STORE=Path('/data2/zhangcy/sp-selector-20261014-v1')
SP=Path('/data2/zhangcy/sp-lightglue-hard1502-20261013-v1')
DATA=Path('/data2/zhangcy/leader-v2-full-trajectory-20261001/NCLT')
SCAN_ROOT=Path('/data2/dataset/NCLT')
V7=Path('/data2/zhangcy/v7-full-20261006/epoch10-evaluation-20261009/evaluation-epoch10')
VIS=Path('/data2/zhangcy/visual-pose-likelihood-20261014-v5')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):(STORE/n).write_text(json.dumps(v,allow_nan=False))
def main():
 assert os.getuid()==1003
 STORE.mkdir(exist_ok=True)
 for n in ['match','features','decisions','ref-projection','cls']:(STORE/n).mkdir(exist_ok=True)
 lib=json.loads((SP/'library.json').read_text());inp=json.loads((VIS/'inputs.json').read_text());hard=json.loads((SP/'queries.json').read_text())['queries'];e=np.load(V7/'errors.npz');control=[];bindings={str(SP/'library.json'):sha(SP/'library.json'),str(SP/'queries.json'):sha(SP/'queries.json'),str(V7/'errors.npz'):sha(V7/'errors.npz')};hardkeys={q['key'] for q in hard}
 pos={str(k):i for i,k in enumerate(e['frame_keys'])};err=e['V7'];offset=0;chunks={}
 for date in ['2012-02-12','2012-02-19','2012-03-31','2012-05-26']:
  mp=DATA/date/'manifest.json';bindings[str(mp)]=sha(mp);frames=json.loads(mp.read_text())['frames'];eligible=[]
  for i,r in enumerate(frames):
   key=date+'/'+str(r['timestamp_us']);j=pos[key]
   if key not in hardkeys and err[j,0]<=.5 and err[j,1]<2:eligible.append(i)
  ids=np.asarray(eligible)[np.linspace(0,len(eligible)-1,250).round().astype(int)]
  for i in ids:
   r=frames[int(i)];key=date+'/'+str(r['timestamp_us']);j=offset+int(i);f=V7/('chunk-%d.json'%(j//64*64))
   if f not in chunks:
    d=json.loads(f.read_text());assert d['binding']['checkpoint_sha256']=='79e3de5c861316ca5909f07e855366889d7fba26bd4a37b1dbf87597d52cc7bf';chunks[f]={x['frame_key']:x for x in d['rows']};bindings[str(f)]=sha(f)
   b=chunks[f][key]['T_WB_pred'];control.append(dict(key=key,canonical_index=j,scan_path=str(SCAN_ROOT/date/r['scan']['path']),scan_sha256=r['scan']['sha256'],image_path=str(DATA/date/r['image']['path']),image_sha256=r['image']['sha256'],baseline_V7=b,original_SC2=b,crossvote_refit=b,group='normal'))
  offset+=len(frames)
 train=[];trainmeta=[]
 for date in sorted({r['key'].split('/')[0] for r in inp['references']}):
  ids=[i for i,r in enumerate(inp['references']) if r['key'].startswith(date+'/')];chosen=np.asarray(ids)[np.linspace(0,len(ids)-1,32).round().astype(int)]
  for k,j in enumerate(chosen):
   r=inp['references'][int(j)];T=np.asarray(r['T_WB']);rng=np.random.RandomState(20+int(j));baselines=[]
   for distance,angle in [(0.03,.2),(.15,.8),(.4,1.8),(.8,3),(1.5,5),(5,10),(30,30),(450,120)]:
    d=rng.normal(size=3);d/=np.linalg.norm(d);axis=rng.normal(size=3);axis/=np.linalg.norm(axis);b=T.copy();b[:3,3]+=distance*d;b[:3,:3]=Rotation.from_rotvec(axis*np.deg2rad(angle)).as_matrix()@T[:3,:3];baselines.append(b.tolist())
   train.append(dict(**{n:r[n] for n in ['key','scan_path','scan_sha256','image_path','image_sha256']},canonical_index=int(j),baseline_V7=baselines[3],original_SC2=baselines[3],crossvote_refit=baselines[3],group='train',reference_index=int(j)))
   trainmeta.append(dict(key=r['key'],pose=T.tolist(),baseline_variants=baselines,split='fit' if k%2==0 else 'calibration'))
 rows=train+control+[dict(q,group='hard',original_index=q['index']) for q in hard]
 for i,r in enumerate(rows):r['index']=i
 assert len(rows)==2630 and len(set(r['key'] for r in rows))==2630
 save('library.json',lib);save('train-baselines.json',dict(rows=[dict(key=x['key'],baseline_variants=x['baseline_variants']) for x in trainmeta]));save('train-labels.json',dict(rows=trainmeta,note='Train known poses and controlled baselines only; no real Train V7 predictions available. Actual SP+LightGlue registration hypotheses supply candidate negatives. Calibration cannot guarantee normal Dev false-switch rate.'))
 save('queries-pending.json',dict(queries=rows,T_BC=lib['T_BC']))
 save('params.json',dict(seed=20,normal_frames=1000,query_voxel=.4,query_cap=2048,map_voxel=.5,map_radius=25.,scales=[.15,.3,.6,1.2],azimuth_bins=36,spatial_validation='1m query voxel parity split; no raw Query LiDAR participates in visual PnP',normal_neighbors=12,probe_translation=.2,probe_rotation_deg=.5,fit_frames=64,calibration_frames=64,model='HistGradientBoosting max_iter100 depth3 lr.05 l2=1',alpha=.01,threshold_from_Dev=False,cohort_repeated_Dev=True))
 hashes={}
 for q in hard:
  for d in ['match','verify']:
   p=SP/d/('%06d.json'%q['index']);hashes[str(p)]=sha(p)
 save('parent-candidate-freeze.json',dict(files=hashes,fixed_hard_candidate_pool=True));save('input-freeze.json',dict(bindings=bindings,train_inputs_sha=sha(VIS/'inputs.json'),parent_pool_sha=sha(STORE/'parent-candidate-freeze.json'),params_sha=sha(STORE/'params.json')))
 print(json.dumps(dict(complete=True,train=len(train),normal=len(control),hard=len(hard))))
if __name__=='__main__':main()
