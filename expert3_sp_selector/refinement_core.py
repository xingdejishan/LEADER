import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

def transform(points,T):return points@T[:3,:3].T+T[:3,3]
def refine(points,target,initial):
 T=np.asarray(initial,dtype=np.float64).copy();initial=T.copy();cells=np.floor(points).astype(np.int64);fold=(cells[:,0]+3*cells[:,1]+5*cells[:,2])&1;fit=points[fold==0];validation=points[fold==1];tree=cKDTree(target)
 if len(fit)<64 or len(validation)<64 or len(target)<12:return T,dict(status='insufficient_points')
 def validation_error(pose):return float(np.minimum(tree.query(transform(validation,pose),workers=1)[0],1.5).mean())
 before=validation_error(T);iterations=0;directions=0
 for k in range(12):
  placed=transform(fit,T);d,ids=tree.query(placed,workers=1);chosen=np.flatnonzero(d<1.)
  if len(chosen)<64:break
  _,nn=tree.query(target[ids[chosen]],k=12,workers=1);near=target[nn];center=near.mean(1);offset=near-center[:,None];cov=np.einsum('nki,nkj->nij',offset,offset)/12;ev,v=np.linalg.eigh(cov);mask=ev[:,0]/np.maximum(ev.sum(1),1e-12)<.1;chosen=chosen[mask];n=v[mask,:,0];center=center[mask]
  if len(chosen)<64:break
  r=np.einsum('nd,nd->n',n,placed[chosen]-center);w=np.minimum(1.,.2/np.maximum(np.abs(r),1e-12));lever=(placed[chosen]-T[:3,3])/20.;J=np.column_stack([n,np.cross(lever,n)]);H=J.T@(w[:,None]*J)/len(J);g=J.T@(w*r)/len(J);e,V=np.linalg.eigh(H);observable=e>max(1e-5,float(e[-1])*.005);directions=int(observable.sum())
  if directions<3:break
  step=-(V[:,observable]@((V[:,observable].T@g)/e[observable]));translation=step[:3];omega=step[3:]/20.;norm=np.linalg.norm(translation);angle=np.linalg.norm(omega);scale=min(1.,.25/max(norm,1e-12),np.deg2rad(.5)/max(angle,1e-12));translation*=scale;omega*=scale;new=T.copy();new[:3,:3]=Rotation.from_rotvec(omega).as_matrix()@T[:3,:3];new[:3,3]+=translation;total_angle=np.rad2deg(np.arccos(np.clip((np.trace(new[:3,:3].T@initial[:3,:3])-1)/2,-1,1)))
  if np.linalg.norm(new[:3,3]-initial[:3,3])>2 or total_angle>5:break
  T=new;iterations=k+1
  if np.linalg.norm(step)*scale<1e-5:break
 after=validation_error(T);accept=iterations>0 and after<before-1e-4
 return (T if accept else initial),dict(status='accepted' if accept else 'rejected',iterations=iterations,observable_directions=directions,validation_before=before,validation_after=after,fit_points=len(fit),validation_points=len(validation))
