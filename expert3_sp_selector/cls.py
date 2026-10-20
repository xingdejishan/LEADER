import json,os,sys,time,hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parent;STORE=Path('/data2/zhangcy/sp-selector-20261014-v1');GLOBAL=Path('/data2/zhangcy/v7-dinov2-recall-20261012-v1');SOURCE=Path('/home/zhangcy/projects/v7-dinov2-recall-20261012-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def image(path):
 with Image.open(path) as im:
  im=im.convert('RGB').transpose(Image.Transpose.ROTATE_270);w,h=im.size;s=min(1.,560/max(w,h));rw,rh=round(w*s),round(h*s);im=im.resize((rw,rh),Image.Resampling.BICUBIC);cw,ch=rw//14*14,rh//14*14;left,top=(rw-cw)//2,(rh-ch)//2;im=im.crop((left,top,left+cw,top+ch));x=torch.from_numpy(np.array(im).copy()).permute(2,0,1).float()/255
 return (x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
def main():
 shard=int(sys.argv[1]);count=int(sys.argv[2]);torch.set_num_threads(1);torch.manual_seed(20);os.environ['XFORMERS_DISABLED']='1';assets=json.loads((GLOBAL/'assets-freeze.json').read_text());assert sha(GLOBAL/'dinov2_vitb14_pretrain.pth')==assets['weights_sha256']
 for n,h in assets['source_files'].items():assert sha(SOURCE/'dinov2-source'/n)==h
 device='cuda' if torch.cuda.is_available() else 'cpu'
 if device=='cuda':torch.cuda.set_per_process_memory_fraction(.1);assert torch.cuda.mem_get_info()[0]>2*2**30
 model=torch.hub.load(str(SOURCE/'dinov2-source'),'dinov2_vitb14',source='local',pretrained=False);model.load_state_dict(torch.load(GLOBAL/'dinov2_vitb14_pretrain.pth',map_location='cpu'));model.to(device).eval().requires_grad_(False)
 qs=json.loads((STORE/'queries-pending.json').read_text())['queries'];start=time.monotonic();done=0
 gate=json.loads((GLOBAL/'image-plan.json').read_text())['queries'][0];expected=np.load(GLOBAL/'query-cls.npy')[0]
 with torch.inference_mode():g=model.forward_features(image(gate['image_path'])[None].to(device))['x_norm_clstoken'][0].cpu().numpy();g/=np.linalg.norm(g)
 assert float(g@expected)>.9999
 for q in qs:
  if q['group']!='normal' or q['index']%count!=shard:continue
  out=STORE/'cls'/('%06d.npy'%q['index'])
  if out.exists():continue
  assert sha(q['image_path'])==q['image_sha256']
  with torch.inference_mode():a=model.forward_features(image(q['image_path'])[None].to(device))['x_norm_clstoken'][0].cpu().numpy();a/=np.linalg.norm(a)
  assert np.isfinite(a).all();np.save(out,a);done+=1
  if done%25==0:print(json.dumps(dict(shard=shard,frames=done,seconds=time.monotonic()-start)),flush=True)
 print(json.dumps(dict(complete=True,shard=shard,frames=done,seconds=time.monotonic()-start)))
if __name__=='__main__':main()
