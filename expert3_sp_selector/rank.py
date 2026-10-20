import json,hashlib
from pathlib import Path
import numpy as np
P=Path('/data2/zhangcy/sp-selector-20261014-v1');G=Path('/data2/zhangcy/v7-dinov2-recall-20261012-v1');SP=Path('/data2/zhangcy/sp-lightglue-hard1502-20261013-v1')
q=json.loads((P/'queries-pending.json').read_text());r=json.loads((P/'library.json').read_text())['references'];features=np.load(G/'reference-cls.npy');freeze={}
for row in q['queries']:
 if row['group']=='hard':
  m=json.loads((SP/'match'/('%06d.json'%row['original_index'])).read_text());row['pair_references']=m['pair_references'];m['index']=row['index'];(P/'match'/('%06d.json'%row['index'])).write_text(json.dumps(m));continue
 a=features[row['reference_index']] if row['group']=='train' else np.load(P/'cls'/('%06d.npy'%row['index']));scores=features@a
 if row['group']=='train':scores[np.array([x['key'].split('/')[0]==row['key'].split('/')[0] for x in r])]=-np.inf
 ix=np.argsort(-scores,kind='stable')[:5];row['pair_references']=ix.tolist()
 if row['group']=='normal':f=P/'cls'/('%06d.npy'%row['index']);freeze[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
(P/'queries.json').write_text(json.dumps(q));(P/'cls-freeze.json').write_text(json.dumps(dict(files=freeze,reference_cls_sha=hashlib.sha256((G/'reference-cls.npy').read_bytes()).hexdigest())));(P/'match-references.json').write_text(json.dumps(dict(indices=sorted({j for x in q['queries'] for j in x['pair_references']}))))
print(json.dumps(dict(complete=True,queries=len(q['queries']),reference_projections=len(json.loads((P/'match-references.json').read_text())['indices']))))
