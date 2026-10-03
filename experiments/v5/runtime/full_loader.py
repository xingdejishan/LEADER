import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from data.leader_nclt_manifest import ManifestLEADERDataset,TRAIN_DATES
from backend_runtime import backend_sample
from geometry_cache import GeometryCache
from v5_paths import DATA,CACHE as STORE


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*2**20),b''):h.update(block)
    return h.hexdigest()

class FullLoader:
    def __init__(self,config):
        catalog=json.loads((DATA/'catalog.json').read_text())
        assert sha(DATA/'catalog.json')=='0c3303c80203a70e059abc36f98cbe5eb285858b779e70f88c2279d207423d56'
        assert catalog['splits']['train']['frames']==107339 and catalog['splits']['dev']['frames']==104732
        self.datasets={}
        self.initials={}
        self.geometry={}
        self.records=[]
        self.center=torch.as_tensor(torch.load(config['base_checkpoint'],map_location='cpu')['center_t'],dtype=torch.float32).reshape(3)
        self.manifests={}
        for date in TRAIN_DATES:
            ds=ManifestLEADERDataset([DATA/'NCLT'/date/'manifest.json'],DATA/'NCLT','train',strict_frozen=False,sam_cache_manifest=DATA/'sam'/date/'cache_manifest.json',cache_memory_frames=0)
            initial_path=DATA/'initial'/date/'initial_cache.json'
            initial=json.loads(initial_path.read_text())
            assert initial['manifest_sha256']==ds.manifest_hashes and set(initial['frames'])==set(ds.keys)
            assert initial['ground_truth_used_for_prediction'] is False and not initial.get('failures')
            assert initial['backend']=='raw_SC2-PCR_no_full_pool_refine'
            assert initial['base_checkpoint_sha256']==config['base_checkpoint_sha256']
            assert all(config['source_sha256'][n]==h for n,h in initial['producer_source_sha256'].items())
            np.testing.assert_array_equal(np.asarray(initial['center_t'],np.float32),self.center.numpy())
            geometry=GeometryCache(STORE/'geometry'/date,ds.manifest_hashes[date],sha(initial_path))
            assert geometry.manifest['parent_source_sha256']==config['source_sha256']
            self.datasets[date]=ds
            self.initials[date]=initial
            self.geometry[date]=geometry
            self.manifests[date]={'train_manifest_sha256':ds.manifest_hashes[date],'initial_sha256':sha(initial_path),'sam_sha256':sha(DATA/'sam'/date/'cache_manifest.json'),'geometry_manifest_sha256':sha(STORE/'geometry'/date/'completed.json')}
            self.records.extend((date,index,r.frame_key) for index,r in enumerate(ds.records))
        assert len(self.records)==len({r[2] for r in self.records})==107339

    def frame(self,index,pin=False):
        date,local,key=self.records[index]
        ds=self.datasets[date]
        record=ds.records[local]
        item=self.initials[date]['frames'][key]
        path=DATA/'initial'/date/item['file']
        assert item['scan_sha256']==record.scan_sha256 and item['image_sha256']==record.image_sha256
        assert sha(path)==item['sha256']
        with np.load(path,allow_pickle=False) as archive:
            initial,source,target=[torch.from_numpy(archive[n].copy()) for n in ('initial','source','target')]
        sample=backend_sample(ds,local)
        geometry=self.geometry[date].load(key,record.scan_sha256,item['sha256'])
        assert len(geometry['structure'])==len(source)
        valid=geometry['visible_ids']>=0
        assert torch.equal(geometry['points'][valid],source[geometry['visible_ids'][valid]])
        with Image.open(record.image_path) as image:
            rgb=torch.from_numpy(np.asarray(image.convert('RGB')).copy()).permute(2,0,1).float()/255
        truth=torch.from_numpy(sample['T_WB'].copy())
        truth[:3,3]-=self.center
        frame=dict(geometry,key=key,initial=initial,source=source,target=target,raw=torch.from_numpy(sample['points']),rgb=rgb,sam=sample['sam_features'].float(),hw=sample['sam_metadata']['resized_size_hw'],truth=truth)
        if pin:frame={n:v.pin_memory() if torch.is_tensor(v) else v for n,v in frame.items()}
        return frame

    def batch(self,indices,pin=False):return [self.frame(index,pin) for index in indices]

    def close(self):
        for reader in self.geometry.values():reader.close()
