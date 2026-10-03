import hashlib,json,os
from pathlib import Path
import numpy as np
import torch

LAYOUT=(('points','<f4',3),('pixels','<f4',2),('visible_ids','<i8',1),('structure','<f4',9))

def encode(values):
    arrays=[np.ascontiguousarray(values[name].detach().cpu().numpy(),dtype=dtype) for name,dtype,_ in LAYOUT]
    return b''.join(a.tobytes() for a in arrays),len(arrays[0]),len(arrays[-1])

def decode(data,point_count,source_count):
    offset=0
    result={}
    for name,dtype,channels in LAYOUT:
        count=source_count if name=='structure' else point_count
        size=np.dtype(dtype).itemsize*count*channels
        shape=(count,3,3) if name=='structure' else (count,channels) if channels>1 else (count,)
        value=np.frombuffer(data,dtype=dtype,count=count*channels,offset=offset).copy().reshape(shape)
        result[name]=torch.from_numpy(value)
        offset+=size
    assert offset==len(data)
    return result

class GeometryCache:
    def __init__(self,directory,manifest_sha256,initial_sha256):
        self.directory=Path(directory)
        self.manifest=json.loads((self.directory/'completed.json').read_text())
        assert self.manifest['complete'] and self.manifest['manifest_sha256']==manifest_sha256 and self.manifest['initial_manifest_sha256']==initial_sha256
        assert self.manifest['range_m']==[1.,100.] and self.manifest['max_points']==2048
        self.stream=(self.directory/'geometry.bin').open('rb')
        assert os.fstat(self.stream.fileno()).st_size==self.manifest['bytes']
    def load(self,key,scan_sha256,initial_sha256):
        row=self.manifest['frames'][key]
        assert row['scan_sha256']==scan_sha256 and row['initial_sha256']==initial_sha256
        self.stream.seek(row['offset'])
        data=self.stream.read(row['bytes'])
        assert hashlib.sha256(data).hexdigest()==row['sha256']
        return decode(data,row['points'],row['source_points'])
    def close(self):self.stream.close()
