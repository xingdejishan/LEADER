import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile
import time

from common import ROOT,STORE,PLAN,atomic,guard,sha,verify_source,prohibit_dev_gt
import cv2
import numpy as np
from resume_tar_stream import ResumeTarStream

class Reader:
    def __init__(self,path,offset):
        self.file=Path(path).open('rb');self.file.seek(offset)
        self.initial_position=offset;self.position=offset;self.bytes=0;self.start=time.monotonic()
    def read(self,size):
        assert 0<=size<=8*1024**2
        value=self.file.read(size);self.bytes+=len(value);self.position+=len(value)
        delay=self.bytes/(PLAN['archive_read_limit_mib_s']*1024**2)-(time.monotonic()-self.start)
        if delay>0:time.sleep(delay)
        return value
    def close(self):self.file.close()

def open_before(path,date,camera):
    low=0;high=path.stat().st_size
    for probe in range(14):
        offset=((low+high)//2//(4*1024**2))*(4*1024**2)
        stream=ResumeTarStream(Reader(path,offset),date)
        first=int(re.search(r'/Cam([0-5])/',stream.first_member).group(1));stream.close()
        if first<camera:low=offset
        else:high=offset
        atomic(STORE/'extract-progress.json',{'date':date,'phase':'locating_camera','camera':camera,'probe':probe,'first_camera':first,'low':low,'high':high,'time':time.time()})
        if high-low<64*1024**2:break
    offset=max(0,low-32*1024**2)
    if offset==0:return Reader(path,0),'r|gz',None
    stream=ResumeTarStream(Reader(path,offset),date)
    assert int(re.search(r'/Cam([0-5])/',stream.first_member).group(1))<camera
    return stream,'r|',stream.first_member

verify_source();guard();prohibit_dev_gt();cv2.setNumThreads(1)
lock=(STORE/'extract.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
six=Path(PLAN['six_root']);cal=six/'calibration';meta=json.loads((cal/'manifest.json').read_text())
assert sha(cal/'manifest.json')==PLAN['calibration_manifest_sha256']
maps={}
for c in range(1,5):
    pair=[]
    for letter in ('u','v'):
        p=cal/f'Cam{c}_map{letter}.npy';assert sha(p)==meta['files'][p.name]
        a=np.load(p,mmap_mode='r',allow_pickle=False);assert a.shape==(1232,1616) and a.dtype==np.float32;pair.append(a)
    maps[c]=pair
raw=Path(PLAN['archive_root']);status=json.loads((raw/'status.json').read_text())
receipts=[]
for date in PLAN['train_dates']+PLAN['dev_dates']:
    guard();original=Path(PLAN['data_root'])/date/'manifest.json';assert sha(original)==PLAN['manifest_sha256'][date]
    selected=json.loads(original.read_text());assert len(selected['frames'])==300
    targets={r['timestamp_us'] for r in selected['frames']};assert len(targets)==300
    dest=STORE/'images'/date/'manifest.json'
    if dest.exists():value=json.loads(dest.read_text())
    else:
        value={'date':date,'files':{},'source_manifest_sha256':sha(original),'complete':False}
        old=six/'images'/date/'manifest.json'
        if old.exists():
            for key,r in json.loads(old.read_text()).get('files',{}).items():
                if r['camera_id'] not in maps or r['timestamp'] not in targets:continue
                p=six/r['path'];assert sha(p)==r['sha256']
                value['files'][key]=dict(r,absolute_path=str(p),reused=True)
        atomic(dest,value)
    records=value['files'];missing={f'Cam{c}/{t}' for c in maps for t in targets}-set(records)
    if missing:
        relative=f'archives/images/{date}_lb3.tar.gz';spec=status['archives'][relative]
        path=raw/relative;assert spec['complete'] and path.stat().st_size==spec['size_bytes'] and path.stat().st_uid==os.getuid()
        minimum=min(int(k[3]) for k in missing);reader,mode,first=open_before(path,date,minimum)
        count={str(c):0 for c in range(minimum,5)};new=0;anchors=0;last=time.time()
        archive=tarfile.open(fileobj=reader,mode=mode)
        try:
            for member in archive:
                archive.members.clear()
                match=re.search(r'/Cam([0-5])/(\d+)\.tiff?$',member.name,re.I)
                if not match or not member.isfile():continue
                camera,timestamp=map(int,match.groups())
                if camera>=5:break
                if camera<minimum:continue
                count[str(camera)]+=1
                key=f'Cam{camera}/{timestamp}'
                if timestamp not in targets:continue
                if key not in missing and anchors>=3:continue
                guard();assert member.size<32*1024**2
                with archive.extractfile(member) as stream:payload=stream.read()
                assert len(payload)==member.size
                digest=hashlib.sha256(payload).hexdigest()
                if key not in missing:
                    assert digest==records[key]['raw_tiff_sha256'];anchors+=1;continue
                image=cv2.imdecode(np.frombuffer(payload,np.uint8),cv2.IMREAD_UNCHANGED)
                assert image is not None and image.shape==(1232,1616,3)
                image=cv2.resize(cv2.remap(image,*maps[camera],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT),(808,616),interpolation=cv2.INTER_AREA)
                ok,jpg=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,95]);assert ok
                output=STORE/'images'/date/f'Cam{camera}'/f'{timestamp}.jpg';output.parent.mkdir(exist_ok=True,parents=True)
                data=jpg.tobytes()
                if output.exists():assert sha(output)==hashlib.sha256(data).hexdigest()
                else:output.write_bytes(data)
                records[key]={'camera_id':camera,'timestamp':timestamp,'absolute_path':str(output),'sha256':sha(output),'raw_tiff_sha256':digest,'archive_member':member.name,'reused':False,'u2d_text_sha256':meta['cameras'][str(camera)]['u2d_text_sha256']}
                missing.remove(key);new+=1
                if new%25==0 or time.time()-last>45:
                    atomic(dest,value);atomic(STORE/'extract-progress.json',{'date':date,'phase':'extracting','ready':len(records),'remaining':len(missing),'new':new,'archive_position':reader.position,'camera_counts':count,'time':time.time()});last=time.time()
                    print(json.dumps({'date':date,'new':new,'remaining':len(missing)}),flush=True)
        finally:archive.close();reader.close()
        assert not missing,(date,len(missing))
        assert all(count[str(c)]==spec['member_counts']['camera_members'][str(c)] for c in range(minimum,5)),count
        value['archive_receipt']={'archive_sha256':spec['sha256'],'archive_size_bytes':spec['size_bytes'],'first_member':first,'camera_counts':count,'reused_raw_anchors_verified':anchors}
    assert len(records)==1200 and not missing
    value['complete']=True;atomic(dest,value)
    receipts.append({'date':date,'manifest_sha256':sha(dest),'camera_inputs':len(records)})
    print(json.dumps({'event':'date_images_complete','date':date,'inputs':len(records)}),flush=True)
atomic(STORE/'images-complete.json',{'dates':receipts,'side_camera_inputs':9600,'time':time.time()})
