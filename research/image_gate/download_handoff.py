import argparse
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--extract',action='store_true')
    parser.add_argument('--group',choices=['all','checkpoints','replay-features','paired-inputs'],default='all')
    args=parser.parse_args()
    manifest=json.loads(Path(__file__).with_name('assets_manifest.json').read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    root=args.output.resolve()
    for asset in manifest['archives']:
        if args.group!='all' and asset['name']!=args.group+'.tar.gz':
            continue
        name=asset['name']
        assert Path(name).name==name
        target=root/name
        if not target.exists():
            pending=target.with_suffix('.pending')
            url=f"https://github.com/xingdejishan/LEADER/releases/download/{manifest['tag']}/{name}"
            urllib.request.urlretrieve(url,pending)
            pending.replace(target)
        if target.stat().st_size!=asset['size'] or digest(target)!=asset['sha256']:
            raise ValueError('Archive checksum mismatch: '+name)
        print('Verified '+name,flush=True)
        if args.extract:
            with tarfile.open(target,'r:gz') as archive:
                for member in archive:
                    destination=(root/member.name).resolve()
                    if not destination.is_relative_to(root) or not member.isfile():
                        raise ValueError('Unsafe archive entry: '+member.name)
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    with archive.extractfile(member) as source,destination.open('wb') as output:
                        for block in iter(lambda:source.read(4*1024*1024),b''):
                            output.write(block)
            for record in manifest['files']:
                if record['archive']==name and digest(root/record['path'])!=record['sha256']:
                    raise ValueError('Extracted checksum mismatch: '+record['path'])
            print('Extracted and verified '+name,flush=True)


if __name__=='__main__':main()
