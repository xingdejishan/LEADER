import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    for name in ('base-checkpoint','data-root','train-sam','dev-sam','output'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/'experiments/far/config.example.json').read_text())
    for name in ('base_checkpoint','train_sam','dev_sam'):
        path=Path(getattr(args,name)).resolve(strict=True)
        observed=digest(path)
        if observed!=config[name+'_sha256']:raise ValueError('asset differs from frozen experiment: '+name)
        config[name]=str(path)
    config['data_root']=str(Path(args.data_root).resolve(strict=True))
    manifest=json.loads((root/'experiments/far/source_manifest.json').read_text())
    config['source_sha256']={}
    for name,expected in manifest['source_sha256_lf'].items():
        path=root/name
        if hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=expected:
            raise ValueError('published source differs: '+name)
        config['source_sha256'][name]=digest(path)
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(config,indent=2)+'\n')


if __name__=='__main__':main()
