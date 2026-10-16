import argparse
import hashlib
import json
import os
from pathlib import Path

import requests


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    metadata=json.loads((root/'checkpoint.json').read_text())
    path=Path(args.output) if args.output else root/'weights'/metadata['asset_name']
    session=requests.Session()
    token=os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')
    if token:
        session.headers['Authorization']='Bearer '+token
    session.headers['Accept']='application/vnd.github+json'
    release=session.get('https://api.github.com/repos/'+metadata['repository']+'/releases/tags/'+metadata['release_tag'],timeout=30)
    release.raise_for_status()
    asset=next(a for a in release.json()['assets'] if a['name']==metadata['asset_name'])
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.partial')
    digest=hashlib.sha256()
    response=session.get(asset['url'],headers={'Accept':'application/octet-stream'},stream=True,timeout=(30,180))
    response.raise_for_status()
    with temporary.open('wb') as stream:
        for chunk in response.iter_content(1048576):
            if chunk:
                stream.write(chunk)
                digest.update(chunk)
    if digest.hexdigest()!=metadata['sha256']:
        raise RuntimeError('Downloaded checkpoint SHA-256 mismatch; partial file retained')
    temporary.replace(path)
    print(path)


if __name__=='__main__':
    main()
