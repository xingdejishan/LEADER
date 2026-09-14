import argparse
from pathlib import Path
import tarfile
import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    wanted = {'model.safetensors', 'extra.json'}
    if all((args.output / name).exists() for name in wanted):
        return
    url = 'https://huggingface.co/wushing001/LEADER/resolve/main/nclt_checkpoint_epoch49.tar.gz'
    with requests.get(url, stream=True, timeout=(20, 120)) as response:
        response.raise_for_status()
        with tarfile.open(fileobj=response.raw, mode='r|gz') as archive:
            for entry in archive:
                name = Path(entry.name).name
                print(entry.name, entry.size, flush=True)
                if name not in wanted or not entry.isfile():
                    continue
                target = args.output / name
                with archive.extractfile(entry) as source, target.with_suffix('.pending').open('wb') as output:
                    while block := source.read(1024 * 1024):
                        output.write(block)
                target.with_suffix('.pending').replace(target)
                wanted.remove(name)
                if not wanted:
                    break
    if wanted:
        raise RuntimeError(f'Missing checkpoint files: {wanted}')


if __name__ == '__main__':
    main()
