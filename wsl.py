import json
from pathlib import Path
import subprocess
import sys


def run(arguments):
    root = Path(__file__).resolve().parent
    config = json.loads((root / 'WSL_ENVIRONMENT.json').read_text(encoding='utf-8'))
    prefix = ['wsl.exe', '-d', config['distribution']]

    def convert(path):
        return subprocess.check_output(prefix + ['--exec', 'wslpath', '-u', str(path).replace('\\', '/')], text=True).strip()

    linux_root = convert(root)
    forwarded = [convert(value) if len(value) > 2 and value[1] == ':' else value.replace('\\', '/') for value in arguments]
    subprocess.run(prefix + ['--cd', linux_root, '--exec', 'env', 'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=2',
        config['python'], linux_root + '/local.py'] + forwarded, check=True)


if __name__ == '__main__':
    run(sys.argv[1:])
