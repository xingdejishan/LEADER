import os
from pathlib import Path
import sys


def private_root():
    root_value=os.environ.get('RAYBUNDLE_ROOT')
    prefix_value=os.environ.get('RAYBUNDLE_ENV_PREFIX')
    if not root_value or not prefix_value:
        raise RuntimeError('set RAYBUNDLE_ROOT and RAYBUNDLE_ENV_PREFIX explicitly')
    root=Path(root_value).resolve(strict=True)
    prefix=Path(prefix_value).resolve(strict=True)
    if not root.is_dir() or root.stat().st_uid!=os.getuid():
        raise ValueError('private project owner mismatch')
    if Path(sys.prefix).resolve()!=prefix:
        raise RuntimeError('execute only inside the configured existing environment')
    return root
