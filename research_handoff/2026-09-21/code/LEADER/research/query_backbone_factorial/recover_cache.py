import json
from pathlib import Path
import zipfile

import numpy as np

root=Path('/home/zhang/leader-query-backbone-factorial').resolve()
records=[]
for index,path in enumerate(sorted((root/'raw').glob('*.npz'))):
    assert path.resolve().parent==root/'raw'
    try:
        with np.load(path) as data:
            values={k:data[k] for k in ['dedode','dino','mask','direction']}
        assert values['dedode'].shape[:3]==values['mask'].shape
        assert values['dino'].shape[:3]==values['mask'].shape
    except (OSError,EOFError,ValueError,zipfile.BadZipFile,KeyError):
        path.unlink()
        records.append(dict(file=path.name,status='removed incomplete newly generated cache; re-extract'))
        continue
    before=path.stat().st_size
    temporary=path.with_suffix('.pending.npz')
    np.savez_compressed(temporary,**values)
    with np.load(temporary) as checked:
        assert all(np.array_equal(v,checked[k]) for k,v in values.items())
    temporary.replace(path)
    records.append(dict(file=path.name,before=before,after=path.stat().st_size,exact_array_parity=True))
    print(f'cache recovery {index+1}: {before/1e6:.1f} -> {path.stat().st_size/1e6:.1f} MB',flush=True)
(root/'storage_recovery.json').write_text(json.dumps(records,indent=2))
