import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path('/home/zhang/benefit-channel-modulation')
CACHE = Path('/home/zhang/leader-image-gate-raw')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


rows = json.loads((CACHE / 'manifest.json').read_text())
result = dict(source={p.name: digest(p) for p in HERE.glob('*.py')},
              proposal=digest(HERE / 'proposal.md'), manifest=digest(CACHE / 'manifest.json'), frames=[])
for row in rows:
    result['frames'].append(dict(frame_id=row['frame_id'], **{kind: digest(CACHE / kind / (row['frame_id'] + '.npz')) for kind in ['lidar', 'visual']}))
result['original_checkpoint'] = digest(HERE.parents[3] / 'research/image_gate_checkpoint/model.safetensors')
result['line1_checkpoint'] = digest(CACHE / 'aligned.pt')
result['trained_heads'] = {str(p.relative_to(OUT)): digest(p) for p in sorted(OUT.glob('*/*.pt'))}
(OUT / 'provenance.json').write_text(json.dumps(result, indent=2))
print('Recorded source, checkpoint and 96 paired cache hashes')
