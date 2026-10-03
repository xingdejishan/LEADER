import json,os
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'source'
RUNTIME=ROOT/'runtime'
BASE=ROOT/'local'
SETTINGS=Path(os.environ.get('V5_PATHS',str(BASE/'paths.json')))
if not SETTINGS.is_file():raise RuntimeError('Run configure.py with the prepared data and asset paths first')
settings=json.loads(SETTINGS.read_text(encoding='utf-8'))
DATA=Path(settings['data']).resolve()
STORE=Path(settings['output']).resolve()
CACHE=Path(settings['geometry_cache']).resolve()
RAW_GT=Path(settings['raw_gt']).resolve()
