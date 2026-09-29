import argparse
import json
from pathlib import Path
import runpy
import sys


parser=argparse.ArgumentParser()
parser.add_argument('--source',required=True)
parser.add_argument('--worker',required=True)
parser.add_argument('--task',required=True)
parser.add_argument('--job',required=True)
args=parser.parse_args()
sys.path.insert(0,str(Path(args.source).resolve()))
sys.argv=[args.worker,'--task',args.task,'--job',args.job]
namespace=runpy.run_path(args.worker,run_name='raybundle.external_validation_job')
namespace['execute'](json.loads(Path(args.task).read_text()),args.job)
