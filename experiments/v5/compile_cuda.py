import argparse
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--arch',default='86')
args=parser.parse_args()
assert args.arch.isdigit() and len(args.arch) in (2,3)
nvcc=shutil.which('nvcc')
assert nvcc is not None,'CUDA compiler nvcc must be available in the current environment'
command=[nvcc,str(ROOT/'runtime/hash_fused.cu'),'-o',str(ROOT/'runtime/hash_fused.so'),'-shared','-Xcompiler','-fPIC','-O3','--fmad=false','--prec-div=true',
    '-gencode','arch=compute_{0},code=sm_{0}'.format(args.arch),'-gencode','arch=compute_{0},code=compute_{0}'.format(args.arch),'-cudart','shared']
prefix=Path(nvcc).resolve().parents[1]
for directory in (prefix/'lib',prefix/'lib64'):
    if (directory/'libcudart.so').exists():
        command+=['-L'+str(directory),'-Xlinker','-rpath','-Xlinker',str(directory)]
        break
subprocess.run(command,check=True)
