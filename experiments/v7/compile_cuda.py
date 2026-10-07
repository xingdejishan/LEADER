import argparse
import shutil
import subprocess
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--arch',default='86')
args=parser.parse_args()
assert args.arch.isdigit() and len(args.arch) in [2,3]
nvcc=shutil.which('nvcc')
assert nvcc is not None
root=Path(__file__).resolve().parent/'runtime'
for name in ['mixed_gradient_terms','mixed_backward_complete_v1','target_mixed_backward_v1']:
    command=[nvcc,str(root/(name+'.cu')),'-o',str(root/(name+'.so')),'-shared','-Xcompiler','-fPIC','-O3','--fmad=false','--prec-div=true','-gencode','arch=compute_{0},code=sm_{0}'.format(args.arch),'-gencode','arch=compute_{0},code=compute_{0}'.format(args.arch),'-cudart','shared']
    for directory in [Path(nvcc).resolve().parents[1]/'lib',Path(nvcc).resolve().parents[1]/'lib64']:
        if (directory/'libcudart.so').exists():
            command+=['-L'+str(directory),'-Xlinker','-rpath','-Xlinker',str(directory)]
            break
    subprocess.run(command,check=True)
