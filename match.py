import argparse
from pathlib import Path

import numpy as np
import torch

from model import load_features, load_geometry, load_model, match_features


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--query',required=True)
    parser.add_argument('--references',nargs='+',required=True)
    parser.add_argument('--geometry')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--threshold',type=float,default=.1)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model=load_model(args.checkpoint,args.device)
    query=load_features(args.query,args.device)
    references=[load_features(path,args.device) for path in args.references]
    geometry=load_geometry(args.geometry,references,args.device) if args.geometry else None
    matches=match_features(model,query,references,geometry,args.threshold)
    output={}
    for index,match in enumerate(matches):
        output.update({f'reference_{index}_{key}':value for key,value in match.items()})
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**output)
    print('matches per reference:',[len(value['query_index']) for value in matches])


if __name__=='__main__':
    main()
