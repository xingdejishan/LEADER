import argparse
import os
from pathlib import Path

import torch

from models.model_mink import LEADER
from tools.train_local905 import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=37)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.manual_seed(args.seed)
    model = LEADER(in_channels=3, out_channels=4, feat_channels=512, magic=True)
    fusion = model.magic_fusion
    if torch.count_nonzero(fusion.aggregate.output.weight) or torch.count_nonzero(
            fusion.aggregate.output.bias):
        raise ValueError('Fusion residual must initialize to zero')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix('.tmp')
    torch.save({'fusion': fusion.state_dict(), 'seed': args.seed,
                'base_checkpoint_sha256': digest(args.base_checkpoint)}, temporary)
    os.replace(temporary, args.out)
    print(digest(args.out), flush=True)


if __name__ == '__main__':
    main()
