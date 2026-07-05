#!/usr/bin/env python
"""
Training entry point for IC Conv-TasNet (AmbiDrop and baseline modes).

Public interface:
    main — parse arguments, build datasets/model/optimizer, and run Solver.train()

Examples:
    # AmbiDrop mode (9 SH channels, with dropout)
    python ConvTasNet/src/train.py --mode ambidrop --mic_num 9 \
        --train_dir si_tr_s_preprocessed_full --valid_dir si_dt_05_preprocessed_full

    # Baseline mode (7 mic channels, no dropout)
    python ConvTasNet/src/train.py --mode baseline --mic_num 7 \
        --train_dir datasets/experiment_full_anm/mic_train_ds_preprocessed_merged \
        --valid_dir datasets/experiment_full_anm/mic_val_ds_preprocessed_merged
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader

from ConvTasNet.datasets import MergedDataset, MatDataset, SimDS_preprocessed
from ConvTasNet.solver import Solver
import ConvTasNet.model as conv_tasnet_model
import wandb

wandb.login()

parser = argparse.ArgumentParser("IC Conv-TasNet Training")

parser.add_argument('--mode', choices=['baseline', 'ambidrop'], default='ambidrop')

parser.add_argument('--train_dir', type=str, default=None)
parser.add_argument('--valid_dir', type=str, default=None)
parser.add_argument('--sample_rate', default=16000, type=int)
parser.add_argument('--segment', default=4, type=float)
parser.add_argument('--cv_maxlen', default=6, type=float)

parser.add_argument('--mic_num', default=9, type=int)
parser.add_argument('--ch_dim', default=8, type=int)
parser.add_argument('--enc_dim', default=512, type=int)
parser.add_argument('--feature_dim', default=128, type=int)
parser.add_argument('--win', default=16, type=int)
parser.add_argument('--layer', default=8, type=int)
parser.add_argument('--stack', default=1, type=int)
parser.add_argument('--kernel', default=3, type=int)
parser.add_argument('--num_spk', default=1, type=int)
parser.add_argument('--causal', type=bool, default=False)

parser.add_argument('--dropout_type', default='SHChannelDropout',
                    choices=['SHChannelDropout', 'PerChDropout'])
parser.add_argument('--drop_prob', default=0.4, type=float)
parser.add_argument('--max_drop', default=3, type=int)
parser.add_argument('--drop_probs', type=str, default=None,
                    help='Comma-separated per-channel probabilities')

parser.add_argument('--use_cuda', type=int, default=1)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--half_lr', default=1, type=int)
parser.add_argument('--early_stop', default=1, type=int)
parser.add_argument('--max_norm', default=5, type=float)
parser.add_argument('--shuffle', default=0, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--optimizer', default='adam', type=str, choices=['sgd', 'adam'])
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--momentum', default=0.0, type=float)
parser.add_argument('--l2', default=0.0, type=float)

base_save_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'ConvTasNet')
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
save_folder = os.path.join(base_save_path, f"run_{timestamp}")

parser.add_argument('--save_folder', default=save_folder)
parser.add_argument('--checkpoint', default=0, type=int)
parser.add_argument('--continue_from', default='')
parser.add_argument('--model_path', default='final.pth.tar')
parser.add_argument('--print_freq', default=10, type=int)
parser.add_argument('--no_wandb', action='store_true')


def main(args):
    drop_probs = None
    if args.drop_probs:
        drop_probs = [float(x) for x in args.drop_probs.split(',')]

    if not args.no_wandb:
        project = f"ConvTasNet_{args.mode}"
        wandb.init(project=project, entity="tatarjit-ben-gurion-university-of-the-negev",
                   name=f"run_{timestamp}", config=vars(args))
        if args.mode == 'ambidrop':
            wandb.config.update({
                "drop_prob": args.drop_prob,
                "max_drop": args.max_drop,
                "dropout_type": args.dropout_type,
            })

    has_mat = any(f.endswith('.mat') for f in os.listdir(args.train_dir))
    has_pt = any(f.endswith('.pt') for f in os.listdir(args.train_dir))

    if has_mat:
        tr_dataset = MatDataset(args.train_dir)
        cv_dataset = MatDataset(args.valid_dir)
    elif has_pt:
        tr_dataset = SimDS_preprocessed(args.train_dir, '.', mode=args.mode)
        cv_dataset = SimDS_preprocessed(args.valid_dir, '.', mode=args.mode)
    else:
        tr_dataset = MergedDataset(args.train_dir)
        cv_dataset = MergedDataset(args.valid_dir)

    tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size,
                           shuffle=True, num_workers=args.num_workers)
    cv_loader = DataLoader(cv_dataset, batch_size=1,
                           shuffle=False, num_workers=args.num_workers)

    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}
    print(f'Data loaded: {len(tr_dataset)} train, {len(cv_dataset)} val')

    model = conv_tasnet_model.TasNet(
        args.mic_num, args.num_spk, args.enc_dim, args.feature_dim, args.ch_dim,
        args.sample_rate, args.win, args.layer, args.stack, args.kernel, args.causal,
        mode=args.mode,
        drop_prob=args.drop_prob,
        max_drop=args.max_drop,
        drop_probs=drop_probs,
        dropout_type=args.dropout_type,
    )

    k = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'# of parameters: {k:,}')

    if args.use_cuda and torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
        model.cuda()

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                     momentum=args.momentum, weight_decay=args.l2)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    solver = Solver(data, model, optimizer, args)
    solver.train()

    if not args.no_wandb:
        wandb.finish()


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
