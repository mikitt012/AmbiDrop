#!/usr/bin/env python

# Created on 2018/12
# Author: Kaituo XU

import argparse

import torch

from data import AudioDataset, AudioDataLoader, MergedDataset, MatDataset
from solver import Solver
from conv_tasnet import ConvTasNet
import conv_tasnet_ic

from torch.utils.data import Dataset, DataLoader
import os
from datetime import datetime
import wandb
wandb.login()


parser = argparse.ArgumentParser(
    "Fully-Convolutional Time-domain Audio Separation Network (Conv-TasNet) "
    "with Permutation Invariant Training")
# General config
# Task related
parser.add_argument('--train_dir', type=str, default=None,
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--valid_dir', type=str, default=None,
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--sample_rate', default=16000, type=int,
                    help='Sample rate')
parser.add_argument('--segment', default=4, type=float,
                    help='Segment length (seconds)')
parser.add_argument('--cv_maxlen', default=6, type=float,
                    help='max audio length (seconds) in cv, to avoid OOM issue.')
# Network architecture
    # model_mic_num: int = 4
    # model_ch_dim: int = 8
    # model_enc_dim: int = 512
    # model_feature_dim: int = 128
    # model_win: int = 16
    # model_layer: int = 8
    # model_stack: int = 1
    # model_kernel: int = 3
    # model_num_spk: int = 1
    # model_causal: bool = False
parser.add_argument('--mic_num', default=7, type=int)
parser.add_argument('--ch_dim', default=8, type=int)
parser.add_argument('--enc_dim', default=512, type=int)
parser.add_argument('--feature_dim', default=128, type=int)
parser.add_argument('--win', default=16, type=int)
parser.add_argument('--layer', default=8, type=int)
parser.add_argument('--stack', default=1, type=int)
parser.add_argument('--kernel', default=3, type=int)
parser.add_argument('--num_spk', default=1, type=int)
parser.add_argument('--causal', type=bool, default=False)

# Training config
parser.add_argument('--use_cuda', type=int, default=1,
                    help='Whether use GPU')
parser.add_argument('--epochs', default=100, type=int,
                    help='Number of maximum epochs')
parser.add_argument('--half_lr', dest='half_lr', default=1, type=int,
                    help='Halving learning rate when get small improvement')
parser.add_argument('--early_stop', dest='early_stop', default=1, type=int,
                    help='Early stop training when no improvement for 10 epochs')
parser.add_argument('--max_norm', default=5, type=float,
                    help='Gradient norm threshold to clip')
# minibatch
parser.add_argument('--shuffle', default=0, type=int,
                    help='reshuffle the data at every epoch')
parser.add_argument('--batch_size', default=64, type=int,
                    help='Batch size')
parser.add_argument('--num_workers', default=0, type=int,
                    help='Number of workers to generate minibatch')
# optimizer
parser.add_argument('--optimizer', default='adam', type=str,
                    choices=['sgd', 'adam'],
                    help='Optimizer (support sgd and adam now)')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='Init learning rate')
parser.add_argument('--momentum', default=0.0, type=float,
                    help='Momentum for optimizer')
parser.add_argument('--l2', default=0.0, type=float,
                    help='weight decay (L2 penalty)')
# save and load model
# save and load model
base_save_path = '/gpfs0/bgu-br/users/tatarjit/speech-enhancement/ConvTasNet_baseline/checkpoints'
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
save_folder = os.path.join(base_save_path, f"run_{timestamp}")

parser.add_argument('--save_folder', default=save_folder, help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', default=0, type=int,
                    help='Enables checkpoint saving of model')
parser.add_argument('--continue_from', default='',
                    help='Continue from checkpoint model')
parser.add_argument('--model_path', default='final.pth.tar',
                    help='Location to save best validation model')
# logging
parser.add_argument('--print_freq', default=10, type=int,
                    help='Frequency of printing training infomation')
parser.add_argument('--visdom', dest='visdom', type=int, default=0,
                    help='Turn on visdom graphing')
parser.add_argument('--visdom_epoch', dest='visdom_epoch', type=int, default=0,
                    help='Turn on visdom graphing each epoch')
parser.add_argument('--visdom_id', default='TasNet training',
                    help='Identifier for visdom run')


def main(args):
    wandb.init(
            project=f"ConvTasNet_baseline",
            entity="tatarjit-ben-gurion-university-of-the-negev", 
            name=f"run_{timestamp}",
            config=vars(args) # Logs all parser arguments automatically
        )
    # data
    tr_dataset = MergedDataset('/gpfs0/bgu-br/users/tatarjit/speech-enhancement/FaSNet/mic_train_ds_processed')
    cv_dataset = MergedDataset('/gpfs0/bgu-br/users/tatarjit/speech-enhancement/FaSNet/mic_val_ds_processed')

    # tr_dataset.file_list = tr_dataset.file_list[:10] # Directly slice the list
    # cv_dataset.file_list = cv_dataset.file_list[:10] # Directly slice the list

    tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    cv_loader = DataLoader(cv_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}
    print('data is loaded')

    # model
    # model = ConvTasNet(args.N, args.L, args.B, args.H, args.P, args.X, args.R,
    #                    args.C, drop_prob=drop_prob, max_drop=max_drop, drop_probs=drop_probs, dropout=dropout, norm_type=args.norm_type, causal=args.causal,
    #                    mask_nonlinear=args.mask_nonlinear)

    model = conv_tasnet_ic.TasNet(args.mic_num, args.num_spk, args.enc_dim, args.feature_dim, args.ch_dim,
                         args.sample_rate, args.win, args.layer, args.stack, args.kernel, args.causal)

    k = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('# of parameters:', k)

    if args.use_cuda:
        model = torch.nn.DataParallel(model)
        model.cuda()
    # optimizer
    if args.optimizer == 'sgd':
        optimizier = torch.optim.SGD(model.parameters(),
                                     lr=args.lr,
                                     momentum=args.momentum,
                                     weight_decay=args.l2)
    elif args.optimizer == 'adam':
        optimizier = torch.optim.Adam(model.parameters(),
                                      lr=args.lr,
                                      weight_decay=args.l2)
    else:
        print("Not support optimizer")
        return

    # solver
    solver = Solver(data, model, optimizier, args)
    solver.train()
    wandb.finish()


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)

#runai-bgu submit python -n convtasnet-baseline-train5 -c 20 -m 80G -g 1 --conda venv -- "python /gpfs0/bgu-br/users/tatarjit/speech-enhancement/ConvTasNet_baseline/src/train.py"
