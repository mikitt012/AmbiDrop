#!/usr/bin/env python
"""
Unified IC Conv-TasNet evaluation script.

Available checkpoints:
    run_2026-04-09_08-35  AmbiDrop, SHChannelDropout (p=0.4, max=3)
    run_2026-04-07_15-27  AmbiDrop, PerChDropout (th=-3.4 dB, probs=[0,0.1,0.45,0.1,0.45,1,0.75,1,0.45])
    run_2026-04-09_10-55  Baseline (no dropout, 7 mic channels)

Examples:
    # AmbiDrop with SHChannelDropout
    python ConvTasNet/src/evaluate.py --mode ambidrop \
        --model_path checkpoints/ConvTasNet/run_2026-04-09_08-35/final.pth.tar \
        --data_dir datasets/experiment_full_anm/test_of_train_ds

    # AmbiDrop with PerChDropout
    python ConvTasNet/src/evaluate.py --mode ambidrop \
        --model_path checkpoints/ConvTasNet/run_2026-04-07_15-27/final.pth.tar \
        --data_dir datasets/experiment_full_anm/test_of_test_ds \
        --dropout_type PerChDropout \
        --drop_probs "0,0.1,0.45,0.1,0.45,1,0.75,1,0.45"

    # Baseline
    python ConvTasNet/src/evaluate.py --mode baseline \
        --model_path checkpoints/ConvTasNet/run_2026-04-09_10-55/final.pth.tar \
        --data_dir datasets/experiment_full_anm/test_of_test_ds_preprocessed
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.io import loadmat

from ConvTasNet.datasets import SimDS_preprocessed, MatDatasetTest_ASM
import ConvTasNet.model as conv_tasnet_model
from ConvTasNet.utils import remove_pad
from ambidrop.constants import REF_IDX_MAP
from pesq import pesq
from pystoi import stoi
import wandb

wandb.login()

parser = argparse.ArgumentParser('Evaluate IC Conv-TasNet')
parser.add_argument('--mode', choices=['baseline', 'ambidrop'], default='ambidrop')
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--use_cuda', type=int, default=1)
parser.add_argument('--sample_rate', default=16000, type=int)
parser.add_argument('--dropout_type', default='SHChannelDropout')
parser.add_argument('--drop_prob', default=0.4, type=float)
parser.add_argument('--max_drop', default=3, type=int)
parser.add_argument('--drop_probs', type=str, default=None)
parser.add_argument('--no_wandb', action='store_true')


def cal_sisnr(ref_sig, out_sig, eps=1e-8):
    """Numpy SI-SNR between two 1D signals."""
    assert len(ref_sig) == len(out_sig)
    ref_sig = ref_sig - np.mean(ref_sig)
    out_sig = out_sig - np.mean(out_sig)
    ref_energy = np.sum(ref_sig ** 2) + eps
    proj = np.sum(ref_sig * out_sig) * ref_sig / ref_energy
    noise = out_sig - proj
    return 10 * np.log10(np.sum(proj ** 2) / (np.sum(noise ** 2) + eps))


def get_ref_id(test_type):
    """Get 0-based reference mic index from array name."""
    clean_name = test_type.removesuffix("_preprocessed")
    key = test_type if test_type in REF_IDX_MAP else clean_name + "_preprocessed"
    return REF_IDX_MAP.get(key, 1) - 1


def evaluate(args):
    drop_probs = [float(x) for x in args.drop_probs.split(',')] if args.drop_probs else None

    model = conv_tasnet_model.TasNet.load_model(
        args.model_path, mode=args.mode, dropout_type=args.dropout_type,
        drop_prob=args.drop_prob, max_drop=args.max_drop, drop_probs=drop_probs)
    print(f"Model loaded: mode={args.mode}")
    model.eval()
    if args.use_cuda and torch.cuda.is_available():
        model.cuda()

    steering_dir = os.path.join(os.path.dirname(__file__), '..',
                                'datasets', 'experiment_full_anm', 'steering')
    grid_path = os.path.join(os.path.dirname(__file__), '..',
                             'datasets', 'experiment_full_anm', 'utils', 'Lebvedev2702.mat')

    test_types = sorted([
        d for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d)) and not d.startswith('.')
    ])

    for test_type in test_types:
        array_name = test_type.removesuffix("_preprocessed")
        ref_id = get_ref_id(test_type)

        if args.mode == 'ambidrop':
            steer_path = os.path.join(steering_dir, f"{array_name}.mat")
            if not os.path.exists(steer_path):
                print(f"Skipping {test_type}: steering not found")
                continue
            V = loadmat(steer_path)["V"]
            grid_mat = loadmat(grid_path)
            th, ph = grid_mat["th"].squeeze(), grid_mat["ph"].squeeze()
            test_ds = MatDatasetTest_ASM(os.path.join(args.data_dir, test_type), V, th, ph)
        else:
            test_ds = SimDS_preprocessed(os.path.join(args.data_dir, test_type), '.', mode='baseline')

        data_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

        metrics = {k: [] for k in ['sisdr_noisy', 'sisdr_enhanced',
                                     'pesq_noisy', 'pesq_enhanced',
                                     'stoi_noisy', 'stoi_enhanced']}

        if not args.no_wandb:
            wandb.init(project=f"ConvTasNet_{args.mode}_test",
                       entity="tatarjit-ben-gurion-university-of-the-negev",
                       name=array_name, reinit=True)

        with torch.no_grad():
            for data in data_loader:
                noisy_mic_batch = None

                if isinstance(data, (tuple, list)) and len(data) == 4:
                    noisy_mic_batch, clean_mic, noisy_batch, clean_batch = data
                    ref_ids_tensor = None
                elif isinstance(data, (tuple, list)) and len(data) >= 5:
                    noisy_batch, clean_batch, ref_ids_tensor, _, _ = data
                    batch_idx = torch.arange(clean_batch.shape[0])
                    clean_batch = clean_batch[batch_idx, ref_ids_tensor, :]
                else:
                    noisy_batch, clean_batch = data[0], data[1]
                    ref_ids_tensor = torch.full((noisy_batch.shape[0],), ref_id, dtype=torch.long)
                    if clean_batch.dim() == 3 and clean_batch.shape[1] > 1:
                        clean_batch = clean_batch[:, ref_id, :]
                    elif clean_batch.dim() == 3:
                        clean_batch = clean_batch.squeeze(1)

                clean_energy = torch.sqrt(torch.mean(clean_batch**2, dim=-1))
                if (clean_energy < 1e-4).any():
                    continue

                batch_size = noisy_batch.shape[0]
                num_samples = noisy_batch.shape[2]
                mixture_lengths = torch.full((batch_size,), num_samples, dtype=torch.int64)

                if args.use_cuda and torch.cuda.is_available():
                    noisy_batch = noisy_batch.cuda()
                    mixture_lengths = mixture_lengths.cuda()
                    clean_batch = clean_batch.cuda()
                    if ref_ids_tensor is not None:
                        ref_ids_tensor = ref_ids_tensor.cuda()
                    if noisy_mic_batch is not None:
                        noisy_mic_batch = noisy_mic_batch.cuda()

                padded_source = clean_batch.unsqueeze(1)
                estimate_source = model(noisy_batch, ref_ids=ref_ids_tensor)

                if noisy_mic_batch is not None:
                    mixture_ref = noisy_mic_batch[:, ref_id, :].unsqueeze(1)
                    clean_ref_for_noisy = clean_mic[:, ref_id, :] if clean_mic.dim() == 3 else clean_mic
                else:
                    mixture_ref = noisy_batch[:, ref_id, :].unsqueeze(1)
                    clean_ref_for_noisy = clean_batch

                mixture_ref = mixture_ref.view(batch_size, -1)
                noisy_mixture_lengths = torch.full((batch_size,), mixture_ref.shape[1], dtype=torch.int64)

                mixture = remove_pad(mixture_ref, noisy_mixture_lengths)
                source = remove_pad(padded_source, mixture_lengths)
                estimate_source = remove_pad(estimate_source, mixture_lengths)

                if noisy_mic_batch is not None:
                    clean_for_noisy = clean_ref_for_noisy / (clean_ref_for_noisy.abs().max() + 1e-8)
                    noisy_src_ref = remove_pad(clean_for_noisy.unsqueeze(1), noisy_mixture_lengths)
                else:
                    noisy_src_ref = source

                for idx_sample in range(len(mixture)):
                    mix = np.squeeze(mixture[idx_sample]).real.astype('float32')
                    src_ref = np.squeeze(source[idx_sample])
                    src_est = np.squeeze(estimate_source[idx_sample])
                    mix = mix / (np.abs(mix).max() + 1e-8)
                    src_ref = src_ref / (np.abs(src_ref).max() + 1e-8)
                    src_est = src_est / (np.abs(src_est).max() + 1e-8)

                    if noisy_mic_batch is not None:
                        noisy_clean = np.squeeze(noisy_src_ref[idx_sample])
                        noisy_clean = noisy_clean / (np.abs(noisy_clean).max() + 1e-8)
                    else:
                        noisy_clean = src_ref

                    metrics['sisdr_noisy'].append(cal_sisnr(noisy_clean, mix))
                    metrics['sisdr_enhanced'].append(cal_sisnr(src_ref, src_est))
                    metrics['stoi_noisy'].append(stoi(noisy_clean, mix, 16000, extended=False))
                    metrics['pesq_noisy'].append(pesq(16000, noisy_clean, mix, mode="wb"))
                    metrics['stoi_enhanced'].append(stoi(src_ref, src_est, 16000, extended=False))
                    metrics['pesq_enhanced'].append(pesq(16000, src_ref, src_est, mode="wb"))

        if not args.no_wandb:
            wandb.log({f"test/{k}": float(np.mean(v)) for k, v in metrics.items() if v})
            wandb.finish()

        si_sdri = np.mean(metrics['sisdr_enhanced']) - np.mean(metrics['sisdr_noisy'])
        print(f"---------- {array_name} ----------")
        print(f"  SI-SDR: {np.mean(metrics['sisdr_noisy']):.2f} -> {np.mean(metrics['sisdr_enhanced']):.2f} ({si_sdri:+.2f})")
        print(f"  PESQ:   {np.mean(metrics['pesq_noisy']):.2f} -> {np.mean(metrics['pesq_enhanced']):.2f}")
        print(f"  STOI:   {np.mean(metrics['stoi_noisy']):.3f} -> {np.mean(metrics['stoi_enhanced']):.3f}")


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    evaluate(args)
