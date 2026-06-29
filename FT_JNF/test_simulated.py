"""
Unified testing script for simulated array data (baseline and AmbiDrop).

Examples:
    # Test AmbiDrop on test arrays
    python scripts/test_simulated.py --mode ambidrop \
        --checkpoint checkpoints/SH_FT_JNF,2025-12-01_10-08-18.pt \
        --data-dir datasets/experiment_full_anm/test_of_test_ds_preprocessed \
        --input-dim 18 --hidden1 64 --hidden2 64 \
        --dropout-type SHChannelDropout --drop-prob 0.4 --max-drop 3 \
        --epoch 200

    # Test baseline on train arrays
    python scripts/test_simulated.py --mode baseline \
        --checkpoint checkpoints/FT_JNF,2026-03-25_13-37-42.pt \
        --data-dir datasets/experiment_full_anm/test_of_train_ds_preprocessed \
        --input-dim 14 --hidden1 64 --hidden2 64

    # Test specific array only
    python scripts/test_simulated.py --mode ambidrop \
        --checkpoint checkpoints/SH_FT_JNF,2025-12-01_10-08-18.pt \
        --data-dir datasets/experiment_full_anm/test_of_train_ds_preprocessed \
        --test-type "full circle (rigid) radius = 0.1_preprocessed" \
        --input-dim 18 --hidden1 64 --hidden2 64 \
        --dropout-type SHChannelDropout --drop-prob 0.4 --max-drop 3

    # Test with random channel zeroing (quick mic failure test)
    python scripts/test_simulated.py --mode ambidrop \
        --checkpoint ... --zero-channels 2
"""

import os
import sys
import argparse

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import numpy as np
from torch.utils.data import DataLoader
from pesq import pesq
from pystoi import stoi
import wandb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from FT_JNF.model import FT_JNF
from ambidrop.losses import si_snr
from FT_JNF.datasets import SimDS_preprocessed
from ambidrop.checkpoint import load_checkpoint
from ambidrop.signal_utils import zero_random_channels
from ambidrop.constants import (
    REF_IDX_MAP, N_FFT, HOP_LENGTH, WIN_LENGTH, get_device
)


def parse_args():
    p = argparse.ArgumentParser(description="Test FT-JNF on simulated arrays")
    p.add_argument('--mode', choices=['baseline', 'ambidrop'], required=True)

    p.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    p.add_argument('--epoch', type=int, default=None, help='Target epoch to load')

    p.add_argument('--data-dir', required=True,
                   help='Directory containing array subdirectories')
    p.add_argument('--test-type', default=None,
                   help='Test only a specific array (subdirectory name)')
    p.add_argument('--zero-channels', type=int, default=0,
                   help='Number of random channels to zero (mic failure simulation)')

    p.add_argument('--input-dim', type=int, default=18)
    p.add_argument('--hidden1', type=int, default=64)
    p.add_argument('--hidden2', type=int, default=64)
    p.add_argument('--output-dim', type=int, default=2)

    p.add_argument('--dropout-type', default=None)
    p.add_argument('--drop-prob', type=float, default=0.4)
    p.add_argument('--max-drop', type=int, default=3)
    p.add_argument('--drop-probs', type=str, default=None)

    p.add_argument('--output-csv', default=None, help='Save results to CSV')

    p.add_argument('--wandb-project', default='AmbiDrop_test')
    p.add_argument('--wandb-entity', default='tatarjit-ben-gurion-university-of-the-negev')
    p.add_argument('--no-wandb', action='store_true')

    return p.parse_args()


def evaluate_array(net, test_type, data_dir, mode, device, zero_channels=0):
    """Evaluate a model on one array type. Returns dict of metric arrays."""
    test_ds = SimDS_preprocessed(data_dir, test_type)
    testloader = DataLoader(test_ds, batch_size=1, shuffle=False)

    num_ch = None
    ref_id = REF_IDX_MAP.get(test_type, 1) - 1

    metrics = {
        'si_sdr_noisy': [], 'si_sdr_enhanced': [],
        'pesq_noisy': [], 'pesq_enhanced': [],
        'stoi_noisy': [], 'stoi_enhanced': [],
    }

    win = torch.hamming_window(WIN_LENGTH, device=device)

    for i, data in enumerate(testloader):
        has_both = 'noisy_mic' in data

        if mode == 'ambidrop':
            x = data['noisy'].to(device)
            s1 = data['clean'].to(device)
            if has_both:
                noisy_mic = data['noisy_mic'].to(device)
                clean_mic = data['clean_mic'].to(device)
            else:
                noisy_mic = None
                clean_mic = None
        else:
            if has_both:
                x = data['noisy_mic'].to(device)
                s1 = data['clean_mic'].to(device)
            else:
                x = data['noisy'].to(device)
                s1 = data['clean'].to(device)
            noisy_mic = None
            clean_mic = None

        num_ch = x.shape[-1] // 2

        if zero_channels > 0:
            x = zero_random_channels(x, n=zero_channels)

        M = net(x)
        Ms = M[..., 0] + 1j * M[..., 1]
        Ms = Ms.squeeze()

        if mode == 'ambidrop':
            ref_ch = x[:, :, :, 0] + 1j * x[:, :, :, num_ch]
        else:
            ref_ch = x[:, :, :, ref_id] + 1j * x[:, :, :, num_ch + ref_id]

        Y = ref_ch.squeeze(0)
        S_hat = Ms * Y

        s_hat = torch.istft(S_hat.T, n_fft=N_FFT, hop_length=HOP_LENGTH,
                            win_length=WIN_LENGTH, window=win, center=True,
                            normalized=False, onesided=True, return_complex=False,
                            length=s1.shape[-1])
        y = torch.istft(Y.T, n_fft=N_FFT, hop_length=HOP_LENGTH,
                        win_length=WIN_LENGTH, window=win, center=True,
                        normalized=False, onesided=True, return_complex=False,
                        length=s1.shape[-1])

        s_hat = s_hat / s_hat.max()
        s1_eval = s1.squeeze(0)
        if s1_eval.dim() == 2:
            if mode == 'baseline':
                s1_eval = s1_eval[ref_id, :]
            else:
                s1_eval = s1_eval[0, :]
        s1_eval = s1_eval / s1_eval.max()
        y = y / y.max()

        s1_cpu = s1_eval.detach().cpu()
        s_hat_cpu = s_hat.detach().cpu()
        y_cpu = y.detach().cpu()

        if noisy_mic is not None and clean_mic is not None:
            s1_mic = clean_mic[:, ref_id, :].squeeze()
            s1_mic = s1_mic / s1_mic.max()
            num_mic_ch = noisy_mic.shape[-1] // 2
            ref_mic = noisy_mic[:, :, :, ref_id] + 1j * noisy_mic[:, :, :, num_mic_ch + ref_id]
            Y_mic = ref_mic.squeeze(0)
            y_mic = torch.istft(Y_mic.T, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                win_length=WIN_LENGTH, window=win, center=True,
                                normalized=False, onesided=True, return_complex=False,
                                length=s1_mic.shape[-1])
            y_mic = y_mic / y_mic.max()
            s1_mic_cpu = s1_mic.detach().cpu()
            y_noisy_cpu = y_mic.detach().cpu()
        else:
            s1_mic_cpu = s1_cpu
            y_noisy_cpu = y_cpu

        metrics['stoi_noisy'].append(stoi(s1_mic_cpu.numpy(), y_noisy_cpu.numpy(), 16000, extended=False))
        metrics['si_sdr_noisy'].append(si_snr(y_noisy_cpu.unsqueeze(0), s1_mic_cpu.unsqueeze(0)).item())
        metrics['pesq_noisy'].append(pesq(16000, s1_mic_cpu.numpy(), y_noisy_cpu.numpy(), mode="wb"))

        metrics['stoi_enhanced'].append(stoi(s1_cpu.numpy(), s_hat_cpu.numpy(), 16000, extended=False))
        metrics['si_sdr_enhanced'].append(si_snr(s_hat_cpu.unsqueeze(0), s1_cpu.unsqueeze(0)).item())
        metrics['pesq_enhanced'].append(pesq(16000, s1_cpu.numpy(), s_hat_cpu.numpy(), mode="wb"))

    for k in metrics:
        metrics[k] = np.array(metrics[k])

    return metrics


def main():
    args = parse_args()
    device = get_device()

    drop_probs = None
    if args.drop_probs:
        drop_probs = [float(x) for x in args.drop_probs.split(',')]

    dropout_type = args.dropout_type
    if args.mode == 'ambidrop' and dropout_type is None:
        dropout_type = 'SHChannelDropout'

    net = FT_JNF(
        input_dim=args.input_dim,
        hidden1_dim=args.hidden1,
        hidden2_dim=args.hidden2,
        output_dim=args.output_dim,
        dropout_type=dropout_type,
        drop_prob=args.drop_prob,
        max_drop=args.max_drop,
        drop_probs=drop_probs,
    ).to(device)

    info = load_checkpoint(args.checkpoint, target_epoch=args.epoch, net=net)
    print(f"Loaded epoch {info['epoch']}")

    net.eval()

    if args.test_type:
        test_types = [args.test_type]
    else:
        test_types = sorted([
            d for d in os.listdir(args.data_dir)
            if os.path.isdir(os.path.join(args.data_dir, d)) and d in REF_IDX_MAP
        ])

    if not test_types:
        print(f"No recognized array types found in {args.data_dir}")
        return

    all_results = []

    for test_type in test_types:
        print(f"\nEvaluating: {test_type}")
        metrics = evaluate_array(net, test_type, args.data_dir, args.mode, device,
                                 zero_channels=args.zero_channels)

        print(f"  Noisy  -> SI-SDR: {metrics['si_sdr_noisy'].mean():.2f} dB, "
              f"PESQ: {metrics['pesq_noisy'].mean():.2f}, "
              f"STOI: {metrics['stoi_noisy'].mean():.3f}")
        print(f"  Enhanced -> SI-SDR: {metrics['si_sdr_enhanced'].mean():.2f} dB, "
              f"PESQ: {metrics['pesq_enhanced'].mean():.2f}, "
              f"STOI: {metrics['stoi_enhanced'].mean():.3f}")

        all_results.append({
            'array': test_type,
            **{k: v.mean() for k, v in metrics.items()},
        })

        if not args.no_wandb:
            wandb.login()
            wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                       name=test_type, reinit=True)
            wandb.log({f"test/{k}": float(v.mean()) for k, v in metrics.items()})
            wandb.finish()

    if args.output_csv:
        import pandas as pd
        df = pd.DataFrame(all_results)
        df.to_csv(args.output_csv, index=False)
        print(f"\nResults saved to {args.output_csv}")

    print("\n=== Summary (averaged over all arrays) ===")
    avg_noisy = np.mean([r['si_sdr_noisy'] for r in all_results])
    avg_enhanced = np.mean([r['si_sdr_enhanced'] for r in all_results])
    avg_pesq = np.mean([r['pesq_enhanced'] for r in all_results])
    avg_stoi = np.mean([r['stoi_enhanced'] for r in all_results])
    print(f"  Noisy SI-SDR:    {avg_noisy:.2f} dB")
    print(f"  Enhanced SI-SDR: {avg_enhanced:.2f} dB  (SI-SDRi: {avg_enhanced - avg_noisy:.2f} dB)")
    print(f"  Enhanced PESQ:   {avg_pesq:.2f}")
    print(f"  Enhanced STOI:   {avg_stoi:.3f}")


if __name__ == '__main__':
    main()
