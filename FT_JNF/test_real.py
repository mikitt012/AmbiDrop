"""
Evaluation script for FT-JNF on real-world Project Aria glasses recordings.

Tests AmbiDrop or baseline FT-JNF on recordings from Project Aria glasses,
with on-the-fly ASM encoding, temporal alignment, and configurable ATF sources.

Public interface:
    preprocess_single_example — load one ex_* folder, apply ASM, return STFT tensors
    load_simulated_atf — load steering matrix V and grid angles from .mat files
    load_measured_atf — load measured ATF from a SOFA file
    select_clean_channel — select a target channel from a multichannel clean signal
    align_with_best_shift — temporally align three signals using a precomputed lag
    find_best_shift_correlation — find temporal alignment lag via cross-correlation

Examples:
    # AmbiDrop with simulated ATF, compute cnm on-the-fly with Tikhonov
    python scripts/test_real.py --mode ambidrop \
        --checkpoint checkpoints/SH_FT_JNF,2025-12-01_10-08-18.pt \
        --aria-data-dir datasets/aria_ds/mixed_data \
        --atf simulated --cnm-source compute --regularization tikhonov

    # AmbiDrop with measured ATF
    python scripts/test_real.py --mode ambidrop \
        --checkpoint checkpoints/SH_FT_JNF,2025-12-01_10-08-18.pt \
        --aria-data-dir datasets/aria_ds/mixed_data \
        --atf measured

    # Baseline on Aria data
    python scripts/test_real.py --mode baseline \
        --checkpoint checkpoints/FT_JNF,2026-03-25_13-37-42.pt \
        --aria-data-dir datasets/aria_ds/mixed_data

    # With specific scenarios
    python scripts/test_real.py --mode ambidrop \
        --checkpoint ... --scenarios not_blocked blocked
"""

import os
import sys
import re
import argparse
import pickle
from math import factorial, pi, sqrt

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
import sofar
from scipy.io import loadmat
from scipy.signal import resample_poly, correlate, correlation_lags
from scipy.special import lpmv
try:
    from scipy.special import sph_harm
except ImportError:
    from scipy.special import sph_harm_y
    def sph_harm(m, n, phi, theta):
        return sph_harm_y(n, m, theta, phi)
from pesq import pesq
from pystoi import stoi
from tqdm import tqdm
import wandb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from FT_JNF.model import FT_JNF
from ambidrop.losses import si_snr
from ambidrop.checkpoint import load_checkpoint
from ambidrop.constants import get_device, N_FFT, HOP_LENGTH, WIN_LENGTH
from ambidrop.asm import encode_ambisonics, apply_asm_filters

import matplotlib
matplotlib.use("Agg")


# ── Spherical Harmonics ──────────────────────────────────────────────────────

def sh2(N, theta, phi):
    """Compute complex spherical harmonics up to order N using Legendre functions."""
    theta = np.atleast_1d(theta)
    phi = np.atleast_1d(phi)
    if len(theta) != len(phi):
        raise ValueError("Lengths of theta and phi must be equal!")
    L = len(theta)
    Y = [np.sqrt(1/(4*pi)) * np.ones(L, dtype=complex)]
    j = 1j
    for n in range(1, N+1):
        Y1 = []
        for m in range(0, n+1):
            a = sqrt((2*n+1)/(4*pi) * factorial(n-m)/factorial(n+m))
            Pnm = lpmv(m, n, np.cos(theta))
            Ynm = a * Pnm * np.exp(j*m*phi)
            Y1.append(Ynm)
        Y1 = np.vstack(Y1)
        Y2 = []
        for m in range(-n, 0):
            Ynm = (-1)**m * np.conj(Y1[-m-1, :])
            Y2.append(Ynm)
        if Y2:
            Y2 = np.vstack(Y2)
            Y_stack = np.vstack([Y2, Y1])
        else:
            Y_stack = Y1
        Y.append(Y_stack)
    Y = np.vstack(Y)
    return Y


# ── Temporal alignment ───────────────────────────────────────────────────────


def align_with_best_shift(s1, y, s_hat, best_shift):
    """Align three signals using the computed best shift."""
    s1, y, s_hat = np.asarray(s1), np.asarray(y), np.asarray(s_hat)
    T = min(len(s1), len(y), len(s_hat))
    s1, y, s_hat = s1[:T], y[:T], s_hat[:T]
    k = int(best_shift)
    if k > 0:
        s1_new, y_new, s_hat_new = s1[k:], y[:-k], s_hat[:-k]
    elif k < 0:
        k = -k
        s1_new, y_new, s_hat_new = s1[:-k], y[k:], s_hat[k:]
    else:
        s1_new, y_new, s_hat_new = s1, y, s_hat
    L = min(len(s1_new), len(y_new), len(s_hat_new))
    return s1_new[:L], y_new[:L], s_hat_new[:L]


def find_best_shift_correlation(ref_sig, est_sig):
    """Find best alignment shift via cross-correlation."""
    ref_sig = np.asarray(ref_sig).flatten()
    est_sig = np.asarray(est_sig).flatten()
    corr = correlate(ref_sig, est_sig, mode='full')
    lags = correlation_lags(len(ref_sig), len(est_sig), mode='full')
    best_idx = np.argmax(np.abs(corr))
    return lags[best_idx]


# ── Data helpers ─────────────────────────────────────────────────────────────

def select_clean_channel(clean_time_mic, idx=0):
    """Select a target channel from multichannel clean signal."""
    if clean_time_mic.ndim == 1:
        return clean_time_mic.unsqueeze(0)
    if clean_time_mic.ndim == 2 and clean_time_mic.shape[0] == 1:
        return clean_time_mic.unsqueeze(0)
    C, T = clean_time_mic.shape
    return clean_time_mic[idx, :].unsqueeze(0)


def preprocess_single_example(folder_path, V, th, ph, nfft, fs,
                               regularization='tikhonov', cnm_source='compute',
                               precomputed_cnm=None, asm_nfft=None, device="cpu",
                               encode_fn=None):
    """
    Process one real-world example folder.
    Expected files: p.wav, s.wav, best_shift.txt

    encode_fn : callable (M, T_48k) -> (nm, T_48k), optional
        When provided, used instead of encode_ambisonics for the 'compute' path.
        Intended for measured ATF via shroom's ASM class (encodes at 48 kHz, the
        caller is responsible for resampling). V/th/ph are unused in this case.
    """
    array_file = os.path.join(folder_path, "p.wav")
    direct_file = os.path.join(folder_path, "s.wav")

    with open(os.path.join(folder_path, "best_shift.txt"), "r") as f:
        best_shift = int(f.read())

    noisy_speech_mic, fs_rec = sf.read(array_file)
    noisy_speech_mic = noisy_speech_mic.T  # (C, T) at 48kHz

    clean_speech_mic, fs_clean = sf.read(direct_file)
    clean_speech_mic = clean_speech_mic.T  # already 16kHz

    if cnm_source == 'compute':
        if encode_fn is not None:
            # measured ATF via shroom: encode at 48 kHz, then downsample
            noisy_speech_anm = encode_fn(noisy_speech_mic)                          # (nm, T_48k)
            noisy_speech_anm = resample_poly(noisy_speech_anm, up=1, down=3, axis=1)
            noisy_speech_mic = resample_poly(noisy_speech_mic, up=1, down=3, axis=1)
        else:
            # simulated ATF: downsample first, then encode at 16 kHz with Tikhonov/SVD
            noisy_speech_mic = resample_poly(noisy_speech_mic, up=1, down=3, axis=1)
            noisy_speech_anm, _ = encode_ambisonics(
                noisy_speech_mic, V, sh_order=2, th=th, ph=ph, method=regularization
            )

    elif cnm_source == 'precomputed':
        filt_len = asm_nfft if asm_nfft is not None else nfft
        noisy_speech_anm = apply_asm_filters(noisy_speech_mic, precomputed_cnm, filt_samp=filt_len)

        noisy_speech_anm = resample_poly(noisy_speech_anm, up=1, down=3, axis=1)
        noisy_speech_mic = resample_poly(noisy_speech_mic, up=1, down=3, axis=1)

    else:
        raise ValueError(f"Unknown cnm_source: {cnm_source}")

    noisy_speech_mic = torch.from_numpy(noisy_speech_mic).to(torch.complex64).to(device)
    clean_speech_mic = torch.from_numpy(clean_speech_mic).to(torch.complex64).to(device)
    noisy_speech_anm = torch.from_numpy(noisy_speech_anm).to(torch.complex64).to(device)

    n_fft, hop, win_len = 512, 256, 512
    win = torch.hamming_window(window_length=win_len, device=noisy_speech_mic.device)

    noisy_tf_mic = torch.stft(noisy_speech_mic, n_fft=n_fft, hop_length=hop, win_length=win_len,
                               window=win, center=True, normalized=False, return_complex=True).transpose(0, 2)
    noisy_tf_anm = torch.stft(noisy_speech_anm, n_fft=n_fft, hop_length=hop, win_length=win_len,
                               window=win, center=True, normalized=False, return_complex=True).transpose(0, 2)

    noisy_tf_mic = noisy_tf_mic[:, :n_fft//2+1, :]
    noisy_tf_anm = noisy_tf_anm[:, :n_fft//2+1, :]

    max_val_mic = noisy_tf_mic.abs().max().item() or 1.0
    noisy_tf_mic = noisy_tf_mic / max_val_mic
    clean_speech_mic = clean_speech_mic / max_val_mic

    max_val_anm = noisy_tf_anm.abs().max().item() or 1.0
    noisy_tf_anm = noisy_tf_anm / max_val_anm

    noisy_tf_mic = torch.cat((noisy_tf_mic.real, noisy_tf_mic.imag), dim=2)
    noisy_tf_anm = torch.cat((noisy_tf_anm.real, noisy_tf_anm.imag), dim=2)

    clean_time_mic = clean_speech_mic.float()

    return noisy_tf_anm, noisy_tf_mic, clean_time_mic, best_shift


# ── ATF Loading ──────────────────────────────────────────────────────────────

def load_simulated_atf(steering_path, grid_path):
    """Load simulated ATF from .mat files."""
    steer_mat = loadmat(steering_path)
    V = steer_mat["V"]
    grid_mat = loadmat(grid_path)
    th = grid_mat["th"].squeeze()
    ph = grid_mat["ph"].squeeze()
    return V, th, ph, 16000


def load_measured_atf(sofa_path, atf_npy_path=None):
    """Load measured ATF from SOFA file (and optional precomputed .npy)."""
    sofa_data = sofar.read_sofa(sofa_path)
    ir = sofa_data.Data_IR
    fs = sofa_data.Data_SamplingRate
    ir = ir.transpose(1, 2, 0)
    directions = sofa_data.SourcePosition.T
    ph = directions[0, :]
    th = directions[1, :]
    ir_t = torch.from_numpy(ir)
    n_fft = 332
    ir_fft_full = torch.fft.fft(ir_t, n=n_fft, dim=1)
    V = ir_fft_full[:, :n_fft//2 + 1, :].detach().cpu().numpy()

    if atf_npy_path and os.path.exists(atf_npy_path):
        V_yo = np.load(atf_npy_path)
        V_yo = V_yo.transpose(0, 2, 1)
        V_yo = V_yo[:, :n_fft//2 + 1, :]
        V = V_yo

    return V, th, ph, fs


# ── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Test FT-JNF on real-world Aria glasses data")

    p.add_argument('--mode', choices=['baseline', 'ambidrop'], required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--epoch', type=int, default=200)

    p.add_argument('--aria-data-dir', required=True,
                   help='Directory containing scenario subdirectories with ex_* folders')
    p.add_argument('--scenarios', nargs='+', default=None,
                   help='Scenario names to test (default: auto-detect)')

    p.add_argument('--atf', choices=['simulated', 'measured'], default='simulated')
    p.add_argument('--cnm-source', choices=['compute', 'precomputed'], default='compute')
    p.add_argument('--regularization', choices=['tikhonov', 'svd'], default='tikhonov')

    p.add_argument('--steering-path', default=None,
                   help='Path to simulated steering .mat (default: auto)')
    p.add_argument('--grid-path', default=None,
                   help='Path to Lebedev grid .mat (default: auto)')
    p.add_argument('--sofa-path', default=None,
                   help='Path to measured SOFA file (default: auto)')
    p.add_argument('--atf-npy-path', default=None,
                   help='Path to precomputed ATF .npy (default: auto)')
    p.add_argument('--cnm-path', default=None,
                   help='Path to precomputed cnm .npy (for --cnm-source precomputed)')

    p.add_argument('--closest-mic', type=int, default=3,
                   help='1-based index of closest microphone to target')

    p.add_argument('--input-dim', type=int, default=18)
    p.add_argument('--hidden1', type=int, default=64)
    p.add_argument('--hidden2', type=int, default=64)
    p.add_argument('--dropout-type', default=None)
    p.add_argument('--drop-prob', type=float, default=0.4)
    p.add_argument('--max-drop', type=int, default=3)
    p.add_argument('--drop-probs', type=str, default=None)

    p.add_argument('--output-csv', default=None)
    p.add_argument('--no-wandb', action='store_true')
    p.add_argument('--wandb-project', default='Lab_Experiment')
    p.add_argument('--wandb-entity', default='tatarjit-ben-gurion-university-of-the-negev')

    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = get_device()
    torch.set_default_device(device)

    base_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'experiment_full_anm')

    if args.atf == 'simulated':
        steering_path = args.steering_path or os.path.join(
            base_dir, 'steering', 'Aria on rigid sphere (simulated).mat')
        grid_path = args.grid_path or os.path.join(base_dir, 'utils', 'Lebvedev2702.mat')
        V, th, ph, fs = load_simulated_atf(steering_path, grid_path)
        print("---- simulated ATF ----")
    else:
        sofa_path = args.sofa_path or os.path.join(
            os.path.dirname(__file__), '..', 'datasets', 'aria_ds', 'aria_atfs_fixed.sofa')
        atf_npy = args.atf_npy_path or os.path.join(
            os.path.dirname(__file__), '..', 'datasets', 'aria_ds', 'ATF.npy')
        V, th, ph, fs = load_measured_atf(sofa_path, atf_npy)
        print("---- measured ATF ----")

    precomputed_cnm = None
    asm_nfft = 512
    if args.cnm_source == 'precomputed' and args.cnm_path:
        cnm_raw = np.load(args.cnm_path)
        print(f"Loaded precomputed cnm from {args.cnm_path}, raw shape: {cnm_raw.shape}")
        # cnm_raw is (M, nm, F) -> transpose to (nm, F, M)
        precomputed_cnm = cnm_raw.transpose(1, 2, 0)
        asm_nfft = precomputed_cnm.shape[1]  # F dimension = nfft (e.g. 332)
        # Slice to positive frequencies only
        precomputed_cnm = precomputed_cnm[:, :asm_nfft // 2 + 1, :]
        print(f"  -> (nm, F_pos, M) = {precomputed_cnm.shape}, asm_nfft={asm_nfft}")

    drop_probs = None
    if args.drop_probs:
        drop_probs = [float(x) for x in args.drop_probs.split(',')]

    ch_num = args.input_dim
    dropout_type = args.dropout_type
    if args.mode == 'ambidrop' and dropout_type is None:
        dropout_type = 'SHChannelDropout'

    net = FT_JNF(
        input_dim=ch_num, hidden1_dim=args.hidden1, hidden2_dim=args.hidden2,
        output_dim=2, dropout_type=dropout_type,
        drop_prob=args.drop_prob, max_drop=args.max_drop, drop_probs=drop_probs,
    ).to(device)

    info = load_checkpoint(args.checkpoint, target_epoch=args.epoch, net=net)
    print(f"Loaded epoch {info['epoch']}")
    net.eval()

    closest_mic_idx = args.closest_mic
    sample_rate = 16000
    n_fft, hop, win_len = 512, 256, 512

    if args.scenarios:
        scenarios = args.scenarios
    else:
        scenarios = sorted([
            d for d in os.listdir(args.aria_data_dir)
            if os.path.isdir(os.path.join(args.aria_data_dir, d))
        ])

    all_results = []

    for scenario in scenarios:
        data_type = scenario

        input_path = os.path.join(args.aria_data_dir, data_type)
        if not os.path.isdir(input_path):
            print(f"Skipping {scenario}: directory not found at {input_path}")
            continue

        subfolders = sorted(
            [d for d in os.listdir(input_path)
             if os.path.isdir(os.path.join(input_path, d)) and d.startswith("ex_")],
            key=lambda f: int(re.sub(r'\D', '', f))
        )

        if not subfolders:
            print(f"Skipping {scenario}: no ex_* folders")
            continue

        metrics = {k: [] for k in ['si_sdr_noisy', 'si_sdr_enhanced', 'pesq_noisy',
                                     'pesq_enhanced', 'stoi_noisy', 'stoi_enhanced']}

        if not args.no_wandb:
            wandb.login()
            wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                       name=f"{scenario}_{args.atf}_{args.mode}", reinit=True)

        for folder in tqdm(subfolders, desc=f"Processing {scenario}"):
            folder_path = os.path.join(input_path, folder)

            noisy_tf_anm, noisy_tf_mic, clean_time_mic, measured_shift = preprocess_single_example(
                folder_path=folder_path, V=V, th=th, ph=ph, nfft=n_fft, fs=fs,
                regularization=args.regularization,
                cnm_source=args.cnm_source, precomputed_cnm=precomputed_cnm,
                asm_nfft=asm_nfft, device=device,
            )

            clean_signal = select_clean_channel(clean_time_mic, idx=closest_mic_idx - 1)

            if args.mode == 'ambidrop':
                x, s = noisy_tf_anm, clean_signal
            else:
                x, s = noisy_tf_mic, clean_signal

            s1 = s.to(device)
            x = x.to(device)

            if x.dim() == 3:
                x = x.unsqueeze(0)

            M = net(x)
            Ms = M[..., 0] + 1j * M[..., 1]
            Ms = Ms.squeeze()

            C = ch_num // 2
            if args.mode == 'ambidrop':
                ref_ch = x[:, :, :, 0] + 1j * x[:, :, :, C]
            else:
                ref_ch = x[:, :, :, closest_mic_idx-1] + 1j * x[:, :, :, C+closest_mic_idx-1]
            Y = ref_ch.squeeze(0)
            S_hat = Ms * Y

            sig_len = s1.shape[-1]
            win = torch.hamming_window(window_length=win_len, device=device)
            y = torch.istft(Y.T, n_fft=n_fft, hop_length=hop, win_length=win_len,
                            window=win, center=True, normalized=False,
                            onesided=True, return_complex=False, length=sig_len)
            s_hat = torch.istft(S_hat.T, n_fft=n_fft, hop_length=hop, win_length=win_len,
                                window=win, center=True, normalized=False,
                                onesided=True, return_complex=False, length=sig_len)
            s_hat = s_hat / s_hat.max()
            s1 = s1 / s1.max()
            y = y / y.max()

            s1 = s1.squeeze(0).detach().cpu()
            s_hat = s_hat.detach().cpu()
            y = y.detach().cpu()
            clean_mic = s1

            best_shift = find_best_shift_correlation(s1, s_hat)
            s1_aligned, y_aligned, s_hat_aligned = align_with_best_shift(s1, y, s_hat, best_shift)

            y_a = torch.from_numpy(y_aligned).cpu()
            s1_a = torch.from_numpy(s1_aligned).cpu()
            s_hat_a = torch.from_numpy(s_hat_aligned).cpu()

            noisy_mic = noisy_tf_mic.to(device)
            num_mic_ch = noisy_mic.shape[-1] // 2
            ref_mic = noisy_mic[:, :, closest_mic_idx-1] + 1j * noisy_mic[:, :, num_mic_ch+closest_mic_idx-1]
            Y_mic = ref_mic.squeeze(0)
            y_mic = torch.istft(Y_mic.T, n_fft=512, hop_length=256, win_length=512,
                                window=torch.hamming_window(512, device=device),
                                center=True, normalized=False, onesided=True,
                                return_complex=False, length=clean_mic.shape[0])
            y_mic = (y_mic / y_mic.max()).detach().cpu()

            best_shift_mic = find_best_shift_correlation(clean_mic, y_mic)
            s1_mic, y_mic_aligned, _ = align_with_best_shift(clean_mic.numpy(), y_mic.numpy(), y_mic.numpy(), best_shift_mic)
            s1_mic = torch.from_numpy(s1_mic).cpu()
            y_mic_aligned = torch.from_numpy(y_mic_aligned).cpu()

            metrics['stoi_noisy'].append(stoi(s1_mic, y_mic_aligned, sample_rate, extended=False))
            metrics['si_sdr_noisy'].append(si_snr(y_mic_aligned.unsqueeze(0), s1_mic.unsqueeze(0)).item())
            metrics['pesq_noisy'].append(pesq(sample_rate, s1_mic.numpy(), y_mic_aligned.numpy(), mode="wb"))

            metrics['stoi_enhanced'].append(stoi(s1_a, s_hat_a, sample_rate, extended=False))
            metrics['si_sdr_enhanced'].append(si_snr(s_hat_a.unsqueeze(0), s1_a.unsqueeze(0)).item())
            metrics['pesq_enhanced'].append(pesq(sample_rate, s1_a.numpy(), s_hat_a.numpy(), mode="wb"))

        for k in metrics:
            metrics[k] = np.array(metrics[k])

        print(f"\n{scenario}:")
        print(f"  Noisy    -> SI-SDR: {metrics['si_sdr_noisy'].mean():.2f} dB, "
              f"PESQ: {metrics['pesq_noisy'].mean():.2f}, STOI: {metrics['stoi_noisy'].mean():.3f}")
        print(f"  Enhanced -> SI-SDR: {metrics['si_sdr_enhanced'].mean():.2f} dB, "
              f"PESQ: {metrics['pesq_enhanced'].mean():.2f}, STOI: {metrics['stoi_enhanced'].mean():.3f}")

        si_sdri = metrics['si_sdr_enhanced'].mean() - metrics['si_sdr_noisy'].mean()
        print(f"  SI-SDRi: {si_sdri:.2f} dB")

        all_results.append({
            'scenario': scenario, 'atf': args.atf,
            **{k: v.mean() for k, v in metrics.items()},
        })

        if not args.no_wandb:
            wandb.log({f"test/{k}": float(v.mean()) for k, v in metrics.items()})
            wandb.finish()

    if args.output_csv and all_results:
        import pandas as pd
        pd.DataFrame(all_results).to_csv(args.output_csv, index=False)
        print(f"\nResults saved to {args.output_csv}")


if __name__ == '__main__':
    main()
