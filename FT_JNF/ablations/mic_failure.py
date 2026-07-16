"""
Microphone Failure Resilience — Figure 7

Evaluates AmbiDrop and baseline performance as microphones are randomly
deactivated. For AmbiDrop, removed channels are excluded from the steering
matrix before ASM encoding (recomputes Ambisonics from fewer mics).
For baseline, removed channels are zeroed in the mic signal.

Requires raw test data with p.wav, pDirect.wav, anm.mat per example folder,
plus steering matrices.

Usage:
    python FT_JNF/ablations/mic_failure.py \
        --data-dir datasets/experiment_full_anm/test_of_train_ds \
        --output figures/fig7_mic_failure.png

    # From pre-computed CSV
    python FT_JNF/ablations/mic_failure.py \
        --from-csv results/mic_failure.csv \
        --output figures/fig7_mic_failure.png
"""

import os
import sys
import re
import argparse

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import soundfile as sf
import scipy.io
from scipy.io import loadmat
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from FT_JNF.model import FT_JNF
from FT_JNF.constants import CHECKPOINT_REGISTRY
from ambidrop.losses import si_snr
from ambidrop.checkpoint import load_checkpoint
from ambidrop.constants import REF_IDX_MAP, get_device, N_FFT, HOP_LENGTH, WIN_LENGTH


def array_ambisonics_time(p, V, th, ph, N):
    """Compute Ambisonics from mic signals using Tikhonov ASM."""
    from ASM.tikhonov import tikhonov
    try:
        from scipy.special import sph_harm
    except ImportError:
        from scipy.special import sph_harm_y
        def sph_harm(m, n, phi, theta):
            return sph_harm_y(n, m, theta, phi)

    num_harmonics = (N + 1) ** 2
    num_samples = ph.size
    Y = np.zeros((num_harmonics, num_samples), dtype=complex)
    idx = 0
    for n_ord in range(N + 1):
        for m in range(-n_ord, n_ord + 1):
            Y[idx, :] = sph_harm(m, n_ord, ph, th)
            idx += 1

    V_t = V.T
    cnm = np.zeros((num_harmonics, V_t.shape[1], V_t.shape[2]), dtype=np.complex128)
    for nm in range(num_harmonics):
        for f in range(V_t.shape[1]):
            cnm[nm, f] = tikhonov(A=V_t[:, f, :].conj(), b=Y[nm, :])

    T = p.shape[1]
    N_mic = p.shape[0]
    filt_samp = 512
    anmt = np.zeros((num_harmonics, T), dtype=np.float32)
    for j in range(num_harmonics):
        c_f = cnm[j, :, :].T
        c_time = np.fft.irfft(c_f, n=filt_samp, axis=1)
        c_time_cs = np.roll(c_time, filt_samp // 2, axis=1)
        c_filter = np.concatenate([c_time_cs[:, [0]], c_time_cs[:, :0:-1]], axis=1)
        tmp = np.zeros(T, dtype=np.float64)
        for m_idx in range(N_mic):
            conv = np.convolve(p[m_idx, :].astype(np.float64),
                               c_filter[m_idx, :].astype(np.float64), mode='full')
            tmp += conv[filt_samp // 2: filt_samp // 2 + T]
        anmt[j, :] = tmp
    return anmt


def preprocess_with_mic_failure(folder_path, V, th, ph, num_ch_to_cancel, ref_id, device="cpu"):
    """
    Process one example with mic failure simulation.
    Removes channels from V before ASM, zeros them in mic signal.
    """
    noisy_mic, _ = sf.read(os.path.join(folder_path, "p.wav"), dtype='float64')
    noisy_mic = noisy_mic.T
    clean_mic, _ = sf.read(os.path.join(folder_path, "pDirect.wav"))
    clean_mic = clean_mic.T

    mat_data = scipy.io.loadmat(os.path.join(folder_path, "anm.mat"))
    clean_anm = mat_data["anmtDirect"].T

    num_total_ch = noisy_mic.shape[0]
    all_indices = np.arange(num_total_ch)
    pool = all_indices[all_indices != ref_id]
    num_to_draw = min(num_ch_to_cancel, len(pool))
    target_indices = np.random.choice(pool, size=num_to_draw, replace=False)

    noisy_mic_zeroed = noisy_mic.copy()
    noisy_mic_zeroed[target_indices, :] = 0

    keep_indices = np.setdiff1d(all_indices, target_indices)
    mic_for_asm = noisy_mic[np.sort(keep_indices), :]
    V_reduced = V[keep_indices, :, :]

    noisy_anm = array_ambisonics_time(mic_for_asm, V_reduced, th, ph, N=2)

    noisy_mic_t = torch.from_numpy(noisy_mic_zeroed).to(torch.complex64).to(device)
    noisy_anm_t = torch.from_numpy(noisy_anm).to(torch.complex64).to(device)
    clean_mic_t = torch.from_numpy(clean_mic).to(torch.complex64).to(device)
    clean_anm_t = torch.from_numpy(clean_anm).to(torch.complex64).to(device)

    win = torch.hamming_window(WIN_LENGTH, device=device)
    noisy_tf_mic = torch.stft(noisy_mic_t, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                               window=win, center=True, normalized=False, return_complex=True).transpose(0, 2)
    noisy_tf_anm = torch.stft(noisy_anm_t, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                               window=win, center=True, normalized=False, return_complex=True).transpose(0, 2)

    noisy_tf_mic = noisy_tf_mic[:, :N_FFT // 2 + 1, :]
    noisy_tf_anm = noisy_tf_anm[:, :N_FFT // 2 + 1, :]

    max_mic = noisy_tf_mic.abs().max().item() or 1.0
    noisy_tf_mic = noisy_tf_mic / max_mic
    clean_mic_t = clean_mic_t / max_mic

    max_anm = noisy_tf_anm.abs().max().item() or 1.0
    noisy_tf_anm = noisy_tf_anm / max_anm
    clean_anm_t = clean_anm_t / max_anm

    noisy_tf_mic = torch.cat((noisy_tf_mic.real, noisy_tf_mic.imag), dim=2)
    noisy_tf_anm = torch.cat((noisy_tf_anm.real, noisy_tf_anm.imag), dim=2)

    return noisy_tf_mic, clean_mic_t.float(), noisy_tf_anm, clean_anm_t.float()


def evaluate_mic_failure(net, mode, data_dir, V, th, ph, num_ch_to_cancel, ref_id, device):
    """Evaluate one array with a given number of cancelled mics."""
    subfolders = sorted(
        [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("ex_")],
        key=lambda f: int(re.sub(r'\D', '', f))
    )

    si_sdr_noisy_list, si_sdr_enhanced_list = [], []
    ch_num = 18 if mode == 'ambidrop' else 14
    win = torch.hamming_window(WIN_LENGTH, device=device)

    for folder in subfolders:
        folder_path = os.path.join(data_dir, folder)
        noisy_tf_mic, clean_mic, noisy_tf_anm, clean_anm = preprocess_with_mic_failure(
            folder_path, V, th, ph, num_ch_to_cancel, ref_id, device)

        if mode == 'ambidrop':
            x = noisy_tf_anm.to(device)
            s = clean_anm[0, :] if clean_anm.dim() == 2 else clean_anm
        else:
            x = noisy_tf_mic.to(device)
            s = clean_mic[ref_id, :] if clean_mic.dim() == 2 else clean_mic

        s1 = s.to(device)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        M = net(x)
        Ms = (M[..., 0] + 1j * M[..., 1]).squeeze()

        C = ch_num // 2
        if mode == 'ambidrop':
            ref_ch = x[:, :, :, 0] + 1j * x[:, :, :, C]
        else:
            ref_ch = x[:, :, :, ref_id] + 1j * x[:, :, :, C + ref_id]
        Y = ref_ch.squeeze(0)
        S_hat = Ms * Y

        y = torch.istft(Y.T, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                        window=win, center=True, normalized=False, onesided=True,
                        return_complex=False, length=s1.shape[-1])
        s_hat = torch.istft(S_hat.T, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                            window=win, center=True, normalized=False, onesided=True,
                            return_complex=False, length=s1.shape[-1])

        s_hat = (s_hat / s_hat.max()).detach().cpu()
        s1 = (s1 / s1.max()).squeeze().detach().cpu()
        y = (y / y.max()).detach().cpu()

        si_sdr_noisy_list.append(si_snr(y.unsqueeze(0), s1.unsqueeze(0)).item())
        si_sdr_enhanced_list.append(si_snr(s_hat.unsqueeze(0), s1.unsqueeze(0)).item())

    return np.mean(si_sdr_enhanced_list) - np.mean(si_sdr_noisy_list)


def plot_mic_failure(results_df, output_path):
    """Plot Fig. 7: SI-SDRi vs number of available channels."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in results_df['method'].unique():
        subset = results_df[results_df['method'] == method].sort_values('available_channels', ascending=False)
        ax.plot(subset['available_channels'], subset['si_sdri'],
                'o-', linewidth=2, markersize=8, label=method)
    ax.set_xlabel("Number of Available Channels", fontsize=14)
    ax.set_ylabel("SI-SDRi [dB]", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.invert_xaxis()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")


def main():
    p = argparse.ArgumentParser(description="Microphone failure ablation (Fig. 7)")
    p.add_argument('--data-dir', default=None,
                   help='Root dir with raw array subdirectories (needs p.wav, pDirect.wav, anm.mat)')
    p.add_argument('--arrays', nargs='+', default=None,
                   help='Specific array folders to test (default: all recognized)')
    p.add_argument('--ambidrop-checkpoint', default='SH_FT_JNF,2025-12-01_10-08-18.pt')
    p.add_argument('--baseline-checkpoint', default='FT_JNF,2026-03-25_13-37-42.pt')
    p.add_argument('--checkpoint-dir', default='checkpoints/FT_JNF')
    p.add_argument('--from-csv', default=None)
    p.add_argument('--output', default='figures/fig7_mic_failure.png')
    p.add_argument('--save-csv', default=None)
    args = p.parse_args()

    if args.from_csv:
        results_df = pd.read_csv(args.from_csv)
        plot_mic_failure(results_df, args.output)
        return

    if args.data_dir is None:
        print("Error: --data-dir required when not using --from-csv")
        return

    device = get_device()

    steering_dir = os.path.join(os.path.dirname(__file__), '..', '..',
                                'utils', 'steering')
    grid_path = os.path.join(os.path.dirname(__file__), '..', '..',
                             'utils', 'Lebvedev2702.mat')
    grid_mat = loadmat(grid_path)
    th, ph = grid_mat["th"].squeeze(), grid_mat["ph"].squeeze()

    if args.arrays:
        array_folders = args.arrays
    else:
        array_folders = sorted([
            d for d in os.listdir(args.data_dir)
            if os.path.isdir(os.path.join(args.data_dir, d)) and not d.startswith('.')
        ])

    rows = []
    for mode, ckpt_name in [('ambidrop', args.ambidrop_checkpoint), ('baseline', args.baseline_checkpoint)]:
        config = CHECKPOINT_REGISTRY[ckpt_name]
        net = FT_JNF(
            input_dim=config["input_dim"], hidden1_dim=config["hidden1"],
            hidden2_dim=config["hidden2"], output_dim=2,
            dropout_type=config.get("dropout"), drop_prob=config.get("drop_prob", 0.0),
            max_drop=config.get("max_drop", 0), drop_probs=config.get("drop_probs"),
        ).to(device)
        load_checkpoint(os.path.join(args.checkpoint_dir, ckpt_name), target_epoch=300, net=net)
        net.eval()

        method_name = 'FT-JNF + AmbiDrop' if mode == 'ambidrop' else 'FT-JNF (Baseline)'

        for n_cancel in range(0, 6):
            available = 7 - n_cancel
            print(f"\n{method_name}: {available} channels (cancelling {n_cancel})")

            sdri_per_array = []
            for array_name in array_folders:
                key = array_name + "_preprocessed"
                if key not in REF_IDX_MAP:
                    continue
                ref_id = REF_IDX_MAP[key] - 1

                steer_path = os.path.join(steering_dir, f"{array_name}.mat")
                if not os.path.exists(steer_path):
                    continue
                V = loadmat(steer_path)["V"]

                data_path = os.path.join(args.data_dir, array_name)
                if not os.path.isdir(data_path):
                    continue

                sdri = evaluate_mic_failure(net, mode, data_path, V, th, ph, n_cancel, ref_id, device)
                sdri_per_array.append(sdri)
                print(f"  {array_name}: SI-SDRi = {sdri:.2f} dB")

            mean_sdri = np.mean(sdri_per_array) if sdri_per_array else 0
            rows.append({'method': method_name, 'available_channels': available, 'si_sdri': mean_sdri})
            print(f"  Mean SI-SDRi: {mean_sdri:.2f} dB")

    results_df = pd.DataFrame(rows)
    if args.save_csv:
        os.makedirs(os.path.dirname(args.save_csv) or '.', exist_ok=True)
        results_df.to_csv(args.save_csv, index=False)

    plot_mic_failure(results_df, args.output)


if __name__ == '__main__':
    main()
