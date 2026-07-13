"""
Top-level wrapper for evaluating FT-JNF on real-world Project Aria glasses recordings.

Provides a simpler interface than FT_JNF/test_real.py: set paths once in the USER
CONFIG block, then run with a single command. Architecture is resolved automatically
from the checkpoint filename via CHECKPOINT_REGISTRY.

Public interface:
    main — parse CLI args, resolve architecture from CHECKPOINT_REGISTRY, load ATF, and run evaluation

Usage:
    # AmbiDrop with simulated ATF (default):
    python run_Real_World.py --aria-data-dir datasets/aria_ds/mixed_data_1_5int

    # Override checkpoint:
    python run_Real_World.py --checkpoint checkpoints/FT_JNF/my_model.pt

    # AmbiDrop with measured ATF:
    python run_Real_World.py --atf measured --atf-path datasets/aria_ds/aria_atfs_fixed.sofa

    # Baseline mode:
    python run_Real_World.py --mode baseline --checkpoint checkpoints/FT_JNF/FT_JNF,2026-03-25_13-37-42.pt

Architecture is resolved automatically from the checkpoint filename using
FT_JNF/constants.py CHECKPOINT_REGISTRY. Unrecognised checkpoints fall back
to the AmbiDrop defaults (input_dim=18, hidden1=64, hidden2=64,
dropout=SHChannelDropout p=0.4 max_drop=3).
"""

import os
import sys
import re
import argparse

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from pesq import pesq
from pystoi import stoi
from tqdm import tqdm

from shroom.encoders.asm import ASM
from shroom.utils.file_utils import load_file

from FT_JNF.model import FT_JNF
from FT_JNF.constants import CHECKPOINT_REGISTRY
from FT_JNF.test_real import (
    preprocess_single_example,
    load_simulated_atf,
    load_measured_atf,
    select_clean_channel,
    align_with_best_shift,
    find_best_shift_correlation,
)
from ambidrop.checkpoint import load_checkpoint
from ambidrop.constants import get_device
from ambidrop.losses import si_snr


# ============================================================
# === USER CONFIG — edit these before running ================
# ============================================================

# Default FT-JNF checkpoint to evaluate (preferred AmbiDrop model).
CHECKPOINT = "checkpoints/FT_JNF/SH_FT_JNF,2025-12-01_10-08-18.pt"

# Directory whose immediate sub-directories are named scenarios; each scenario
# folder contains ex_* sub-folders with p.wav (noisy, 48 kHz), s.wav (clean,
# 16 kHz), and best_shift.txt.
# Example layout: datasets/aria_ds/mixed_data_1_5int/ex_1/{p.wav, s.wav, ...}
# → set ARIA_DATA_DIR = "datasets/aria_ds" and each sub-dir is a scenario.
ARIA_DATA_DIR = "datasets/aria_ds"

# ATF source: "simulated" (default) or "measured".
ATF = "simulated"

# Simulated ATF files — only needed when ATF="simulated".
STEERING_PATH = "datasets/experiment_full_anm/steering/Aria on rigid sphere (simulated).mat"
GRID_PATH     = "datasets/experiment_full_anm/utils/Lebvedev2702.mat"

# Measured ATF files — only needed when ATF="measured".
SOFA_PATH    = "datasets/aria_ds/aria_atfs_fixed.sofa"
ATF_NPY_PATH = "datasets/aria_ds/ATF.npy"

# 1-based index of the Aria microphone closest to the target speaker.
REF_MIC_IDX = 3

# ============================================================

_AMBIDROP_DEFAULTS = {
    "mode": "ambidrop",
    "input_dim": 18,
    "hidden1": 64,
    "hidden2": 64,
    "dropout": "SHChannelDropout",
    "drop_prob": 0.4,
    "max_drop": 3,
}
_BASELINE_DEFAULTS = {
    "mode": "baseline",
    "input_dim": 14,
    "hidden1": 64,
    "hidden2": 64,
    "dropout": None,
    "drop_prob": 0.0,
    "max_drop": 0,
}


def _resolve_arch(checkpoint_path, mode):
    """
    Look up architecture config from CHECKPOINT_REGISTRY by filename.
    Falls back to mode-appropriate defaults if the checkpoint is not registered.
    """
    fname = os.path.basename(checkpoint_path)
    if fname in CHECKPOINT_REGISTRY:
        cfg = CHECKPOINT_REGISTRY[fname]
        return {
            "mode":       cfg.get("mode", mode),
            "input_dim":  cfg.get("input_dim", 18 if mode == "ambidrop" else 14),
            "hidden1":    cfg.get("hidden1", 64),
            "hidden2":    cfg.get("hidden2", 64),
            "dropout":    cfg.get("dropout", "SHChannelDropout" if mode == "ambidrop" else None),
            "drop_prob":  cfg.get("drop_prob", 0.4 if mode == "ambidrop" else 0.0),
            "max_drop":   cfg.get("max_drop", 3 if mode == "ambidrop" else 0),
            "drop_probs": cfg.get("drop_probs", None),
        }
    print(f"[warn] '{fname}' not in CHECKPOINT_REGISTRY — using {mode} defaults")
    defaults = _AMBIDROP_DEFAULTS if mode == "ambidrop" else _BASELINE_DEFAULTS
    return {**defaults, "drop_probs": None}


def _load_atf(args):
    """Load steering matrix V, grid angles th/ph, and sample rate."""
    if args.atf == "simulated":
        steering = args.steering_path or STEERING_PATH
        grid     = args.grid_path or GRID_PATH
        V, th, ph, fs = load_simulated_atf(steering, grid)
        print(f"ATF: simulated  ({steering})")
    else:
        sofa    = args.sofa_path or SOFA_PATH
        npy     = args.atf_npy_path or ATF_NPY_PATH
        V, th, ph, fs = load_measured_atf(sofa, npy if os.path.exists(npy) else None)
        print(f"ATF: measured   ({sofa})")
    return V, th, ph, fs


def _run_eval(net, arch, V, th, ph, fs, args, device):
    """Main evaluation loop over all scenarios in aria_data_dir."""
    aria_dir = args.aria_data_dir
    ref_mic  = args.ref_mic or REF_MIC_IDX
    mode     = arch["mode"]
    n_fft, hop, win_len = 512, 256, 512
    sample_rate = 16000

    precomputed_cnm = None
    asm_nfft = n_fft
    if args.cnm_path:
        cnm_raw = np.load(args.cnm_path)
        precomputed_cnm = cnm_raw.transpose(1, 2, 0)
        asm_nfft = precomputed_cnm.shape[1]
        precomputed_cnm = precomputed_cnm[:, :asm_nfft // 2 + 1, :]
        print(f"Loaded precomputed cnm: {precomputed_cnm.shape}")

    cnm_source = "precomputed" if precomputed_cnm is not None else "compute"
    regularization = args.regularization

    # Build shroom ASM encoder once when using measured ATF (compute path only).
    # Encodes at 48 kHz; preprocess_single_example handles downsampling.
    encode_fn = None
    if args.atf == "measured" and cnm_source == "compute":
        sofa_path = args.sofa_path or SOFA_PATH
        print(f"[shroom] Building ASM encoder from {sofa_path}")
        _array = load_file(sofa_path)
        _array.toFreq()
        _asm = ASM(sh_order=2, array=_array, fs=_array.fs)
        _asm.calculate()
        _offset = _asm.cnm.data.shape[-1] // 2
        encode_fn = lambda mic: (
            _asm.encode_amb(mic.T).data[0, :, _offset : _offset + mic.shape[1]].real.astype(np.float32)
        )
        print(f"[shroom] cnm: {_asm.cnm.data.shape}  (M, nm, F_full)\n")

    if args.scenarios:
        scenarios = args.scenarios
    else:
        scenarios = sorted([
            d for d in os.listdir(aria_dir)
            if os.path.isdir(os.path.join(aria_dir, d))
        ])

    all_results = []

    for scenario in scenarios:
        input_path = os.path.join(aria_dir, scenario)
        if not os.path.isdir(input_path):
            print(f"Skipping {scenario}: not a directory")
            continue

        subfolders = sorted(
            [d for d in os.listdir(input_path)
             if os.path.isdir(os.path.join(input_path, d)) and d.startswith("ex_")],
            key=lambda f: int(re.sub(r"\D", "", f)),
        )
        if not subfolders:
            print(f"Skipping {scenario}: no ex_* folders")
            continue

        metrics = {k: [] for k in [
            "si_sdr_noisy", "si_sdr_enhanced",
            "pesq_noisy",   "pesq_enhanced",
            "stoi_noisy",   "stoi_enhanced",
        ]}

        ch_num = arch["input_dim"]
        C = ch_num // 2

        for folder in tqdm(subfolders, desc=f"  {scenario}"):
            folder_path = os.path.join(input_path, folder)

            noisy_tf_anm, noisy_tf_mic, clean_time_mic, _ = preprocess_single_example(
                folder_path=folder_path, V=V, th=th, ph=ph, nfft=n_fft, fs=fs,
                regularization=regularization, cnm_source=cnm_source,
                precomputed_cnm=precomputed_cnm, asm_nfft=asm_nfft, device=device,
                encode_fn=encode_fn,
            )

            clean_signal = select_clean_channel(clean_time_mic, idx=ref_mic - 1)

            x = noisy_tf_anm if mode == "ambidrop" else noisy_tf_mic
            s1 = clean_signal.reshape(-1).to(device)  # ensure 1D for alignment
            x  = x.to(device)
            if x.dim() == 3:
                x = x.unsqueeze(0)

            with torch.no_grad():
                M = net(x)
            Ms = M[..., 0] + 1j * M[..., 1]
            Ms = Ms.squeeze()

            if mode == "ambidrop":
                ref_ch = x[:, :, :, 0] + 1j * x[:, :, :, C]
            else:
                ref_ch = x[:, :, :, ref_mic - 1] + 1j * x[:, :, :, C + ref_mic - 1]
            Y_ch = ref_ch.squeeze(0)
            S_hat = Ms * Y_ch

            sig_len = s1.shape[-1]
            win = torch.hamming_window(window_length=win_len, device=device)
            istft_kwargs = dict(n_fft=n_fft, hop_length=hop, win_length=win_len,
                                window=win, center=True, normalized=False,
                                onesided=True, return_complex=False, length=sig_len)
            y_td    = torch.istft(Y_ch.T, **istft_kwargs)
            s_hat_td = torch.istft(S_hat.T, **istft_kwargs)

            s_hat_td = (s_hat_td / s_hat_td.abs().max()).detach().cpu()
            s1_td    = (s1      / s1.abs().max()     ).detach().cpu()
            y_td     = (y_td    / y_td.abs().max()   ).detach().cpu()

            best_shift = find_best_shift_correlation(s1_td, s_hat_td)
            s1_a, _, s_hat_a = align_with_best_shift(s1_td, y_td, s_hat_td, best_shift)

            # ── noisy reference mic ───────────────────────────────────────────
            noisy_mic_tf = noisy_tf_mic.to(device)
            nm_C = noisy_mic_tf.shape[-1] // 2
            ref_mic_tf = noisy_mic_tf[:, :, ref_mic - 1] + 1j * noisy_mic_tf[:, :, nm_C + ref_mic - 1]
            y_mic = torch.istft(ref_mic_tf.squeeze(0).T, **istft_kwargs)
            y_mic = (y_mic / y_mic.abs().max()).detach().cpu()
            best_shift_mic = find_best_shift_correlation(s1_td, y_mic)
            s1_mic, y_mic_a, _ = align_with_best_shift(s1_td.numpy(), y_mic.numpy(), y_mic.numpy(), best_shift_mic)
            s1_mic = torch.from_numpy(s1_mic).cpu()
            y_mic_a = torch.from_numpy(y_mic_a).cpu()

            # align_with_best_shift returns numpy; convert for si_snr (torch) and pesq/stoi (numpy)
            s1_a_t    = torch.from_numpy(np.asarray(s1_a,    dtype=np.float32)).cpu()
            s_hat_a_t = torch.from_numpy(np.asarray(s_hat_a, dtype=np.float32)).cpu()

            metrics["si_sdr_noisy"].append(    si_snr(y_mic_a.unsqueeze(0),    s1_mic.unsqueeze(0)).item())
            metrics["pesq_noisy"].append(      pesq(sample_rate, s1_mic.numpy(),    y_mic_a.numpy(),    mode="wb"))
            metrics["stoi_noisy"].append(      stoi(s1_mic.numpy(),                 y_mic_a.numpy(),    sample_rate, extended=False))
            metrics["si_sdr_enhanced"].append( si_snr(s_hat_a_t.unsqueeze(0),      s1_a_t.unsqueeze(0)).item())
            metrics["pesq_enhanced"].append(   pesq(sample_rate, s1_a_t.numpy(),   s_hat_a_t.numpy(),  mode="wb"))
            metrics["stoi_enhanced"].append(   stoi(s1_a_t.numpy(),                s_hat_a_t.numpy(),  sample_rate, extended=False))

        for k in metrics:
            metrics[k] = np.array(metrics[k])

        si_sdri = metrics["si_sdr_enhanced"].mean() - metrics["si_sdr_noisy"].mean()
        print(f"\n{scenario}:")
        print(f"  Noisy     SI-SDR: {metrics['si_sdr_noisy'].mean():.2f} dB  "
              f"PESQ: {metrics['pesq_noisy'].mean():.2f}  "
              f"STOI: {metrics['stoi_noisy'].mean():.3f}")
        print(f"  Enhanced  SI-SDR: {metrics['si_sdr_enhanced'].mean():.2f} dB  "
              f"PESQ: {metrics['pesq_enhanced'].mean():.2f}  "
              f"STOI: {metrics['stoi_enhanced'].mean():.3f}")
        print(f"  SI-SDRi:  {si_sdri:.2f} dB")

        all_results.append({
            "scenario": scenario,
            "atf": args.atf,
            **{k: float(v.mean()) for k, v in metrics.items()},
            "si_sdri": float(si_sdri),
        })

    return all_results


def main():
    p = argparse.ArgumentParser(
        description="Evaluate FT-JNF on real-world Aria glasses recordings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--checkpoint", default=None,
                   help=f"FT-JNF checkpoint path (default: {CHECKPOINT})")
    p.add_argument("--aria-data-dir", default=None,
                   help=f"Directory with scenario sub-directories (default: {ARIA_DATA_DIR})")
    p.add_argument("--atf", choices=["simulated", "measured"], default=None,
                   help=f"ATF source (default: {ATF})")
    p.add_argument("--atf-path", default=None,
                   help="Path to SOFA file when --atf measured (overrides SOFA_PATH)")
    p.add_argument("--mode", choices=["ambidrop", "baseline"], default=None,
                   help="Inference mode (default: resolved from CHECKPOINT_REGISTRY)")

    p.add_argument("--ref-mic", type=int, default=None,
                   help=f"1-based closest mic index to target (default: {REF_MIC_IDX})")
    p.add_argument("--scenarios", nargs="+", default=None,
                   help="Scenario names to test (default: auto-detect all)")
    p.add_argument("--regularization", choices=["tikhonov", "svd"], default="tikhonov",
                   help="ASM regularization method")
    p.add_argument("--cnm-path", default=None,
                   help="Path to precomputed cnm .npy — skips ASM coefficient computation")

    p.add_argument("--steering-path", default=None, help="Override STEERING_PATH")
    p.add_argument("--grid-path", default=None, help="Override GRID_PATH")
    p.add_argument("--sofa-path", default=None, help="Override SOFA_PATH")
    p.add_argument("--atf-npy-path", default=None, help="Override ATF_NPY_PATH")

    p.add_argument("--output-csv", default=None, help="Save results to a CSV file")
    args = p.parse_args()

    # Apply USER CONFIG defaults for args not provided on the command line.
    ckpt_path  = args.checkpoint or CHECKPOINT
    aria_dir   = args.aria_data_dir or ARIA_DATA_DIR
    atf_source = args.atf or ATF
    args.atf           = atf_source
    args.aria_data_dir = aria_dir
    if args.atf == "measured" and args.atf_path:
        args.sofa_path = args.atf_path

    device = get_device()

    # ── Resolve architecture from registry ──────────────────────────────────
    mode_hint = args.mode or ("ambidrop" if "SH_" in os.path.basename(ckpt_path) else "baseline")
    arch = _resolve_arch(ckpt_path, mode_hint)
    if args.mode:
        arch["mode"] = args.mode

    print(f"Checkpoint : {ckpt_path}")
    print(f"Mode       : {arch['mode']}")
    print(f"Arch       : input_dim={arch['input_dim']}  "
          f"hidden1={arch['hidden1']}  hidden2={arch['hidden2']}  "
          f"dropout={arch['dropout']}")

    # ── Build and load model ─────────────────────────────────────────────────
    net = FT_JNF(
        input_dim=arch["input_dim"],
        hidden1_dim=arch["hidden1"],
        hidden2_dim=arch["hidden2"],
        output_dim=2,
        dropout_type=arch["dropout"],
        drop_prob=arch["drop_prob"],
        max_drop=arch["max_drop"],
        drop_probs=arch.get("drop_probs"),
    ).to(device)

    info = load_checkpoint(ckpt_path, net=net)
    print(f"Loaded epoch {info.get('epoch', '?')}\n")
    net.eval()

    # ── Load ATF ────────────────────────────────────────────────────────────
    V, th, ph, fs = _load_atf(args)

    # ── Run evaluation ───────────────────────────────────────────────────────
    all_results = _run_eval(net, arch, V, th, ph, fs, args, device)

    if not all_results:
        print("No results — check --aria-data-dir and --scenarios.")
        return

    print("\n" + "=" * 60)
    overall_sdri = np.mean([r["si_sdri"] for r in all_results])
    print(f"Overall SI-SDRi: {overall_sdri:.2f} dB  ({len(all_results)} scenario(s))")

    if args.output_csv:
        import pandas as pd
        pd.DataFrame(all_results).to_csv(args.output_csv, index=False)
        print(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
