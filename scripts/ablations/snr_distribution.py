"""
SI-SDR Distribution Histogram — Figure 4

Generates a histogram showing the distribution of noisy SI-SDR values
across all recordings, with mean SI-SDR improvement (SI-SDRi) annotated
per bin.

Usage:
    # From pre-computed .npy files (generates CSV then plots)
    python scripts/ablations/snr_distribution.py \
        --noisy-npy snr_distribution_fixed2/master_si_sdr_noisy.npy \
        --enhanced-npy snr_distribution_fixed2/master_si_sdr_enhanced.npy \
        --output figures/fig4_snr_distribution.png

    # From pre-computed CSV (skip .npy step)
    python scripts/ablations/snr_distribution.py \
        --from-csv "snr_distribution_fixed2/si-sdr distribution across examples and arrays.csv" \
        --output figures/fig4_snr_distribution.png

    # Run inference to generate data
    python scripts/ablations/snr_distribution.py \
        --checkpoint checkpoints/SH_FT_JNF,2025-12-01_10-08-18.pt \
        --data-dir datasets/experiment_full_anm/test_of_train_ds_preprocessed \
        --output figures/fig4_snr_distribution.png
"""

import os
import sys
import argparse

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def run_and_collect(checkpoint_path, data_dir):
    """Run inference and collect per-example SI-SDR values."""
    from ambidrop.models import FT_JNF
    from ambidrop.checkpoint import load_checkpoint
    from ambidrop.constants import CHECKPOINT_REGISTRY, REF_IDX_MAP, get_device
    from scripts.test_simulated import evaluate_array

    ckpt_name = os.path.basename(checkpoint_path)
    config = CHECKPOINT_REGISTRY.get(ckpt_name, {
        "input_dim": 18, "hidden1": 64, "hidden2": 64,
        "dropout": "SHChannelDropout", "drop_prob": 0.4, "max_drop": 3
    })
    device = get_device()

    net = FT_JNF(
        input_dim=config["input_dim"],
        hidden1_dim=config["hidden1"],
        hidden2_dim=config["hidden2"],
        output_dim=2,
        dropout_type=config.get("dropout"),
        drop_prob=config.get("drop_prob", 0.0),
        max_drop=config.get("max_drop", 0),
        drop_probs=config.get("drop_probs"),
    ).to(device)

    load_checkpoint(checkpoint_path, target_epoch=200, net=net)
    net.eval()

    test_types = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d in REF_IDX_MAP
    ])

    all_noisy, all_enhanced = [], []
    for test_type in test_types:
        metrics = evaluate_array(net, test_type, data_dir, "ambidrop", device)
        all_noisy.extend(metrics['si_sdr_noisy'].tolist())
        all_enhanced.extend(metrics['si_sdr_enhanced'].tolist())

    return np.array(all_noisy), np.array(all_enhanced)


def npy_to_csv(noisy_arr, enhanced_arr):
    """Convert raw .npy arrays into a 1-dB binned summary DataFrame."""
    improvement = enhanced_arr - noisy_arr
    df = pd.DataFrame({
        'Noisy_SI_SDR': noisy_arr,
        'Enhanced_SI_SDR': enhanced_arr,
        'Improvement': improvement,
    })

    min_bin = int(np.floor(noisy_arr.min()))
    max_bin = int(np.ceil(noisy_arr.max()))
    bins = np.arange(min_bin, max_bin + 1, 1)

    df['Bin'] = pd.cut(df['Noisy_SI_SDR'], bins=bins, right=False)
    summary = df.groupby('Bin', observed=True).agg({
        'Noisy_SI_SDR': ['count', 'mean'],
        'Enhanced_SI_SDR': 'mean',
        'Improvement': 'mean'
    }).reset_index()

    summary.columns = [
        'Bin [dB]',
        'Sample Count',
        'Mean Noisy SI-SDR [dB]',
        'Mean Enhanced SI-SDR [dB]',
        'Mean Improvement [dB]'
    ]
    return summary


def plot_snr_distribution(csv_df, output_path, bin_width=2):
    """
    Plot Fig. 4: horizontal histogram of SI-SDR distribution with
    mean improvement annotated per bin. Matches the style from a.py.
    """
    df = csv_df.copy()

    # Extract numeric lower bound from the bin string (e.g. "[-10, -9)" -> -10.0)
    df['bin_numeric'] = df['Bin [dB]'].astype(str).str.extract(r'\[(.*?),').astype(float)

    # Re-aggregate into wider bins
    bins_agg = np.arange(-30, 7, bin_width)
    df['agg_group'] = pd.cut(df['bin_numeric'], bins=bins_agg, right=False)

    agg_df = df.groupby('agg_group', observed=True).agg({
        'Sample Count': 'sum',
        'Mean Improvement [dB]': 'mean'
    }).reset_index()

    agg_df['bin_label'] = agg_df['agg_group'].astype(str)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    bars = ax.barh(agg_df["bin_label"], agg_df["Sample Count"],
                   color='skyblue', edgecolor='black')

    ax.set_xscale('log')

    max_count = agg_df["Sample Count"].max()
    ax.set_xlim(right=max_count * 3)

    for bar, mean_val in zip(bars, agg_df["Mean Improvement [dB]"]):
        width = bar.get_width()
        if width > 0:
            ax.text(width * 1.05, bar.get_y() + bar.get_height() / 2,
                    f'{mean_val:.1f} dB',
                    va='center', fontweight='bold', color='darkblue', fontsize=20)

    ax.set_xlabel("Sample Count (Log Scale)", fontsize=23, labelpad=15)
    ax.set_ylabel("Input SI-SDR Bin [dB]", fontsize=23, labelpad=15)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")


def main():
    p = argparse.ArgumentParser(description="SI-SDR distribution histogram (Fig. 4)")
    p.add_argument('--noisy-npy', default=None, help='Path to master_si_sdr_noisy.npy')
    p.add_argument('--enhanced-npy', default=None, help='Path to master_si_sdr_enhanced.npy')
    p.add_argument('--from-csv', default=None, help='Pre-computed 1dB-binned CSV')
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--data-dir', default=None)
    p.add_argument('--output', default='figures/fig4_snr_distribution.png')
    p.add_argument('--bin-width', type=int, default=2)
    p.add_argument('--save-csv', default=None)
    args = p.parse_args()

    if args.from_csv:
        csv_df = pd.read_csv(args.from_csv)
    elif args.noisy_npy and args.enhanced_npy:
        noisy = np.load(args.noisy_npy).flatten()
        enhanced = np.load(args.enhanced_npy).flatten()
        csv_df = npy_to_csv(noisy, enhanced)
        if args.save_csv:
            os.makedirs(os.path.dirname(args.save_csv) or '.', exist_ok=True)
            csv_df.to_csv(args.save_csv, index=False)
            print(f"CSV saved to {args.save_csv}")
    elif args.checkpoint and args.data_dir:
        noisy, enhanced = run_and_collect(args.checkpoint, args.data_dir)
        csv_df = npy_to_csv(noisy, enhanced)
        if args.save_csv:
            os.makedirs(os.path.dirname(args.save_csv) or '.', exist_ok=True)
            csv_df.to_csv(args.save_csv, index=False)
    else:
        print("Error: provide --from-csv, (--noisy-npy + --enhanced-npy), or (--checkpoint + --data-dir)")
        return

    plot_snr_distribution(csv_df, args.output, bin_width=args.bin_width)


if __name__ == '__main__':
    main()
