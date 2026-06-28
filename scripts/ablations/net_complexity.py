"""
Network Complexity vs Performance — Figure 8

Plots SI-SDRi as a function of the total number of model parameters.
Reads per-size CSV results or runs inference for each model size.

Usage:
    # From pre-computed CSVs in net_size_comparison/
    python scripts/ablations/net_complexity.py \
        --results-dir net_size_comparison \
        --output figures/fig8_net_complexity.png

    # Run inference for each checkpoint size
    python scripts/ablations/net_complexity.py \
        --data-dir datasets/experiment_full_anm/test_of_train_ds_preprocessed \
        --checkpoint-dir checkpoints \
        --output figures/fig8_net_complexity.png
"""

import os
import sys
import argparse
import glob

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ambidrop.constants import CHECKPOINT_REGISTRY

SIZE_CHECKPOINTS = [
    "checkpoint_size_1223170.pt",
    "checkpoint_size_547330.pt",
    "checkpoint_size_316674.pt",
    "checkpoint_size_142594.pt",
    "checkpoint_size_84610.pt",
    "checkpoint_size_38530.pt",
    "checkpoint_size_23874.pt",
    "checkpoint_size_11074.pt",
    "checkpoint_size_7330.pt",
    "checkpoint_size_3490.pt",
]


def run_size_evaluation(checkpoint_name, data_dir, checkpoint_dir):
    """Evaluate one model size, return (total_params, mean_si_sdri)."""
    from scripts.test_simulated import evaluate_array
    from ambidrop.models import FT_JNF
    from ambidrop.checkpoint import load_checkpoint
    from ambidrop.constants import REF_IDX_MAP, get_device

    config = CHECKPOINT_REGISTRY[checkpoint_name]
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

    total_params = sum(p.numel() for p in net.parameters())

    ckpt_path = os.path.join(checkpoint_dir, checkpoint_name)
    load_checkpoint(ckpt_path, target_epoch=200, net=net)
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

    si_sdri = np.mean(all_enhanced) - np.mean(all_noisy)
    return total_params, si_sdri


def load_from_csvs(results_dir):
    """Load results from per-size CSV files (metrics_params_*.csv)."""
    csv_files = glob.glob(os.path.join(results_dir, "metrics_params_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No metrics_params_*.csv found in {results_dir}")

    rows = []
    for f in csv_files:
        df = pd.read_csv(f)
        params = df['Model_Params'].iloc[0]
        si_sdri = (df['SI_SDR_Enhanced'] - df['SI_SDR_Noisy']).mean()
        rows.append({'params': params, 'si_sdri': si_sdri})

    return pd.DataFrame(rows).sort_values('params', ascending=False)


def plot_net_complexity(results_df, output_path):
    """Plot Fig. 8: SI-SDRi vs total parameters."""
    results_df = results_df.sort_values('params', ascending=False)

    x_labels = [f"{int(p):,}" for p in results_df['params']]
    x_indices = range(len(results_df))

    plt.figure(figsize=(10, 6))
    plt.plot(x_indices, results_df['si_sdri'].values, color='#2980B9',
             marker='o', linewidth=3, markersize=10,
             markerfacecolor='white', markeredgewidth=2)

    plt.ylabel("SI-SDRi [dB]", fontsize=20)
    plt.xlabel("Number of Parameters", fontsize=20, labelpad=10)
    plt.xticks(x_indices, x_labels, rotation=45, fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")


def main():
    p = argparse.ArgumentParser(description="Network complexity ablation (Fig. 8)")
    p.add_argument('--results-dir', default=None,
                   help='Directory with metrics_params_*.csv files')
    p.add_argument('--data-dir', default=None,
                   help='Test data directory (for running inference)')
    p.add_argument('--checkpoint-dir', default='checkpoints')
    p.add_argument('--output', default='figures/fig8_net_complexity.png')
    p.add_argument('--save-csv', default=None)
    args = p.parse_args()

    if args.results_dir:
        results_df = load_from_csvs(args.results_dir)
    elif args.data_dir:
        rows = []
        for ckpt_name in SIZE_CHECKPOINTS:
            ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
            if not os.path.exists(ckpt_path):
                print(f"Skipping {ckpt_name}: not found")
                continue
            print(f"Evaluating {ckpt_name}...")
            params, si_sdri = run_size_evaluation(ckpt_name, args.data_dir, args.checkpoint_dir)
            rows.append({'params': params, 'si_sdri': si_sdri})
            print(f"  {params:,} params -> SI-SDRi: {si_sdri:.2f} dB")

        results_df = pd.DataFrame(rows)

        if args.save_csv:
            os.makedirs(os.path.dirname(args.save_csv) or '.', exist_ok=True)
            results_df.to_csv(args.save_csv, index=False)
    else:
        print("Error: provide --results-dir or --data-dir")
        return

    plot_net_complexity(results_df, args.output)


if __name__ == '__main__':
    main()
