"""
Dropout Ablation Study — Figure 6

Evaluates multiple dropout configurations (uniform and per-channel) across
simulated arrays and Aria data, then plots SI-SDRi for each configuration.

Usage:
    # Run evaluation across all dropout configs on simulated test data
    python scripts/ablations/dropout_ablation.py \
        --data-dir datasets/experiment_full_anm/test_of_train_ds_preprocessed \
        --output figures/fig6_dropout_ablation.png

    # Use pre-computed CSV results instead of running inference
    python scripts/ablations/dropout_ablation.py \
        --from-csv results/dropout_ablation.csv \
        --output figures/fig6_dropout_ablation.png
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

from ambidrop.constants import CHECKPOINT_REGISTRY

UNIFORM_CONFIGS = [
    {"label": "(7)\n(0.7)", "checkpoint": "SH_FT_JNF,2025-12-04_15-20-29.pt"},
    {"label": "(7)\n(0.6)", "checkpoint": "SH_FT_JNF,2025-12-04_21-18-51.pt"},
    {"label": "(7)\n(0.4)", "checkpoint": "SH_FT_JNF,2025-12-21_17-36-40.pt"},
    {"label": "(6)\n(0.5)", "checkpoint": "SH_FT_JNF,2025-12-05_03-31-01.pt"},
    {"label": "(6)\n(0.3)", "checkpoint": "SH_FT_JNF,2025-12-22_14-53-29.pt"},
    {"label": "(3)\n(0.7)", "checkpoint": "SH_FT_JNF,2025-12-23_05-06-54.pt"},
    {"label": "(3)\n(0.4)", "checkpoint": "SH_FT_JNF,2025-12-01_10-08-18.pt"},
    {"label": "(0)\n(0)",   "checkpoint": "SH_FT_JNF,2025-12-23_13-38-21.pt"},
]

PERCHANNEL_CONFIGS = [
    {"label": "-10.0", "checkpoint": "SH_FT_JNF,2025-12-04_15-45-32.pt"},
    {"label": "-8.2",  "checkpoint": "SH_FT_JNF,2025-12-04_21-44-11.pt"},
    {"label": "-7.2",  "checkpoint": "SH_FT_JNF,2025-12-05_03-53-11.pt"},
    {"label": "-5.7",  "checkpoint": "SH_FT_JNF,2025-12-05_09-45-29.pt"},
    {"label": "-5.0",  "checkpoint": "SH_FT_JNF,2025-12-05_15-44-31.pt"},
    {"label": "-4.2",  "checkpoint": "SH_FT_JNF,2025-12-05_21-56-06.pt"},
    {"label": "-3.4",  "checkpoint": "SH_FT_JNF,2025-12-06_04-07-37.pt"},
    {"label": "-2.4",  "checkpoint": "SH_FT_JNF,2025-12-06_09-56-40.pt"},
    {"label": "-1.4",  "checkpoint": "SH_FT_JNF,2025-12-06_15-58-32.pt"},
    {"label": "0.0",   "checkpoint": "SH_FT_JNF,2025-12-06_22-04-29.pt"},
]


def run_evaluation(checkpoint_name, data_dir, checkpoint_dir):
    """Run test_simulated evaluation for one checkpoint, return mean SI-SDRi."""
    from scripts.test_simulated import evaluate_array
    from ambidrop.models import FT_JNF
    from ambidrop.checkpoint import load_checkpoint
    from ambidrop.constants import REF_IDX_MAP, get_device

    config = CHECKPOINT_REGISTRY[checkpoint_name]
    device = get_device()

    drop_probs = config.get("drop_probs", None)
    net = FT_JNF(
        input_dim=config["input_dim"],
        hidden1_dim=config["hidden1"],
        hidden2_dim=config["hidden2"],
        output_dim=2,
        dropout_type=config.get("dropout"),
        drop_prob=config.get("drop_prob", 0.0),
        max_drop=config.get("max_drop", 0),
        drop_probs=drop_probs,
    ).to(device)

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

    mean_noisy = np.mean(all_noisy)
    mean_enhanced = np.mean(all_enhanced)
    return mean_enhanced - mean_noisy


def plot_dropout_ablation(results_df, output_path):
    """Plot Fig. 6: SI-SDRi across dropout configurations."""
    uniform = results_df[results_df['type'] == 'uniform']
    perchannel = results_df[results_df['type'] == 'perchannel']

    fig, ax = plt.subplots(figsize=(14, 5))

    n_uniform = len(uniform)
    n_perchannel = len(perchannel)
    x_uniform = np.arange(n_uniform)
    x_perchannel = np.arange(n_uniform + 1, n_uniform + 1 + n_perchannel)

    ax.plot(x_uniform, uniform['si_sdri'].values, 'o-', color='black', linewidth=2, markersize=8, label='Simulated Arrays')

    ax.plot(x_perchannel, perchannel['si_sdri'].values, 'o-', color='black', linewidth=2, markersize=8)

    ax.set_xticks(np.concatenate([x_uniform, x_perchannel]))
    ax.set_xticklabels(
        uniform['label'].tolist() + perchannel['label'].tolist(),
        fontsize=9
    )

    mid_uniform = x_uniform[len(x_uniform)//2]
    mid_perchannel = x_perchannel[len(x_perchannel)//2]
    ax.text(mid_uniform, ax.get_ylim()[1] + 0.3, "Uniform Dropout", ha='center', fontsize=12, fontweight='bold')
    ax.text(mid_perchannel, ax.get_ylim()[1] + 0.3, "Per Channel Dropout", ha='center', fontsize=12, fontweight='bold')

    ax.set_ylabel("SI-SDRi [dB]", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")


def main():
    p = argparse.ArgumentParser(description="Dropout ablation study (Fig. 6)")
    p.add_argument('--data-dir', default=None, help='Test data directory')
    p.add_argument('--checkpoint-dir', default='checkpoints')
    p.add_argument('--from-csv', default=None, help='Load pre-computed results from CSV')
    p.add_argument('--output', default='figures/fig6_dropout_ablation.png')
    p.add_argument('--save-csv', default=None, help='Save results to CSV')
    args = p.parse_args()

    if args.from_csv:
        results_df = pd.read_csv(args.from_csv)
    else:
        if args.data_dir is None:
            print("Error: --data-dir required when not using --from-csv")
            return

        rows = []
        for cfg in UNIFORM_CONFIGS:
            print(f"Evaluating uniform: {cfg['label'].replace(chr(10), ', ')}")
            si_sdri = run_evaluation(cfg['checkpoint'], args.data_dir, args.checkpoint_dir)
            rows.append({'type': 'uniform', 'label': cfg['label'], 'checkpoint': cfg['checkpoint'], 'si_sdri': si_sdri})
            print(f"  SI-SDRi: {si_sdri:.2f} dB")

        for cfg in PERCHANNEL_CONFIGS:
            print(f"Evaluating per-channel: threshold={cfg['label']} dB")
            si_sdri = run_evaluation(cfg['checkpoint'], args.data_dir, args.checkpoint_dir)
            rows.append({'type': 'perchannel', 'label': cfg['label'], 'checkpoint': cfg['checkpoint'], 'si_sdri': si_sdri})
            print(f"  SI-SDRi: {si_sdri:.2f} dB")

        results_df = pd.DataFrame(rows)

        if args.save_csv:
            os.makedirs(os.path.dirname(args.save_csv) or '.', exist_ok=True)
            results_df.to_csv(args.save_csv, index=False)
            print(f"Results saved to {args.save_csv}")

    plot_dropout_ablation(results_df, args.output)


if __name__ == '__main__':
    main()
