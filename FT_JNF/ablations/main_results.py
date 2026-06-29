"""
Main Results Tables — Tables I & II

Generates the main performance comparison tables:
  - Table I:  Simulated data (training arrays + test arrays)
  - Table II: Real-world Aria data (simulated vs measured ATF, normal vs mispositioned)

Usage:
    # From pre-computed CSV
    python scripts/ablations/main_results.py \
        --from-csv results/main_results.csv \
        --output figures/table1_simulated.png

    # Run inference for Table I
    python scripts/ablations/main_results.py \
        --train-data-dir datasets/experiment_full_anm/test_of_train_ds_preprocessed \
        --test-data-dir datasets/experiment_full_anm/test_of_test_ds_preprocessed \
        --output figures/table1_simulated.png
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

from FT_JNF.constants import CHECKPOINT_REGISTRY

METHODS = {
    "FT-JNF (Baseline)": {
        "checkpoint": "FT_JNF,2026-03-25_13-37-42.pt",
        "mode": "baseline",
    },
    "FT-JNF + AmbiDrop": {
        "checkpoint": "SH_FT_JNF,2025-12-01_10-08-18.pt",
        "mode": "ambidrop",
    },
}


def run_table_evaluation(methods, data_dir, checkpoint_dir, dataset_label):
    """Evaluate all methods on a dataset, return list of result rows."""
    from FT_JNF.test_simulated import evaluate_array
    from FT_JNF.model import FT_JNF
    from ambidrop.checkpoint import load_checkpoint
    from ambidrop.constants import REF_IDX_MAP, get_device

    device = get_device()

    test_types = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d in REF_IDX_MAP
    ])

    rows = []

    # Noisy baseline (no model)
    noisy_si_sdr, noisy_pesq, noisy_stoi = [], [], []
    for test_type in test_types:
        # Run with any model just to get noisy metrics
        first_method = list(methods.values())[0]
        config = CHECKPOINT_REGISTRY[first_method["checkpoint"]]

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
        ckpt_path = os.path.join(checkpoint_dir, first_method["checkpoint"])
        load_checkpoint(ckpt_path, target_epoch=200, net=net)
        net.eval()

        metrics = evaluate_array(net, test_type, data_dir, first_method["mode"], device)
        noisy_si_sdr.extend(metrics['si_sdr_noisy'].tolist())
        noisy_pesq.extend(metrics['pesq_noisy'].tolist())
        noisy_stoi.extend(metrics['stoi_noisy'].tolist())
        break  # only need noisy once

    rows.append({
        'Dataset': dataset_label,
        'Method': 'Noisy (Input)',
        'SI-SDR (dB)': round(np.mean(noisy_si_sdr), 2),
        'PESQ': round(np.mean(noisy_pesq), 2),
        'STOI': round(np.mean(noisy_stoi), 2),
    })

    for method_name, method_cfg in methods.items():
        config = CHECKPOINT_REGISTRY[method_cfg["checkpoint"]]

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
        ckpt_path = os.path.join(checkpoint_dir, method_cfg["checkpoint"])
        load_checkpoint(ckpt_path, target_epoch=200, net=net)
        net.eval()

        all_si_sdr, all_pesq, all_stoi = [], [], []
        for test_type in test_types:
            metrics = evaluate_array(net, test_type, data_dir, method_cfg["mode"], device)
            all_si_sdr.extend(metrics['si_sdr_enhanced'].tolist())
            all_pesq.extend(metrics['pesq_enhanced'].tolist())
            all_stoi.extend(metrics['stoi_enhanced'].tolist())

        rows.append({
            'Dataset': dataset_label,
            'Method': method_name,
            'SI-SDR (dB)': round(np.mean(all_si_sdr), 2),
            'PESQ': round(np.mean(all_pesq), 2),
            'STOI': round(np.mean(all_stoi), 2),
        })

    return rows


def plot_table(results_df, output_path, title="Performance Comparison"):
    """Render a styled table as a PNG image."""
    fig, ax = plt.subplots(figsize=(12, len(results_df) * 0.6 + 2))
    ax.axis('off')

    tbl = ax.table(
        cellText=results_df.values,
        colLabels=results_df.columns,
        cellLoc='center',
        loc='center',
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.2, 1.8)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2c3e50')
        cell.set_edgecolor('#ABB2B9')

    plt.title(title, fontsize=16, fontweight='bold', pad=20)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    p = argparse.ArgumentParser(description="Main results tables (Tables I & II)")
    p.add_argument('--train-data-dir', default=None)
    p.add_argument('--test-data-dir', default=None)
    p.add_argument('--checkpoint-dir', default='checkpoints/FT_JNF')
    p.add_argument('--from-csv', default=None)
    p.add_argument('--output', default='figures/table1_simulated.png')
    p.add_argument('--save-csv', default=None)
    args = p.parse_args()

    if args.from_csv:
        results_df = pd.read_csv(args.from_csv)
    else:
        all_rows = []
        if args.train_data_dir:
            print("=== Evaluating on Training Arrays ===")
            rows = run_table_evaluation(METHODS, args.train_data_dir,
                                         args.checkpoint_dir, "Training Arrays")
            all_rows.extend(rows)

        if args.test_data_dir:
            print("\n=== Evaluating on Test Arrays ===")
            rows = run_table_evaluation(METHODS, args.test_data_dir,
                                         args.checkpoint_dir, "Test Arrays")
            all_rows.extend(rows)

        if not all_rows:
            print("Error: provide --train-data-dir and/or --test-data-dir")
            return

        results_df = pd.DataFrame(all_rows)

        if args.save_csv:
            os.makedirs(os.path.dirname(args.save_csv) or '.', exist_ok=True)
            results_df.to_csv(args.save_csv, index=False)
            print(f"Results saved to {args.save_csv}")

    print("\n" + results_df.to_string(index=False))
    plot_table(results_df, args.output)


if __name__ == '__main__':
    main()
