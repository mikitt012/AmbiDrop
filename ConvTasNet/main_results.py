"""
ConvTasNet Main Results Table

Generates performance comparison table for IC Conv-TasNet baseline vs AmbiDrop
on simulated training and test arrays.

Usage:
    python ConvTasNet/main_results.py --output figures/table_convtasnet.png
"""

import os
import sys
import argparse
import subprocess

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def run_evaluate(mode, model_path, data_dir, dropout_type='SHChannelDropout',
                 drop_prob=0.4, max_drop=3, drop_probs=None):
    """Run ConvTasNet/evaluate.py and parse output."""
    cmd = [
        sys.executable, 'ConvTasNet/evaluate.py',
        '--mode', mode,
        '--model_path', model_path,
        '--data_dir', data_dir,
        '--use_cuda', '0',
        '--no_wandb',
    ]
    if mode == 'ambidrop':
        cmd += ['--dropout_type', dropout_type, '--drop_prob', str(drop_prob), '--max_drop', str(max_drop)]
    if drop_probs:
        cmd += ['--drop_probs', drop_probs]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                            env={**os.environ, 'KMP_DUPLICATE_LIB_OK': 'TRUE'})
    output = result.stdout + result.stderr

    si_sdr_noisy = si_sdr_enh = pesq_n = pesq_e = stoi_n = stoi_e = None
    for line in output.split('\n'):
        line = line.strip()
        if '->' in line:
            parts = line.split('->')
            if 'SI-SDR:' in line:
                si_sdr_noisy = float(parts[0].split(':')[1].strip())
                si_sdr_enh = float(parts[1].strip().split()[0])
            elif 'PESQ:' in line:
                pesq_n = float(parts[0].split(':')[1].strip())
                pesq_e = float(parts[1].strip().split()[0])
            elif 'STOI:' in line:
                stoi_n = float(parts[0].split(':')[1].strip())
                stoi_e = float(parts[1].strip().split()[0])

    return {
        'SI-SDR Noisy': si_sdr_noisy, 'SI-SDR Enhanced': si_sdr_enh,
        'PESQ Noisy': pesq_n, 'PESQ Enhanced': pesq_e,
        'STOI Noisy': stoi_n, 'STOI Enhanced': stoi_e,
    }


def plot_table(results_df, output_path, title):
    fig, ax = plt.subplots(figsize=(14, len(results_df) * 0.6 + 2))
    ax.axis('off')
    tbl = ax.table(cellText=results_df.values, colLabels=results_df.columns,
                   cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
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
    p = argparse.ArgumentParser(description="ConvTasNet main results table")
    p.add_argument('--train-data-dir', default='datasets/experiment_full_anm/test_of_train_ds')
    p.add_argument('--test-data-dir', default='datasets/experiment_full_anm/test_of_test_ds')
    p.add_argument('--train-data-dir-baseline', default='datasets/experiment_full_anm/test_of_train_ds_preprocessed')
    p.add_argument('--test-data-dir-baseline', default='datasets/experiment_full_anm/test_of_test_ds_preprocessed')
    p.add_argument('--ambidrop-checkpoint', default='checkpoints/ConvTasNet/run_2026-04-09_08-35/final.pth.tar')
    p.add_argument('--baseline-checkpoint', default='checkpoints/ConvTasNet/run_2026-04-09_10-55/final.pth.tar')
    p.add_argument('--output', default='figures/table_convtasnet.png')
    p.add_argument('--from-csv', default=None)
    args = p.parse_args()

    if args.from_csv:
        results_df = pd.read_csv(args.from_csv)
    else:
        configs = [
            ("Training Arrays", "IC-ConvTasNet (Baseline)", args.baseline_checkpoint, args.train_data_dir_baseline, "baseline"),
            ("Training Arrays", "IC-ConvTasNet + AmbiDrop", args.ambidrop_checkpoint, args.train_data_dir, "ambidrop"),
            ("Test Arrays", "IC-ConvTasNet (Baseline)", args.baseline_checkpoint, args.test_data_dir_baseline, "baseline"),
            ("Test Arrays", "IC-ConvTasNet + AmbiDrop", args.ambidrop_checkpoint, args.test_data_dir, "ambidrop"),
        ]

        rows = []
        for dataset_label, method_name, ckpt, data_dir, mode in configs:
            print(f"\n{dataset_label} - {method_name}")
            metrics = run_evaluate(mode, ckpt, data_dir)

            if dataset_label == "Training Arrays" and mode == "baseline" and metrics['SI-SDR Noisy'] is not None:
                rows.append({
                    'Dataset': dataset_label, 'Method': 'Noisy (Input)',
                    'SI-SDR (dB)': metrics['SI-SDR Noisy'], 'PESQ': metrics['PESQ Noisy'], 'STOI': metrics['STOI Noisy']
                })
            if dataset_label == "Test Arrays" and mode == "baseline" and metrics['SI-SDR Noisy'] is not None:
                rows.append({
                    'Dataset': dataset_label, 'Method': 'Noisy (Input)',
                    'SI-SDR (dB)': metrics['SI-SDR Noisy'], 'PESQ': metrics['PESQ Noisy'], 'STOI': metrics['STOI Noisy']
                })

            rows.append({
                'Dataset': dataset_label, 'Method': method_name,
                'SI-SDR (dB)': metrics['SI-SDR Enhanced'], 'PESQ': metrics['PESQ Enhanced'], 'STOI': metrics['STOI Enhanced']
            })
            print(f"  SI-SDR: {metrics['SI-SDR Noisy']} -> {metrics['SI-SDR Enhanced']}")

        results_df = pd.DataFrame(rows)

    print("\n" + results_df.to_string(index=False))
    plot_table(results_df, args.output, title="IC Conv-TasNet Performance")


if __name__ == '__main__':
    main()
