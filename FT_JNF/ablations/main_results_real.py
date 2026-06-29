"""
Main Results Table II — Real-World Aria Glasses

Generates Table II from the paper: AmbiDrop performance on real-world Aria
recordings with simulated and measured ATFs, per scenario.

Usage:
    python FT_JNF/ablations/main_results_real.py \
        --aria-data-dir datasets/aria_ds \
        --scenarios mixed_data_1_5int \
        --output figures/table2_real.png
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def run_test_real_single(mode, checkpoint, aria_data_dir, scenario, atf, cnm_source,
                          input_dim, hidden1, hidden2, dropout_type=None, drop_prob=0.4,
                          max_drop=3, cnm_path=None):
    """Run FT_JNF/test_real.py for a single scenario and parse output."""
    cmd = [
        sys.executable, 'FT_JNF/test_real.py',
        '--mode', mode,
        '--checkpoint', checkpoint,
        '--aria-data-dir', aria_data_dir,
        '--scenarios', scenario,
        '--atf', atf,
        '--cnm-source', cnm_source,
        '--input-dim', str(input_dim),
        '--hidden1', str(hidden1),
        '--hidden2', str(hidden2),
        '--no-wandb',
    ]
    if dropout_type:
        cmd += ['--dropout-type', dropout_type, '--drop-prob', str(drop_prob), '--max-drop', str(max_drop)]
    if cnm_path:
        cmd += ['--cnm-path', cnm_path]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                            env={**os.environ, 'KMP_DUPLICATE_LIB_OK': 'TRUE'})
    output = result.stdout + result.stderr

    si_sdri = pesq_val = stoi_val = None
    for line in output.split('\n'):
        if 'Enhanced' in line and 'SI-SDR' in line:
            parts = line.split(',')
            for p in parts:
                p = p.strip()
                if 'SI-SDR' in p and 'dB' in p:
                    si_sdri_enh = float(p.split(':')[1].strip().replace('dB', '').strip())
                elif 'PESQ' in p:
                    pesq_val = float(p.split(':')[1].strip())
                elif 'STOI' in p:
                    stoi_val = float(p.split(':')[1].strip())
        if 'SI-SDRi' in line and 'dB' in line and 'Noisy' not in line and 'Enhanced' not in line:
            si_sdri = float(line.split(':')[1].strip().replace('dB', '').strip())

    return {'SI-SDRi': si_sdri, 'PESQ': pesq_val, 'STOI': stoi_val}


def plot_table(results_df, output_path, title="Performance Comparison"):
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
    p = argparse.ArgumentParser(description="Table II: Real-world Aria results")
    p.add_argument('--aria-data-dir', default='datasets/aria_ds')
    p.add_argument('--scenarios', nargs='+', default=None,
                   help='Scenario folders to test (default: auto-detect all)')
    p.add_argument('--checkpoint', default='checkpoints/FT_JNF/SH_FT_JNF,2025-12-01_10-08-18.pt')
    p.add_argument('--baseline-checkpoint', default='checkpoints/FT_JNF/FT_JNF,2026-03-25_13-37-42.pt')
    p.add_argument('--output', default='figures/table2_real.png')
    p.add_argument('--from-csv', default=None)
    p.add_argument('--save-csv', default=None)
    args = p.parse_args()

    if args.from_csv:
        results_df = pd.read_csv(args.from_csv)
    else:
        if args.scenarios:
            scenarios = args.scenarios
        else:
            scenarios = sorted([
                d for d in os.listdir(args.aria_data_dir)
                if os.path.isdir(os.path.join(args.aria_data_dir, d))
            ])

        atf_configs = [
            {"label": "Baseline", "mode": "baseline", "ckpt": args.baseline_checkpoint,
             "atf": "simulated", "cnm_source": "compute", "cnm_path": None,
             "input_dim": 14, "hidden1": 64, "hidden2": 64, "dropout_type": None},
            {"label": "AmbiDrop + Sim ATF", "mode": "ambidrop", "ckpt": args.checkpoint,
             "atf": "simulated", "cnm_source": "compute", "cnm_path": None,
             "input_dim": 18, "hidden1": 64, "hidden2": 64, "dropout_type": "SHChannelDropout"},
            {"label": "AmbiDrop + Meas ATF", "mode": "ambidrop", "ckpt": args.checkpoint,
             "atf": "measured", "cnm_source": "precomputed",
             "cnm_path": os.path.join(args.aria_data_dir, "cnm.npy"),
             "input_dim": 18, "hidden1": 64, "hidden2": 64, "dropout_type": "SHChannelDropout"},
        ]

        rows = []
        for scenario in scenarios:
            print(f"\n{'='*60}")
            print(f"Scenario: {scenario}")
            print(f"{'='*60}")

            for cfg in atf_configs:
                print(f"  {cfg['label']}...")
                metrics = run_test_real_single(
                    mode=cfg['mode'], checkpoint=cfg['ckpt'],
                    aria_data_dir=args.aria_data_dir, scenario=scenario,
                    atf=cfg['atf'], cnm_source=cfg['cnm_source'],
                    input_dim=cfg['input_dim'], hidden1=cfg['hidden1'], hidden2=cfg['hidden2'],
                    dropout_type=cfg['dropout_type'], cnm_path=cfg.get('cnm_path'),
                )
                rows.append({
                    'Dataset': scenario,
                    'Method': cfg['label'],
                    'SI-SDRi (dB)': round(metrics['SI-SDRi'], 2) if metrics['SI-SDRi'] is not None else 'N/A',
                    'PESQ': round(metrics['PESQ'], 2) if metrics['PESQ'] is not None else 'N/A',
                    'STOI': round(metrics['STOI'], 3) if metrics['STOI'] is not None else 'N/A',
                })
                print(f"    SI-SDRi: {metrics['SI-SDRi']}, PESQ: {metrics['PESQ']}, STOI: {metrics['STOI']}")

        results_df = pd.DataFrame(rows)

        if args.save_csv:
            os.makedirs(os.path.dirname(args.save_csv) or '.', exist_ok=True)
            results_df.to_csv(args.save_csv, index=False)
            print(f"CSV saved to {args.save_csv}")

    print("\n" + results_df.to_string(index=False))
    plot_table(results_df, args.output, title="Table II: Real-World Aria Glasses")


if __name__ == '__main__':
    main()
