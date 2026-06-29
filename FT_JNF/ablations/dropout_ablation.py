"""
Dropout Ablation Study — Figure 6

Plots SI-SDRi across 18 dropout configurations (8 uniform + 10 per-channel)
for simulated arrays and Aria data (multiple ATF configurations).

Usage:
    # Plot with hardcoded results from full cluster evaluation
    python scripts/ablations/dropout_ablation.py \
        --output figures/fig6_dropout_ablation.png

    # Run evaluation on local data (simulated arrays only)
    python scripts/ablations/dropout_ablation.py \
        --data-dir datasets/experiment_full_anm/test_of_train_ds_preprocessed \
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from FT_JNF.constants import CHECKPOINT_REGISTRY

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

# Pre-computed results from full cluster evaluation (all 20 arrays, 300 examples each)
PRECOMPUTED = {
    "simulated_arrays": {
        "noisy": -6.28,
        "si_sdr_enh_test": np.array([3.85, 4.17, 4.77, 4.16, 4.22, 4.12, 4.77, 2.43,
                                      2.63, 2, 4.64, 4.73, 4.45, 4.75, 4.75, 5.11, 4.86, -0.07]),
        "si_sdr_enh_train": np.array([3.512, 4.34, 4.84, 4.44, 4.97, 4.84, 5.06, 4.29,
                                       2.63, 2.68, 4.85, 5.05, 5.08, 5.18, 5.15, 5.22, 5.06, 3.38]),
    },
    "simulated_aria_simATF": np.array([8.8, 9.4, 10.1, 9.8, 10.3, 10.2, 9.6, 8.9,
                                        8.5, 8.0, 10.2, 10.2, 10.3, 10.4, 10.0, 10.3, 9.5, 8.1]),
    "measured_aria_simATF": np.array([8.15, 8.84, 8.74, 8.38, 9.25, 8.88, 7.34, 8.65,
                                       7.93, 7.95, 9.18, 9.23, 8.97, 8.85, 9.16, 8.57, 8.44, 7.99]),
    "measured_aria_measATF": np.array([5.61, 6.47, 6.53, 5.95, 7.12, 6.89, 5.79, 6.37,
                                        6.92, 6.9, 6.38, 6.69, 6.31, 6.1, 6.75, 6.14, 6.18, 5.69]),
}


def insert_gap(data, index):
    """Insert NaN at index to create visual gap between uniform and per-channel."""
    return np.insert(data.astype(float), index, np.nan)


def plot_dropout_ablation(output_path, inference_sdri=None):
    """
    Plot Fig. 6.
    If inference_sdri is provided, plots only the simulated curve from local data.
    Otherwise plots all 4 curves from pre-computed full-dataset results.
    """
    max_drop = [7, 7, 7, 6, 6, 3, 3, 0]
    drop_prob = [0.7, 0.6, 0.4, 0.5, 0.3, 0.7, 0.4, 0]
    chosen_th = [-10.0, -8.2, -7.2, -5.7, -5.0, -4.2, -3.4, -2.4, -1.4, 0.0]

    t = np.arange(18)
    gap_idx = 8

    if inference_sdri is not None:
        si_sdri_sim = inference_sdri
    else:
        sim = PRECOMPUTED["simulated_arrays"]
        mean_si_sdr = 0.5 * (sim["si_sdr_enh_train"] + sim["si_sdr_enh_test"])
        si_sdri_sim = mean_si_sdr - sim["noisy"]

    t = insert_gap(t, gap_idx)
    si_sdri_sim = insert_gap(si_sdri_sim, gap_idx)

    plt.figure(figsize=(15, 8))

    plt.plot(t, si_sdri_sim, '-s', color='black', linewidth=2, markersize=7,
             label='Simulated Arrays')

    if inference_sdri is None:
        sdri_sim_aria = insert_gap(PRECOMPUTED["simulated_aria_simATF"], gap_idx)
        sdri_meas_sim = insert_gap(PRECOMPUTED["measured_aria_simATF"], gap_idx)
        sdri_meas_meas = insert_gap(PRECOMPUTED["measured_aria_measATF"], gap_idx)
        plt.plot(t, sdri_sim_aria, '-^', color='#2ca02c', linewidth=2, markersize=8,
                 label='Simulated Aria data with simulated Aria ATF')
        plt.plot(t, sdri_meas_sim, '-s', color='#ff7f0e', linewidth=2, markersize=8,
                 label='Measured Aria data with simulated Aria ATF')
        plt.plot(t, sdri_meas_meas, '-o', color='#1f77b4', linewidth=2, markersize=8,
                 label='Measured Aria data with measured Aria ATF')

    plt.axvline(x=7.5, color='black', linestyle='--', linewidth=2.5)
    plt.text(3.5, 12.5, 'Uniform Dropout', ha='center', weight='bold', fontsize=20)
    plt.text(12.5, 12.5, 'Per Channel Dropout', ha='center', weight='bold', fontsize=20)

    for k in range(8):
        if k != 7:
            label = f"({max_drop[k]})\n({drop_prob[k]:.1f})"
        else:
            label = f"({max_drop[k]})\n({drop_prob[k]:.0f})"
        plt.axvline(x=k, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        plt.text(k, 11.65, label, ha='center', va='bottom', fontsize=16,
                 color='black', fontweight='semibold')

    for i, x_val in enumerate(range(8, 18)):
        plt.axvline(x=x_val, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        plt.text(x_val, 11.8, chosen_th[i], ha='center', fontsize=16,
                 color='black', alpha=0.8, fontweight='semibold')

    plt.ylabel('SI-SDRi [dB]', fontsize=20)
    plt.legend(loc='lower left', fontsize=16, frameon=True, shadow=True)
    plt.xticks([])
    plt.yticks(fontsize=16)
    plt.grid(True, axis='y', linestyle=':', alpha=0.6)
    plt.xlim([-1, 18])
    plt.ylim([4, 13])

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")


def run_evaluation(checkpoint_name, data_dir, checkpoint_dir):
    """Run test_simulated evaluation for one checkpoint, return mean SI-SDRi."""
    from FT_JNF.test_simulated import evaluate_array
    from FT_JNF.model import FT_JNF
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

    return np.mean(all_enhanced) - np.mean(all_noisy)


def main():
    p = argparse.ArgumentParser(description="Dropout ablation study (Fig. 6)")
    p.add_argument('--data-dir', default=None, help='Test data directory (for local inference)')
    p.add_argument('--checkpoint-dir', default='checkpoints/FT_JNF')
    p.add_argument('--output', default='figures/fig6_dropout_ablation.png')
    p.add_argument('--use-precomputed', action='store_true', default=True,
                   help='Use pre-computed results from full cluster evaluation (default)')
    p.add_argument('--run-inference', action='store_true',
                   help='Run local inference instead of using pre-computed results')
    args = p.parse_args()

    inference_sdri = None
    if args.run_inference and args.data_dir:
        print("Running local inference (simulated arrays only)...")
        all_configs = UNIFORM_CONFIGS + PERCHANNEL_CONFIGS
        si_sdri_values = []
        for cfg in all_configs:
            label = cfg['label'].replace('\n', ', ')
            print(f"  Evaluating: {label}")
            si_sdri = run_evaluation(cfg['checkpoint'], args.data_dir, args.checkpoint_dir)
            si_sdri_values.append(si_sdri)
            print(f"    SI-SDRi: {si_sdri:.2f} dB")
        inference_sdri = np.array(si_sdri_values)

    plot_dropout_ablation(args.output, inference_sdri=inference_sdri)


if __name__ == '__main__':
    main()
