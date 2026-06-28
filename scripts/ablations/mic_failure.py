"""
Microphone Failure Resilience — Figure 7

Evaluates AmbiDrop and baseline performance as microphones are randomly
deactivated (1 to 5 channels removed from the 7-channel input).

The ASM encoding is informed of which channels are missing so it adjusts
the steering matrix accordingly.

Usage:
    python scripts/ablations/mic_failure.py \
        --ambidrop-checkpoint checkpoints/SH_FT_JNF,2025-12-01_10-08-18.pt \
        --baseline-checkpoint checkpoints/FT_JNF,2026-03-25_13-37-42.pt \
        --data-dir datasets/experiment_full_anm/test_of_train_ds_preprocessed \
        --output figures/fig7_mic_failure.png

    # From pre-computed CSV
    python scripts/ablations/mic_failure.py \
        --from-csv results/mic_failure.csv \
        --output figures/fig7_mic_failure.png
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

from ambidrop.constants import CHECKPOINT_REGISTRY, get_device


def run_mic_count_evaluation(checkpoint_name, data_dir, checkpoint_dir,
                              num_channels_to_zero, mode):
    """Run test with N channels zeroed, return mean SI-SDRi."""
    from scripts.test_simulated import evaluate_array
    from ambidrop.models import FT_JNF
    from ambidrop.checkpoint import load_checkpoint
    from ambidrop.constants import REF_IDX_MAP

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

    ckpt_path = os.path.join(checkpoint_dir, checkpoint_name)
    load_checkpoint(ckpt_path, target_epoch=200, net=net)
    net.eval()

    test_types = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d in REF_IDX_MAP
    ])

    all_noisy, all_enhanced = [], []
    for test_type in test_types:
        metrics = evaluate_array(net, test_type, data_dir, mode, device,
                                  zero_channels=num_channels_to_zero)
        all_noisy.extend(metrics['si_sdr_noisy'].tolist())
        all_enhanced.extend(metrics['si_sdr_enhanced'].tolist())

    mean_noisy = np.mean(all_noisy)
    mean_enhanced = np.mean(all_enhanced)
    return mean_enhanced - mean_noisy


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
    p.add_argument('--ambidrop-checkpoint', default='SH_FT_JNF,2025-12-01_10-08-18.pt')
    p.add_argument('--baseline-checkpoint', default='FT_JNF,2026-03-25_13-37-42.pt')
    p.add_argument('--data-dir', default=None)
    p.add_argument('--checkpoint-dir', default='checkpoints')
    p.add_argument('--from-csv', default=None)
    p.add_argument('--output', default='figures/fig7_mic_failure.png')
    p.add_argument('--save-csv', default=None)
    args = p.parse_args()

    if args.from_csv:
        results_df = pd.read_csv(args.from_csv)
    else:
        if args.data_dir is None:
            print("Error: --data-dir required when not using --from-csv")
            return

        rows = []
        for n_zero in range(0, 6):
            available = 7 - n_zero
            print(f"\n=== {available} available channels (zeroing {n_zero}) ===")

            print(f"  AmbiDrop...")
            si_sdri = run_mic_count_evaluation(
                args.ambidrop_checkpoint, args.data_dir, args.checkpoint_dir,
                n_zero, mode='ambidrop')
            rows.append({'method': 'FT-JNF + AmbiDrop', 'available_channels': available, 'si_sdri': si_sdri})
            print(f"  SI-SDRi: {si_sdri:.2f} dB")

            print(f"  Baseline...")
            si_sdri = run_mic_count_evaluation(
                args.baseline_checkpoint, args.data_dir, args.checkpoint_dir,
                n_zero, mode='baseline')
            rows.append({'method': 'FT-JNF (Baseline)', 'available_channels': available, 'si_sdri': si_sdri})
            print(f"  SI-SDRi: {si_sdri:.2f} dB")

        results_df = pd.DataFrame(rows)

        if args.save_csv:
            os.makedirs(os.path.dirname(args.save_csv) or '.', exist_ok=True)
            results_df.to_csv(args.save_csv, index=False)
            print(f"Results saved to {args.save_csv}")

    plot_mic_failure(results_df, args.output)


if __name__ == '__main__':
    main()
