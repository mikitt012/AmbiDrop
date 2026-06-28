# AmbiDrop Code Reorganization Plan

## Context

The AmbiDrop codebase was developed incrementally on a RunAI cluster, resulting in massive code duplication: `si_snr` is copy-pasted in 16 files, `SHChannelDropout` in 15, `FT_JNF` in 11, and dataset classes in 9. There are also 5 experimental dropout classes that are defined but never used. The goal is to consolidate everything into a clean package structure with shared modules, unified configurable train/test scripts, and dedicated ablation scripts that each produce a specific paper figure — without changing any logic or data formats.

## Target Directory Structure

```
AmbiDrop/
├── ambidrop/                          # Shared package (NEW)
│   ├── __init__.py
│   ├── losses.py                      # si_snr
│   ├── dropouts.py                    # SHChannelDropout, PerChDropout
│   ├── models.py                      # FT_JNF (unified)
│   ├── datasets.py                    # All dataset classes
│   ├── signal_utils.py                # pad_or_truncate, add_white_noise, etc.
│   ├── checkpoint.py                  # load_checkpoint, save_checkpoint
│   ├── inference.py                   # reconstruct_signal, evaluate_sample
│   └── constants.py                   # REF_IDX_MAP, STFT params, checkpoint registry
│
├── scripts/                           # Runnable scripts (NEW)
│   ├── train.py                       # Unified training (baseline + AmbiDrop)
│   ├── test_simulated.py              # Testing on simulated arrays (baseline + AmbiDrop)
│   ├── test_real.py                   # Testing on real-world Aria data (+ baseline)
│   └── ablations/
│       ├── dropout_ablation.py        # → Fig. 6
│       ├── mic_failure.py             # → Fig. 7
│       ├── net_complexity.py          # → Fig. 8
│       ├── snr_distribution.py        # → Fig. 4
│       └── main_results.py           # → Tables I & II
│
├── ASM/                               # Unchanged
├── ConvTasNet/                        # Refactored (baseline merged in)
├── utils/                             # Data generation / preprocessing (kept)
├── smoke_test.py                      # Updated to import from ambidrop/
├── checkpoints/
├── datasets/
└── figures/                           # Output directory for ablation plots
```

## Phase 1: Create `ambidrop/` Package

Extract shared code into reusable modules. No existing files are deleted yet — old scripts still work.

### `ambidrop/losses.py`
- Extract `si_snr(estimate, reference, epsilon=1e-8, debug=False)` — source from any file (all 16 copies are identical).
- Extract `complex_si_snr()` from `test_SH_FT_JNF.py`.

### `ambidrop/dropouts.py`
- Extract `SHChannelDropout` — remove the hardcoded assert (`C == 10` or `C == 18`), replace with `assert C % 2 == 0`. No learnable parameters, so checkpoint-safe.
- Extract `PerChDropout` — identical across all copies.
- **Delete** the 5 experimental dropout classes (`LearnableFreqDropout`, `SmoothLPFFreqDropout`, `MixedSHFreqDropout`, `ProgressiveDeterministicFreqDropout`, `MixedSHLearnableFreqDropout`). They are never used — always commented out at instantiation.

### `ambidrop/models.py`
- Unified `FT_JNF` class with optional dropout:
  ```python
  class FT_JNF(nn.Module):
      def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim,
                   dropout_type=None, drop_prob=0.0, max_drop=0, drop_probs=None):
  ```
- `dropout_type=None` → baseline (no dropout layer). `"SHChannelDropout"` or `"PerChDropout"` → creates the layer.
- `forward()` is identical across all copies: dropout → LSTM1 (across freq) → LSTM2 (across time) → linear.
- **Remove `training_step()` from the model**. It differs between baseline and AmbiDrop (different ref channel logic) and belongs in the training script.
- Checkpoint compatibility: baseline checkpoints lack `channel_dropout.*` keys — load with `strict=False`. AmbiDrop checkpoints have matching keys when constructed with the same dropout type.

### `ambidrop/datasets.py`
- `SimDS_preprocessed` — auto-detects `.pt` format (2-tuple, 4-tuple, or dict) and returns a normalized dict: `{'noisy': ..., 'clean': ..., 'ref_id': ..., 'array_name': ..., 'ex_id': ...}`.
- `SimDS` — raw `.mat` loader (from `test_SH_FT_JNF.py`, with `ambisonics` flag).
- `PreprocessedSHDataset` — STFT-preprocessed SH data loader.

### `ambidrop/signal_utils.py`
- `add_white_noise_torch(signal, snr_db)` — from `train_FT_JNF.py`
- `add_white_noise_numpy(signal, snr_db)` — from `test_SH_FT_JNF.py` (handles complex signals)
- `pad_or_truncate_torch(signal, target_length)` — from `Test_FT_JNF.py`
- `pad_or_truncate_numpy(signal, target_length)` — from `test_SH_FT_JNF.py`
- `pad_to_length(x, target_len)` — torch, pad-only version
- `process_segment(noisy, clean, target_samples, threshold)`
- `zero_random_channels(x, n)`
- `find_max_length(data_dir, data_type, ambisonics)`
- `unwrap_model(model)`, `get_lr(optimizer)`

### `ambidrop/checkpoint.py`
- Unified `load_checkpoint(path, target_epoch, net, optimizer, scheduler)` — merges training version (returns loss) and test version (returns chosen_epoch, closest-epoch logic). Returns a dict with all info.
- `save_checkpoint(path, epoch, net, optimizer, loss, lr)` — extracted from training loops.

### `ambidrop/constants.py`
- `REF_IDX_MAP` — the array-name-to-reference-index mapping (currently duplicated in `Test_FT_JNF.py`, `test_SH_FT_JNF.py`, `mic_data_process.py`).
- STFT defaults: `N_FFT=512, HOP_LENGTH=256, WIN_LENGTH=512, SAMPLE_RATE=16000`.
- `get_device()` helper.
- **Checkpoint registry** — maps each checkpoint file to its configuration so you always know what model/dropout config a checkpoint was trained with:

```python
CHECKPOINT_REGISTRY = {
    # Baseline FT-JNF (microphone input, 14 channels, no dropout)
    "FT_JNF,2025-11-30_14-41-59.pt":   {"mode": "baseline", "input_dim": 14, "hidden1": 256, "hidden2": 128},
    "FT_JNF,2025-12-01_09-21-58.pt":   {"mode": "baseline", "input_dim": 14, "hidden1": 64,  "hidden2": 64},
    "FT_JNF,2025-12-29_14-41-04.pt":   {"mode": "baseline", "input_dim": 14, "hidden1": 64,  "hidden2": 64},
    "FT_JNF,2026-03-25_13-37-42.pt":   {"mode": "baseline", "input_dim": 14, "hidden1": 64,  "hidden2": 64},

    # AmbiDrop FT-JNF — SHChannelDropout (18 channels = 9 SH × 2)
    "SH_FT_JNF,2025-12-01_09-21-33.pt": {"mode": "ambidrop", "input_dim": 18, "hidden1": 256, "hidden2": 128, "dropout": "SHChannelDropout", "drop_prob": 0.4, "max_drop": 3},
    "SH_FT_JNF,2025-12-01_10-08-18.pt": {"mode": "ambidrop", "input_dim": 18, "hidden1": 64,  "hidden2": 64,  "dropout": "SHChannelDropout", "drop_prob": 0.4, "max_drop": 3},
    "SH_FT_JNF,2025-12-04_15-20-29.pt": {"mode": "ambidrop", "input_dim": 18, "hidden1": 64,  "hidden2": 64,  "dropout": "SHChannelDropout", "drop_prob": 0.7, "max_drop": 7},
    "SH_FT_JNF,2025-12-04_21-18-51.pt": {"mode": "ambidrop", "input_dim": 18, "hidden1": 64,  "hidden2": 64,  "dropout": "SHChannelDropout", "drop_prob": 0.6, "max_drop": 7},
    "SH_FT_JNF,2025-12-05_03-31-01.pt": {"mode": "ambidrop", "input_dim": 18, "hidden1": 64,  "hidden2": 64,  "dropout": "SHChannelDropout", "drop_prob": 0.5, "max_drop": 6},
    "SH_FT_JNF,2025-12-21_17-36-40.pt": {"mode": "ambidrop", "input_dim": 18, "hidden1": 64,  "hidden2": 64,  "dropout": "SHChannelDropout", "drop_prob": 0.4, "max_drop": 7},
    "SH_FT_JNF,2025-12-22_14-53-29.pt": {"mode": "ambidrop", "input_dim": 18, "hidden1": 64,  "hidden2": 64,  "dropout": "SHChannelDropout", "drop_prob": 0.3, "max_drop": 6},
    "SH_FT_JNF,2025-12-23_05-06-54.pt": {"mode": "ambidrop", "input_dim": 18, "hidden1": 64,  "hidden2": 64,  "dropout": "SHChannelDropout", "drop_prob": 0.7, "max_drop": 3},
    "SH_FT_JNF,2025-12-23_13-38-21.pt": {"mode": "ambidrop", "input_dim": 18, "hidden1": 64,  "hidden2": 64,  "dropout": "SHChannelDropout", "drop_prob": 0.0, "max_drop": 0},

    # AmbiDrop FT-JNF — PerChDropout (per-channel probabilities, threshold-based)
    "SH_FT_JNF,2025-12-04_15-45-32.pt": {"mode": "ambidrop", "input_dim": 18, "hidden1": 64, "hidden2": 64, "dropout": "PerChDropout", "drop_probs": [0, 0.55, 0.9, 0.6, 1, 1, 1, 1, 1], "threshold_dB": -10.0},
    "SH_FT_JNF,2025-12-04_21-44-11.pt": {"mode": "ambidrop", "input_dim": 18, "hidden1": 64, "hidden2": 64, "dropout": "PerChDropout", "drop_probs": [0, 0.35, 0.75, 0.2, 1, 1, 1, 1, 1], "threshold_dB": -8.2},
    "SH_FT_JNF,2025-12-05_03-53-11.pt": {"mode": "ambidrop", "input_dim": 18, "hidden1": 64, "hidden2": 64, "dropout": "PerChDropout", "drop_probs": [0, 0.15, 0.6, 0.2, 0.95, 1, 0.95, 1, 0.95], "threshold_dB": -7.2},
    "SH_FT_JNF,2025-12-05_09-45-29.pt": {"mode": "ambidrop", "input_dim": 18, "hidden1": 64, "hidden2": 64, "dropout": "PerChDropout", "drop_probs": [0, 0.15, 0.55, 0.15, 0.8, 1, 0.95, 1, 0.85], "threshold_dB": -5.7},
    "SH_FT_JNF,2025-12-05_15-44-31.pt": {"mode": "ambidrop", "input_dim": 18, "hidden1": 64, "hidden2": 64, "dropout": "PerChDropout", "drop_probs": [0, 0.1, 0.45, 0.15, 0.7, 1, 0.85, 1, 0.65], "threshold_dB": -5.0},
    "SH_FT_JNF,2025-12-05_21-56-06.pt": {"mode": "ambidrop", "input_dim": 18, "hidden1": 64, "hidden2": 64, "dropout": "PerChDropout", "drop_probs": [0, 0.1, 0.45, 0.1, 0.55, 1, 0.85, 1, 0.55], "threshold_dB": -4.2},
    "SH_FT_JNF,2025-12-06_04-07-37.pt": {"mode": "ambidrop", "input_dim": 18, "hidden1": 64, "hidden2": 64, "dropout": "PerChDropout", "drop_probs": [0, 0.1, 0.45, 0.1, 0.45, 1, 0.75, 1, 0.45], "threshold_dB": -3.4},
    "SH_FT_JNF,2025-12-06_09-56-40.pt": {"mode": "ambidrop", "input_dim": 18, "hidden1": 64, "hidden2": 64, "dropout": "PerChDropout", "drop_probs": [0, 0.05, 0.45, 0.05, 0.4, 0.95, 0.5, 0.95, 0.4], "threshold_dB": -2.4},
    "SH_FT_JNF,2025-12-06_15-58-32.pt": {"mode": "ambidrop", "input_dim": 18, "hidden1": 64, "hidden2": 64, "dropout": "PerChDropout", "drop_probs": [0, 0.05, 0.45, 0.05, 0.1, 0.75, 0.4, 0.75, 0.1], "threshold_dB": -1.4},
    "SH_FT_JNF,2025-12-06_22-04-29.pt": {"mode": "ambidrop", "input_dim": 18, "hidden1": 64, "hidden2": 64, "dropout": "PerChDropout", "drop_probs": [0, 0.05, 0.35, 0.05, 0, 0.4, 0, 0.3, 0], "threshold_dB": 0.0},

    # Network size ablation checkpoints (all AmbiDrop, SHChannelDropout p=0.4 max=3)
    "checkpoint_size_3490.pt":    {"mode": "ambidrop", "input_dim": 18, "hidden1": 8,   "hidden2": 8,   "dropout": "SHChannelDropout", "drop_prob": 0.4, "max_drop": 3},
    "checkpoint_size_7330.pt":    {"mode": "ambidrop", "input_dim": 18, "hidden1": 16,  "hidden2": 8,   "dropout": "SHChannelDropout", "drop_prob": 0.4, "max_drop": 3},
    "checkpoint_size_11074.pt":   {"mode": "ambidrop", "input_dim": 18, "hidden1": 16,  "hidden2": 16,  "dropout": "SHChannelDropout", "drop_prob": 0.4, "max_drop": 3},
    "checkpoint_size_23874.pt":   {"mode": "ambidrop", "input_dim": 18, "hidden1": 32,  "hidden2": 16,  "dropout": "SHChannelDropout", "drop_prob": 0.4, "max_drop": 3},
    "checkpoint_size_38530.pt":   {"mode": "ambidrop", "input_dim": 18, "hidden1": 32,  "hidden2": 32,  "dropout": "SHChannelDropout", "drop_prob": 0.4, "max_drop": 3},
    "checkpoint_size_84610.pt":   {"mode": "ambidrop", "input_dim": 18, "hidden1": 64,  "hidden2": 32,  "dropout": "SHChannelDropout", "drop_prob": 0.4, "max_drop": 3},
    "checkpoint_size_142594.pt":  {"mode": "ambidrop", "input_dim": 18, "hidden1": 64,  "hidden2": 64,  "dropout": "SHChannelDropout", "drop_prob": 0.4, "max_drop": 3},
    "checkpoint_size_316674.pt":  {"mode": "ambidrop", "input_dim": 18, "hidden1": 128, "hidden2": 64,  "dropout": "SHChannelDropout", "drop_prob": 0.4, "max_drop": 3},
    "checkpoint_size_547330.pt":  {"mode": "ambidrop", "input_dim": 18, "hidden1": 128, "hidden2": 128, "dropout": "SHChannelDropout", "drop_prob": 0.4, "max_drop": 3},
    "checkpoint_size_1223170.pt": {"mode": "ambidrop", "input_dim": 18, "hidden1": 256, "hidden2": 128, "dropout": "SHChannelDropout", "drop_prob": 0.4, "max_drop": 3},

    # Named experiment checkpoints
    "AmbiDrop_full_experiment.pt":          {"mode": "ambidrop", "input_dim": 18, "hidden1": 256, "hidden2": 128, "dropout": "SHChannelDropout", "drop_prob": 0.4, "max_drop": 3},
    "AmbiDrop_full_experiment_smallnet.pt": {"mode": "ambidrop", "input_dim": 18, "hidden1": 64,  "hidden2": 64,  "dropout": "SHChannelDropout", "drop_prob": 0.4, "max_drop": 3},
    "AmbiDrop_experiment2.pt":              {"mode": "ambidrop", "input_dim": 18, "hidden1": 64,  "hidden2": 64,  "dropout": "SHChannelDropout", "drop_prob": 0.4, "max_drop": 3},
    "baseline_experiment2.pt":              {"mode": "baseline", "input_dim": 14, "hidden1": 64,  "hidden2": 64},
}
```

### `ambidrop/inference.py`
- `reconstruct_signal(net, x, num_channels, ref_id)` — forward pass → mask → iSTFT. Extracted from the common test loop pattern shared by simulated tests.
- `evaluate_sample(s_hat, s_clean, y_noisy, sr)` — computes SI-SDR, PESQ, STOI. Returns metrics dict.

## Phase 2: Create Unified Scripts

### `scripts/train.py`
Replaces: `train_FT_JNF.py`, `train_SH_FT_JNF.py`, `train_SH_FT_JNF_with_dropouts.py`, `net_size_comparison/SH_net_sizes_training.py`.

Uses argparse. Key args:
- `--mode {baseline,ambidrop}` — determines dataset format and ref channel logic
- `--data-dir`, `--train-split`, `--val-split` — data paths
- `--input-dim`, `--hidden1`, `--hidden2` — network architecture
- `--dropout-type`, `--drop-prob`, `--max-drop`, `--drop-probs` — dropout config
- `--epochs`, `--lr`, `--batch-size`, `--weight-decay` — training params
- `--checkpoint` — resume from checkpoint
- `--save-dir` — where to save checkpoints
- `--wandb-project`, `--wandb-entity` — logging

Training step logic:
- **baseline**: unpacks `(noisy, clean, ref_id, ...)`, selects per-sample ref channel: `Y = x[b, :, :, ref_id] + 1j * x[b, :, :, num_ch + ref_id]`
- **ambidrop**: unpacks `(noisy, clean)`, always uses channel 0 (a00): `Y = x[:,:,:,0] + 1j * x[:,:,:, num_sh_ch]`

### `scripts/test_simulated.py` — Simulated Array Testing
Replaces: `Test_FT_JNF.py`, `test_SH_FT_JNF.py`.

Tests on preprocessed simulated data. Both baseline and AmbiDrop modes. Loops over array subdirectories, uses `REF_IDX_MAP` for reference channel selection.

Key args:
- `--mode {baseline,ambidrop}`
- `--checkpoint`, `--epoch`
- `--data-dir` — test data (loops over all array subdirectories)
- `--test-type` — optional: test only a specific array
- `--output-csv` — save results to CSV
- `--zero-channels N` — optionally zero N random channels (for quick mic failure test)
- Network and dropout args (same as train)

### `scripts/test_real.py` — Real-World Aria Glasses Testing
Replaces: `test_aria_glasses.py`, `test_aria_glasses_baseline.py`.

This is a **separate script** because the Aria test pipeline is fundamentally different from simulated tests:
- **Data loading**: Reads `.wav` recordings + downsamples 48kHz→16kHz (not preprocessed `.pt`)
- **ASM encoding**: Computes Ambisonics encoding at inference time (not pre-encoded)
- **ATF source toggle**: Rigid sphere model vs SOFA-file measured ATF
- **ASM coefficient source**: Compute c_nm filters on-the-fly OR load precomputed c_nm from data
- **Regularization method**: Tikhonov regularization OR SVD-based inversion for the least-squares problem
- **Time synchronization**: Correlation-based alignment between clean reference and enhanced output (real recordings have unknown timing offset)
- **Glasses positioning**: Normal vs mispositioned — tests robustness to physical fit
- **Sensor failure modes**: Optionally zeros nose-bridge microphone before ASM
- **Unique functions preserved**: `sh2()`, `compute_spherical_harmonics_matrix()`, `array_ambisonics_time_domain()`, `sweep_alignment_sisdr()`, `align_with_best_shift()`, `shifted_overlap()`

Key args:
- `--mode {baseline,ambidrop}` — baseline uses mic signals directly, AmbiDrop uses ASM-encoded Ambisonics
- `--checkpoint`, `--epoch`
- `--aria-data-dir` — path to Aria recordings
- `--atf {simulated,measured}` — which steering matrix to use (rigid sphere .mat vs SOFA file)
- `--cnm-source {compute,precomputed}` — compute c_nm filters on-the-fly or load precomputed from data
- `--regularization {tikhonov,svd}` — which solver for the ASM least-squares problem
- `--positioning {normal,mispositioned}`
- `--cancel-nose` — simulate nose sensor failure (zeros channel before ASM)
- Network and dropout args

### `scripts/ablations/` — One Script Per Figure

Each ablation script:
1. Either runs inference itself (calling shared logic from `ambidrop/`) or reads pre-computed CSV/npy results
2. Produces the corresponding paper figure as a PNG in `figures/`

| Script | Paper Figure | What it does |
|--------|-------------|--------------|
| `dropout_ablation.py` | Fig. 6 | Evaluates multiple dropout configs across simulated + Aria data, plots SI-SDRi |
| `mic_failure.py` | Fig. 7 | Sweeps channel count (contains on-the-fly ASM with channel removal), plots SI-SDRi vs available channels |
| `net_complexity.py` | Fig. 8 | Reads per-size CSVs from test runs, plots SI-SDRi vs parameters |
| `snr_distribution.py` | Fig. 4 | Histogram of SI-SDR with mean improvement per bin |
| `main_results.py` | Tables I & II | Formats results into styled tables |

## Phase 3: ConvTasNet Cleanup

- Delete `ConvTasNet_baseline/` entirely (6 of 11 files are byte-identical copies)
- Modify `ConvTasNet/src/conv_tasnet_ic.py` to accept optional dropout (import from `ambidrop.dropouts`)
- Modify `ConvTasNet/src/train.py` to accept `--mode {baseline,ambidrop}`
- Modify `ConvTasNet/src/solver.py` and `data.py` to handle both modes via flags

## Phase 4: Cleanup

- Delete old root-level scripts: `train_FT_JNF.py`, `train_SH_FT_JNF.py`, `train_SH_FT_JNF_with_dropouts.py`, `Test_FT_JNF.py`, `test_SH_FT_JNF.py`, `test_aria_glasses.py`, `test_aria_glasses_baseline.py`, `test_mic_count.py`, `a.py`, `b.py`
- Delete `net_size_comparison/SH_net_sizes_training.py`, `SH_net_sizes_testing.py`, `results.py`
- Delete `snr_distribution_fixed/SH_train_distribution.py`
- Update `smoke_test.py` to import from `ambidrop/`
- Update `.gitignore` to include `figures/`

**Note**: wandb API key cleanup is deferred — will be done later as a separate task.

## Verification

After each phase:
1. Run `python smoke_test.py` — confirms model forward/backward works
2. Run `python scripts/test_simulated.py --mode ambidrop --checkpoint checkpoints/SH_FT_JNF,2025-12-01_10-08-18.pt --data-dir datasets/experiment_full_anm/test_of_train_ds_preprocessed --input-dim 18 --hidden1 64 --hidden2 64 --dropout-type SHChannelDropout --drop-prob 0.4 --max-drop 3 --epoch 200` — confirms test produces same metrics as current `test_SH_FT_JNF.py`
3. Run `python scripts/train.py --mode ambidrop ... --epochs 1` — confirms AmbiDrop training loop works
4. Run `python scripts/train.py --mode baseline ... --epochs 1` — confirms baseline training works

## Implementation Order

Phase 1 → Phase 2 → smoke test verification → Phase 3 → Phase 4 → full verification

Each phase is independently testable. Old scripts remain functional until Phase 4 cleanup.
