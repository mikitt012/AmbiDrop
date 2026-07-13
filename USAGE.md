# AmbiDrop — Usage Reference

_Last updated: 2026-07-13_

Complete reference for every script, function, and CLI flag in the project. For architecture internals and codebase structure, see `CODEBASE_OVERVIEW.md`. For a project overview, see `README.md`.

---

## Table of Contents

1. [Setup](#1-setup)
2. [Wrapper Scripts](#2-wrapper-scripts)
   - [run_FT_JNF.py](#run_ft_jnfpy)
   - [run_ConvTasNet.py](#run_convtasnetpy)
   - [run_Real_World.py](#run_real_worldpy)
3. [Data Generation](#3-data-generation)
   - [generate_ambidrop_train_ds.py — Type A](#generate_ambidrop_train_dspy--type-a)
   - [generate_baseline_train_ds.py — Type B](#generate_baseline_train_dspy--type-b)
   - [generate_inference_ds.py — Type C](#generate_inference_dspy--type-c)
   - [Paper Arrays](#paper-arrays)
4. [Preprocessing](#4-preprocessing)
5. [ASM (Ambisonics Signal Matching)](#5-asm-ambisonics-signal-matching)
6. [FT-JNF — Direct Training](#6-ft-jnf--direct-training)
7. [FT-JNF — Direct Evaluation (Simulated)](#7-ft-jnf--direct-evaluation-simulated)
8. [Conv-TasNet — Direct Training](#8-conv-tasnet--direct-training)
9. [Conv-TasNet — Direct Evaluation](#9-conv-tasnet--direct-evaluation)
10. [Ablation Scripts](#10-ablation-scripts)
11. [Checkpoint Registry](#11-checkpoint-registry)

---

## 1. Setup

**Conda (recommended):**

```bash
conda env create -f environment.yml
conda activate venv
```

**pip alternative (Python 3.9):**

```bash
pip install -r requirements.txt
```

The `shroom` library (rigid-sphere / ATF simulation) is pulled from GitHub automatically by both methods. To install it manually:

```bash
pip install git+https://github.com/Yhonatangayer/shroom.git
```

**WSJ0 requirement.** Data generation scripts require the WSJ0 corpus. Set the path in the `# === USER CONFIG ===` block at the top of the wrapper script you're running, or pass it via `--speech-dir` directly to the generator scripts.

**Reference mic index.** Most preprocessing and test functions require a 0-based reference mic index (the mic closest to the target speaker at azimuth 0). Use `get_ref_idx` from `ambidrop/constants.py` to look it up by array name:

```python
from ambidrop.constants import REF_IDX_MAP, get_ref_idx

ref_1based = get_ref_idx("uniform sphere (rigid) radius = 0.1")  # → 2
ref_0based  = ref_1based - 1
```

`get_ref_idx(name)` strips the `_preprocessed` suffix automatically, so it works with both bare array names and on-disk directory names. For arrays not in `REF_IDX_MAP`, use `find_ref_mic` from `ambidrop/signal_utils.py` which selects the mic closest to azimuth 0 given mic positions.

---

## 2. Wrapper Scripts

The three wrapper scripts are the recommended entry point. They orchestrate data generation, preprocessing, training, and evaluation in sequence.

---

### `run_FT_JNF.py`

End-to-end pipeline for the FT-JNF model.

**Edit the USER CONFIG block at the top of the file before running:**

```python
WSJ0_ROOT   = "/path/to/wsj0"
DATA_ROOT   = "datasets/run_ftjnf"
CKPT_DIR    = "checkpoints/FT_JNF"
ARRAYS_TRAIN = [...]   # list of ArraySpec — use PAPER_ARRAYS_TRAIN to reproduce paper
ARRAYS_TEST  = [...]   # list of ArraySpec — use PAPER_ARRAYS_TEST to reproduce paper
```

**Flags**

| Flag | Default | Description |
|------|---------|-------------|
| `--mode {ambidrop,baseline,both}` | required | Training and evaluation mode |
| `--actions ACTION [ACTION ...]` | required | One or more of: `generate`, `preprocess`, `train`, `test` |
| `--checkpoint PATH` | auto | Override checkpoint for `test` (single-mode only) |
| `--checkpoint-baseline PATH` | — | Baseline checkpoint when `--mode both` |
| `--checkpoint-ambidrop PATH` | — | AmbiDrop checkpoint when `--mode both` |
| `--test-arrays {test,train,both}` | `both` | Which array geometries to generate/evaluate at test time |
| `--test-raw-dir-test PATH` | auto | Raw Type-C data directory for test arrays |
| `--test-raw-dir-train PATH` | — | Raw Type-C data directory for train arrays; enables train-array evaluation |
| `--legacy-eval-dir PATH` | — | Evaluate on a pre-existing preprocessed directory (skips generate/preprocess) |
| `--raw-baseline-train PATH` | auto | Existing raw baseline training directory (skips generate for this split) |
| `--raw-baseline-val PATH` | auto | Existing raw baseline validation directory |
| `--raw-ambidrop-train PATH` | auto | Existing raw AmbiDrop training directory (skips generate for this split) |
| `--raw-ambidrop-val PATH` | auto | Existing raw AmbiDrop validation directory |
| `--prep-baseline-train PATH` | — | Already-preprocessed baseline training directory (skips preprocess) |
| `--prep-baseline-val PATH` | — | Already-preprocessed baseline validation directory |
| `--prep-ambidrop-train PATH` | — | Already-preprocessed AmbiDrop training directory (skips preprocess) |
| `--prep-ambidrop-val PATH` | — | Already-preprocessed AmbiDrop validation directory |
| `--wandb-project STR` | `FT_JNF` | W&B project name |
| `--wandb-entity STR` | — | W&B entity / team |
| `--wandb-run-name STR` | — | Prefix for W&B run names |
| `--no-wandb` | False | Disable W&B logging entirely |

**Examples**

```bash
# Full pipeline from scratch
python run_FT_JNF.py --mode ambidrop --actions generate preprocess train test

# Both modes in one run (direct comparison)
python run_FT_JNF.py --mode both --actions generate preprocess train test

# Evaluate pre-existing checkpoints (no generation or training)
python run_FT_JNF.py --mode both --actions test

# Use existing raw data, skip generation
python run_FT_JNF.py --mode ambidrop --actions preprocess train test

# Evaluate a specific checkpoint on fresh data
python run_FT_JNF.py --mode ambidrop --actions generate preprocess test \
    --checkpoint checkpoints/FT_JNF/SH_FT_JNF,2025-12-01_10-08-18.pt

# Evaluate on both test AND train arrays
python run_FT_JNF.py --mode both --actions generate preprocess test --test-arrays both

# Reuse existing raw data from a different location
python run_FT_JNF.py --mode both --actions preprocess train \
    --raw-ambidrop-train datasets/my_run/raw/ambidrop_train \
    --raw-ambidrop-val   datasets/my_run/raw/ambidrop_val \
    --raw-baseline-train datasets/my_run/raw/baseline_train \
    --raw-baseline-val   datasets/my_run/raw/baseline_val

# Train directly from already-preprocessed .pt files
python run_FT_JNF.py --mode ambidrop --actions train \
    --prep-ambidrop-train /path/to/preprocessed/train \
    --prep-ambidrop-val   /path/to/preprocessed/val
```

---

### `run_ConvTasNet.py`

End-to-end pipeline for IC Conv-TasNet. Same `--mode` and `--actions` structure as `run_FT_JNF.py` with the following differences:

- **No W&B flags** — `--wandb-*` and `--no-wandb` are not available
- **No `--legacy-eval-dir`** — legacy directory evaluation is not supported
- **`--test-arrays` defaults to `test`** (not `both` as in `run_FT_JNF.py`)
- **Preprocessing is time-domain**, not STFT: Conv-TasNet operates on raw waveforms
- **AmbiDrop test preprocessing encodes ASM on-the-fly** from `p.wav` via `ConvTasNet/preprocess.py` rather than reading pre-stored `anmt_array` from `anm.mat`

**Edit the USER CONFIG block at the top of the file before running** (same fields as `run_FT_JNF.py`).

**Flags**

| Flag | Default | Description |
|------|---------|-------------|
| `--mode {ambidrop,baseline,both}` | required | Training and evaluation mode |
| `--actions ACTION [ACTION ...]` | required | One or more of: `generate`, `preprocess`, `train`, `test` |
| `--checkpoint PATH` | auto | Override checkpoint for `test` (single-mode only) |
| `--checkpoint-baseline PATH` | — | Baseline checkpoint when `--mode both` |
| `--checkpoint-ambidrop PATH` | — | AmbiDrop checkpoint when `--mode both` |
| `--test-arrays {test,train,both}` | `test` | Which array geometries to evaluate at test time |
| `--test-raw-dir-test PATH` | auto | Raw Type-C data directory for test arrays |
| `--test-raw-dir-train PATH` | — | Raw Type-C data directory for train arrays; enables train-array evaluation |
| `--raw-baseline-train PATH` | auto | Existing raw baseline training directory |
| `--raw-baseline-val PATH` | auto | Existing raw baseline validation directory |
| `--raw-ambidrop-train PATH` | auto | Existing raw AmbiDrop training directory |
| `--raw-ambidrop-val PATH` | auto | Existing raw AmbiDrop validation directory |
| `--prep-baseline-train PATH` | — | Already-preprocessed baseline training directory (skips preprocess) |
| `--prep-baseline-val PATH` | — | Already-preprocessed baseline validation directory |
| `--prep-ambidrop-train PATH` | — | Already-preprocessed AmbiDrop training directory (skips preprocess) |
| `--prep-ambidrop-val PATH` | — | Already-preprocessed AmbiDrop validation directory |

```bash
# Full pipeline
python run_ConvTasNet.py --mode ambidrop --actions generate preprocess train test

# Evaluate pre-existing checkpoints
python run_ConvTasNet.py --mode ambidrop --actions test

# Evaluate on both test AND train arrays
python run_ConvTasNet.py --mode both --actions generate preprocess test --test-arrays both

# Reuse existing raw data
python run_ConvTasNet.py --mode both --actions preprocess train \
    --raw-ambidrop-train datasets/my_run/raw/ambidrop_train \
    --raw-ambidrop-val   datasets/my_run/raw/ambidrop_val
```

---

### `run_Real_World.py`

Evaluates an FT-JNF checkpoint on real Project Aria glasses recordings. The model architecture is resolved automatically from the checkpoint filename via `CHECKPOINT_REGISTRY`.

**Requires:** `datasets/aria_ds/` directory with scenario subdirectories and ATF files.

**Flags**

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint PATH` | preferred AmbiDrop ckpt | FT-JNF checkpoint file |
| `--aria-data-dir PATH` | `datasets/aria_ds` | Root dir containing scenario subdirectories |
| `--atf {simulated,measured}` | `simulated` | ATF source for ASM |
| `--sofa-path PATH` | `datasets/aria_ds/aria_atfs_fixed.sofa` | SOFA file for measured ATF path |
| `--atf-npy-path PATH` | `datasets/aria_ds/ATF.npy` | Precomputed ATF `.npy` for simulated path |
| `--cnm-path PATH` | — | Precomputed cnm `.npy` (shape `M × nm × F_full`); activates precomputed path |
| `--mode {ambidrop,baseline}` | from registry | Override model type |
| `--scenarios NAME [NAME ...]` | all found | Scenario subdirectory names to evaluate |
| `--ref-mic INT` | `3` | 1-based reference mic index (closest to target speaker) |
| `--regularization {tikhonov,svd}` | `tikhonov` | ASM solver (simulated ATF path only) |
| `--output-csv PATH` | — | Save per-scenario results to a CSV |

**Three ASM encoding paths**

| Path | Flag | Notes |
|------|------|-------|
| Simulated ATF | `--atf simulated` (default) | Tikhonov inversion at 16 kHz via `ambidrop/asm.py` |
| Measured ATF | `--atf measured` | Shroom ASM at 48 kHz, then resampled |
| Precomputed cnm | `--cnm-path FILE` | Skips coefficient computation; fastest |

```bash
# Default: simulated ATF
python run_Real_World.py

# Measured ATF (requires SOFA file)
python run_Real_World.py --atf measured

# Precomputed cnm
python run_Real_World.py --cnm-path datasets/aria_ds/cnm_shroom.npy

# Specific scenarios, save results
python run_Real_World.py --scenarios scenario_1 scenario_2 --output-csv results.csv
```

**Generating `cnm_shroom.npy` from the measured ATF:**

```python
import numpy as np
from shroom.encoders.asm import ASM
from shroom.utils.file_utils import load_file

array = load_file("datasets/aria_ds/aria_atfs_fixed.sofa")
array.toFreq()
asm = ASM(sh_order=2, array=array, fs=array.fs)
asm.calculate()
np.save("datasets/aria_ds/cnm_shroom.npy", asm.cnm.data)  # shape: (M, nm, F_full)
```

---

## 3. Data Generation

Three generator scripts produce the raw data for training and evaluation. All share the same room simulation pipeline: random rooms via `pyroomacoustics` (ISM method), array ATFs from `shroom`, speech from WSJ0. Output is organised into `ex_0/`, `ex_1/`, … subdirectories.

---

### `generate_ambidrop_train_ds.py` — Type A

Generates **ideal Ambisonics** training data (no physical microphone array). Output per example: `anm.mat` containing `anmt` (complex 9-channel SH signals) and `anmtDirect` (clean a₀₀ target).

Used for: FT-JNF and Conv-TasNet AmbiDrop training.

**Flags**

| Flag | Default | Description |
|------|---------|-------------|
| `--n-examples N` | 6000 | Number of training examples |
| `--seed N` | fixed | RNG seed for reproducibility |
| `--output-dir PATH` | configured root | Output directory for training examples |
| `--speech-dir PATH` | WSJ0 train split | Path to speech files |
| `--n-val N` | 1000 | Number of validation examples |
| `--val-output-dir PATH` | auto | Output directory for validation examples |
| `--val-speech-dir PATH` | WSJ0 val split | Path to validation speech files |
| `--val-seed N` | auto | Separate RNG seed for validation set |

```bash
python datagenerator/generate_ambidrop_train_ds.py \
    --speech-dir /path/to/wsj0/train \
    --output-dir datasets/ambidrop_train \
    --n-examples 6000
```

---

### `generate_baseline_train_ds.py` — Type B

Generates **microphone array signals** for baseline training. Output per example: `p.wav` (7-ch noisy), `pDirect.wav` (7-ch clean). Array geometry is sampled from the configured list at each example.

Used for: FT-JNF and Conv-TasNet baseline training.

**Flags** (same structure as Type A)

| Flag | Default | Description |
|------|---------|-------------|
| `--n-examples N` | 6000 | Number of training examples |
| `--seed N` | fixed | RNG seed |
| `--output-dir PATH` | configured root | Output directory |
| `--speech-dir PATH` | WSJ0 train split | Path to speech files |
| `--n-val N` | 1000 | Validation examples |
| `--val-output-dir PATH` | auto | Validation output directory |
| `--val-speech-dir PATH` | WSJ0 val split | Validation speech |
| `--val-seed N` | auto | Validation RNG seed |

```bash
python datagenerator/generate_baseline_train_ds.py \
    --speech-dir /path/to/wsj0/train \
    --output-dir datasets/baseline_train \
    --n-examples 6000
```

---

### `generate_inference_ds.py` — Type C

Generates the **evaluation dataset**: microphone signals (Type B) plus ASM-encoded Ambisonics (`anmt_array`) saved in `anm.mat`. This is the key evaluation format — it tests whether a model trained on ideal SH (Type A) can enhance ASM-encoded signals from a real array.

**Flags**

| Flag | Default | Description |
|------|---------|-------------|
| `--n-examples N` | 500 | Number of test examples |
| `--seed N` | fixed | RNG seed |
| `--output-dir PATH` | configured root | Output directory |
| `--speech-dir PATH` | WSJ0 test split | Path to speech files |

```bash
python datagenerator/generate_inference_ds.py \
    --speech-dir /path/to/wsj0/test \
    --output-dir datasets/inference \
    --n-examples 500
```

---

### Paper Arrays

**File:** `datagenerator/paper_arrays.py`

Defines the 21 microphone array geometries from the paper as `ArraySpec` instances.

```python
from datagenerator.paper_arrays import (
    PAPER_ARRAYS_TRAIN,   # 10 training arrays (Fig. 2 in paper)
    PAPER_ARRAYS_TEST,    # 11 test arrays (Fig. 3 in paper)
    PAPER_ARRAYS_ALL,     # all 21 combined
)
```

Each `ArraySpec` contains mic positions, array type (free-field / rigid sphere), and array name. Assign these to the `ARRAYS_TRAIN` / `ARRAYS_TEST` variables in the wrapper USER CONFIG to reproduce exact paper results.

To visualise the array geometries:

```python
from datagenerator.paper_arrays import plot_paper_arrays
plot_paper_arrays()   # generates geometry figures for all 21 arrays
```

---

## 4. Preprocessing

**File:** `ambidrop/preprocess.py`

Converts raw data folders into `.pt` tensor files for training and evaluation. All functions extract a fixed-length window anchored to speech onset and normalise by peak amplitude.

**Per-example functions**

| Function | Input type | Output | Used by |
|----------|-----------|--------|---------|
| `preprocess_mic(ex_dir, ref_id, train)` | Type B raw dir | `(T, 257, 14)` STFT + `(T_s,)` clean | FT-JNF baseline |
| `preprocess_sh_stft(ex_dir, anm_source, train)` | Type A (`"ideal"`) or C (`"asm"`) | `(T, 257, 18)` STFT + `(T_s,)` clean a₀₀ | FT-JNF AmbiDrop |
| `preprocess_sh_time(ex_dir, train)` | Type A raw dir | `(9, T_s)` real ACN + `(T_s,)` clean | Conv-TasNet AmbiDrop |

- `train=True` → 6 s window; `train=False` → 7.5 s window
- `anm_source="ideal"` reads `anmt` from `anm.mat`; `anm_source="asm"` reads `anmt_array`

**Batch functions**

```python
from ambidrop.preprocess import preprocess_dataset, preprocess_dataset_multi

# Process all ex_N/ subdirs in one raw_dir → save .pt files to out_dir
preprocess_dataset(raw_dir, out_dir, preprocess_fn, train=True, **fn_kwargs)

# Process multiple array directories and merge into one combined output dir
preprocess_dataset_multi(array_dirs, out_dir, preprocess_fn, train=True, **fn_kwargs)
```

**When to use which batch function**

| Scenario | Function | Reason |
|----------|----------|--------|
| Type A / B training or validation (multiple arrays) | `preprocess_dataset_multi` | You want one merged dataset so the model trains across all arrays simultaneously |
| Type C inference / test (multiple arrays) | `preprocess_dataset` — once per array | You want each array's results to stay separate so you can compare per-array performance |

---

### `ConvTasNet/preprocess.py` — Time-domain preprocessing (Conv-TasNet only)

These two functions produce time-domain `.pt` dicts (no STFT) for the IC Conv-TasNet pipeline. Both are compatible with `preprocess_dataset` / `preprocess_dataset_multi`.

| Function | Input | Output (`format` key) | Used by |
|----------|-------|-----------------------|---------|
| `preprocess_mic_time(ex_dir, ref_id, ...)` | Type B raw dir | `noisy (M,T)`, `clean (M,T)`, `ref_id` — `format='time'` | Conv-TasNet baseline train/test |
| `preprocess_ambisonics_time(ex_dir, V, th, ph, ...)` | Type C raw dir + steering matrix | `noisy_mic (M,T)`, `clean_mic (M,T)`, `anmt (9,T)`, `clean_anm (T,)` — `format='ambidrop_test'` | Conv-TasNet AmbiDrop test |

`preprocess_ambisonics_time` encodes Ambisonics on-the-fly from raw `p.wav` via ASM (no pre-stored `anmt_array` needed). It corrects the group delay introduced by the array IR + ASM filters using `estimate_delay` / `align_to_lag` from `datagenerator/helpers.py`. The steering matrix `V` (shape `M × F_pos × Q`) is built by `_build_steering_matrix` in `run_ConvTasNet.py`.

`SimDS_preprocessed` in `ConvTasNet/datasets.py` dispatches on the `format` key: `'time'` returns a 5-tuple (noisy, clean, ref_id, array_name, ex_id), `'ambidrop_test'` returns a 4-tuple (noisy_mic, clean_mic, anmt, clean_anm, ref_id), and the absence of a dict (raw 2-tuple) is used for AmbiDrop training data from `preprocess_sh_time`.

**Example — preprocess a Type A training set from multiple arrays (merged):**

```python
from ambidrop.preprocess import preprocess_sh_stft, preprocess_dataset_multi

preprocess_dataset_multi(
    array_dirs=[
        "datasets/ambidrop_train/array_1",
        "datasets/ambidrop_train/array_2",
    ],
    out_dir="datasets/ambidrop_train_preprocessed_merged",
    preprocess_fn=preprocess_sh_stft,
    train=True,
    anm_source="ideal",
)
```

**Example — preprocess a Type C test set, keeping each array separate:**

```python
from ambidrop.preprocess import preprocess_sh_stft, preprocess_dataset

for array_name in ["array_1", "array_2", "array_3"]:
    preprocess_dataset(
        raw_dir=f"datasets/inference/{array_name}",
        out_dir=f"datasets/inference_preprocessed/{array_name}",
        preprocess_fn=preprocess_sh_stft,
        train=False,
        anm_source="asm",
    )
# evaluate each out_dir separately → per-array metrics
```

---

## 5. ASM (Ambisonics Signal Matching)

**File:** `ambidrop/asm.py`

Encodes microphone signals into the Ambisonics domain at inference time using Tikhonov-regularised steering matrix inversion.

**Public API**

```python
from ambidrop.asm import encode_ambisonics, compute_asm_coefficients, apply_asm_filters
```

---

### `encode_ambisonics` — unified entry point

```python
encoded_anm, cnm = encode_ambisonics(
    mic_signals,        # (M, T) numpy array — M microphone signals, T samples
    V,                  # (M, F, Q) — array steering matrix (complex)
    sh_order=2,         # Ambisonics order; output will have (sh_order+1)^2 channels
    th=None,            # (Q,) elevation angles (radians) of Q plane-wave directions
    ph=None,            # (Q,) azimuth angles (radians)
    method="tikhonov",  # solver: "tikhonov" or "svd"
    cnm=None,           # pass precomputed cnm to skip coefficient computation
    sh_type="complex",  # "complex" or "real" (for Conv-TasNet)
    filt_samp=512,      # filter length in samples
)
# Returns:
#   encoded_anm  — (nm, T) encoded Ambisonics signals
#   cnm          — ((N+1)^2, F, M) filter coefficients (reusable)
```

**With precomputed cnm (faster at inference):**

```python
# First call: compute and cache cnm
encoded_anm, cnm = encode_ambisonics(mic_signals, V, sh_order=2, th=th, ph=ph)

# Subsequent calls: reuse cnm, skip coefficient computation
encoded_anm, _ = encode_ambisonics(mic_signals_new, V, cnm=cnm)
```

---

### `compute_asm_coefficients` — compute filters only

```python
cnm = compute_asm_coefficients(
    V,                  # (M, F, Q) steering matrix
    sh_order=2,
    th,                 # (Q,) elevation angles
    ph,                 # (Q,) azimuth angles
    method="tikhonov",
    filt_samp=512,
)
# Returns: cnm — ((N+1)^2, F, M) filter coefficients
```

---

### `apply_asm_filters` — apply precomputed filters

```python
encoded_anm = apply_asm_filters(
    mic_signals,   # (M, T)
    cnm,           # ((N+1)^2, F, M) precomputed coefficients
)
# Returns: encoded_anm — ((N+1)^2, T)
```

---

## 6. FT-JNF — Direct Training

**File:** `FT_JNF/train.py`

Train the FT-JNF model directly without the wrapper script. Useful when you want fine-grained control over data paths and hyperparameters.

**Flags**

| Flag | Default | Description |
|------|---------|-------------|
| `--mode {baseline,ambidrop}` | required | Training mode |
| `--data-dir PATH` | `datasets/experiment_full_anm` | Root directory containing preprocessed splits |
| `--train-split NAME` | `mic_train_ds_preprocessed_merged` | Subdirectory name for training data |
| `--val-split NAME` | `mic_val_ds_preprocessed_merged` | Subdirectory name for validation data |
| `--input-dim N` | 14 | Input feature dimension (14 for baseline, 18 for ambidrop) |
| `--hidden1 N` | 64 | BiLSTM-1 hidden size |
| `--hidden2 N` | 64 | BiLSTM-2 hidden size |
| `--output-dim N` | 2 | Output dimension (always 2: real + imag IRM) |
| `--dropout-type TYPE` | None | `SHChannelDropout` or `PerChDropout` (ambidrop only) |
| `--drop-prob F` | 0.4 | Channel drop probability |
| `--max-drop N` | 3 | Max channels to drop per example |
| `--drop-probs STR` | None | Comma-separated per-channel probabilities (PerChDropout) |
| `--epochs N` | 300 | Maximum training epochs |
| `--batch-size N` | 8 | Batch size |
| `--lr F` | 0.001 | Learning rate |
| `--weight-decay F` | 1e-6 | L2 regularisation |
| `--max-batches N` | None | Limit batches per epoch (for debugging) |
| `--checkpoint PATH` | None | Resume from checkpoint |
| `--save-dir PATH` | `checkpoints/FT_JNF` | Directory to save checkpoints |
| `--wandb-project STR` | `speech-enhancement` | W&B project name |
| `--wandb-entity STR` | — | W&B entity/username |
| `--no-wandb` | False | Disable W&B logging |

```bash
# AmbiDrop training with SHChannelDropout
python FT_JNF/train.py \
    --mode ambidrop \
    --input-dim 18 \
    --dropout-type SHChannelDropout \
    --drop-prob 0.4 --max-drop 3 \
    --data-dir datasets/experiment_full_anm \
    --no-wandb

# Baseline training
python FT_JNF/train.py \
    --mode baseline \
    --input-dim 14 \
    --data-dir datasets/experiment_full_anm \
    --no-wandb
```

---

## 7. FT-JNF — Direct Evaluation (Simulated)

**File:** `FT_JNF/test_simulated.py`

Evaluates an FT-JNF checkpoint on a preprocessed simulated test set. Reports per-array SI-SDR, PESQ, and STOI.

**Flags**

| Flag | Default | Description |
|------|---------|-------------|
| `--mode {baseline,ambidrop}` | required | Model mode |
| `--checkpoint PATH` | required | Path to `.pt` checkpoint file |
| `--epoch N` | None (latest) | Specific epoch to load |
| `--data-dir PATH` | required | Directory with array subdirectories of preprocessed `.pt` files |
| `--test-type STR` | None (auto) | Override data format detection |
| `--zero-channels N` | 0 | Force-zero N random SH channels at test time (mic failure simulation) |
| `--input-dim N` | 18 | Model input dimension |
| `--hidden1 N` | 64 | BiLSTM-1 hidden size |
| `--hidden2 N` | 64 | BiLSTM-2 hidden size |
| `--output-dim N` | 2 | Output dimension |
| `--dropout-type TYPE` | None | Dropout type (must match trained model) |
| `--drop-prob F` | 0.4 | Drop probability |
| `--max-drop N` | 3 | Max channels to drop |
| `--drop-probs STR` | None | Per-channel probabilities (PerChDropout) |
| `--output-csv PATH` | None | Save results to CSV |
| `--no-wandb` | False | Disable W&B logging |

```bash
python FT_JNF/test_simulated.py \
    --mode ambidrop \
    --checkpoint checkpoints/FT_JNF/SH_FT_JNF,2025-12-01_10-08-18.pt \
    --data-dir datasets/experiment_full_anm/test_of_test_ds_preprocessed \
    --no-wandb
```

---

## 8. Conv-TasNet — Direct Training

**File:** `ConvTasNet/train.py`

**Flags**

| Flag | Default | Description |
|------|---------|-------------|
| `--mode {baseline,ambidrop}` | `ambidrop` | Training mode |
| `--train_dir PATH` | None | Preprocessed training data directory |
| `--valid_dir PATH` | None | Preprocessed validation data directory |
| `--mic_num N` | 9 | Input channel count (9 for ambidrop, 7 for baseline) |
| `--enc_dim N` | 512 | Encoder output dimension |
| `--feature_dim N` | 128 | TCN feature dimension |
| `--ch_dim N` | 8 | Inter-channel attention dimension |
| `--win N` (ms) | 16 | Encoder window size in milliseconds |
| `--layer N` | 8 | TCN dilated convolution layers |
| `--stack N` | 1 | Number of TCN stacks |
| `--kernel N` | 3 | TCN kernel size |
| `--dropout_type TYPE` | `SHChannelDropout` | Dropout type |
| `--drop_prob F` | 0.4 | Drop probability |
| `--max_drop N` | 3 | Max channels to drop |
| `--drop_probs STR` | None | Per-channel probabilities (PerChDropout) |
| `--epochs N` | 100 | Training epochs |
| `--batch_size N` | 64 | Batch size |
| `--lr F` | 1e-3 | Learning rate |
| `--optimizer STR` | `adam` | Optimizer (`adam` or `sgd`) |
| `--save_folder PATH` | auto | Checkpoint save directory |
| `--continue_from PATH` | `""` | Resume from checkpoint |
| `--no_wandb` | False | Disable W&B logging |

```bash
python ConvTasNet/train.py \
    --mode ambidrop \
    --train_dir datasets/ambidrop_train_preprocessed \
    --valid_dir datasets/ambidrop_val_preprocessed \
    --no_wandb
```

---

## 9. Conv-TasNet — Direct Evaluation

**File:** `ConvTasNet/evaluate.py`

**Flags**

| Flag | Default | Description |
|------|---------|-------------|
| `--mode {baseline,ambidrop}` | `ambidrop` | Model mode |
| `--model_path PATH` | required | Path to `.pth.tar` checkpoint |
| `--data_dir PATH` | required | Preprocessed test data directory |
| `--dropout_type TYPE` | `SHChannelDropout` | Dropout type (must match trained model) |
| `--drop_prob F` | 0.4 | Drop probability |
| `--max_drop N` | 3 | Max channels to drop |
| `--drop_probs STR` | None | Per-channel probabilities |
| `--use_cuda N` | 1 | 1=use GPU, 0=CPU only |
| `--no_wandb` | False | Disable W&B |

```bash
python ConvTasNet/evaluate.py \
    --mode ambidrop \
    --model_path checkpoints/ConvTasNet/run_2026-04-09_08-35/final.pth.tar \
    --data_dir datasets/ambidrop_test_preprocessed \
    --no_wandb
```

To reproduce Table III from the paper without the wrapper:

```bash
python ConvTasNet/main_results.py \
    --ambidrop-checkpoint checkpoints/ConvTasNet/run_2026-04-09_08-35/final.pth.tar \
    --baseline-checkpoint checkpoints/ConvTasNet/run_2026-04-09_10-55/final.pth.tar
```

---

## 10. Ablation Scripts

All scripts are in `FT_JNF/ablations/`. Each is standalone — no wrapper needed. Checkpoints are resolved from `CHECKPOINT_REGISTRY` in `FT_JNF/constants.py`.

---

### `main_results.py` — Reproduce Table I (Simulated)

Evaluates baseline and AmbiDrop FT-JNF on training and test arrays.

| Flag | Default | Description |
|------|---------|-------------|
| `--train-data-dir PATH` | `datasets/experiment_full_anm/test_of_train_ds_preprocessed` | Preprocessed training-array test data |
| `--test-data-dir PATH` | `datasets/experiment_full_anm/test_of_test_ds_preprocessed` | Preprocessed test-array test data |
| `--checkpoint-dir PATH` | `checkpoints/FT_JNF` | Directory containing checkpoints |
| `--from-csv PATH` | None | Load pre-computed results from CSV |

```bash
python FT_JNF/ablations/main_results.py \
    --train-data-dir datasets/experiment_full_anm/test_of_train_ds_preprocessed \
    --test-data-dir datasets/experiment_full_anm/test_of_test_ds_preprocessed
```

---

### `main_results_real.py` — Reproduce Table II (Real-world Aria)

Evaluates baseline, AmbiDrop + simulated ATF, and AmbiDrop + measured ATF on real Aria recordings.

| Flag | Default | Description |
|------|---------|-------------|
| `--aria-data-dir PATH` | `datasets/aria_ds` | Aria data root |
| `--scenarios NAME [...]` | all | Scenario subdirectory names |
| `--checkpoint PATH` | preferred AmbiDrop ckpt | AmbiDrop checkpoint |
| `--baseline-checkpoint PATH` | preferred baseline ckpt | Baseline checkpoint |
| `--output PATH` | `figures/table2_real.png` | Output figure path |
| `--from-csv PATH` | None | Load pre-computed results from CSV |
| `--save-csv PATH` | None | Save results to CSV |

```bash
python FT_JNF/ablations/main_results_real.py \
    --aria-data-dir datasets/aria_ds
```

---

### `dropout_ablation.py` — Dropout Strategies (Fig. 6)

Compares uniform `SHChannelDropout` (various max_drop / drop_prob settings) against per-channel `PerChDropout` (various error thresholds). Loads all relevant checkpoints from `CHECKPOINT_REGISTRY` automatically.

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir PATH` | `datasets/experiment_full_anm/test_of_train_ds_preprocessed` | Test data directory |
| `--output PATH` | `figures/fig6_dropout_ablation.png` | Output figure |

```bash
python FT_JNF/ablations/dropout_ablation.py \
    --data-dir datasets/experiment_full_anm/test_of_train_ds_preprocessed \
    --output figures/fig6_dropout_ablation.png
```

---

### `net_complexity.py` — Network Complexity (Fig. 8)

Evaluates all 10 `checkpoint_size_*.pt` checkpoints (3.5K–1.2M parameters) to plot SI-SDRi vs. parameter count.

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir PATH` | `datasets/experiment_full_anm/test_of_train_ds_preprocessed` | Test data directory |
| `--checkpoint-dir PATH` | `checkpoints` | Root checkpoint directory |
| `--output PATH` | `figures/fig8_net_complexity.png` | Output figure |

```bash
python FT_JNF/ablations/net_complexity.py \
    --data-dir datasets/experiment_full_anm/test_of_train_ds_preprocessed
```

---

### `mic_failure.py` — Microphone Failure (Fig. 7)

Evaluates how AmbiDrop and the baseline degrade when 1, 2, or 3 microphones are artificially removed at inference time.

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir PATH` | required | Raw (unpreprocessed) test data directory |
| `--arrays NAME [...]` | all in REF_IDX_MAP | Array names to evaluate |
| `--ambidrop-checkpoint NAME` | `SH_FT_JNF,2025-12-01_10-08-18.pt` | AmbiDrop checkpoint filename |
| `--baseline-checkpoint NAME` | `FT_JNF,2026-03-25_13-37-42.pt` | Baseline checkpoint filename |
| `--checkpoint-dir PATH` | `checkpoints/FT_JNF` | Checkpoint directory |
| `--from-csv PATH` | None | Load pre-computed results from CSV |
| `--output PATH` | `figures/fig7_mic_failure.png` | Output figure |
| `--save-csv PATH` | None | Save results to CSV |

```bash
python FT_JNF/ablations/mic_failure.py \
    --data-dir datasets/experiment_full_anm/test_of_test_ds
```

---

### `snr_distribution.py` — SNR Distribution (Fig. 4)

Analyses the SNR distribution of the inference dataset and plots AmbiDrop performance across SNR bins.

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint PATH` | None | AmbiDrop checkpoint path |
| `--data-dir PATH` | None | Preprocessed test data directory |
| `--from-csv PATH` | None | Load pre-computed 1 dB-binned CSV |
| `--noisy-npy PATH` | None | Path to `master_si_sdr_noisy.npy` |
| `--enhanced-npy PATH` | None | Path to `master_si_sdr_enhanced.npy` |
| `--output PATH` | `figures/fig4_snr_distribution.png` | Output figure |
| `--bin-width N` | 2 | SNR bin width in dB |
| `--save-csv PATH` | None | Save binned results to CSV |

```bash
# Run from scratch
python FT_JNF/ablations/snr_distribution.py \
    --checkpoint checkpoints/FT_JNF/SH_FT_JNF,2025-12-01_10-08-18.pt \
    --data-dir datasets/experiment_full_anm/test_of_test_ds_preprocessed

# Load pre-computed results
python FT_JNF/ablations/snr_distribution.py \
    --from-csv figures/fig4_snr_distribution.csv
```

---

## 11. Checkpoint Registry

**File:** `FT_JNF/constants.py`

`CHECKPOINT_REGISTRY` maps each checkpoint filename to its full model architecture configuration. The evaluation scripts use this to reconstruct the model without requiring the user to specify architecture flags manually.

**Structure of each entry:**

```python
CHECKPOINT_REGISTRY = {
    "SH_FT_JNF,2025-12-01_10-08-18.pt": {
        "mode":       "ambidrop",
        "input_dim":  18,
        "hidden1":    64,
        "hidden2":    64,
        "dropout":    "SHChannelDropout",
        "drop_prob":  0.4,
        "max_drop":   3,
    },
    "FT_JNF,2026-03-25_13-37-42.pt": {
        "mode":       "baseline",
        "input_dim":  14,
        "hidden1":    64,
        "hidden2":    64,
    },
    # ...
}
```

**Adding a new checkpoint:**

1. Train the model and note the checkpoint filename.
2. Add an entry to `CHECKPOINT_REGISTRY` with the correct `mode`, `input_dim`, `hidden1`, `hidden2`, and (if applicable) `dropout`, `drop_prob`, `max_drop`.
3. The evaluation scripts and `run_Real_World.py` will pick it up automatically.

**Preferred checkpoints for paper experiments:**

| Purpose | Checkpoint |
|---------|-----------|
| FT-JNF AmbiDrop (preferred) | `SH_FT_JNF,2025-12-01_10-08-18.pt` |
| FT-JNF Baseline (preferred) | `FT_JNF,2026-03-25_13-37-42.pt` |
| Conv-TasNet AmbiDrop | `checkpoints/ConvTasNet/run_2026-04-09_08-35/final.pth.tar` |
| Conv-TasNet Baseline | `checkpoints/ConvTasNet/run_2026-04-09_10-55/final.pth.tar` |
