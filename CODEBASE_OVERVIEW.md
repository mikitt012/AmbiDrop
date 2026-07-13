# AmbiDrop — Codebase Overview

_Last updated: 2026-07-13_

> **Living document.** Update this file whenever the codebase changes (new modules, renamed files, new checkpoints, changed APIs).

This document describes the four key aspects of the AmbiDrop codebase: the neural networks used, how data is handled and preprocessed, where checkpoints live, and how ASM is computed.

---

## 1. Neural Networks

### FT-JNF (Frequency-Time Joint Non-linear Filter)
**File:** `FT_JNF/model.py`

A two-stage bidirectional LSTM network operating in the STFT domain.

```
Input (B, T, F, 2C)
  └─ BiLSTM-1: across frequency dimension (hidden1_dim × 2 outputs per frame)
  └─ BiLSTM-2: across time dimension (hidden2_dim × 2 outputs per frequency)
  └─ Linear:   → complex IRM mask (B, T, F, 2)
Output: mask applied to reference channel STFT → enhanced speech
```

- **AmbiDrop mode** — input C=9 SH channels → 18 real/imag dims; `SHChannelDropout` applied at train time
- **Baseline mode** — input C=7 mic channels → 14 real/imag dims; no dropout
- The mask is always applied to channel 0 (a₀₀, the omnidirectional SH component), which is also the training target
- STFT parameters: N_FFT=512, HOP=256, WIN=512, fs=16000 → (T, 257, 2C) tensors

### IC Conv-TasNet (Improved Convolutional Time-domain Audio Separation Network)
**File:** `ConvTasNet/model.py`

A time-domain encoder-decoder with a dilated TCN core.

```
Input (B, C, T)  — real ACN time-domain signals (C=9 AmbiDrop, C=7 baseline)
  └─ Conv1d encoder:       (B, C, T) → (B, C × enc_dim, T/stride)
  └─ TCN with attention:   dilated depthwise-separable convolutions (8 layers, 1 stack)
  └─ ConvTranspose1d:      → enhanced speech (B, 1, T)
```

- Operates on **real-valued ACN** signals (complex SH converted via `complex_acn_to_real_acn` in `ambidrop/signal_utils.py`)
- Default config: enc_dim=512, feature_dim=128, ch_dim=8, kernel=3, win=16ms

---

## 2. Data Pipeline

The pipeline has three data types and three preprocessing functions.

### Data Types

| Type | Generator | Contents | Used for |
|------|-----------|----------|---------|
| **A** — Ideal Ambisonics | `datagenerator/generate_ambidrop_train_ds.py` | `anm.mat`: `anmt` (T,9) complex SH, `anmtDirect` (T,1) a₀₀ clean | FT-JNF + Conv-TasNet AmbiDrop **training** |
| **B** — Mic signals | `datagenerator/generate_baseline_train_ds.py` | `p.wav` (T,7) noisy mics, `pDirect.wav` (T,7) clean mics | FT-JNF + Conv-TasNet **baseline training** |
| **C** — Full inference | `datagenerator/generate_inference_ds.py` | All of B plus `anmt_array` (T,9) ASM-encoded SH in `anm.mat` | AmbiDrop **test / inference** |

All three types share the same scene randomisation: room dimensions (2.5–5 × 3–9 × 2.2–3.5 m), T60 (0.2–0.5 s), one target speaker at azimuth 0 and 5 interferers. The scene is simulated in the SH domain at order 20 using `shroom.acoustics.room.Room` (pyroomacoustics backend), then the target is rotated to azimuth 0 via a Wigner-D matrix before truncating to order 2.

The 21 microphone array geometries used in the paper are predefined in `datagenerator/paper_arrays.py` as `PAPER_ARRAYS_TRAIN` (10 arrays), `PAPER_ARRAYS_TEST` (11 arrays), and `PAPER_ARRAYS_ALL` (21 combined).

### Preprocessing Functions

**File:** `ambidrop/preprocess.py` — FT-JNF and Conv-TasNet AmbiDrop training

| Function | Input | Output shape | Used by |
|----------|-------|-------------|---------|
| `preprocess_mic(ex_dir, ref_id, train)` | Type B raw dir | `(T, 257, 14)` noisy STFT + `(T_s,)` clean mic | FT-JNF baseline |
| `preprocess_sh_stft(ex_dir, anm_source, train)` | Type A (`"ideal"`) or C (`"asm"`) | `(T, 257, 18)` noisy SH STFT + `(T_s,)` clean a₀₀ | FT-JNF AmbiDrop |
| `preprocess_sh_time(ex_dir, train)` | Type A raw dir | `(9, T_s)` real ACN + `(T_s,)` direct a₀₀ | Conv-TasNet AmbiDrop training |

All functions extract a 6 s (train) or 7.5 s (test) window anchored to speech onset, normalise by peak amplitude, and save as `.pt` files.

**File:** `ConvTasNet/preprocess.py` — Conv-TasNet time-domain preprocessing

| Function | Input | Output (`.pt` dict) | Used by |
|----------|-------|---------------------|---------|
| `preprocess_mic_time(ex_dir, ref_id, train)` | Type B raw dir | `noisy (M,T)`, `clean (M,T)`, `ref_id`, `format='time'` | Conv-TasNet baseline |
| `preprocess_ambisonics_time(ex_dir, V, th, ph, ...)` | Type C raw dir + steering matrix | `noisy_mic (M,T)`, `clean_mic (M,T)`, `anmt (9,T)`, `clean_anm (T,)`, `format='ambidrop_test'` | Conv-TasNet AmbiDrop test |

`preprocess_ambisonics_time` encodes ASM on-the-fly from raw `p.wav` (no pre-stored `anmt_array` needed) and corrects group delay via `estimate_delay` / `align_to_lag` from `datagenerator/helpers.py`. `SimDS_preprocessed` in `ConvTasNet/datasets.py` dispatches on the `format` key to return the correct tuple shape.

Two batch helpers exist:
- `preprocess_dataset(raw_dir, out_dir, fn, ...)` — processes one array directory; use this for **Type C test data** so each array stays in its own output directory and per-array metrics can be computed separately.
- `preprocess_dataset_multi(array_dirs, out_dir, fn, ...)` — merges multiple array directories into one combined output; use this for **Type A / B training and validation** data where the model should see all arrays in a single shuffled dataset.

### Reference Microphone
The baseline mode and the AmbiDrop test evaluation both need a **reference microphone index** — the mic physically closest to the target speaker (at azimuth 0 after scene rotation).

- For the 25 pre-registered arrays: `REF_IDX_MAP` in `ambidrop/constants.py` stores the 1-based index by array name.
- For user-defined arrays: `find_ref_mic(mics_az)` in `ambidrop/signal_utils.py` returns the 0-based index of the mic closest to azimuth 0.

---

## 3. Checkpoints

### FT-JNF
**Location:** `checkpoints/FT_JNF/`
**Registry:** `FT_JNF/constants.py` — `CHECKPOINT_REGISTRY` maps each filename to its full architecture config (mode, input_dim, hidden1, hidden2, dropout type/prob).

| Group | Files | Notes |
|-------|-------|-------|
| **Preferred baseline** | `FT_JNF,2026-03-25_13-37-42.pt` | 64/64, no dropout |
| **Preferred AmbiDrop** | `SH_FT_JNF,2025-12-01_10-08-18.pt` | 64/64, SHDrop p=0.4 max=3 |
| Other AmbiDrop (SHDrop) | `SH_FT_JNF,2025-12-*.pt`, `SH_FT_JNF,2026-03-09_*.pt` | Various p/max combinations |
| PerChDropout ablation | `SH_FT_JNF,2025-12-04_15-45-32.pt` through `SH_FT_JNF,2025-12-06_22-04-29.pt` | 10 thresholds, −10 dB → 0 dB |
| Network size ablation | `checkpoint_size_3490.pt` → `checkpoint_size_1223170.pt` | 10 sizes, 3.5K–1.2M params |
| Other baseline | `FT_JNF,2025-11-30_*.pt`, `FT_JNF,2025-12-01_*.pt`, `FT_JNF,2025-12-29_*.pt` | Earlier training runs |
| Named experiments | `AmbiDrop_full_experiment.pt`, `AmbiDrop_full_experiment_smallnet.pt`, `AmbiDrop_experiment2.pt`, `baseline_experiment2.pt` | — |

### Conv-TasNet
**Location:** `checkpoints/ConvTasNet/`

| Directory | Purpose |
|-----------|---------|
| `run_2026-04-09_10-55/final.pth.tar` | Baseline |
| `run_2026-04-09_08-35/final.pth.tar` | AmbiDrop (SHDrop p=0.4 max=3) |
| `run_2026-04-07_15-27/` | Earlier run |

### Loading Checkpoints
**File:** `ambidrop/checkpoint.py`

```python
from ambidrop.checkpoint import load_checkpoint, save_checkpoint

# Load latest epoch (or a specific one):
epoch, loss, lr = load_checkpoint(path, target_epoch=None, net=net)

# Save (called by training loop when validation loss improves):
save_checkpoint(path, epoch, net, optimizer, loss)
```

---

## 4. ASM Computation

**Ambisonics Signal Matching** encodes microphone signals into the Ambisonics domain at inference time.

### Formula

For each SH index nm and frequency bin f:

```
c_nm[f] = tikhonov(V[:, f, :].conj(), Y[nm, :])
```

where:
- `V` is the array steering matrix `(M, F, Q)` — transfer function from Q plane-wave directions to M microphones
- `Y` is the SH basis matrix `((N+1)², Q)` evaluated at the same Q directions
- `tikhonov(A, b)` solves the regularised least squares `(A^H A + λ²I) x = A^H b` via an augmented system

The result `c_nm` has shape `((N+1)², F, M)`. For Na=2 that is `9 × 257 × 7`.

To get time-domain Ambisonics: IFFT each filter row to length `filt_samp` (default 512), then for each SH channel sum the convolutions of the M mic signals with their M filter taps.

### Public API
**File:** `ambidrop/asm.py`

```python
from ambidrop.asm import encode_ambisonics, compute_asm_coefficients, apply_asm_filters

# Compute coefficients and encode in one call:
encoded_anm, cnm = encode_ambisonics(mic_signals, V, sh_order=2, th=th, ph=ph)

# Reuse precomputed coefficients (skips coefficient computation):
encoded_anm, cnm = encode_ambisonics(mic_signals, V, cnm=precomputed_cnm)

# Conv-TasNet (real-valued output):
encoded_anm, cnm = encode_ambisonics(mic_signals, V, sh_type="real", th=th, ph=ph)

# Separate steps:
cnm = compute_asm_coefficients(V, sh_order=2, th=th, ph=ph)
encoded_anm = apply_asm_filters(mic_signals, cnm)
```

### Where ASM Runs

| Location | When |
|----------|------|
| `datagenerator/generate_inference_ds.py` | Building Type C dataset (stores `anmt_array` in `anm.mat`) |
| `FT_JNF/test_real.py` | Real-world Aria inference |
| `ConvTasNet/preprocess.py` (`preprocess_ambisonics_time`) | Conv-TasNet AmbiDrop test preprocessing (on-the-fly from `p.wav`) |

### Channel Dropout (Training Only)
**File:** `ambidrop/dropouts.py`

`SHChannelDropout(drop_prob=0.4, max_drop=3)`: during training, zeros up to 3 randomly selected channels (never channel 0, a₀₀) with probability 0.4. This simulates the near-zero amplitude of poorly estimated ASM channels, making the model robust to encoding errors at inference.

`PerChDropout`: variant where each channel has its own drop probability, derived from empirical per-channel error rates across the array dataset (used in dropout ablation experiments).

---

## File Map

```
AmbiDrop/
├── run_FT_JNF.py         — end-to-end wrapper: generate / preprocess / train / test (FT-JNF)
├── run_ConvTasNet.py     — end-to-end wrapper: same four phases for Conv-TasNet
├── run_Real_World.py     — wrapper: evaluate FT-JNF on real Aria glasses recordings
│
├── ambidrop/             — core library
│   ├── __init__.py       — public re-exports
│   ├── constants.py      — STFT params, get_device(), REF_IDX_MAP, get_ref_idx()
│   ├── checkpoint.py     — save_checkpoint / load_checkpoint
│   ├── preprocess.py     — preprocess_mic / preprocess_sh_stft / preprocess_sh_time
│   ├── asm.py            — encode_ambisonics (unified ASM API + tikhonov solver)
│   ├── dropouts.py       — SHChannelDropout, PerChDropout
│   ├── losses.py         — si_snr, complex_si_snr
│   └── signal_utils.py   — STFT helpers, complex_acn_to_real_acn, find_ref_mic
│
├── FT_JNF/               — FT-JNF architecture and experiments
│   ├── model.py          — FT_JNF_model (2× BiLSTM + linear)
│   ├── train.py          — training_step_baseline / training_step_ambidrop
│   ├── datasets.py       — SimDS_preprocessed
│   ├── test_simulated.py — evaluate_array (SI-SDR / PESQ / STOI)
│   ├── test_real.py      — evaluate on real Aria recordings
│   ├── constants.py      — CHECKPOINT_REGISTRY, SAMPLE_RATE
│   └── ablations/        — paper figure scripts
│       ├── main_results.py       — reproduce Table I (simulated)
│       ├── main_results_real.py  — reproduce Table II (real-world)
│       ├── dropout_ablation.py   — dropout strategy study (Fig. 6)
│       ├── net_complexity.py     — network size ablation (Fig. 8)
│       ├── mic_failure.py        — mic failure robustness (Fig. 7)
│       └── snr_distribution.py  — SNR distribution analysis (Fig. 4)
│
├── ConvTasNet/           — IC Conv-TasNet architecture and experiments
│   ├── model.py          — ConvTasNet (encoder → TCN → decoder)
│   ├── modules.py        — cLN, DepthConv1d, DepthConv2d_Attention
│   ├── solver.py         — Solver class (training loop)
│   ├── train.py          — training entry point
│   ├── datasets.py       — SimDS_preprocessed (format-sentinel dispatch)
│   ├── preprocess.py     — preprocess_mic_time / preprocess_ambisonics_time
│   ├── evaluate.py       — test evaluation helpers
│   ├── loss.py           — SI-SNR + PIT loss
│   └── main_results.py   — reproduce Table III results
│
├── datagenerator/        — synthetic data generation
│   ├── helpers.py        — build_array(), geometry utilities
│   ├── generate_ambidrop_train_ds.py  — Type A: ideal Ambisonics (SH order 2)
│   ├── generate_baseline_train_ds.py  — Type B: 7-mic array signals
│   ├── generate_inference_ds.py       — Type C: mic signals + ASM-encoded Ambisonics
│   └── paper_arrays.py               — PAPER_ARRAYS_TRAIN / _TEST / _ALL (21 arrays)
│
├── checkpoints/
│   ├── FT_JNF/           — 40 .pt checkpoint files (see CHECKPOINT_REGISTRY)
│   └── ConvTasNet/       — 3 run directories with final.pth.tar
│
├── CODEBASE_OVERVIEW.md  — this file
├── USAGE.md              — full CLI and API reference
└── environment.yml       — Conda environment
```
