# AmbiDrop: Array-Agnostic Speech Enhancement via Ambisonics Training

AmbiDrop decouples a speech enhancement DNN from microphone array geometry by training exclusively on ideal Ambisonics signals with simulated channel dropout (*AmbiDrop*), then using **Array Signal Matching (ASM)** at inference to encode any real microphone array into the Ambisonics domain. A single trained model generalises to unseen arrays — no retraining or fine-tuning required.

Two DNN architectures are provided:
- **FT-JNF** — bidirectional LSTM in frequency and time, applies a complex IRM mask to the a₀₀ channel (STFT domain)
- **IC Conv-TasNet** — dilated TCN with Conv1d encoder/decoder operating directly in the time domain on 9-channel real ACN signals

---

## Quick Start

```bash
# Test FT-JNF on pre-existing evaluation data (both AmbiDrop and baseline modes)
python run_ftjnf.py --mode both --actions test

# Test IC Conv-TasNet
python run_convtasnet.py --mode ambidrop --actions test

# Evaluate AmbiDrop FT-JNF on real Aria glasses recordings
python run_real_world.py
```

Expected results (see [RESULTS.md](RESULTS.md)):

| Model | Array | SI-SDRi |
|---|---|---|
| FT-JNF AmbiDrop | Seen (rigid sphere r=0.1) | +8.79 dB |
| FT-JNF AmbiDrop | Unseen (Aria, simulated ATF) | +7.10 dB |
| FT-JNF Baseline | Unseen (random sphere) | −19.55 dB |

---

## Setup

```bash
conda env create -f environment.yml
conda activate venv
```

Edit the `# === USER CONFIG ===` block at the top of the wrapper script you want to run:

```python
# run_ftjnf.py
WSJ0_ROOT = "/path/to/wsj0"          # WSJ0 corpus root
DATA_ROOT = "datasets/run_ftjnf"     # generated data goes here
CKPT_DIR  = "checkpoints/FT_JNF"    # newly trained models saved here
```

The `shroom` library (array simulation) must be installed separately:

```bash
pip install -e /path/to/mashroom
```

---

## Wrapper Scripts

| Script | What it does | Key flags |
|---|---|---|
| `run_ftjnf.py` | End-to-end FT-JNF: generate → preprocess → train → test | `--mode {baseline,ambidrop,both}` `--actions` |
| `run_convtasnet.py` | End-to-end IC Conv-TasNet: same pipeline | `--mode {baseline,ambidrop,both}` `--actions` |
| `run_real_world.py` | Evaluate AmbiDrop FT-JNF on real Aria recordings | `--atf {simulated,measured}` `--checkpoint` |

### Common `--actions` combinations

```bash
# Full pipeline from scratch (generate + preprocess + train + test)
python run_ftjnf.py --mode ambidrop --actions generate preprocess train test

# Skip generation, use existing raw data
python run_ftjnf.py --mode ambidrop --actions preprocess train test

# Evaluate only (uses pre-existing EVAL_TRAIN_DIR / EVAL_TEST_DIR)
python run_ftjnf.py --mode both --actions test

# Generate fresh data and evaluate with a specific checkpoint
python run_ftjnf.py --mode ambidrop --actions generate preprocess test \
    --checkpoint checkpoints/FT_JNF/SH_FT_JNF,2025-12-01_10-08-18.pt

# Real-world evaluation with measured ATF
python run_real_world.py --atf measured --atf-path datasets/aria_ds/aria_atfs_fixed.sofa
```

---

## Results

Full benchmark tables are in [RESULTS.md](RESULTS.md).

**Table I — Simulated test set (5 unseen array geometries):**

| Model | SI-SDRi (dB) |
|---|---|
| FT-JNF AmbiDrop (h=64,64) | +8.79 |
| FT-JNF Baseline (h=64,64) | −19.55 (random sphere4) |
| IC Conv-TasNet AmbiDrop | see RESULTS.md |

**Table II — Real-world (Aria glasses, 5 examples):**

| Model | ATF | SI-SDRi (dB) |
|---|---|---|
| FT-JNF AmbiDrop | Simulated | +7.10 |
| FT-JNF AmbiDrop | Measured | see RESULTS.md |

---

## Project Structure

```
AmbiDrop/
├── ambidrop/                        # Core package (ASM, preprocessing, losses, dropouts)
│   ├── asm.py                       # ASM encoding: tikhonov solver + filter application
│   ├── checkpoint.py                # Checkpoint save/load (list-of-dicts format)
│   ├── constants.py                 # STFT params, REF_IDX_MAP, get_device()
│   ├── dropouts.py                  # SHChannelDropout (training regularisation)
│   ├── losses.py                    # SI-SNR loss
│   ├── preprocess.py                # STFT/time-domain preprocessing pipeline
│   └── signal_utils.py              # Shared tensor/numpy utilities
│
├── FT_JNF/                          # FT-JNF model
│   ├── model.py                     # FT_JNF_model (2× BiLSTM + mask)
│   ├── train.py                     # training_step_ambidrop / training_step_baseline
│   ├── datasets.py                  # SimDS_preprocessed (multi-format loader)
│   ├── test_simulated.py            # evaluate_array() — simulated test evaluation
│   ├── test_real.py                 # Real-world Aria evaluation helpers
│   ├── constants.py                 # CHECKPOINT_REGISTRY, SAMPLE_RATE
│   └── ablations/                   # Ablation study scripts (dropout, net size, SNR)
│
├── ConvTasNet/                      # IC Conv-TasNet model
│   ├── model.py                     # ConvTasNet (encoder → TCN → decoder)
│   ├── modules.py                   # cLN, DepthConv1d, DepthConv2d_Attention
│   ├── solver.py                    # Training loop
│   ├── train.py                     # Training entry point
│   ├── datasets.py                  # Dataset classes (precomputed + on-the-fly ASM)
│   ├── evaluate.py                  # Evaluation helpers
│   ├── loss.py                      # SI-SNR loss
│   └── main_results.py              # Reproduce paper Table II results
│
├── datagenerator/                   # Dataset generation
│   ├── helpers.py                   # build_array() + geometry utilities
│   ├── generate_ambidrop_train_ds.py  # Type A: ideal Ambisonics (SH order 2)
│   ├── generate_baseline_train_ds.py  # Type B: 7-mic array signals
│   └── generate_inference_ds.py     # Type C: mic + ASM-encoded Ambisonics
│
├── run_ftjnf.py                     # End-to-end FT-JNF wrapper (USER CONFIG here)
├── run_convtasnet.py                # End-to-end Conv-TasNet wrapper
├── run_real_world.py                # Real Aria evaluation wrapper
├── smoke_test.py                    # Quick import / forward-pass sanity check
├── environment.yml                  # Conda environment
├── RESULTS.md                       # Benchmark results table
└── docs/                            # Internal documentation
    ├── CODEBASE_OVERVIEW.md         # Architecture & data format reference
    ├── DEAD_CODE_AUDIT.md           # Dead-code analysis (Phase 2)
    └── ...
```

---

## Advanced: DNN Internals & Ablations

- Architecture details (layer sizes, STFT params, ASM formula): [`docs/CODEBASE_OVERVIEW.md`](docs/CODEBASE_OVERVIEW.md)
- Ablation scripts (dropout type, network size, SNR distribution): `FT_JNF/ablations/`
- Conv-TasNet paper-table reproduction: `ConvTasNet/main_results.py`

---

## Citation

```bibtex
@inproceedings{ambidrop2026,
  title   = {AmbiDrop: Array-Agnostic Speech Enhancement via Ambisonics Training with Channel Dropout},
  author  = {},
  year    = {2026},
}
```
