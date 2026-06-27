# AmbiDrop: Codebase Understanding

## Paper Summary

AmbiDrop is a framework for **array-agnostic multichannel speech enhancement**. The core idea is to decouple the DNN training from specific microphone array geometries by using **ideal Ambisonics signals** (spherical harmonic representations of the sound field) as the DNN input instead of raw microphone signals. During training, a **channel-wise dropout layer** simulates the encoding errors that occur when real microphone arrays are converted to the Ambisonics domain. During inference, microphone signals from any arbitrary array are first transformed into Ambisonics via **Ambisonics Signal Matching (ASM)**, then processed by the trained DNN.

The result: a single trained model works with any microphone array geometry, whereas baseline models trained on raw microphone signals fail when presented with unseen array configurations.

---

## High-Level Pipeline

### Training Phase
```
Ideal Ambisonics signals (9 channels, order Na=2)
        |
  Channel-wise Dropout (simulates ASM encoding errors)
        |
     DNN (FT-JNF or IC Conv-TasNet)
        |
   Enhanced Speech (compared against clean a00 reference)
```

### Inference Phase
```
Microphone signals (from arbitrary array)
        |
     ASM Encoding (microphones → Ambisonics domain)
        |
     DNN (same trained model, dropout disabled)
        |
   Enhanced Speech
```

---

## Directory Structure

```
AmbiDrop/
├── train_FT_JNF.py                  # Baseline FT-JNF training (microphone input)
├── train_SH_FT_JNF.py               # AmbiDrop FT-JNF training (Ambisonics input + dropout)
├── train_SH_FT_JNF_with_dropouts.py # Variant with different dropout configs
├── Test_FT_JNF.py                   # Baseline FT-JNF testing
├── test_SH_FT_JNF.py                # AmbiDrop FT-JNF testing
├── test_aria_glasses.py             # Real-world Aria glasses evaluation
├── test_aria_glasses_baseline.py    # Baseline evaluation on Aria data
├── test_mic_count.py                # Microphone failure resilience study
├── a.py, b.py                       # Misc/scratch scripts
│
├── ASM/                              # Ambisonics Signal Matching module
│   ├── asm.py                        # Core ASM class
│   ├── tikhonov.py                   # Tikhonov regularization solver
│   ├── bf_filers_base_object.py      # Base class for beamformer-based filters
│   ├── utils.py                      # Utility functions (convolution, etc.)
│   └── validate.py                   # Validation utilities
│
├── ConvTasNet/                       # IC Conv-TasNet (AmbiDrop variant)
│   └── src/
│       ├── conv_tasnet_ic.py         # IC Conv-TasNet with SH dropout
│       ├── conv_tasnet.py            # Base Conv-TasNet architecture
│       ├── solver.py                 # Training loop / solver
│       ├── data.py                   # Data loading
│       └── ...
│
├── ConvTasNet_baseline/              # IC Conv-TasNet (baseline, microphone input)
│   └── src/ (same structure)
│
├── utils/
│   ├── data_gen.py                   # Simulated data generation (mic signals only)
│   ├── data_gen_wSH.py              # Data generation with SH/ASM computation
│   ├── SH_data_process.py           # Preprocessing: .mat → .pt (Ambisonics data)
│   ├── mic_data_process.py          # Preprocessing: .mat → .pt (microphone data)
│   ├── merge_folders.py             # Merge preprocessed folders
│   └── clean_change.py              # Data cleanup utility
│
├── net_size_comparison/              # Network complexity ablation study
│   ├── SH_net_sizes_training.py     # Train AmbiDrop with varying BLSTM sizes
│   ├── SH_net_sizes_testing.py      # Test each network size
│   └── results.py                   # Plot results (Fig. 8 in paper)
│
├── snr_distribution_fixed/          # SI-SDR distribution analysis
│   └── SH_train_distribution.py     # Generate histogram (Fig. 4 in paper)
│
├── checkpoints/                      # Saved model weights (.pt files)
└── datasets/                         # Data directory (on cluster)
```

---

## File-by-File Explanation

### Core Training Scripts

#### `train_FT_JNF.py` — Baseline FT-JNF Training
- **Purpose**: Trains the FT-JNF network on raw **microphone signals** from multiple array geometries. This is the geometry-dependent baseline.
- **Key components**:
  - `SimDS_preprocessed`: Dataset class loading preprocessed `.pt` files containing STFT-domain microphone signals and clean references. Each sample includes `noisy_tf` (T x F x 2C tensor, real+imag concatenated), `clean` (multichannel time-domain), `ref_id` (reference microphone index), `array_name`, and `ex_id`.
  - `FT_JNF`: The neural network — two bidirectional LSTM layers followed by a linear layer. Takes concatenated real/imaginary STFT as input (14 channels = 7 mics x 2), outputs a 2D complex Ideal Ratio Mask (cIRM).
  - `si_snr()`: Scale-Invariant Signal-to-Noise Ratio loss function.
  - `training_step()`: Forward pass — applies learned mask to reference channel STFT, reconstructs time-domain signal via iSTFT, computes negative SI-SNR loss.
  - `input_dim=14`: 7 microphone channels x 2 (real + imaginary parts).
  - Training uses **10,000 examples** across 10 training array geometries (1,000 per array).
  - **No dropout** is applied — the network sees raw microphone channels.
  - `add_white_noise()`: Adds sensor noise at 30 dB SNR to simulate realistic conditions.

#### `train_SH_FT_JNF.py` — AmbiDrop FT-JNF Training
- **Purpose**: Trains the FT-JNF network on **ideal Ambisonics signals** with channel-wise dropout. This is the AmbiDrop model.
- **Key differences from baseline**:
  - `input_dim=18`: 9 Ambisonics channels (order Na=2) x 2 (real + imaginary) = 18 input features.
  - `SHChannelDropout`: The core AmbiDrop innovation — randomly zeros out entire Ambisonics channels during training (both real and imaginary parts together). Channel a00 (index 0) is **never dropped** since it serves as the reference. Default: `drop_prob=0.4`, `max_drop=3`.
  - `PerChDropout`: Alternative dropout where each channel has its own dropout probability derived from ASM encoding error analysis (Table IV in the paper).
  - Multiple experimental dropout classes are defined but commented out:
    - `LearnableFreqDropout`: Learnable per-frequency dropout probabilities
    - `SmoothLPFFreqDropout`: Sigmoid-based frequency-dependent dropout
    - `MixedSHFreqDropout`: Combined channel + frequency dropout
    - `ProgressiveDeterministicFreqDropout`: Epoch-dependent progressive dropout
    - `MixedSHLearnableFreqDropout`: Learnable frequency + channel dropout
  - Training uses **6,000 examples** of ideal Ambisonics (not tied to any specific array).
  - The reference signal is the omnidirectional channel a00.
  - The `training_step()` uses channel 0 and channel 9 (real and imaginary parts of a00) to build the reference STFT Y.

### Core Testing Scripts

#### `Test_FT_JNF.py` — Baseline Testing
- **Purpose**: Evaluates the baseline FT-JNF model on various test array geometries.
- **Key behavior**:
  - Loads preprocessed test data for each array type (e.g., "ULA along X-axis", "random sphere", "Aria glasses", etc.).
  - For each test array, a `ref_idx` (reference microphone closest to target) is manually specified via a lookup table.
  - Computes SI-SDR, PESQ, and STOI metrics for both noisy input and enhanced output.
  - Applies the mask M to the reference channel: `S_hat = Ms * Y`, then reconstructs via iSTFT.
  - Results are logged to Weights & Biases (wandb).

#### `test_SH_FT_JNF.py` — AmbiDrop Testing
- **Purpose**: Evaluates the AmbiDrop FT-JNF model across simulated arrays and various dropout configurations.
- **Key behavior**:
  - Iterates over 18 different dropout configurations (indexed by `t`), each with its own trained checkpoint. Configurations 0-3 use uniform `SHChannelDropout`, 4-17 use `PerChDropout` with varying thresholds (corresponding to Tables III and IV in the paper).
  - Tests on both "training arrays" (seen by baseline, unseen by AmbiDrop) and "test arrays" (unseen by both).
  - Uses `zero_random_channels()` to optionally simulate microphone failures during testing (set to `n=0` by default).
  - Computes per-example noisy SI-SDR using the **microphone-domain** reference channel (for fair comparison with baseline), and enhanced SI-SDR using the **Ambisonics-domain** enhanced output.
  - Stores results in a `master_si_sdr` matrix (arrays x examples) for distribution analysis.
  - Can generate SI-SDR distribution histograms (Fig. 4 in paper) when `plot_snr_dist=True`.

#### `test_aria_glasses.py` — Real-World Aria Glasses Evaluation
- **Purpose**: Tests AmbiDrop on real-world recordings from Project Aria glasses (Section VI of paper).
- **Key behavior**:
  - Loads measured audio from Aria glasses mounted on a KEMAR manikin.
  - Performs ASM encoding using either a **simulated ATF** (rigid sphere model) or **measured ATF** (from CHiME-8 challenge).
  - Implements the full ASM pipeline: spherical harmonics computation → steering matrix construction → Tikhonov-regularized encoding.
  - Evaluates both correctly positioned and mispositioned glasses.
  - Uses temporal grid search to align enhanced output with clean reference (addresses variable recording latency).

#### `test_aria_glasses_baseline.py` — Baseline on Aria Data
- **Purpose**: Tests the baseline (microphone-input) FT-JNF on Aria recordings for comparison.

#### `test_mic_count.py` — Microphone Failure Resilience
- **Purpose**: Tests AmbiDrop's robustness when microphones are randomly deactivated (Section VII-B of paper).
- **Key behavior**:
  - For each test, randomly deactivates N microphones from the 7-channel input.
  - The ASM encoding stage is informed of which channels are missing (adjusts steering matrix).
  - Evaluates SI-SDR as a function of available channels (Fig. 7 in paper).

### Data Generation

#### `utils/data_gen.py` — Microphone Signal Generation
- **Purpose**: Generates simulated acoustic scenes using the **image method** (pyroomacoustics).
- **Key behavior**:
  - Creates a shoebox room with random dimensions and reverberation time T60.
  - Places a 4-microphone array at a random position and orientation.
  - Positions 1 target speaker (front, 0.3-1m) and 5 interferers at various angles.
  - Simulates three signal types:
    - `p`: Full mixture (target + interference + reverberation)
    - `pTarget`: Target only (with reverberation)
    - `pDirect`: Target direct path only (no reflections, max_order=0)
  - Speech signals are drawn from the WSJ0 dataset at 16 kHz.
  - Saves output as `.mat` files.

#### `utils/data_gen_wSH.py` — Data Generation with Spherical Harmonics
- **Purpose**: Extends `data_gen.py` to support multiple array geometries and includes ASM computation infrastructure.
- **Key additions**:
  - `sh2()`: Computes complex spherical harmonics up to order N for given angles.
  - `array_ambisonics_time_domain()`: Computes time-domain Ambisonics signals from microphone signals using ASM (frequency-domain → time-domain conversion via IFFT and convolution).
  - `generate_microphone_arrays()`: Creates 12 different array geometries — uniform circles, semi-circles, ULAs, X-shape, plus-shape, and random arrays.
  - `shift_closest_to_phi0()`: Reorders microphones so the one closest to phi=0 (positive x-axis, facing the target) is first.
  - Generates training and test splits from different array subsets.

### Data Preprocessing

#### `utils/SH_data_process.py` — Ambisonics Data Preprocessing
- **Purpose**: Converts raw `.mat` files (containing both microphone and Ambisonics signals) into preprocessed `.pt` (PyTorch tensor) files for efficient training.
- **Pipeline**:
  1. Load `.mat` file containing `anmt_array` (ASM-encoded Ambisonics) and `anmtDirect` (clean Ambisonics)
  2. Add white noise at 30 dB SNR
  3. Pad/truncate to fixed length (120,000 samples = 7.5s, or 96,000 = 6s for training)
  4. Compute STFT (512-point FFT, 256 hop, Hamming window)
  5. Split into real and imaginary parts, concatenate along channel dimension
  6. Normalize by max absolute value
  7. Save as `.pt` tuple: `(noisy_tf_mic, clean_time_mic, noisy_tf_anm, clean_time_anm)`

#### `utils/mic_data_process.py` — Microphone Data Preprocessing
- **Purpose**: Same as above but for microphone-domain data only (baseline models).
- **Key difference**: Loads `p.wav` and `pDirect.wav` files, stores `ref_id` (reference microphone index) alongside the data.
- **Output format**: `.pt` dict with keys `noisy`, `clean`, `ref_id`, `array_name`, `ex_id`.

### ASM Module

#### `ASM/asm.py` — Ambisonics Signal Matching
- **Purpose**: Implements the core ASM algorithm (Eq. 4-7 in paper).
- **`asm` class**:
  - `_calculate_coefficients()`: Computes ASM filter coefficients c_nm for each frequency bin using Tikhonov regularization. For each SH channel (n,m), solves: `c_nm = (V V^H + λI)^{-1} V y_nm`
  - `calc_ambisonics()`: Applies the computed filters to microphone signals to produce Ambisonics output via convolution.
- **`asmMSE` class**: Computes the normalized MSE between ideal and ASM-estimated Ambisonics (Eq. 5-6 in paper).
- **`asmBinMse` class**: Computes binaural MSE (for HRTF-based evaluation).
- **`asmMagnitude` class**: Computes magnitude response of ASM encoding.

#### `ASM/tikhonov.py` — Tikhonov Regularization
- **Purpose**: Solves the regularized least-squares problem for ASM encoding (Eq. 7 in paper).
- **Algorithm**: Constructs augmented system `[A; λL]x = [b; 0]` and solves via `np.linalg.lstsq`.
- **Auto-lambda**: If no regularization parameter is specified, uses 1% of the maximum singular value of A as a heuristic.

### Network Complexity Study

#### `net_size_comparison/SH_net_sizes_training.py`
- **Purpose**: Trains AmbiDrop FT-JNF with varying BLSTM hidden dimensions to study the parameter count vs. performance tradeoff (Section VII-C, Table V).
- **Configurations**: (256,128), (128,128), (128,64), (64,64), (64,32), (32,32), (32,16), (16,16), (16,8), (8,8) — ranging from 1.2M to 3.5K parameters.

#### `net_size_comparison/SH_net_sizes_testing.py`
- **Purpose**: Tests each trained network size variant and computes metrics.

#### `net_size_comparison/results.py`
- **Purpose**: Generates the SI-SDRi vs. parameter count plot (Fig. 8 in paper).

### ConvTasNet Variants

#### `ConvTasNet/src/conv_tasnet_ic.py` — IC Conv-TasNet + AmbiDrop
- **Purpose**: Modified Inter-Channel Conv-TasNet that accepts **real-valued Ambisonics** input with channel-wise dropout.
- **Key differences**: Operates in time domain (not STFT), uses encoder-separator-decoder architecture with TCN blocks. The `SHChannelDropout` operates on `(B, C, T)` tensors (different dimension ordering than FT-JNF).

#### `ConvTasNet_baseline/src/conv_tasnet_ic.py` — Baseline IC Conv-TasNet
- **Purpose**: Standard IC Conv-TasNet operating on raw microphone signals.

---

## Key Functions

### Loss Function: `si_snr()`
Present in all training/testing files. Implements **Scale-Invariant Signal-to-Noise Ratio (SI-SNR)** [Eq. 25 in ref]:
1. Zero-mean normalize estimate and reference
2. Compute projection: `s_target = (<s_hat, s> / ||s||^2) * s`
3. Compute noise: `e_noise = s_hat - s_target`
4. SI-SNR = `10 * log10(||s_target||^2 / ||e_noise||^2)`

### Dropout: `SHChannelDropout`
- Input shape: `(B, T, F, C)` where C = 2 * SH_channels (real + imaginary)
- For each batch element:
  1. Sample Bernoulli mask for channels 1..SH_C-1 (skip a00)
  2. Cap number of dropped channels at `max_drop`
  3. Zero both real and imaginary parts of selected channels
- **Never drops channel 0** (a00 — omnidirectional, used as reference)

### Network: `FT_JNF`
- **Architecture**: LSTM1 (across frequency) → LSTM2 (across time) → Linear → complex mask
- **Forward pass**:
  1. Apply channel dropout (AmbiDrop only)
  2. LSTM1: Process each time frame independently across frequency bins (B*T, F, C)
  3. LSTM2: Process each frequency bin independently across time frames (B*F, T, H)
  4. Linear: Map to 2D output (real + imaginary parts of complex mask)
- **Mask application**: `S_hat = M * Y_ref` where Y_ref is the reference channel STFT
- **Reconstruction**: `s_hat = iSTFT(S_hat)`

### Data Loading: `SimDS_preprocessed`
- Loads pre-computed `.pt` files
- Returns STFT tensors ready for network input
- Two variants: one for microphone data (returns ref_id for per-array reference channel selection), one for Ambisonics data (always uses channel 0)

---

## STFT Configuration
- **FFT size**: 512 (32 ms at 16 kHz)
- **Hop size**: 256 (50% overlap)
- **Window**: Hamming
- **Frequency bins**: 257 (one-sided)
- **Input format**: Real and imaginary parts concatenated along channel dimension

## Training Configuration
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-6)
- **Loss**: Negative SI-SNR
- **Batch size**: 8
- **Epochs**: 250-300
- **Checkpoint**: Saves best model (lowest validation loss)
- **Signal length**: 6 seconds (96,000 samples at 16 kHz) for training, 7.5 seconds (120,000) for testing
- **Logging**: TensorBoard + Weights & Biases

## Array Configurations
- **Training arrays** (10 total): 1D/2D free-field + rigid sphere arrays
- **Test arrays** (10 total): Different unseen geometries
- All arrays: 7 microphones, constrained within 0.1m radius sphere
- **Ambisonics order**: Na=2 → 9 channels ((2+1)^2 = 9)

---

## Execution Environment
- **Cluster**: BGU HPC with GPU (NVIDIA) via RunAI scheduler
- **Framework**: PyTorch with CUDA
- **Commands** (at bottom of files): `runai-bgu submit python -n <name> -c <cpus> -m <mem> -g <gpus> --conda venv -- "python <script>.py"`
