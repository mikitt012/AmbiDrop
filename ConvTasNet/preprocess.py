"""
Time-domain preprocessing for IC Conv-TasNet.

Two functions cover the two cases where preprocessing is needed:

preprocess_mic_time   — Baseline train/test: saves raw mic waveforms.
preprocess_ambisonics_time — AmbiDrop test: encodes real-ACN Ambisonics
                            from p.wav via ASM, saves alongside mic signals
                            needed for the noisy SI-SDR reference.

Both produce dicts with a 'format' sentinel that SimDS_preprocessed uses to
return the correct tuple shape.  All kwargs are compatible with
preprocess_dataset / preprocess_dataset_multi.
"""

import os

import numpy as np
import soundfile as sf
import torch

_TRAIN_LEN = 96_000     # 6 s at 16 kHz  (matches ambidrop/preprocess.py)
_TEST_LEN  = 120_000    # 7.5 s at 16 kHz


def preprocess_mic_time(ex_dir, ref_id: int, array_name: str = "", ex_id: str = "",
                        train: bool = True) -> dict:
    """
    Preprocess microphone-domain data for IC Conv-TasNet Baseline.

    Reads p.wav (noisy, sensor noise already present) and pDirect.wav (clean).
    Both files must exist in ex_dir (Type B or Type C folder).

    Unlike preprocess_mic, signals are kept in the time domain — no STFT is
    computed. Normalization is by peak waveform amplitude.

    Parameters
    ----------
    ex_dir     : path to an ex_N/ folder
    ref_id     : reference microphone index (0-based)
    array_name : stored in the output dict for logging
    ex_id      : stored in the output dict for logging; defaults to basename of ex_dir
    train      : True → onset-based 6 s window; False → full 7.5 s from start

    Returns
    -------
    dict with keys:
        noisy      : Tensor(M, T_samples) — noisy multichannel mic signals
        clean      : Tensor(M, T_samples) — clean direct-path mic signals
        ref_id     : int
        array_name : str
        ex_id      : str
        format     : 'time'  ← sentinel for SimDS_preprocessed
    """
    noisy_np, _ = sf.read(os.path.join(ex_dir, "p.wav"))
    clean_np, _ = sf.read(os.path.join(ex_dir, "pDirect.wav"))
    noisy_np = noisy_np.T.astype(np.float32)   # (M, T)
    clean_np = clean_np.T.astype(np.float32)   # (M, T)

    length = _TRAIN_LEN if train else _TEST_LEN
    if train:
        above = np.where(np.abs(clean_np[0]) > 1e-3)[0]
        onset = int(above[0]) if len(above) > 0 else 0
    else:
        onset = 0

    noisy_np = _slice(noisy_np, onset, length)
    clean_np = _slice(clean_np, onset, length)

    max_val = float(np.abs(noisy_np).max())
    if max_val == 0.0:
        max_val = 1.0

    return {
        "noisy":      torch.from_numpy(noisy_np / max_val),   # (M, T_samples)
        "clean":      torch.from_numpy(clean_np / max_val),   # (M, T_samples)
        "ref_id":     ref_id,
        "array_name": array_name,
        "ex_id":      ex_id or os.path.basename(str(ex_dir)),
        "format":     "time",
    }


def preprocess_ambisonics_time(ex_dir, V, th, ph, ref_id: int = 0,
                               array_name: str = "", ex_id: str = "",
                               train: bool = False) -> dict:
    """
    Preprocess test data for AmbiDrop: encode real-ACN Ambisonics from p.wav
    via a steering matrix (ASM) and save alongside mic signals.

    Unlike preprocess_sh_time (which uses ideal anmt from anm.mat), this
    function computes ambisonics from raw microphone signals — matching what
    a deployed system would do.

    Parameters
    ----------
    ex_dir     : path to an ex_N/ folder (Type C — must have p.wav, pDirect.wav, anm.mat)
    V          : steering matrix (M, F_pos, Q) from _build_steering_matrix
    th, ph     : co-elevation and azimuth grids for the source grid
    ref_id     : 0-based reference mic index (for noisy SI-SDR)
    array_name : stored in output dict for logging
    ex_id      : stored in output dict for logging
    train      : unused (kept for compatibility with preprocess_dataset signature)

    Returns
    -------
    dict with keys:
        noisy_mic  : Tensor(M, T) — raw mic signals (for noisy SI-SDR)
        clean_mic  : Tensor(M, T) — direct-path mic signals (for noisy SI-SDR)
        anmt       : Tensor(9, T) — real-ACN Ambisonics (model input)
        clean_anm  : Tensor(T,)   — a00 direct path (model target)
        ref_id     : int
        array_name : str
        ex_id      : str
        format     : 'ambidrop_test'
    """
    from scipy.io import loadmat
    from ambidrop.asm import encode_ambisonics

    noisy_np, _ = sf.read(os.path.join(ex_dir, "p.wav"))
    clean_np, _ = sf.read(os.path.join(ex_dir, "pDirect.wav"))
    noisy_np = noisy_np.T.astype(np.float32)   # (M, T)
    clean_np = clean_np.T.astype(np.float32)   # (M, T)

    # Truncate to the same window as preprocess_mic_time (train=False) so that
    # noisy SI-SDR is computed over the same samples in both baseline and AmbiDrop tests.
    noisy_np = _slice(noisy_np, 0, _TEST_LEN)
    clean_np = _slice(clean_np, 0, _TEST_LEN)

    anmt, _ = encode_ambisonics(noisy_np, V, sh_order=2, th=th, ph=ph, sh_type="real")

    mat = loadmat(os.path.join(ex_dir, "anm.mat"))
    anmt_direct_ref = mat["anmtDirect"][:, 0].real.astype(np.float32)  # (T,) a00

    # Correct group delay introduced by the array IR + ASM filters, matching
    # the alignment step in generate_inference_ds.py.
    from datagenerator.helpers import estimate_delay, align_to_lag
    lag = estimate_delay(anmt_direct_ref, anmt[0, :])
    anmt_aligned, clean_anm_aligned = align_to_lag(anmt, lag, anmt_direct_ref[np.newaxis, :])
    anmt_aligned      = anmt_aligned[:, :_TEST_LEN]
    clean_anm_aligned = clean_anm_aligned[0, :_TEST_LEN]

    return {
        "noisy_mic":  torch.from_numpy(noisy_np),
        "clean_mic":  torch.from_numpy(clean_np),
        "anmt":       torch.from_numpy(anmt_aligned.astype(np.float32)),
        "clean_anm":  torch.from_numpy(clean_anm_aligned),
        "ref_id":     ref_id,
        "array_name": array_name,
        "ex_id":      ex_id or os.path.basename(str(ex_dir)),
        "format":     "ambidrop_test",
    }


def _slice(arr: np.ndarray, start: int, length: int) -> np.ndarray:
    sliced = arr[:, start:start + length]
    if sliced.shape[1] < length:
        sliced = np.pad(sliced, ((0, 0), (0, length - sliced.shape[1])))
    return sliced[:, :length]
