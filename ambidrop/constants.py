"""
Project-wide constants and device helper for AmbiDrop.

Public interface:
    N_FFT — STFT window size (512)
    HOP_LENGTH — STFT hop length (256)
    WIN_LENGTH — STFT window length (512)
    SAMPLE_RATE — audio sample rate in Hz (16000)
    get_device — return the best available torch device (CUDA or CPU)
    REF_IDX_MAP — dict mapping array folder name to 1-based reference mic index
"""
import torch

# ── STFT defaults ────────────────────────────────────────────────────────────

N_FFT = 512
HOP_LENGTH = 256
WIN_LENGTH = 512
SAMPLE_RATE = 16000


# ── Device helper ────────────────────────────────────────────────────────────

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Reference microphone index map ───────────────────────────────────────────
# Maps array folder name (with "_preprocessed" suffix) to 1-based reference
# microphone index (the mic closest to the target speaker).
# Subtract 1 for 0-based indexing: ref_id = REF_IDX_MAP[name] - 1

REF_IDX_MAP = {
    "front hemisphere1 (rigid) radius = 0.1_preprocessed": 1,
    "full circle (rigid) radius = 0.1_preprocessed": 1,
    "planar_preprocessed": 6,
    "random 2D array1 radius = 0.1_preprocessed": 6,
    "random sphere1 radius = 0.1_preprocessed": 7,
    "random sphere3 (rigid) radius = 0.1_preprocessed": 4,
    "random sphere5 (rigid) radius = 0.05_preprocessed": 2,
    "semi circle planar radius = 0.05_preprocessed": 6,
    "ULA along X-axis_preprocessed": 7,
    "uniform sphere (rigid) radius = 0.1_preprocessed": 2,
    "front hemisphere2 (rigid) radius = 0.1_preprocessed": 1,
    "planar (rot=45deg)_preprocessed": 5,
    "random 2D array2 radius = 0.1_preprocessed": 2,
    "random sphere2 radius = 0.1_preprocessed": 2,
    "random sphere4 (rigid) radius = 0.1_preprocessed": 7,
    "random sphere6 (rigid) radius = 0.05_preprocessed": 4,
    "semi circle (rigid) radius = 0.1_preprocessed": 4,
    "ULA along Z-axis_preprocessed": 4,
    "uniform sphere (rigid) radius = 0.05_preprocessed": 2,
    "semi circle planar radius = 0.1_preprocessed": 6,
    "Aria on rigid sphere (simulated)_preprocessed": 3,
    "ULA along Y-axis (tilt=30deg)_preprocessed": 4,
    "ULA along x-axis (rot=30deg)_preprocessed": 7,
    "ULA along y-axis_preprocessed": 4,
    "ULA along X-axis (tilt=20)_preprocessed": 7,
}

