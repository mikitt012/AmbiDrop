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
# Maps array name to 1-based reference mic index (closest mic to target speaker).
# Keys are stored WITHOUT the "_preprocessed" suffix so the map works for both
# bare array names and directory names that carry the suffix.
# Use get_ref_idx(name) for safe lookups — it strips the suffix automatically.
# Subtract 1 for 0-based indexing.

REF_IDX_MAP = {
    "front hemisphere1 (rigid) radius = 0.1": 1,
    "full circle (rigid) radius = 0.1": 1,
    "planar": 6,
    "random 2D array1 radius = 0.1": 6,
    "random sphere1 radius = 0.1": 7,
    "random sphere3 (rigid) radius = 0.1": 4,
    "random sphere5 (rigid) radius = 0.05": 2,
    "semi circle planar radius = 0.05": 6,
    "ULA along X-axis": 7,
    "uniform sphere (rigid) radius = 0.1": 2,
    "front hemisphere2 (rigid) radius = 0.1": 1,
    "planar (rot=45deg)": 5,
    "random 2D array2 radius = 0.1": 2,
    "random sphere2 radius = 0.1": 2,
    "random sphere4 (rigid) radius = 0.1": 7,
    "random sphere6 (rigid) radius = 0.05": 4,
    "semi circle (rigid) radius = 0.1": 4,
    "ULA along Z-axis": 4,
    "uniform sphere (rigid) radius = 0.05": 2,
    "semi circle planar radius = 0.1": 6,
    "Aria on rigid sphere (simulated)": 3,
    "ULA along Y-axis (tilt=30deg)": 4,
    "ULA along x-axis (rot=30deg)": 7,
    "ULA along y-axis": 4,
    "ULA along X-axis (tilt=20)": 7,
}


def get_ref_idx(name: str, default: int = 1) -> int:
    """Return 1-based reference mic index for an array name.

    Tolerates the '_preprocessed' suffix so the same call works for both raw
    array names and on-disk directory names.
    """
    clean = name.removesuffix("_preprocessed")
    return REF_IDX_MAP.get(clean, REF_IDX_MAP.get(name, default))

