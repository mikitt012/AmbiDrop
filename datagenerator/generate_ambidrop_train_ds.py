"""
Generate Type A (AmbiDrop training) data: array-agnostic ideal 2nd-order Ambisonics.

Simulates multi-talker reverberant scenes via ISM, truncates to SH order 2 (9 ACN
channels), and saves anm.mat (anmt + anmtDirect) with no microphone arrays or ASM.

Public interface:
    generate_dataset — generate N examples into output_root/ex_1/ … ex_N/

Output folder:
  <output_dir>/ex_1/anm.mat
  <output_dir>/ex_2/anm.mat
  ...
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import glob
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import scipy.io
import pyroomacoustics as pra
from tqdm import tqdm
from shroom.acoustics.room import Room
from shroom.utils.rotation_utils import wigner_d_matrix

from datagenerator.helpers import estimate_delay, align_to_lag

# ============================================================
# Parameters
# ============================================================

FS             = 16000
SH_ORDER_SIM   = 20
SH_ORDER_OUT   = 2       # saved Ambisonics order → (2+1)^2 = 9 ACN channels
N_INTERFERERS  = 5
N_SOURCES      = N_INTERFERERS + 1

SPEECH_DIR      = "/Users/mikitatarjitzky/Documents/speech enhancement - ACL/wsj0/si_tr_s"
VAL_SPEECH_DIR  = "/Users/mikitatarjitzky/Documents/speech enhancement - ACL/wsj0/si_dt_05"
OUTPUT_ROOT     = "datasets/ambidrop_train_ds"
N_EXAMPLES      = 6000
N_VAL_EXAMPLES  = 1000
SEED            = 0


# ============================================================
# Scene helpers (mirrors generate_inference_ds.py)
# ============================================================

@dataclass
class Scene:
    T60: float
    L:   np.ndarray
    Xm:  np.ndarray
    phs: float
    rs:  float
    Xs:  np.ndarray
    phi: np.ndarray
    ri:  np.ndarray
    Xi:  np.ndarray


def randomize_scene(rng):
    T60 = 0.2 + 0.3 * rng.random()
    L = np.array([
        2.5 + 2.5 * rng.random(),
        3.0 + 6.0 * rng.random(),
        2.2 + 1.3 * rng.random(),
    ])
    Xm = np.array([
        1 + (L[0] - 2) * rng.random(),
        1 + (L[1] - 2) * rng.random(),
        1.5,
    ])
    phs = 2 * np.pi * rng.random()
    rs  = 0.3 + 0.7 * rng.random()
    Xs  = Xm + np.array([rs * np.cos(phs), rs * np.sin(phs), 0.0])

    ph_segments = phs + np.linspace(np.deg2rad(20), np.deg2rad(340), N_INTERFERERS + 1)[:N_INTERFERERS]
    phi = ph_segments + np.deg2rad(320 / N_INTERFERERS) * rng.random(N_INTERFERERS)

    while True:
        ri = 1 + 7 * rng.random(N_INTERFERERS)
        Xi = Xm + np.column_stack([
            ri * np.cos(phi),
            ri * np.sin(phi),
            0.1 + np.sqrt(0.08) * rng.standard_normal(N_INTERFERERS),
        ])
        if np.all(Xi >= 0) and np.all(Xi <= L):
            break

    return Scene(T60=T60, L=L, Xm=Xm, phs=phs, rs=rs, Xs=Xs, phi=phi, ri=ri, Xi=Xi)


def load_speech_signals(speech_dir, n_signals, rng):
    speaker_dirs = sorted(
        d for d in glob.glob(os.path.join(speech_dir, "*")) if os.path.isdir(d))
    if len(speaker_dirs) < n_signals:
        raise ValueError(f"Need {n_signals} speakers, found {len(speaker_dirs)}")
    chosen = rng.choice(speaker_dirs, size=n_signals, replace=False)

    signals = []
    for d in chosen:
        wavs = sorted(glob.glob(os.path.join(d, "*.wav")))
        if not wavs:
            raise ValueError(f"No .wav in {d}")
        sig, sr = sf.read(wavs[rng.integers(len(wavs))], dtype="float64")
        if sr != FS:
            from scipy.signal import resample
            sig = resample(sig, int(len(sig) * FS / sr))
        if sig.ndim > 1:
            sig = sig[:, 0]
        signals.append(sig)

    max_len = max(len(s) for s in signals)
    padded = []
    for s in signals:
        n_pad = max_len - len(s)
        if n_pad > 0:
            s = np.pad(s, (0, n_pad))
            s = np.roll(s, rng.integers(n_pad))
        padded.append(s)
    return padded


def rotate_sh_z(signal, theta):
    D = wigner_d_matrix(signal.sh_order, theta, 0.0, 0.0)
    signal.data = D @ signal.data


# ============================================================
# Core generation
# ============================================================

def generate_example(scene, speeches, save_dir):
    """Simulate one scene in SH domain and save anm.mat (no array, no ASM)."""
    target_sig, interferer_sigs = speeches[0], speeches[1:]

    room_dims  = scene.L.tolist()
    absorption, _ = pra.inverse_sabine(scene.T60, room_dims)
    receiver_pos   = scene.Xm.tolist()
    target_pos     = scene.Xs.tolist()
    interferer_pos = scene.Xi.tolist()

    # Full scene: target + 5 interferers
    room_scene = Room(
        dimensions=room_dims, absorption=absorption,
        max_ism_order=10, sh_order=SH_ORDER_SIM, fs=FS,
    )
    room_scene.add_source(target_pos, signal=(target_sig, FS))
    for pos, sig_i in zip(interferer_pos, interferer_sigs):
        room_scene.add_source(pos, signal=(sig_i, FS))
    room_scene.set_receiver(receiver_pos)

    # Direct-only: target speaker, no reflections
    room_direct = Room(
        dimensions=room_dims, absorption=absorption,
        max_ism_order=0, sh_order=SH_ORDER_SIM, fs=FS,
    )
    room_direct.add_source(target_pos, signal=(target_sig, FS))
    room_direct.set_receiver(receiver_pos)

    amb_scene  = room_scene.compute_amb()
    amb_direct = room_direct.compute_amb()

    # Rotate both so target lands at azimuth 0 (matches generate_inference_ds.py)
    rotate_sh_z(amb_scene,  -scene.phs)
    rotate_sh_z(amb_direct, -scene.phs)

    # Truncate to 2nd-order output (9 ACN channels)
    anmt_full   = amb_scene.data[0, :(SH_ORDER_OUT + 1)**2, :]   # (9, T)
    anmt_direct = amb_direct.data[0, :1, :]                       # (1, T) — a00 only

    # Align direct vs. scene on a00 channel
    lag = estimate_delay(anmt_direct[0], anmt_full[0])
    anmt_aligned, anmt_direct_aligned = align_to_lag(anmt_full, lag, anmt_direct)[:2]

    save_t = min(anmt_aligned.shape[1], anmt_direct_aligned.shape[1])
    anmt_aligned        = anmt_aligned[:, :save_t]
    anmt_direct_aligned = anmt_direct_aligned[:, :save_t]

    os.makedirs(save_dir, exist_ok=True)
    scipy.io.savemat(os.path.join(save_dir, "anm.mat"), {
        "anmt":       anmt_aligned.T,         # saved as (T, 9)
        "anmtDirect": anmt_direct_aligned.T,  # saved as (T, 1)
    })


def generate_dataset(n_examples, seed, output_root, speech_dir=SPEECH_DIR):
    os.makedirs(output_root, exist_ok=True)
    print(f"Generating {n_examples} examples → {output_root}")
    print(f"  Speech corpus: {speech_dir}")

    for ex in tqdm(range(1, n_examples + 1), desc="Examples"):
        ex_rng = np.random.default_rng(seed * 10000 + ex)
        scene   = randomize_scene(ex_rng)
        speeches = load_speech_signals(speech_dir, N_SOURCES, ex_rng)
        generate_example(scene, speeches, os.path.join(output_root, f"ex_{ex}"))

    print("Done!")


def main():
    p = argparse.ArgumentParser(description="Generate AmbiDrop training dataset (Type A)")
    p.add_argument("--n-examples",     type=int, default=N_EXAMPLES)
    p.add_argument("--seed",           type=int, default=SEED)
    p.add_argument("--output-dir",     default=OUTPUT_ROOT)
    p.add_argument("--speech-dir",     default=SPEECH_DIR,
                   help="WSJ0 split for training data (default: si_tr_s)")
    # Validation split (optional — only generated when --n-val > 0)
    p.add_argument("--n-val",          type=int, default=N_VAL_EXAMPLES,
                   help="Number of validation examples (0 = skip val generation)")
    p.add_argument("--val-output-dir", default=None,
                   help="Output dir for val data (default: <output-dir>_val)")
    p.add_argument("--val-speech-dir", default=VAL_SPEECH_DIR,
                   help="WSJ0 split for val data (default: si_dt_05)")
    p.add_argument("--val-seed",       type=int, default=None,
                   help="RNG seed for val data (default: --seed + 1)")
    args = p.parse_args()

    generate_dataset(args.n_examples, args.seed, args.output_dir, args.speech_dir)

    if args.n_val > 0:
        val_output = args.val_output_dir or (args.output_dir + "_val")
        val_seed   = args.val_seed if args.val_seed is not None else args.seed + 1
        generate_dataset(args.n_val, val_seed, val_output, args.val_speech_dir)


if __name__ == "__main__":
    main()
