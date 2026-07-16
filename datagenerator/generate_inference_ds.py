"""
Generate Type C (inference) data: full dataset with mic signals, ideal Ambisonics, and ASM-encoded Ambisonics.

Each example saves p.wav, pDirect.wav, and anm.mat (anmt + anmt_array + anmtDirect)
for every configured array. The same N_EXAMPLES scenes are reused across all arrays
so models can be compared on identical acoustic conditions.

Public interface:
    generate_dataset — generate N examples per array into output_root/<array_name>/ex_1/ …
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
from math import factorial, pi, sqrt
from scipy.special import lpmv
try:
    from scipy.special import sph_harm
except ImportError:
    from scipy.special import sph_harm_y
    def sph_harm(m, n, phi, theta):
        return sph_harm_y(n, m, theta, phi)
from tqdm import tqdm
from shroom.acoustics.room import Room
from shroom.acoustics.processors import ArrayDecoder
from shroom.geometry.sampling import sphereicalGrid
from shroom.utils.grid_utils import from_fibonacci_grid
from shroom.utils.rotation_utils import wigner_d_matrix
from shroom.encoders.asm import ASM

from datagenerator.helpers import (
    estimate_delay, align_to_lag, build_array, add_sensor_noise,
    RigidSphereArrayConfig, FreeFieldArrayConfig, PrecomputedArrayConfig,
)
from ambidrop.asm import compute_asm_coefficients, apply_asm_filters

# ============================================================
# Technical parameters
# ============================================================

FS = 16000
SH_ORDER_SIM = 20
SH_ORDER_ASM = 2 # resulting in (2+1)^2 = 9 ACN channels for the ASM-encoded ambisonics output)
SPEECH_DIR = "/Users/mikitatarjitzky/Documents/speech enhancement - ACL/wsj0/si_et_05"

N_INTERFERERS = 5
N_SOURCES = N_INTERFERERS + 1  # target + interferers
SOURCE_GRID_POINTS = 480 # For fibonacci grid, 480 points is a good balance between coverage and speed. For Lebedev grids, use 2702 points.
ARRAY_DURATION_SEC = 0.032  # steering-matrix bandwidth; only used to build rigid_sphere/free_field arrays
regularization = "tikhonov"  # "tikhonov" | "svd"

OUTPUT_ROOT = "datasets/inference"
N_EXAMPLES = 1
SEED = 0
SENSOR_SNR_DB = 30


# ============================================================
# Arrays
# ============================================================

@dataclass
class ArraySpec:
    """One microphone array to generate the dataset for."""
    name: str
    array_type: str  # "rigid_sphere" | "free_field" | "precomputed"
    rigid_sphere: RigidSphereArrayConfig = None
    free_field: FreeFieldArrayConfig = None
    precomputed: PrecomputedArrayConfig = None


N_MICS = 7

# rigid_sphere: mics on the surface of a rigid scattering sphere.
RIGID_SPHERE_MICS_GRID = sphereicalGrid(
    az=np.linspace(0, 2 * np.pi, N_MICS, endpoint=False),
    co=np.full(N_MICS, np.pi / 2),
)

# free_field: mics in free space, arbitrary (x, y, z) positions in meters
# relative to the array center — not constrained to a shared radius/sphere
# like rigid_sphere is.
FREE_FIELD_MIC_POSITIONS = np.array([
    [0.10, 0.00, 0.00],
    [0.00, 0.12, 0.05],
    [-0.08, 0.00, 0.10],
    [0.00, -0.15, -0.02],
    [0.07, 0.07, 0.07],
    [-0.05, -0.09, 0.04],
    [0.12, -0.03, -0.06],
])

# The arrays to generate the dataset for. Add/remove ArraySpec entries to
# control which arrays are built; each gets the same N_EXAMPLES scenes.
ARRAYS = [
    ArraySpec(
        name="rigid_sphere_r0.1_7mic",
        array_type="rigid_sphere",
        rigid_sphere=RigidSphereArrayConfig(mics_grid=RIGID_SPHERE_MICS_GRID, mic_radius=0.1),
    ),
    ArraySpec(
        name="free_field_custom_7mic",
        array_type="free_field",
        free_field=FreeFieldArrayConfig(mic_positions=FREE_FIELD_MIC_POSITIONS),
    ),
    ArraySpec(
        name="precomputed_full_circle_r0.1",
        array_type="precomputed",
        precomputed=PrecomputedArrayConfig(
            array_path="utils/steering/full circle (rigid) radius = 0.1.mat",
            grid_path="utils/Lebvedev2702.mat",
        ),
    ),
]


# ============================================================
# Functions
# ============================================================

@dataclass
class Scene:
    """One randomized scene: room + array position + target/interferer placement."""
    T60: float       # reverberation time (s)
    L: np.ndarray     # room dimensions (3,) [x, y, z] in meters
    Xm: np.ndarray     # mic/array center position (3,), >=1m from walls, height 1.5m
    phs: float          # target source azimuth (rad), uniform over [0, 2*pi)
    rs: float            # target source distance from array center (m)
    Xs: np.ndarray         # target source position (3,)
    phi: np.ndarray         # interferer azimuths (N_INTERFERERS,)
    ri: np.ndarray           # interferer distances from array center (N_INTERFERERS,)
    Xi: np.ndarray            # interferer positions (N_INTERFERERS, 3)


def randomize_scene(rng):
    """
    Randomize one scene: room dimensions/T60, array position, target source,
    and N_INTERFERERS interferers spread over angular sectors around the
    target — mirrors experiment_data_gen_3D.m lines 383-400.
    """
    T60 = 0.2 + 0.3 * rng.random()  # reverberation time
    L = np.array([
        2.5 + 2.5 * rng.random(),
        3.0 + 6.0 * rng.random(),
        2.2 + 1.3 * rng.random(),
    ])  # room dimensions

    Xm = np.array([
        1 + (L[0] - 2) * rng.random(),
        1 + (L[1] - 2) * rng.random(),
        1.5,
    ])  # mic position, at least 1m from the walls

    phs = 2 * np.pi * rng.random()  # target source azimuth, uniform over the full circle
    rs = 0.3 + 0.7 * rng.random()  # target source radius
    Xs = Xm + np.array([rs * np.cos(phs), rs * np.sin(phs), 0.0])  # target source position

    # N_INTERFERERS azimuths: one per sector spanning 20-340 deg around phs
    ph_segments = phs + np.linspace(np.deg2rad(20), np.deg2rad(340), N_INTERFERERS + 1)[:N_INTERFERERS]
    phi = ph_segments + np.deg2rad(320 / N_INTERFERERS) * rng.random(N_INTERFERERS)  # ph interference

    while True:
        ri = 1 + 7 * rng.random(N_INTERFERERS)  # r interference
        Xi = Xm + np.column_stack([
            ri * np.cos(phi),
            ri * np.sin(phi),
            0.1 + np.sqrt(0.08) * rng.standard_normal(N_INTERFERERS),
        ])
        if np.all(Xi >= 0) and np.all(Xi <= L):
            break

    return Scene(T60=T60, L=L, Xm=Xm, phs=phs, rs=rs, Xs=Xs, phi=phi, ri=ri, Xi=Xi)


def load_speech_signals(speech_dir, n_signals, rng):
    """
    Pick n_signals random speakers (one random utterance each) from
    speech_dir, resample to FS, and pad + circularly shift to a common
    length. Returns a list of 1D signals, signals[0] is the target.
    """
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


def build_array_processors(array_spec):
    """Build (asm, array_decoder) for one array geometry."""
    # source_grid = from_fibonacci_grid(SOURCE_GRID_POINTS)
    grid_mat = scipy.io.loadmat("utils/Lebvedev2702.mat")
    source_grid = sphereicalGrid(az=grid_mat["ph"].ravel(), co=grid_mat["th"].ravel())
    array_freq = build_array(
        array_spec.array_type, source_grid, FS, ARRAY_DURATION_SEC,
        rigid_sphere=array_spec.rigid_sphere, free_field=array_spec.free_field,
        precomputed=array_spec.precomputed,
    )
    array_duration = array_freq.data.shape[-1] / FS

    _F_full = array_freq.data.shape[-1]
    _F_pos = _F_full // 2 + 1
    V_asm = np.asarray(array_freq.data[:, :, :_F_pos]).transpose(0, 2, 1)  # (M, F_pos, Q)
    th_asm = source_grid.co   # colatitude
    ph_asm = source_grid.az   # azimuth
    cnm_asm = compute_asm_coefficients(V_asm, SH_ORDER_ASM, th_asm, ph_asm, method=regularization)

    array_time_sh = array_freq.copy()
    array_time_sh.toTime()
    array_time_sh.data = np.fft.fftshift(array_time_sh.data, axes=-1)  # centers the array IR
    array_time_sh.toSH(SH_ORDER_SIM)
    array_decoder = ArrayDecoder(array_time_sh, sh_order=SH_ORDER_SIM)

    return cnm_asm, array_decoder


def rotate_sh_z(signal, theta):
    """
    In-place azimuth rotation of an SH-domain SpatialSignal by theta (rad),
    equivalent to signal.rotate_sh_domain(Rotation.from_euler('z', theta)).
    Applies the Wigner-D matrix via a direct matmul instead of shroom's
    generic einsum, which is ~100x faster for long signals like amb_scene
    (the noisy mix, hundreds of thousands of samples).
    """
    D = wigner_d_matrix(signal.sh_order, theta, 0.0, 0.0)
    signal.data = D @ signal.data


def generate_example(cnm, array_decoder, scene, speeches, save_dir):
    """Simulate one scene through one array and save p.wav/pDirect.wav/anm.mat."""
    target_sig, interferer_sigs = speeches[0], speeches[1:]

    room_dims = scene.L.tolist()
    absorption, _ = pra.inverse_sabine(scene.T60, room_dims)
    receiver_pos = scene.Xm.tolist()
    target_pos = scene.Xs.tolist()
    interferer_pos = scene.Xi.tolist()

    # Full scene: target + interferers, normal reflections.
    room_scene = Room(
        dimensions=room_dims, absorption=absorption,
        max_ism_order=10, sh_order=SH_ORDER_SIM, fs=FS,
    )
    room_scene.add_source(target_pos, signal=(target_sig, FS))
    for pos, sig_i in zip(interferer_pos, interferer_sigs):
        room_scene.add_source(pos, signal=(sig_i, FS))
    room_scene.set_receiver(receiver_pos)

    # Direct-only Room, target speaker only: max_ism_order=0 -> image source
    # method returns only the direct path (no reflections, no interferers).
    room_direct_target = Room(
        dimensions=room_dims, absorption=absorption,
        max_ism_order=0, sh_order=SH_ORDER_SIM, fs=FS,
    )
    room_direct_target.add_source(target_pos, signal=(target_sig, FS))
    room_direct_target.set_receiver(receiver_pos)

    # Ideal ambisonics: full scene (truncated to 2nd order, 9 ACN channels)
    # and direct-only ambisonics of the target speaker alone.
    amb_scene = room_scene.compute_amb()
    amb_direct_target = room_direct_target.compute_amb()

    # The target sits at global azimuth scene.phs (by construction of
    # randomize_scene). Rotate both signals by -phs so the target lands at
    # azimuth 0 — equivalent to physically aiming the array at the target,
    # mirroring experiment_data_gen_3D.m's Hnmt_rot/phs convention. This
    # rotates the array (mic signals, via array_decoder below) and the scene
    # (anmt/anmtDirect ground truth) consistently, since both are derived
    # from these same rotated signals.
    rotate_sh_z(amb_scene, -scene.phs)
    rotate_sh_z(amb_direct_target, -scene.phs)

    amb_scene_o2 = amb_scene.data[0, :9, :]

    # Microphone signals: noisy (whole scene) and direct-only target.
    mic_out_scene = array_decoder.process(amb_scene)
    mic_out_direct_target = array_decoder.process(amb_direct_target)

    # Add sensor noise to the noisy mic signals (30 dB SNR), matching
    # generate_single_example.py. mic_scene_data is used for both alignment
    # and ASM encoding so the saved p.wav and anmt_array share one noise realisation.
    mic_scene_data = add_sensor_noise(mic_out_scene.data[:, 0, :], snr_db=SENSOR_SNR_DB)

    mic_lag = estimate_delay(mic_out_direct_target.data[:, 0, :][0], mic_scene_data[0])
    mic_noisy_aligned, mic_direct_aligned = align_to_lag(
        mic_scene_data, mic_lag, mic_out_direct_target.data[:, 0, :]
    )

    # ASM-encoded ambisonics of the scene.
    mic_for_asm = mic_scene_data.T  # (T, M)
    encoded_scene = apply_asm_filters(mic_scene_data, cnm, filt_samp=512)

    # Align encoded_scene with the Room-domain ground truth before saving.
    # encoded_scene picks up extra group delay from the ArrayDecoder FIR and
    # the ASM time-domain filters (cnm); amb_scene_o2/amb_direct_target come
    # straight from Room.compute_amb() with no extra filtering, so they
    # share one timeline. Without this, anmt_array (model input) and
    # anmtDirect (training target) end up offset by ~128 samples.
    lag = estimate_delay(amb_direct_target.data[0, 0, :], encoded_scene[0, :])
    encoded_scene_aligned, amb_scene_o2_aligned, amb_direct_aligned = align_to_lag(
        encoded_scene, lag, amb_scene_o2, amb_direct_target.data[0, :9, :]
    )

    # Trim everything (mic pair + ambisonics triplet) to one shared length,
    # so the saved p.wav/pDirect.wav/anm.mat all cover the same time window.
    save_t = min(
        mic_noisy_aligned.shape[1], mic_direct_aligned.shape[1],
        encoded_scene_aligned.shape[1], amb_scene_o2_aligned.shape[1], amb_direct_aligned.shape[1],
    )
    mic_noisy_aligned = mic_noisy_aligned[:, :save_t]
    mic_direct_aligned = mic_direct_aligned[:, :save_t]
    encoded_scene_aligned = encoded_scene_aligned[:, :save_t]
    amb_scene_o2_aligned = amb_scene_o2_aligned[:, :save_t]
    amb_direct_aligned = amb_direct_aligned[:, :save_t]

    # Save, matching the existing ex_* folder layout: p.wav (noisy mic),
    # pDirect.wav (direct-only target mic), anm.mat with
    # anmt/anmt_array/anmtDirect.
    os.makedirs(save_dir, exist_ok=True)
    mic_scale = np.max(np.abs(mic_noisy_aligned)) + 1e-8  # global max over all channels
    mic_noisy_aligned  = mic_noisy_aligned  / mic_scale
    mic_direct_aligned = mic_direct_aligned / mic_scale
    sf.write(os.path.join(save_dir, "p.wav"), mic_noisy_aligned.T.real, FS)
    sf.write(os.path.join(save_dir, "pDirect.wav"), mic_direct_aligned.T.real, FS)
    scipy.io.savemat(os.path.join(save_dir, "anm.mat"), {
        "anmt": amb_scene_o2_aligned.T,
        "anmt_array": encoded_scene_aligned.T,
        "anmtDirect": amb_direct_aligned[:1, :].T,
    })


def generate_dataset(arrays, n_examples, seed, output_root, speech_dir=SPEECH_DIR):
    for array_spec in arrays:
        print(f"\n{'=' * 60}\nArray: {array_spec.name}\n{'=' * 60}")
        cnm, array_decoder = build_array_processors(array_spec)
        save_root = os.path.join(output_root, array_spec.name)
        os.makedirs(save_root, exist_ok=True)
        print(f"Saving to {save_root}")
        print(f"Speech corpus: {speech_dir}")

        for ex in tqdm(range(1, n_examples + 1), desc="Examples"):
            # Seed depends only on the example index, not the array, so every
            # array sees the exact same N_EXAMPLES scenes/speech.
            ex_rng = np.random.default_rng(seed * 10000 + ex)
            scene = randomize_scene(ex_rng)
            speeches = load_speech_signals(speech_dir, N_SOURCES, ex_rng)
            generate_example(cnm, array_decoder, scene, speeches, os.path.join(save_root, f"ex_{ex}"))

    print("\nDone!")


def main():
    p = argparse.ArgumentParser(description="Generate AmbiDrop inference dataset")
    p.add_argument("--n-examples", type=int, default=N_EXAMPLES)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--output-dir", default=OUTPUT_ROOT)
    p.add_argument("--speech-dir", default=SPEECH_DIR,
                   help="WSJ0 split to draw speech from (default: si_et_05 for test)")
    args = p.parse_args()
    generate_dataset(ARRAYS, args.n_examples, args.seed, args.output_dir, args.speech_dir)


if __name__ == "__main__":
    main()
