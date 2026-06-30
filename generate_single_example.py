"""
Minimal test: shroom's ASM pipeline on a multi-talker scene (single example).
Based on shroom/examples/binaural_using_asm.py.

Scene: target speaker in front of the array + 5 interferers around the room.
Room → Ambisonics (full scene + target-direct-only) → ArrayDecoder (mics of
the whole scene) → ASM encode_amb → compare with original.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import soundfile as sf
import scipy.io
import pyroomacoustics as pra
from scipy.signal import correlate
from shroom.acoustics.room import Room
from shroom.acoustics.spherical_array import SphericalArray
from shroom.acoustics.processors import ArrayDecoder, ASMEncoder
from shroom.geometry.sampling import sphereicalGrid
from shroom.utils.grid_utils import from_fibonacci_grid
from shroom.encoders.asm import ASM

FS = 16000
SH_ORDER_SIM = 20
SH_ORDER_ASM = 2
SPEECH_DIR = "/Users/mikitatarjitzky/Documents/speech enhancement - ACL/wsj0/si_et_05"
TARGET_SPEECH = f"{SPEECH_DIR}/440/440c0201.wav"
INTERFERER_SPEECH = [f"{SPEECH_DIR}/{spk}/{spk}c0201.wav" for spk in (441, 442, 443, 444, 445)]
DEBUG = True  # print/plot NMSE (asm_mse_error) and SI-SDR diagnostics


def estimate_delay(ref_ch0, est_ch0):
    """Cross-correlation lag (samples) of est_ch0 relative to ref_ch0."""
    T0 = min(len(ref_ch0), len(est_ch0))
    xcorr = correlate(est_ch0[:T0], ref_ch0[:T0], mode="full")
    return int(np.argmax(np.abs(xcorr)) - (T0 - 1))


def align_to_lag(est, lag, *refs):
    """Shift `est` by `lag` samples to align with `refs`, then trim all to the common length."""
    if lag >= 0:
        est = est[:, lag:]
    else:
        refs = tuple(r[:, -lag:] for r in refs)
    T = min(est.shape[1], *(r.shape[1] for r in refs))
    return (est[:, :T],) + tuple(r[:, :T] for r in refs)


# 1. Array (full circle, 7 mics, rigid sphere r=0.1)
mics_grid = sphereicalGrid(
    az=np.linspace(0, 2 * np.pi, 7, endpoint=False),
    co=np.full(7, np.pi / 2),
)
source_grid = from_fibonacci_grid(480)

array_freq = SphericalArray(
    fs=FS, duration=0.008,
    r_sphere=0.1,
    r_mics=np.full(7, 0.1),
    source_grid=source_grid,
    mics_grid=mics_grid,
    sphere_type="rigid",
    sh_order_for_sm_calc=14,
    convert_to_time=False,
)

# 2. ASM encoder
asm = ASM(sh_order=SH_ORDER_ASM, array=array_freq, fs=FS, duration=0.008)

# 3. ArrayDecoder (needs time + SH domain copy)
array_time_sh = array_freq.copy()
array_time_sh.toTime()
array_time_sh.toSH(SH_ORDER_SIM)
array_decoder = ArrayDecoder(array_time_sh, sh_order=SH_ORDER_SIM)

# 4. Room + scene: target speaker in front of the array, 5 interferers around it.
# "Front" = the array's own azimuth=0 reference (mics_grid az starts at 0 too).
# Interferers are spread at 60 deg steps over the rest of the circle.
def load_speech(path):
    sig, sr = sf.read(path, dtype="float64")
    if sr != FS:
        from scipy.signal import resample
        sig = resample(sig, int(len(sig) * FS / sr))
    return sig

target_sig = load_speech(TARGET_SPEECH)
interferer_sigs = [load_speech(p) for p in INTERFERER_SPEECH]

ROOM_DIMS = [4.5, 7.0, 3.0]
T60_TARGET = 0.5  # seconds
ABSORPTION, _ = pra.inverse_sabine(T60_TARGET, ROOM_DIMS)  # max_ism_order kept as before (10)
print(f"T60={T60_TARGET}s -> absorption={ABSORPTION:.4f} (Sabine, approximate at max_ism_order=10)")
RECEIVER_POS = [2.25, 3.5, 1.5]  # room center
TARGET_RADIUS = 0.7
INTERFERER_RADIUS = 2.0
TARGET_AZ_DEG = 0
INTERFERER_AZ_DEG = [60, 120, 180, 240, 300]


def polar_to_room_pos(az_deg, radius, center=RECEIVER_POS):
    az = np.deg2rad(az_deg)
    return [center[0] + radius * np.cos(az), center[1] + radius * np.sin(az), center[2]]


TARGET_POS = polar_to_room_pos(TARGET_AZ_DEG, TARGET_RADIUS)
INTERFERER_POS = [polar_to_room_pos(az, INTERFERER_RADIUS) for az in INTERFERER_AZ_DEG]

# Full scene: target + 5 interferers, normal reflections.
room_scene = Room(
    dimensions=ROOM_DIMS,
    absorption=ABSORPTION,
    max_ism_order=10,
    sh_order=SH_ORDER_SIM,
    fs=FS,
)
room_scene.add_source(TARGET_POS, signal=(target_sig, FS))
for pos, sig_i in zip(INTERFERER_POS, interferer_sigs):
    room_scene.add_source(pos, signal=(sig_i, FS))
room_scene.set_receiver(RECEIVER_POS)

# Direct-only Room, target speaker only: max_ism_order=0 -> image source method
# returns only the direct path (no reflections, no interferers).
room_direct_target = Room(
    dimensions=ROOM_DIMS,
    absorption=ABSORPTION,
    max_ism_order=0,
    sh_order=SH_ORDER_SIM,
    fs=FS,
)
room_direct_target.add_source(TARGET_POS, signal=(target_sig, FS))
room_direct_target.set_receiver(RECEIVER_POS)

# 5. Compute Ambisonics: ideal ambisonics of the scene (truncated to 2nd order,
# 9 ACN channels) and direct-only ambisonics of the target speaker alone.
amb_scene = room_scene.compute_amb()
amb_scene_o2 = amb_scene.data[0, :9, :]
print(f"Ideal ambisonics, scene, 2nd order: {amb_scene_o2.shape}")

amb_direct_target = room_direct_target.compute_amb()
print(f"Ideal ambisonics, target direct only: {amb_direct_target.data.shape}, sh_order={amb_direct_target.sh_order}")

# 6. Microphone signals: noisy (whole scene) and direct-only target.
mic_out_scene = array_decoder.process(amb_scene)
print(f"Mic signals, scene: {mic_out_scene.data.shape}, dtype={mic_out_scene.data.dtype}")

mic_out_direct_target = array_decoder.process(amb_direct_target)

# Align the mic pair too (both come from array_decoder.process(), so the lag
# is expected to be ~0, but we compute it rather than assume it).
mic_lag = estimate_delay(mic_out_direct_target.data[:, 0, :][0], mic_out_scene.data[:, 0, :][0])
print(f"Estimated mic noisy-vs-direct delay: {mic_lag} samples ({mic_lag / FS * 1000:.2f} ms)")
mic_noisy_aligned, mic_direct_aligned = align_to_lag(
    mic_out_scene.data[:, 0, :], mic_lag, mic_out_direct_target.data[:, 0, :]
)

# 7. ASM-encoded ambisonics of the scene.
mic_for_asm = mic_out_scene.data[:, 0, :].T  # (T, M)
encoded_scene = asm.encode_amb(mic_for_asm)
print(f"ASM encoded, scene: {encoded_scene.data.shape}, dtype={encoded_scene.data.dtype}")

# Align encoded_scene with the Room-domain ground truth before saving anything.
# encoded_scene picks up extra group delay from the ArrayDecoder FIR and the
# ASM time-domain filters (cnm); amb_scene_o2/amb_direct_target come straight
# from Room.compute_amb() with no extra filtering, so they share one timeline.
# Without this, anmt_array (model input) and anmtDirect (training target) are
# offset by ~128 samples — which is exactly what wrecked the SI-SDR when this
# scene was fed through FT-JNF.
lag = estimate_delay(amb_direct_target.data[0, 0, :], encoded_scene.data[0, 0, :])
print(f"Estimated ASM-vs-direct delay: {lag} samples ({lag / FS * 1000:.2f} ms)")
encoded_scene_aligned, amb_scene_o2_aligned, amb_direct_aligned = align_to_lag(
    encoded_scene.data[0, :, :], lag, amb_scene_o2, amb_direct_target.data[0, :9, :]
)

# Trim everything (mic pair + ambisonics triplet) to one shared length, so the
# saved p.wav/pDirect.wav/anm.mat all cover exactly the same time window.
SAVE_T = min(
    mic_noisy_aligned.shape[1], mic_direct_aligned.shape[1],
    encoded_scene_aligned.shape[1], amb_scene_o2_aligned.shape[1], amb_direct_aligned.shape[1],
)
mic_noisy_aligned = mic_noisy_aligned[:, :SAVE_T]
mic_direct_aligned = mic_direct_aligned[:, :SAVE_T]
encoded_scene_aligned = encoded_scene_aligned[:, :SAVE_T]
amb_scene_o2_aligned = amb_scene_o2_aligned[:, :SAVE_T]
amb_direct_aligned = amb_direct_aligned[:, :SAVE_T]

# Save scene for FT-JNF preprocessing (utils/SH_data_process.py), matching the
# existing ex_* folder layout: p.wav (noisy mic), pDirect.wav (direct-only
# target mic), anm.mat with anmt/anmt_array/anmtDirect. The real part is the
# physical signal (shroom keeps a small complex residual from SH-domain
# processing — see shroom_dev/sound.py's `.real` convention for playback).

SAVE_DIR = f"datasets/experiment_full_anm/shroom_target0.7m_int2.0m_5int_T60_{T60_TARGET}s/ex_1"
os.makedirs(SAVE_DIR, exist_ok=True)

sf.write(os.path.join(SAVE_DIR, "p.wav"), mic_noisy_aligned.T.real, FS)
sf.write(os.path.join(SAVE_DIR, "pDirect.wav"), mic_direct_aligned.T.real, FS)
scipy.io.savemat(os.path.join(SAVE_DIR, "anm.mat"), {
    "anmt": amb_scene_o2_aligned.T,
    "anmt_array": encoded_scene_aligned.T,
    "anmtDirect": amb_direct_aligned[:1, :].T,
})
print(f"Saved scene to {SAVE_DIR} (all components trimmed to {SAVE_T} samples)")

if DEBUG:
    import matplotlib.pyplot as plt
    import torch
    from shroom_dev.errors import asm_mse_error
    from shroom_dev.plot import loglog_plot
    from ambidrop.losses import complex_si_snr

    # NMSE — ASM filter quality vs. true SH basis.
    # Same metric/plot as projects/asm_project/main.py in the shroom repo:
    # |cnm^H V - Y^*|^2 / |Y|^2 per SH channel vs. positive frequency.
    # Operates on the ASM filters (cnm) and steering matrix (array_freq)
    # directly — no room/speech signal needed.
    asm_duration = 0.008
    freqs_asm = np.fft.fftfreq(int(asm_duration * FS), 1 / FS)
    pos_freqs_asm = np.fft.rfftfreq(int(asm_duration * FS), 1 / FS)

    error_mse, var_mse = asm_mse_error(
        asm.cnm.data, array_freq.data, array_freq.grid.Y(SH_ORDER_ASM), freqs_asm,
        return_variance=True,
    )

    loglog_plot(
        freqs=pos_freqs_asm,
        title="ASM | Complex MSE per SH Channel",
        errors={f"ch{ch}": error_mse[ch] for ch in range(error_mse.shape[0])},
        variances={f"ch{ch}": var_mse[ch] for ch in range(var_mse.shape[0])},
        figsize=(9, 5),
        show=True,
    )

    # SI-SDR (ambisonics domain): reuses encoded_scene_aligned / amb_direct_aligned
    # from the delay alignment done above (before saving).
    si_sdr = complex_si_snr(
        torch.from_numpy(encoded_scene_aligned), torch.from_numpy(amb_direct_aligned)
    ).numpy()

    print(f"\n=== SI-SDR, ambisonics (ASM-encoded vs. ideal direct-only, delay-aligned) ===")
    for ch in range(9):
        print(f"  ch{ch}: SI-SDR = {si_sdr[ch]:.2f} dB")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(np.arange(9), si_sdr)
    ax.set_xlabel("Ambisonics channel")
    ax.set_ylabel("SI-SDR (dB)")
    ax.set_title("SI-SDR: ASM-encoded vs. ideal direct-only Ambisonics (delay-aligned)")
    ax.set_xticks(np.arange(9))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()

    # SI-SDR (microphone domain): reuses mic_noisy_aligned / mic_direct_aligned
    # from the delay alignment done above (before saving).
    si_sdr_mic = complex_si_snr(
        torch.from_numpy(mic_noisy_aligned), torch.from_numpy(mic_direct_aligned)
    ).numpy()

    print(f"\n=== SI-SDR, microphone (noisy scene vs. direct-only target, delay-aligned) ===")
    for ch in range(7):
        print(f"  mic{ch}: SI-SDR = {si_sdr_mic[ch]:.2f} dB")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(np.arange(7), si_sdr_mic)
    ax.set_xlabel("Microphone channel")
    ax.set_ylabel("SI-SDR (dB)")
    ax.set_title("SI-SDR: noisy scene mic vs. direct-only target mic (delay-aligned)")
    ax.set_xticks(np.arange(7))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()
