"""
Minimal test: shroom's ASM pipeline on a single source.
Based on shroom/examples/binaural_using_asm.py.

Room → Ambisonics (reverberant + direct-only) → ArrayDecoder (mics, whole
acoustic scene) → ASM encode_amb → compare with original.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import soundfile as sf
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
SPEECH_PATH = "/Users/mikitatarjitzky/Documents/speech enhancement - ACL/wsj0/si_et_05/440/440c0201.wav"
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

# 4. Room + source
sig, sr = sf.read(SPEECH_PATH, dtype="float64")
if sr != FS:
    from scipy.signal import resample
    sig = resample(sig, int(len(sig) * FS / sr))

ROOM_DIMS = [4.5, 7.0, 3.0]
ABSORPTION = 0.7
SOURCE_POS = [3.0, 4.0, 1.5]
RECEIVER_POS = [2.0, 2.0, 1.5]

room = Room(
    dimensions=ROOM_DIMS,
    absorption=ABSORPTION,
    max_ism_order=10,
    sh_order=SH_ORDER_SIM,
    fs=FS,
)
room.add_source(SOURCE_POS, signal=(sig, FS))
room.set_receiver(RECEIVER_POS)

# Direct-only Room: max_ism_order=0 -> image source model returns only the
# direct path (no reflections), giving the ideal direct-sound Ambisonics.
room_direct = Room(
    dimensions=ROOM_DIMS,
    absorption=ABSORPTION,
    max_ism_order=0,
    sh_order=SH_ORDER_SIM,
    fs=FS,
)
room_direct.add_source(SOURCE_POS, signal=(sig, FS))
room_direct.set_receiver(RECEIVER_POS)

# 5. Compute Ambisonics (reverberant = full scene, and direct-only)
amb = room.compute_amb()
print(f"Ambisonics (reverberant): {amb.data.shape}, sh_order={amb.sh_order}")

amb_direct = room_direct.compute_amb()
print(f"Ambisonics (direct only): {amb_direct.data.shape}, sh_order={amb_direct.sh_order}")

# 6. ArrayDecoder → mic signals
mic_out = array_decoder.process(amb)
print(f"Mic signals: {mic_out.data.shape}, dtype={mic_out.data.dtype}")

# 7. ASM encode_amb — pass full signal, no .real
mic_for_asm = mic_out.data[:, 0, :].T  # (T, M)
encoded = asm.encode_amb(mic_for_asm)
print(f"ASM encoded: {encoded.data.shape}, dtype={encoded.data.dtype}")

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

    # SI-SDR (ambisonics domain): ASM-encoded vs. ideal direct-only Ambisonics.
    # encoded picks up extra group delay from the ArrayDecoder FIR and the ASM
    # time-domain filters (cnm) that amb_direct doesn't have, so we align first.
    lag = estimate_delay(amb_direct.data[0, 0, :], encoded.data[0, 0, :])
    print(f"Estimated ASM-vs-direct delay: {lag} samples ({lag / FS * 1000:.2f} ms)")
    est_aligned, ref_aligned = align_to_lag(encoded.data[0, :, :], lag, amb_direct.data[0, :9, :])

    si_sdr = complex_si_snr(torch.from_numpy(est_aligned), torch.from_numpy(ref_aligned)).numpy()

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

    # SI-SDR (microphone domain): noisy mic signals vs. direct-only mic signals.
    mic_out_direct = array_decoder.process(amb_direct)
    mic_lag = estimate_delay(mic_out_direct.data[:, 0, :][0], mic_out.data[:, 0, :][0])
    print(f"Estimated mic noisy-vs-direct delay: {mic_lag} samples ({mic_lag / FS * 1000:.2f} ms)")
    mic_noisy_aligned, mic_direct_aligned = align_to_lag(
        mic_out.data[:, 0, :], mic_lag, mic_out_direct.data[:, 0, :]
    )

    si_sdr_mic = complex_si_snr(
        torch.from_numpy(mic_noisy_aligned), torch.from_numpy(mic_direct_aligned)
    ).numpy()

    print(f"\n=== SI-SDR, microphone (noisy vs. direct-only, delay-aligned) ===")
    for ch in range(7):
        print(f"  mic{ch}: SI-SDR = {si_sdr_mic[ch]:.2f} dB")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(np.arange(7), si_sdr_mic)
    ax.set_xlabel("Microphone channel")
    ax.set_ylabel("SI-SDR (dB)")
    ax.set_title("SI-SDR: noisy mic vs. direct-only mic (delay-aligned)")
    ax.set_xticks(np.arange(7))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()
