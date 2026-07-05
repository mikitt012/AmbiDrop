"""
Shared helpers for all datagenerator scripts: array construction and signal alignment.

Public interface:
    RigidSphereArrayConfig — config dataclass for mics on a rigid scattering sphere
    FreeFieldArrayConfig — config dataclass for mics at arbitrary 3D positions
    PrecomputedArrayConfig — config dataclass for a precomputed steering matrix loaded from disk
    fibonacci_sphere_points — evenly distribute n points over a sphere of radius r
    ula_mic_positions — n mics evenly spaced along the x-axis
    add_sensor_noise — add real-valued white noise at a given SNR to a mic signal
    estimate_delay — cross-correlation lag of one signal relative to another
    align_to_lag — shift a signal by lag samples and trim all signals to a common length
    build_array — build array steering matrix for rigid_sphere / free_field / precomputed type
    steering_vector — compute free-field plane-wave steering vectors for arbitrary mic positions
    load_precomputed_array — load a precomputed steering matrix from a .mat file
"""

from dataclasses import dataclass

import numpy as np
import scipy.io
from scipy.signal import correlate
from shroom.acoustics.spherical_array import SphericalArray
from shroom.acoustics.spatial_signal import SpatialSignal
from shroom.acoustics.physics import SPEED_OF_SOUND
from shroom.geometry.sampling import sphereicalGrid
from shroom.utils.dsp_utils import reconstruct_neg_frequency_spectrum

DEFAULT_PRECOMPUTED_GRID_PATH = "datasets/experiment_full_anm/utils/Lebvedev2702.mat"


@dataclass
class RigidSphereArrayConfig:
    """Mics on the surface of a rigid scattering sphere of radius `mic_radius`."""
    mics_grid: sphereicalGrid
    mic_radius: float


@dataclass
class FreeFieldArrayConfig:
    """Mics in free space at arbitrary (x, y, z) positions in meters."""
    mic_positions: np.ndarray


@dataclass
class PrecomputedArrayConfig:
    """A precomputed steering matrix + its source grid, loaded from disk."""
    array_path: str
    grid_path: str = DEFAULT_PRECOMPUTED_GRID_PATH


def fibonacci_sphere_points(n, r):
    """
    `n` points spread evenly over a sphere of radius `r` (Fibonacci/golden-angle
    spiral). Used for a well-conditioned free_field mic layout: an arbitrary
    clustered/asymmetric placement starves some SH orders of angular diversity,
    which a regularized ASM solve then can't recover regardless of array type.
    """
    golden_angle = np.pi * (3 - np.sqrt(5))
    i = np.arange(n)
    y = 1 - (i / (n - 1)) * 2
    radius_xz = np.sqrt(1 - y**2)
    theta = golden_angle * i
    return np.stack([np.cos(theta) * radius_xz * r, y * r, np.sin(theta) * radius_xz * r], axis=1)


def ula_mic_positions(n, r):
    """
    `n` mics evenly spaced along the x-axis, spanning from -r to +r — matches
    "ULA along X-axis" (array ID 1), one of the paper's actual free_field
    training arrays (see UNDERSTANDING_DATA_GEN.md). A 3D-spread layout like
    fibonacci_sphere_points is a geometry the model never saw, free-field or
    otherwise, even though it's better-conditioned in isolation.
    """
    x = np.linspace(-r, r, n)
    return np.stack([x, np.zeros(n), np.zeros(n)], axis=1)


def add_sensor_noise(mic_signal, snr_db=30.0):
    """
    Add real-valued sensor noise at `snr_db` SNR, matching
    experiment_data_gen_3D.m's `p = p + sqrt(mean(p^2)/SNR_lin) * randn(...)`.
    `mic_signal` is (M, T), possibly complex (shroom keeps a small imaginary
    residual from SH-domain processing) — noise is added to the real
    (physical) part only, sized from that part's own power.
    """
    snr_linear = 10 ** (snr_db / 10)
    sig_power = np.mean(mic_signal.real ** 2)
    noise_std = np.sqrt(sig_power / snr_linear)
    return mic_signal + noise_std * np.random.standard_normal(mic_signal.shape)


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


def build_rigid_sphere_array(mics_grid, source_grid, fs, duration, mic_radius, sh_order_for_sm_calc=14):
    """Mics on the surface of a rigid scattering sphere."""
    return SphericalArray(
        fs=fs, duration=duration,
        r_sphere=mic_radius, r_mics=np.full(mics_grid.n_points, mic_radius),
        source_grid=source_grid, mics_grid=mics_grid,
        sphere_type="rigid", sh_order_for_sm_calc=sh_order_for_sm_calc,
        convert_to_time=False,
    )


def steering_vector(mic_positions, theta_vec, phi_vec, f_vec, c=SPEED_OF_SOUND):
    """
    Regular free-field, far-field steering vectors via plane-wave time delays —
    exact for any mic geometry and frequency (no SH/Bessel truncation, unlike
    routing free-field through SphericalArray's modal expansion, which is only
    accurate for small ka and breaks down for mics far from the array center).

        tau[m,k] = dot(mic_positions[m], u[k]) / c
        v[m,k,f] = exp(+1j * 2*pi * f_vec[f] * tau[m,k])

    Sign verified empirically: for mics constrained to a constant radius (i.e.
    sitting on a sphere), this must match shroom's native
    SphericalArray(sphere_type="open", source_type="plane_wave") to within
    numerical precision — and only the +1j sign does (mean error ~0.005,
    vs ~1.2 with -1j, tested against the same mic/source grids). The textbook
    delay formula is usually written with -1j (matching the MATLAB reference
    this was ported from), but that convention assumes a different sign for
    "look direction" than shroom's Y_source.conj() usage expects.

    Parameters
    ----------
    mic_positions : (M, 3) array, mic Cartesian positions in meters.
    theta_vec : (K,) colatitude in radians (0 at +z), i.e. shroom's "co".
    phi_vec : (K,) azimuth in radians, i.e. shroom's "az".
    f_vec : (F,) frequency in Hz (signed, full two-sided spectrum).
    c : speed of sound in m/s.

    Returns
    -------
    v : (M, K, F) complex steering vectors.
    """
    u = np.stack([
        np.sin(theta_vec) * np.cos(phi_vec),
        np.sin(theta_vec) * np.sin(phi_vec),
        np.cos(theta_vec),
    ], axis=0)  # (3, K)
    tau = (mic_positions @ u) / c  # (M, K)
    return np.exp(1j * 2 * np.pi * f_vec[None, None, :] * tau[:, :, None])  # (M, K, F)


def build_free_field_array(mic_positions, source_grid, fs, duration):
    """Mics in free space at arbitrary 3D positions — uses the closed-form
    steering_vector() directly rather than SphericalArray, so it's exact for
    any mic placement, not just small apertures near the array center."""
    mic_positions = np.asarray(mic_positions, dtype=float)
    n_samples = int(duration * fs)
    freqs = np.fft.fftfreq(n_samples, 1 / fs)
    sm = steering_vector(mic_positions, source_grid.co, source_grid.az, freqs)  # (M, Q, F)
    return SpatialSignal(data=sm, fs=fs, is_time=False, is_space=True, grid=source_grid)


def load_precomputed_array(path, fs, grid_path=DEFAULT_PRECOMPUTED_GRID_PATH):
    """
    Load a precomputed steering matrix from datasets/experiment_full_anm/steering/*.mat
    (key "V", shape (M, F_pos, Q), positive frequencies only, generated at
    Fs=16000/nfft=512 in experiment_data_gen_3D.m) on the shared Lebedev-2702
    source grid, and wrap it in the same frequency-domain, space-domain
    SpatialSignal format ASM/ArrayDecoder expect from a SphericalArray.
    """
    V = scipy.io.loadmat(path)["V"]  # (M, F_pos, Q)
    n_fft = (V.shape[1] - 1) * 2
    sm_pos = V.transpose(0, 2, 1)  # (M, Q, F_pos)
    sm = reconstruct_neg_frequency_spectrum(sm_pos, n_fft, freq_axis=-1)  # (M, Q, F)

    grid_mat = scipy.io.loadmat(grid_path)
    grid = sphereicalGrid(az=grid_mat["ph"].ravel(), co=grid_mat["th"].ravel())

    array = SpatialSignal(data=sm, fs=fs, is_time=False, is_space=True, grid=grid)
    print(f"Loaded precomputed array: {path} -> {array.data.shape}")
    return array


def build_array(array_type, source_grid, fs, duration, rigid_sphere=None,
                 free_field=None, precomputed=None):
    """
    Build the array's frequency-domain steering matrix.

    Only the config matching `array_type` is used; the others can be left None.

    Parameters
    ----------
    array_type : {"rigid_sphere", "free_field", "precomputed"}
    source_grid : sphereicalGrid, shared source-direction grid (rigid_sphere/free_field only).
    rigid_sphere : RigidSphereArrayConfig, required if array_type == "rigid_sphere".
    free_field : FreeFieldArrayConfig, required if array_type == "free_field".
    precomputed : PrecomputedArrayConfig, required if array_type == "precomputed".
    """
    if array_type == "rigid_sphere":
        return build_rigid_sphere_array(
            rigid_sphere.mics_grid, source_grid, fs, duration, rigid_sphere.mic_radius
        )
    if array_type == "free_field":
        return build_free_field_array(free_field.mic_positions, source_grid, fs, duration)
    if array_type == "precomputed":
        return load_precomputed_array(precomputed.array_path, fs, grid_path=precomputed.grid_path)
    raise ValueError(f"Unknown ARRAY_TYPE: {array_type!r}")
