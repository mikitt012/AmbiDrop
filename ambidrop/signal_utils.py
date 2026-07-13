"""
Signal processing helpers shared across FT-JNF and Conv-TasNet pipelines.

Public interface:
    pad_to_length — pad or truncate a (C, T) tensor to a target length
    pad_or_truncate_numpy — pad or truncate a (C, T) numpy array to a target length
    add_white_noise_numpy — add white noise at a given SNR to a numpy signal
    process_segment — onset-detect and window a (noisy, clean) tensor pair
    zero_random_channels — zero n random SH channels (excluding a00) in a [1,T,F,2C] tensor
    get_lr — return the current learning rate from an optimizer
    find_ref_mic — return 0-based index of mic closest to the positive x-axis direction
    complex_acn_to_real_acn — convert complex ACN Ambisonics to real-valued ACN representation
    tensor_to_istft — invert an STFT tensor (T_frames, F, 2C) back to time domain (C, T)
"""
import torch
import torch.nn.functional as F
import numpy as np

from ambidrop.constants import N_FFT, HOP_LENGTH, WIN_LENGTH


# ── Torch helpers ────────────────────────────────────────────────────────────

def pad_to_length(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad a tensor (C, T) to target_len along time dimension. Truncates if longer."""
    C, T = x.shape
    if T >= target_len:
        return x[:, :target_len]
    pad = torch.zeros(C, target_len - T, device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=1)



# ── Numpy variants ───────────────────────────────────────────────────────────

def pad_or_truncate_numpy(signal: np.ndarray, target_length: int = 120000) -> np.ndarray:
    """Pad or truncate a multichannel signal (C, T) to (C, target_length)."""
    C, T = signal.shape
    if T > target_length:
        return signal[:, :target_length]
    elif T < target_length:
        pad_width = ((0, 0), (0, target_length - T))
        return np.pad(signal, pad_width, mode='constant')
    return signal


def add_white_noise_numpy(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Add white noise to a numpy signal. Handles complex signals."""
    is_complex = np.iscomplexobj(signal)
    signal_power = np.mean(np.abs(signal)**2, axis=1, keepdims=True)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    if is_complex:
        noise_real = np.random.normal(0, np.sqrt(noise_power / 2), signal.shape)
        noise_imag = np.random.normal(0, np.sqrt(noise_power / 2), signal.shape)
        noise = noise_real + 1j * noise_imag
    else:
        noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise


# ── Signal processing helpers ────────────────────────────────────────────────

def process_segment(noisy, clean, target_samples=96000, threshold=1e-3):
    """
    Find the first speech onset, slice target_samples from there,
    and pad if the remaining signal is too short.
    Input tensor shape: [Channels, Time] or [Time].
    """
    noisy = noisy.unsqueeze(0) if noisy.dim() == 1 else noisy
    clean = clean.unsqueeze(0) if clean.dim() == 1 else clean

    ref_channel = clean
    mask = (clean.squeeze().abs() > threshold).nonzero()

    first = mask[0].item() if mask.numel() > 0 else 0
    last = first + target_samples

    noisy_processed = noisy[:, first:last]
    clean_processed = ref_channel[:, first:last]

    curr_len = noisy_processed.shape[1]
    if curr_len < target_samples:
        padding_needed = target_samples - curr_len
        noisy_processed = F.pad(noisy_processed, (0, padding_needed), "constant", 0)
        clean_processed = F.pad(clean_processed, (0, padding_needed), "constant", 0)
    else:
        noisy_processed = noisy_processed[:, :target_samples]
        clean_processed = clean_processed[:, :target_samples]

    return noisy_processed.squeeze(), clean_processed.squeeze()


def zero_random_channels(x, n):
    """
    Zero out n random channels (excluding channel 0) in a complex STFT
    input x of shape [1, T, F, 2C].
    """
    assert x.dim() == 4, "Expected input shape [1, T, F, 2C]"
    C = x.shape[-1] // 2
    assert 0 <= n <= C - 1, f"n must be between 0 and {C-1}, got {n}"

    available_channels = list(range(1, C))
    selected = torch.randperm(len(available_channels))[:n]
    channels_to_zero = [available_channels[i] for i in selected]

    for ch in channels_to_zero:
        x[..., ch] = 0
        x[..., ch + C] = 0

    return x



# ── Misc helpers ─────────────────────────────────────────────────────────────

def get_lr(optimizer):
    """Get current learning rate from optimizer."""
    return optimizer.param_groups[0]['lr']


# ── Reference microphone geometry ───────────────────────────────────────────

def find_ref_mic(mic_positions: np.ndarray) -> int:
    """Return 0-based index of mic closest to the target speaker direction.

    The target speaker is assumed to be on the positive x-axis.  The reference
    point is placed at (r_mean, 0, 0) where r_mean is the mean mic-to-origin
    distance, so the comparison is fair across mics that may differ in distance
    from the origin.

    mic_positions: (M, 3) array of 3D Cartesian mic coordinates.
    """
    pos = np.asarray(mic_positions, dtype=float)
    r_mean = np.mean(np.linalg.norm(pos, axis=1))
    ref = np.array([r_mean, 0.0, 0.0])
    return int(np.argmin(np.linalg.norm(pos - ref, axis=1)))


# ── Ambisonics / STFT conversion helpers ────────────────────────────────────

def complex_acn_to_real_acn(complex_acn_buffer: np.ndarray, max_order: int = 2,
                             sn3d: bool = False) -> np.ndarray:
    """
    Convert complex ACN Ambisonics coefficients to real-valued ACN.
    Input shape: (num_channels, T). Output shape: (num_channels, T).
    Works with both numpy arrays and torch tensors.
    """
    num_channels, num_samples = complex_acn_buffer.shape
    real_acn_buffer = np.zeros((num_channels, num_samples))
    for l in range(max_order + 1):
        sn3d_scale = 1.0 / np.sqrt(2 * l + 1) if sn3d else 1.0
        n_mid = l**2 + l
        real_acn_buffer[n_mid, :] = complex_acn_buffer[n_mid, :].real * sn3d_scale
        for m in range(1, l + 1):
            n_pos = l**2 + l + m
            n_neg = l**2 + l - m
            Y_pos = complex_acn_buffer[n_pos, :]
            Y_neg = complex_acn_buffer[n_neg, :]
            real_pos_m = (Y_neg + ((-1) ** m) * Y_pos) / np.sqrt(2)
            real_acn_buffer[n_pos, :] = real_pos_m.real * sn3d_scale
            real_neg_m = 1j / np.sqrt(2) * (Y_pos - ((-1) ** m) * Y_neg)
            real_acn_buffer[n_neg, :] = real_neg_m.real * sn3d_scale
    return real_acn_buffer


def tensor_to_istft(stft_tensor: torch.Tensor, length: int) -> torch.Tensor:
    """
    Convert STFT tensor (T_frames, F, 2C) back to time-domain waveform (C, T_samples).
    Real and imaginary parts are expected concatenated along the last dimension.
    """
    _, _, two_C = stft_tensor.shape
    C = two_C // 2
    stft_complex = torch.complex(stft_tensor[:, :, :C], stft_tensor[:, :, C:])
    stft_complex = stft_complex.permute(2, 1, 0)  # (C, F, T_frames)
    window = torch.hamming_window(WIN_LENGTH, device=stft_tensor.device)
    return torch.istft(
        stft_complex, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        window=window, center=True, normalized=False, onesided=True,
        return_complex=False, length=length,
    )
