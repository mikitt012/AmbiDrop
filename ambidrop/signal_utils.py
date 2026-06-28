import torch
import torch.nn.functional as F
import numpy as np
import os
import scipy.io


# ── Torch variants ───────────────────────────────────────────────────────────

def pad_or_truncate_torch(signal: torch.Tensor, target_length: int = 120000) -> torch.Tensor:
    """Pad or truncate a multichannel signal (C, T) to (C, target_length)."""
    C, T = signal.shape
    if T > target_length:
        return signal[:, :target_length]
    elif T < target_length:
        pad_len = target_length - T
        pad_tensor = torch.zeros(C, pad_len, device=signal.device, dtype=signal.dtype)
        return torch.cat([signal, pad_tensor], dim=1)
    return signal


def pad_to_length(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad a tensor (C, T) to target_len along time dimension. Truncates if longer."""
    C, T = x.shape
    if T >= target_len:
        return x[:, :target_len]
    pad = torch.zeros(C, target_len - T, device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=1)


def add_white_noise_torch(signal: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Add white Gaussian noise to a multichannel signal tensor (C, T)."""
    signal = signal.float()
    if signal.ndim == 1:
        signal = signal.unsqueeze(0)
    signal_power = signal.pow(2).mean(dim=1, keepdim=True)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(signal) * torch.sqrt(noise_power)
    return signal + noise


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


def find_max_length(data_dir, data_type, ambisonics=False):
    """Find the maximum signal length across all .mat files in a dataset directory."""
    folder_path = os.path.join(data_dir, data_type)
    files_list = os.listdir(folder_path)

    if ambisonics:
        mat_files = [f for f in files_list if f.startswith("Ambisonics_") and f.endswith(".mat")]
    else:
        mat_files = [f for f in files_list if 'ex' in f and f.endswith(".mat")]

    max_len = 0
    for f in mat_files:
        file_path = os.path.join(folder_path, f)
        data = scipy.io.loadmat(file_path)
        key = 'anm_t' if ambisonics else 'p'
        signal = data[key].T
        max_len = max(max_len, signal.shape[1])

    return max_len


# ── Misc helpers ─────────────────────────────────────────────────────────────

def unwrap_model(model):
    """Unwrap a DataParallel model."""
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def get_lr(optimizer):
    """Get current learning rate from optimizer."""
    return optimizer.param_groups[0]['lr']
