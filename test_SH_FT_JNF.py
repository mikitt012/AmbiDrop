import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, Subset
import torch.nn.functional as F
import torch.nn as nn
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import ipdb
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pesq import pesq
from pystoi import stoi
import logging
import h5py
import wandb
import os
import pandas as pd
wandb.login()
# import re  
writer = SummaryWriter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# In[2]:
def zero_random_channels(x, n):
    """
    Zero out n random channels (excluding channel 0) in a complex STFT input x of shape [1, T, F, 2C].

    Parameters:
    - x: torch.Tensor, shape [1, T, F, 2C]
    - n: int, number of channels to zero (must be in 0 <= n <= C - 1)
    """
    assert x.dim() == 4, "Expected input shape [1, T, F, 2C]"
    C = x.shape[-1] // 2
    assert 0 <= n <= C - 1, f"n must be between 0 and {C-1}, got {n}"

    # All candidate channels except channel 0
    available_channels = list(range(1, C))
    selected = torch.randperm(len(available_channels))[:n]
    channels_to_zero = [available_channels[i] for i in selected]

    # Zero both real and imaginary parts
    for ch in channels_to_zero:
        x[..., ch] = 0
        x[..., ch + C] = 0

    return x

def find_max_length(data_dir, data_type, ambisonics=False):
    """
    Loads all .mat files in the specified dataset directory, finds the maximum length among signals,
    zero-pads all signals to this maximum length, and overwrites the .mat files with the padded data.

    Parameters:
        data_dir (str): Base directory containing the dataset.
        data_type (str): Subfolder name within data_dir, e.g., 'train_data_SH'.
        ambisonics (bool): Whether the dataset uses ambisonics format ('anm_t' and 'anm_clean_rir_t').
    """
    folder_path = os.path.join(data_dir, data_type)
    files_list = os.listdir(folder_path)

    if ambisonics:
        mat_files = [f for f in files_list if f.startswith("Ambisonics_") and f.endswith(".mat")]
    else:
        mat_files = [f for f in files_list if 'ex' in f and f.endswith(".mat")]

    # Step 1: Determine maximum length
    max_len = 0
    for f in mat_files:
        file_path = os.path.join(folder_path, f)
        data = scipy.io.loadmat(file_path)
        key = 'anm_t' if ambisonics else 'p'
        signal = data[key].T
        max_len = max(max_len, signal.shape[1])

    return max_len

def process_segment(noisy, clean, target_samples=96000, threshold=1e-3):
    """
    Finds the first occurrence of speech, slices target_samples from there,
    and pads if the remaining signal is too short.
    tensor shape: [Channels, Time]
    """
    # print(clean.shape); print(noisy.shape)
    noisy = noisy.unsqueeze(0)
    clean = clean.unsqueeze(0)
    
    # 1. Find the first index where any channel exceeds the threshold
    ref_channel = clean
    mask = (clean.squeeze().abs() > threshold).nonzero()
    
    if mask.numel() > 0:
        first = mask[0].item()
    else:
        first = 0 # Fallback if speech never hits threshold
    
    last = first + target_samples
    print(f"first is {first}, last is {last}")

    # Slice both signals using the SAME 'first' point found on ref mic
    noisy_processed = noisy[:, first:last]
    clean_processed = ref_channel[:, first:last]

    # --- STANDARDIZE LENGTH (Padding/Truncating) ---
    curr_len = noisy_processed.shape[1]
    if curr_len < target_samples:
        padding_needed = target_samples - curr_len
        noisy_processed = F.pad(noisy_processed, (0, padding_needed), "constant", 0)
        clean_processed = F.pad(clean_processed, (0, padding_needed), "constant", 0)
    else:
        noisy_processed = noisy_processed[:, :target_samples]
        clean_processed = clean_processed[:, :target_samples]

    # print(noisy_processed.shape)
        
    return noisy_processed.squeeze(), clean_processed.squeeze()

def pad_or_truncate(signal, target_length=120000):
    """
    Pads or truncates a multichannel signal (C, T) to (C, target_length).
    """
    C, T = signal.shape
    if T > target_length:
        print(f"[Warning] Signal truncated from {T} to {target_length} samples.")
        return signal[:, :target_length]
    elif T < target_length:
        pad_width = ((0, 0), (0, target_length - T))
        return np.pad(signal, pad_width, mode='constant')
    else:
        return signal
        
def pad_to_length(x, target_len):
    C, T = x.shape
    if T < target_len:
        pad = torch.zeros(C, target_len - T, device=x.device, dtype=x.dtype)
        return torch.cat([x, pad], dim=1)
    return x

# def si_snr(estimate, reference, epsilon=1e-8):
#     estimate = estimate - estimate.mean()
#     reference = reference - reference.mean()
#     reference_pow = reference.pow(2).mean(axis=1, keepdim=True)
#     mix_pow = (estimate * reference).mean(axis=1, keepdim=True)
#     scale = mix_pow / (reference_pow + epsilon)

#     reference = scale * reference
#     error = estimate - reference

#     reference_pow = reference.pow(2)
#     error_pow = error.pow(2)
#     reference_pow = reference_pow.mean(axis=1)
#     error_pow = error_pow.mean(axis=1)

#     error_pow = error_pow.clamp(min=1e-8)
#     reference_pow = reference_pow.clamp(min=1e-8)
    
#     si_snr = 10 * torch.log10(reference_pow) - 10 * torch.log10(error_pow)
#     return si_snr

def si_snr(estimate: torch.Tensor, reference: torch.Tensor, epsilon=1e-8, debug=False):
    """
    Compute Scale-Invariant Signal-to-Noise Ratio (SI-SNR) between estimate and reference signals.
    
    Args:
        estimate (torch.Tensor): Estimated signal, shape [B, T]
        reference (torch.Tensor): Ground truth signal, shape [B, T]
        epsilon (float): Small value to avoid division by zero
        debug (bool): If True, print internal debugging info

    Returns:
        si_snr (torch.Tensor): SI-SNR per sample, shape [B]
    """
    if debug:
        print(f"[DEBUG] estimate shape: {estimate.shape}, reference shape: {reference.shape}")

    # 1. Zero-mean normalization (along time dimension)
    estimate = estimate - estimate.mean(dim=1, keepdim=True)
    reference = reference - reference.mean(dim=1, keepdim=True)

    if debug:
        print(f"[DEBUG] After zero-mean -> estimate: {estimate.shape}, reference: {reference.shape}")

    # 2. Compute the scaling factor
    dot = (estimate * reference).sum(dim=1, keepdim=True)  # [B, 1]
    ref_energy = (reference ** 2).sum(dim=1, keepdim=True) + epsilon  # [B, 1]

    scale = dot / ref_energy  # [B, 1]
    projection = scale * reference  # [B, T]

    # 3. Compute the noise (error)
    noise = estimate - projection  # [B, T]

    # 4. Power of target and noise
    target_power = (projection ** 2).sum(dim=1)  # [B]
    noise_power = (noise ** 2).sum(dim=1) + epsilon  # [B]

    si_snr_value = 10 * torch.log10(target_power / noise_power)  # [B]

    if debug:
        print(f"[DEBUG] target_power: {target_power.mean().item():.4f}, noise_power: {noise_power.mean().item():.4f}")
        print(f"[DEBUG] SI-SNR mean: {si_snr_value.mean().item():.4f} dB")

    return si_snr_value

def complex_si_snr(estimate, reference, epsilon=1e-8):
    # Center both signals (mean over time)
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    reference = reference - reference.mean(dim=-1, keepdim=True)

    # Compute scaling factor using complex inner product
    dot = torch.sum(estimate * reference.conj(), dim=-1, keepdim=True)  # <estimate, reference>
    norm = torch.sum(reference * reference.conj(), dim=-1, keepdim=True) + epsilon  # ||reference||^2

    scale = dot / norm
    scaled_reference = scale * reference

    # Error signal
    error = estimate - scaled_reference

    # Power
    reference_power = torch.sum(torch.abs(scaled_reference) ** 2, dim=-1)
    error_power = torch.sum(torch.abs(error) ** 2, dim=-1)

    si_snr = 10 * torch.log10((reference_power + epsilon) / (error_power + epsilon))
    return si_snr

def add_white_noise(signal, snr_db):
    """
    Add white noise to a signal based on a desired SNR in dB.

    Parameters:
    - signal: np.array, the original signal. Can be multi-channel (channels as rows).
    - snr_db: float, the desired signal-to-noise ratio in dB.

    Returns:
    - np.array, the signal with added white noise.
    """
    # Calculate signal power and convert SNR from dB to linear
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

def load_checkpoint(checkpoint_path, target_epoch=None, net=None, optimizer=None, scheduler=None):
    """
    Load the checkpoint for a specific epoch or the latest checkpoint if no epoch is specified.
    Also loads learning rate and scheduler state.
    """
    checkpoint_list = torch.load(checkpoint_path)
    available_epochs = [ckpt["epoch"] for ckpt in checkpoint_list]

    # If no epoch specified, pick the latest
    if target_epoch is None:
        target_epoch = max(available_epochs)
        print(f"No epoch specified. Loading the latest checkpoint from epoch {target_epoch}")

    # Find exact match OR closest
    if target_epoch in available_epochs:
        chosen_epoch = target_epoch
    else:
        # Pick epoch with minimum distance to target
        chosen_epoch = min(available_epochs, key=lambda e: abs(e - target_epoch))
        print(f"Epoch {target_epoch} not found. Using closest epoch: {chosen_epoch}")

    # Retrieve the selected checkpoint
    checkpoint_to_load = next(ckpt for ckpt in checkpoint_list if ckpt["epoch"] == chosen_epoch)
    
    # Load the model and optimizer state
    if net is not None:
        net.load_state_dict(checkpoint_to_load['model_state_dict'])
    # if optimizer is not None:
    #     optimizer.load_state_dict(checkpoint_to_load['optimizer_state_dict'])

    # # Load the learning rate (if you want to log or use it later)
    # if optimizer is not None:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = checkpoint_to_load['learning_rate']
    
    # # Print the learning rate
    # print(f"Loaded learning rate: {optimizer.param_groups[0]['lr']:.6f}")

    # Load the scheduler state (if provided)
    # if scheduler is not None:
    #     scheduler.load_state_dict(checkpoint_to_load['scheduler_state_dict'])

    # Extract the loss or any other metrics you want
    # prev_loss = checkpoint_to_load['loss']
    # print(f"Loaded checkpoint from epoch {target_epoch}, loss: {prev_loss:.4f}")

        return chosen_epoch
    
class SHChannelDropout(nn.Module):
    def __init__(self, drop_prob=0.5, max_drop=4):
        super().__init__()
        self.drop_prob = drop_prob
        self.max_drop = max_drop

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x

        B, T, F, C = x.shape
        assert C == 10, "Expected 10 channels (5 SH real + 5 imag)"
        SH_C = C // 2  # Number of SH channels (real or imag)

        mask = torch.ones(B, T, F, C, device=x.device)

        for b in range(B):
            drop_mask = torch.rand(SH_C - 1, device=x.device) < self.drop_prob  # exclude a00 (index 0)

            if self.max_drop is not None:
                num_to_drop = min(drop_mask.sum().item(), self.max_drop)
                drop_indices = torch.nonzero(drop_mask).view(-1)
                if len(drop_indices) > num_to_drop:
                    selected = drop_indices[torch.randperm(len(drop_indices))[:num_to_drop]]
                    drop_mask[:] = 0
                    drop_mask[selected] = 1

            for i, drop in enumerate(drop_mask, start=1):  # start=1 to skip a00
                if drop:
                    mask[b, :, :, i] = 0.0           # real part
                    mask[b, :, :, i + SH_C] = 0.0    # imag part

        return x * mask

class PerChDropout(nn.Module):
    def __init__(self, drop_probs):
        """
        drop_probs: 1D list/tuple/tensor of length SH_C (number of SH channels).
                    Entry 0 (a00) will be forced to 0.0 internally (never dropped).
        """
        super().__init__()

        drop_probs = torch.as_tensor(drop_probs, dtype=torch.float32)
        # Ensure we never drop channel 0 (a00)
        if drop_probs.numel() < 1:
            raise ValueError("drop_probs must have at least one entry (for a00).")
        drop_probs = drop_probs.clone()
        drop_probs[0] = 0.0

        self.register_buffer("drop_probs", drop_probs)

    def forward(self, x):
        """
        x: Tensor of shape (B, T, F, C)
           where C = 2 * SH_C (real + imag for each SH channel),
           and channel 0 (and SH_C) are the real/imag parts of a00.
        """
        if not self.training:
            return x

        B, T, F, C = x.shape
        SH_C = C // 2  # number of SH channels (real or imag)

        if self.drop_probs.numel() != SH_C:
            raise ValueError(
                f"drop_probs length ({self.drop_probs.numel()}) must match SH_C ({SH_C})."
            )

        # Broadcast drop_probs to device
        drop_probs = self.drop_probs.to(x.device)

        # Create mask of ones
        mask = torch.ones(B, T, F, C, device=x.device)

        # For each batch element, sample independent dropout decisions per SH channel
        for b in range(B):
            # Sample Bernoulli per SH channel (0..SH_C-1)
            # Channel 0 has p=0.0, so it will never drop.
            rand_vals = torch.rand(SH_C, device=x.device)
            drop_mask = rand_vals < drop_probs   # shape: (SH_C,)

            # Apply to real & imag together
            for i, drop in enumerate(drop_mask):
                if drop:
                    # real part index = i
                    # imag part index = i + SH_C
                    mask[b, :, :, i] = 0.0
                    mask[b, :, :, i + SH_C] = 0.0

        return x * mask

class LearnableFreqDropout(nn.Module):
    def __init__(self, num_sh_channels=5, num_freq_bins=257, smooth_weight=1e-4):
        super().__init__()
        self.num_freq_bins = num_freq_bins
        self.num_sh_channels = num_sh_channels
        self.smooth_weight = smooth_weight

        # Logits for non-reference channels (1..num_sh_channels-1)
        self.logits = nn.Parameter(torch.zeros(num_sh_channels - 1, num_freq_bins))

    def forward(self, x):
        """
        x shape: (B, T, F, C), where C = 2*num_sh_channels.
        """
        B, T, F, C = x.shape
        assert C == 2 * self.num_sh_channels, "C must be twice the number of SH channels (real + imag)."

        SH_C = self.num_sh_channels
        real_channels = x[:, :, :, :SH_C]
        imag_channels = x[:, :, :, SH_C:]

        if self.training:
            dropout_probs = torch.sigmoid(self.logits)  # (SH_C-1, F)
            bernoulli_mask = torch.bernoulli(1.0 - dropout_probs).unsqueeze(0).expand(B, -1, -1)

            channel_mask = torch.ones(B, SH_C, F, device=x.device)
            channel_mask[:, 1:, :] = bernoulli_mask

            channel_mask = channel_mask.permute(0, 2, 1).unsqueeze(1)  # (B, 1, F, SH_C)

            real_channels = real_channels * channel_mask
            imag_channels = imag_channels * channel_mask

        # In eval mode → no dropout
        x_out = torch.cat([real_channels, imag_channels], dim=3)
        return x_out

    def smoothness_reg(self):
        """
        Regularization term to encourage smooth frequency masks (low-pass-like).
        """
        diff = self.logits[:, 1:] - self.logits[:, :-1]
        smooth_loss = (diff ** 2).mean()
        return self.smooth_weight * smooth_loss

    def visualize_masks(self, save_path=None):
        """
        Plot sigmoid probabilities for each SH channel (excluding a00).
        """
        with torch.no_grad():
            probs = torch.sigmoid(self.logits).cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 5))
        for ch in range(probs.shape[0]):
            ax.plot(probs[ch], label=f'SH ch {ch+1}')  # ch+1 to skip a00

        ax.set_title("Learned Dropout Probabilities per Frequency")
        ax.set_xlabel("Frequency Bin Index")
        ax.set_ylabel("Dropout Probability")
        ax.legend()
        ax.grid(True)

        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()

class SmoothLPFFreqDropout(nn.Module):
    def __init__(self, num_sh_channels=5, num_freq_bins=257, cutoff_freq_hz=1000, fs=16000, sharpness=0.02):
        super().__init__()
        self.num_freq_bins = num_freq_bins
        self.num_sh_channels = num_sh_channels
        self.fs = fs
        self.cutoff_freq_hz = cutoff_freq_hz
        self.sharpness = sharpness  # Controls sigmoid slope

        # Frequency vector (0 to Nyquist)
        freqs = np.linspace(0, fs / 2, num_freq_bins)

        # Create smooth sigmoid dropout probabilities
        # Higher sharpness = steeper transition, e.g., 0.02 → soft; 0.1 → sharp
        probs = 1 / (1 + np.exp(- (freqs - cutoff_freq_hz) * sharpness))

        # Save as buffer so it is properly moved to device
        self.register_buffer('dropout_probs', torch.from_numpy(probs.astype(np.float32)))

    def forward(self, x):
        """
        x shape: (B, T, F, C), where C = 2*num_sh_channels.
        """
        B, T, F, C = x.shape
        assert C == 2 * self.num_sh_channels, "C must be twice the number of SH channels (real + imag)."

        SH_C = self.num_sh_channels
        real_channels = x[:, :, :, :SH_C]
        imag_channels = x[:, :, :, SH_C:]

        if self.training:
            # Expand to shape (B, 1, F, SH_C-1)
            probs = self.dropout_probs.view(1, 1, F, 1).expand(B, 1, F, SH_C - 1)

            bernoulli_mask = torch.bernoulli(1.0 - probs).to(x.device)

            # Build full mask
            channel_mask = torch.ones(B, 1, F, SH_C, device=x.device)
            channel_mask[:, :, :, 1:] = bernoulli_mask  # a00 always kept

            real_channels = real_channels * channel_mask
            imag_channels = imag_channels * channel_mask

        x_out = torch.cat([real_channels, imag_channels], dim=3)
        return x_out

class MixedSHFreqDropout(nn.Module):
    def __init__(self, num_sh_channels=5, num_freq_bins=257,
                 cutoff_freq_hz=1000, fs=16000, sharpness=0.02,
                 drop_prob=0.5, max_drop=4):
        super().__init__()
        self.num_freq_bins = num_freq_bins
        self.num_sh_channels = num_sh_channels
        self.fs = fs
        self.cutoff_freq_hz = cutoff_freq_hz
        self.sharpness = sharpness

        self.drop_prob = drop_prob
        self.max_drop = max_drop

        # Frequency vector (0 to Nyquist)
        freqs = np.linspace(0, fs / 2, num_freq_bins)

        # Smooth sigmoid dropout probabilities for frequency mask
        freq_probs = 1 / (1 + np.exp(- (freqs - cutoff_freq_hz) * sharpness))

        self.register_buffer('freq_dropout_probs', torch.from_numpy(freq_probs.astype(np.float32)))

    def forward(self, x):
        """
        x shape: (B, T, F, C), where C = 2*num_sh_channels
        """
        if not self.training or self.drop_prob == 0.0:
            return x

        B, T, F, C = x.shape
        SH_C = C // 2
        assert SH_C == self.num_sh_channels, "SH_C mismatch."

        # Prepare base mask: all ones
        mask = torch.ones(B, T, F, C, device=x.device)

        for b in range(B):
            # Decide which higher-order channels to drop (exclude a00, i=0)
            drop_mask = torch.rand(SH_C - 1, device=x.device) < self.drop_prob

            if self.max_drop is not None:
                num_to_drop = min(drop_mask.sum().item(), self.max_drop)
                drop_indices = torch.nonzero(drop_mask).view(-1)
                if len(drop_indices) > num_to_drop:
                    selected = drop_indices[torch.randperm(len(drop_indices))[:num_to_drop]]
                    drop_mask[:] = 0
                    drop_mask[selected] = 1

            # Build frequency-dependent sigmoid mask for each dropped channel
            for i, drop in enumerate(drop_mask, start=1):  # start=1 skips a00
                if drop:
                    # Get frequency-dependent deterministic mask, shape (F,)
                    freq_mask = 1.0 - self.freq_dropout_probs  # zero at high freq

                    # Broadcast to (T, F)
                    freq_mask_2d = freq_mask.unsqueeze(0).expand(T, F)

                    # Apply to real and imag parts
                    mask[b, :, :, i] = freq_mask_2d
                    mask[b, :, :, i + SH_C] = freq_mask_2d

        return x * mask

class ProgressiveDeterministicFreqDropout(nn.Module):
    def __init__(self, num_sh_channels=5, num_freq_bins=257,
                 drop_prob=0.5, max_drop=4,
                 cutoff_freq_hz=1000, fs=16000, sharpness=0.02,
                 total_epochs=100):
        super().__init__()
        self.num_sh_channels = num_sh_channels
        self.num_freq_bins = num_freq_bins
        self.drop_prob = drop_prob
        self.max_drop = max_drop
        self.total_epochs = total_epochs
        self.current_epoch = 0
        
        # Frequency sigmoid mask
        freqs = np.linspace(0, fs / 2, num_freq_bins)
        probs = 1 / (1 + np.exp(- (freqs - cutoff_freq_hz) * sharpness))
        self.register_buffer('freq_mask_det', torch.from_numpy(probs.astype(np.float32)))

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        B, T, F, C = x.shape
        SH_C = C // 2

        alpha = min(self.current_epoch / self.total_epochs, 1.0)

        # Hard channel-level mask
        hard_mask = torch.ones(B, 1, 1, SH_C, device=x.device)
        for b in range(B):
            drop_mask = torch.rand(SH_C - 1, device=x.device) < self.drop_prob
            if self.max_drop is not None:
                num_to_drop = min(drop_mask.sum().item(), self.max_drop)
                drop_indices = torch.nonzero(drop_mask).view(-1)
                if len(drop_indices) > num_to_drop:
                    selected = drop_indices[torch.randperm(len(drop_indices))[:num_to_drop]]
                    drop_mask[:] = 0
                    drop_mask[selected] = 1
            for i, drop in enumerate(drop_mask, start=1):
                if drop:
                    hard_mask[b, :, :, i] = 0.0

        # Deterministic frequency mask
        freq_mask = torch.ones(1, 1, F, SH_C, device=x.device)
        freq_curve = self.freq_mask_det.view(1, 1, F, 1).expand(1, 1, F, SH_C - 1)
        freq_mask[:, :, :, 1:] = freq_curve  # always keep a00

        # Interpolate
        final_mask = (1 - alpha) * hard_mask + alpha * freq_mask

        x_real = x[:, :, :, :SH_C] * final_mask
        x_imag = x[:, :, :, SH_C:] * final_mask
        return torch.cat([x_real, x_imag], dim=3)

class MixedSHLearnableFreqDropout(nn.Module):
    def __init__(self, num_sh_channels=5, num_freq_bins=257,
                 drop_prob=0.5, max_drop=4, smooth_weight=1e-4):
        super().__init__()
        self.num_sh_channels = num_sh_channels
        self.num_freq_bins = num_freq_bins
        self.drop_prob = drop_prob
        self.max_drop = max_drop
        self.smooth_weight = smooth_weight

        # Learnable logits for non-reference channels
        self.logits = nn.Parameter(torch.zeros(num_sh_channels - 1, num_freq_bins))

    def forward(self, x):
        B, T, F, C = x.shape
        SH_C = C // 2
        assert SH_C == self.num_sh_channels

        # --- SH channel dropout mask (hard) ---
        channel_mask = torch.ones(B, 1, 1, SH_C, device=x.device)
        for b in range(B):
            drop_mask = torch.rand(SH_C - 1, device=x.device) < self.drop_prob
            if self.max_drop is not None:
                num_to_drop = min(drop_mask.sum().item(), self.max_drop)
                drop_indices = torch.nonzero(drop_mask).view(-1)
                if len(drop_indices) > num_to_drop:
                    selected = drop_indices[torch.randperm(len(drop_indices))[:num_to_drop]]
                    drop_mask[:] = 0
                    drop_mask[selected] = 1
            for i, drop in enumerate(drop_mask, start=1):
                if drop:
                    channel_mask[b, :, :, i] = 0.0

        # --- Learnable frequency mask ---
        freq_probs = torch.sigmoid(self.logits)  # (SH_C-1, F)
        freq_mask = torch.ones(1, 1, F, SH_C, device=x.device)
        freq_mask[:, :, :, 1:] = freq_probs.permute(1, 0).unsqueeze(0).unsqueeze(1)  # shape (1, 1, F, SH_C-1)

        # --- Combine masks multiplicatively ---
        final_mask = channel_mask * freq_mask

        # --- Apply ---
        x_real = x[:, :, :, :SH_C] * final_mask
        x_imag = x[:, :, :, SH_C:] * final_mask
        return torch.cat([x_real, x_imag], dim=3)

    def smoothness_reg(self):
        """
        Smoothness regularization for learnable logits.
        """
        diff = self.logits[:, 1:] - self.logits[:, :-1]
        smooth_loss = (diff ** 2).mean()
        return self.smooth_weight * smooth_loss


class SimDS(Dataset):
    def __init__(self, data_dir, data_type, ambisonics=False):
        self.data_dir = os.path.join(data_dir, data_type)
        self.ambisonics = ambisonics

        files_list = os.listdir(self.data_dir)
        if ambisonics:
            self.ex_list = [f for f in files_list if f.startswith("Ambisonics_") and f.endswith(".mat")]
        else:
            self.ex_list = [f for f in files_list if 'ex' in f and f.endswith(".mat")]

    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, idx):
        # try:
        #     file_path = os.path.join(self.data_dir, self.ex_list[idx])
        #     ex = scipy.io.loadmat(file_path)
        # except:
        #     print(f"Problem with file: {self.ex_list[idx]}")
        #     return self.__getitem__((idx + 1) % len(self.ex_list))  # wrap around to avoid index error

        file_path = os.path.join(self.data_dir, self.ex_list[idx])
        ex = scipy.io.loadmat(file_path)

        if self.ambisonics:
            # with h5py.File(file_path, 'r') as f:
            #     noisy_speech = np.array(f['anm_t']).T
            #     clean_speech = np.array(f['anm_clean_rir_t']).T
            noisy_speech = ex['anm_t']
            clean_speech = ex['anm_clean_rir_t']

        else:
            noisy_speech = ex['p']
            clean_speech = ex['pDirect']

        # 2) add noise if desired
        noisy_speech = add_white_noise(noisy_speech, 30)

        # 3) transpose → (C, N)
        noisy_speech = noisy_speech.T
        clean_speech = clean_speech.T

        noisy_speech = pad_or_truncate(noisy_speech, 120000)
        clean_speech = pad_or_truncate(clean_speech, 120000)

        # 5) to torch
        noisy_real = torch.from_numpy(noisy_speech.real).float().to(device)
        noisy_imag = torch.from_numpy(noisy_speech.imag).float().to(device)
        clean = torch.from_numpy(clean_speech).to(torch.complex64).to(device)

        # 6) STFT on real & imag parts separately
        win = torch.hamming_window(512).to(device)
        noisy_real_tf = torch.stft(noisy_real, n_fft=512, hop_length=256, win_length=512,
                                    window=win, center=True, normalized=False, onesided=True, return_complex=True)
        noisy_imag_tf = torch.stft(noisy_imag, n_fft=512, hop_length=256, win_length=512,
                                    window=win, center=True, normalized=False, onesided=True, return_complex=True)
        noisy_speech_tf = noisy_real_tf + 1j * noisy_imag_tf

        # 7) transpose → T×F×C
        noisy_speech_tf = noisy_speech_tf.transpose(0, 2)

        # Split complex STFT into real and imaginary parts for model input
        noisy_speech_tf = torch.cat((noisy_speech_tf.real, noisy_speech_tf.imag), dim=2)  # T x F x 2C

        # 10) select first clean channel
        clean_ch0 = clean[0]

        # max_val = noisy_speech_tf.abs().max().item()
        # noisy_speech_tf = noisy_speech_tf / max_val
        # clean_ch0 = clean_ch0/max_val

        # Return clean time-domain real part (e.g. first channel) and complex TF noisy input
        return clean_ch0, noisy_speech_tf
print('finish section', datetime.now())


class PreprocessedSHDataset(Dataset):
    def __init__(self, root: str, split: str):
            """
            root: where your `<split>_pt/` folders live
            split: e.g. 'train_data_SH' or 'val_data_SH'
            """
            self.folder = os.path.join(root, split)
            # grab all .pt files in sorted order without using glob
            self.samples = sorted(
                os.path.join(self.folder, fname)
                for fname in os.listdir(self.folder)
                if fname.endswith(".pt")
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # each .pt contains a tuple (clean_ch0, noisy_speech_tf)
        clean_real_ch0, noisy_speech_tf = torch.load(self.samples[idx])
        max_val = noisy_speech_tf.abs().max().item()
        noisy_speech_tf = noisy_speech_tf / max_val
        clean_real_ch0 = clean_real_ch0/max_val
        return clean_real_ch0, noisy_speech_tf  

class SimDS_preprocessed(Dataset):
    def __init__(self, data_dir, data_type):
        self.data_type = data_type
        self.data_dir = os.path.join(data_dir, data_type)

        self.sample_files = sorted([
            f for f in os.listdir(self.data_dir)
            if f.endswith('.pt') and f.startswith('ex_')
        ])

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        file_name = self.sample_files[idx]
        file_path = os.path.join(self.data_dir, file_name)

        # try:
        #     data = torch.load(file_path, map_location='cpu')  # Load from .pt file
        #     noisy_speech_tf, clean_speech_1ch = data
        #     noisy_speech_tf = noisy_speech_tf[:, :, [0, 1, 3, 4, 8, 9, 10, 12, 13, 17]]
        # except Exception as e:
        #     print(f"[Error loading] {file_name}: {e}")
        #     return self.__getitem__((idx + 1) % len(self))  # try next valid sample

        # return noisy_speech_tf, clean_speech_1ch.float()

        try:
            data = torch.load(file_path, map_location='cpu')  # Load from .pt file
            noisy_tf_mic, clean_mic, noisy_tf_anm, clean_anm = data
            # noisy_speech_tf = noisy_tf_mic
            noisy_speech_tf = noisy_tf_anm
        except Exception as e:
            print(f"[Error loading] {file_name}: {e}")
            return self.__getitem__((idx + 1) % len(self))  # try next valid sample

        return noisy_tf_mic, clean_mic.float(), noisy_speech_tf, clean_anm.float()


class FT_JNF(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim, 
                 drop_prob, max_drop, drop_probs, dropout):
        super().__init__()
        num_sh_channels = input_dim // 2
        if dropout == "SHChannelDropout":
            self.channel_dropout = SHChannelDropout(drop_prob=drop_prob, max_drop=max_drop)  # adjust max_drop depending on array type
        else:
            self.channel_dropout = PerChDropout(drop_probs=drop_probs)
            
        # self.channel_dropout = LearnableFreqDropout(num_sh_channels=num_sh_channels, num_freq_bins=257, smooth_weight=smooth_weight)
        # self.channel_dropout = SmoothLPFFreqDropout(num_sh_channels=num_sh_channels, num_freq_bins=257, cutoff_freq_hz=1000, fs=16000, sharpness=sharpness)
        # self.channel_dropout = MixedSHFreqDropout(num_sh_channels=num_sh_channels, num_freq_bins=257, cutoff_freq_hz=1500, fs=16000, sharpness=sharpness, drop_prob=drop_prob, max_drop=max_drop)
        # self.channel_dropout = ProgressiveDeterministicFreqDropout(
        #     num_sh_channels=num_sh_channels, 
        #     num_freq_bins=257, 
        #     drop_prob=drop_prob, 
        #     max_drop=max_drop, 
        #     cutoff_freq_hz=cutoff_freq_hz,
        #     fs=fs,
        #     sharpness=sharpness,
        #     total_epochs=total_epochs
        # )
        # self.channel_dropout = MixedSHLearnableFreqDropout(
        #     num_sh_channels=num_sh_channels, 
        #     num_freq_bins=257,
        #     drop_prob=drop_prob,
        #     max_drop=max_drop,
        #     smooth_weight=smooth_weight
        # )

        # FC Small model
        #self.ff1 = nn.Linear(input_dim, hidden1_dim)
        #self.ff2 = nn.Linear(hidden1_dim, hidden2_dim)
        #self.linear = nn.Linear(hidden2_dim, output_dim)


        #LSTM model
        self.LSTM1 = nn.LSTM(input_size=input_dim,hidden_size=hidden1_dim,num_layers=1,batch_first=True,bidirectional=True)
        self.LSTM2 = nn.LSTM(input_size=2*hidden1_dim,hidden_size=hidden2_dim,num_layers=1,batch_first=True,bidirectional=True)
        self.linear = nn.Linear(2*hidden2_dim,output_dim)

        self.hamming_window = torch.hamming_window(512, device=device)
        
        # Initialize weights
    
        # FC Small model
                #nn.init.xavier_uniform_(self.ff1.weight)
        #self.ff1.bias.data.fill_(0)
        #nn.init.xavier_uniform_(self.ff2.weight)
        #self.ff2.bias.data.fill_(0)
        #nn.init.xavier_uniform_(self.linear.weight)
        #self.linear.bias.data.fill_(0)
    
        # LSTM model
        for name, param in self.LSTM1.named_parameters():
            if 'weight_ih' in name:  # Input-hidden weights
                nn.init.xavier_uniform_(param.data)  # Xavier initialization
            elif 'weight_hh' in name:  # Hidden-hidden weights
                nn.init.orthogonal_(param.data)  # Orthogonal initialization
            elif 'bias' in name:
                param.data.fill_(0)  # Bias initialized to zero
        for name, param in self.LSTM2.named_parameters():
            if 'weight_ih' in name:  # Input-hidden weights
                nn.init.xavier_uniform_(param.data)  # Xavier initialization
            elif 'weight_hh' in name:  # Hidden-hidden weights
                nn.init.orthogonal_(param.data)  # Orthogonal initialization
            elif 'bias' in name:
                param.data.fill_(0)  # Bias initialized to zero
        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0)


    def forward(self, x):
        # batch LSTM model
        B, T, F, C = x.shape
        x = self.channel_dropout(x)
        x, hidden = self.LSTM1(x.view(B * T, F, C)) #BTxFxC
        x = x.view(B, T, F, -1)     # (B, T, F, hidden1*2)
        x = x.permute(0, 2, 1, 3)   # → (B, F, T, hidden1*2)
        x = x.reshape(B * F, T, -1) # Each frequency bin across time is one sequence
        x, hidden = self.LSTM2(x) #FxTx256
        x = x.view(B, F, T, -1).permute(0, 2, 1, 3)  # → (B, T, F, hidden2*2)
        x = self.linear(x)   

        # LSTM model
        # x, hidden = self.LSTM1(x.view(x.shape[1],x.shape[2],x.shape[3])) #TxFx512 
        # x, hidden = self.LSTM2(x.permute(1,0,2)) #FxTx256
        # x = self.linear(x) #FxTx2
        # x = x.permute(1,0,2) #TxFx2
   
        # FC Small model
        #x = x.view(-1, x.shape[-1])  # Flatten the input for the feed-forward network
        #x = F.relu(self.ff1(x))
        #x = F.relu(self.ff2(x))
        #x = self.linear(x)
        #x = x.view(-1, 257, 2)  # Reshape back to match the output dimensions
        
        return x


    def training_step(self, batch, batch_idx):
        #x, S, V, s, v, x_normalized = batch
        x, s = batch
        x = x.to(device)
        s = s.to(device)
        # Model prediction
        M = self(x)
        # Ms = M[:, :, 0] + 1j * M[:, :, 1]
        Ms = M[:,:,:,0] + 1j*M[:,:,:,1] # BxTxF
        Ms = Ms.squeeze()
        # Mv = 1 - M[:, :, 0] - 1j * M[:, :, 1]
        # Y = x[0, :, :, 0] + 1j * x[0, :, :, 5]
        Y = x[:,:,:,0] + 1j*x[:,:,:,9]

        # Reconstruct signals
        S_hat = Ms * Y
        #V_hat = Mv * Y
        # s_hat = torch.istft(S_hat.T, n_fft=512, hop_length=256, win_length=512,
        #                     window=self.hamming_window, center=True, normalized=False,
        #                     onesided=True, return_complex=False, length=s.shape[1])
        s_hat = torch.istft(S_hat.permute(0, 2, 1), n_fft=512, hop_length=256,
                            win_length=512, window=self.hamming_window, center=True, normalized=False, 
                            onesided=True, return_complex=False, length=s.shape[1])
        #v_hat = torch.istft(V_hat.T, n_fft=512, hop_length=256, win_length=512,
        #                    window=self.hamming_window, center=True, normalized=False,
        #                    onesided=True, return_complex=False, length=v.shape[1])

        if torch.isnan(s_hat).any() or torch.isnan(s).any():
            print(f"[NaN detected] s_hat or s is NaN at batch {batch_idx}")
            return torch.tensor(0.0, requires_grad=True).to(device)

        if s_hat.abs().max() < 1e-6 or s.abs().max() < 1e-6:
            print(f"[Warning] Very low energy in s_hat or s at batch {batch_idx}")

        # Compute losses
        # alpha = 10
        # loss_time = alpha * torch.norm(s_hat - s, p=1) + alpha * torch.norm(v_hat - v, p=1)
        # loss_freq = torch.norm((S.abs() - S_hat.abs()), p=1) + torch.norm(V.abs()-V_hat.abs(),p=1)
        # loss = loss_time + loss_freq
        # loss = -si_snr(s_hat.unsqueeze(0),s)
        loss = -si_snr(s_hat, s, debug=False)
        return loss.mean()
print('finish section', datetime.now())

batch_size = 8
lr = 0.001
weight_decay=1e-6

# --- different dropout ---
for t in range(0,1):
    if t == 0:
        drop_prob = 0.4
        max_drop = 3
        drop_probs = []
        th = 0
        dropout = "SHChannelDropout"
        checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2025-12-01_10-08-18.pt"
    if t == 1:
        drop_prob = 0.7
        max_drop = 7
        drop_probs = []
        th = 0
        dropout = "SHChannelDropout"
        checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2025-12-04_15-20-29.pt"
    if t == 2:
        drop_prob = 0.6
        max_drop = 7
        drop_probs = []
        th = 0
        dropout = "SHChannelDropout"
        checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2025-12-04_21-18-51.pt"
    if t == 3:
        drop_prob = 0.5
        max_drop = 6
        drop_probs = []
        th = 0
        dropout = "SHChannelDropout"
        checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2025-12-05_03-31-01.pt"
    if t == 4:
        drop_probs = [0, 0.55, 0.9, 0.6, 1, 1, 1, 1, 1] # th = -10 dB
        th = -10 # dB
        drop_prob = 0
        max_drop = 0
        dropout = "PerChDropout"
        checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2025-12-04_15-45-32.pt"
    if t == 5:
        drop_probs = [0, 0.35, 0.75, 0.2, 1, 1, 1, 1, 1] # th = -8.2 dB
        th = -8.2 # dB
        drop_prob = 0
        max_drop = 0
        dropout = "PerChDropout"
        checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2025-12-04_21-44-11.pt"
    if t == 6:
        drop_probs = [0, 0.15, 0.6, 0.2, 0.95, 1, 0.95, 1, 0.95] # th = -7 dB
        th = -7 # dB
        drop_prob = 0
        max_drop = 0
        dropout = "PerChDropout"
        checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2025-12-05_03-53-11.pt"
    if t == 7:
        drop_probs = [0, 0.15, 0.55, 0.15, 0.8, 1, 0.95, 1, 0.85] # th = -5.7 dB
        th = -5.7 # dB
        drop_prob = 0
        max_drop = 0
        dropout = "PerChDropout"
        checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2025-12-05_09-45-29.pt"
    if t == 8:
        drop_probs = [0, 0.1, 0.45, 0.15, 0.7, 1, 0.85, 1, 0.65] # th = -5 dB
        th = -5 # dB
        drop_prob = 0
        max_drop = 0
        dropout = "PerChDropout"
        checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2025-12-05_15-44-31.pt"
    if t == 9:
        drop_probs = [0, 0.1, 0.45, 0.1, 0.55, 1, 0.85, 1, 0.55] # th = -4.2 dB
        th = -4.2 # dB
        drop_prob = 0
        max_drop = 0
        dropout = "PerChDropout"
        checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2025-12-05_21-56-06.pt"
    if t == 10:
        drop_probs = [0, 0.1, 0.45, 0.1, 0.45, 1, 0.75, 1, 0.45] # th = -3.4 dB
        th = -3.4 # dB
        drop_prob = 0
        max_drop = 0
        dropout = "PerChDropout"
        checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2025-12-06_04-07-37.pt"
    if t == 11:
        drop_probs = [0, 0.05, 0.45, 0.05, 0.4, 0.95, 0.5, 0.95, 0.4] # th = -2.4 dB
        th = -2.4 # dB
        drop_prob = 0
        max_drop = 0
        dropout = "PerChDropout"
        checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2025-12-06_09-56-40.pt"
    if t == 12:
        drop_probs = [0, 0.05, 0.45, 0.05, 0.1, 0.75, 0.4, 0.75, 0.1] # th = -1.4 dB
        th = -1.4 # dB
        drop_prob = 0
        max_drop = 0
        dropout = "PerChDropout"
        checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2025-12-06_15-58-32.pt"
    if t == 13:
        drop_probs = [0, 0.05, 0.35, 0.05, 0, 0.4, 0, 0.3, 0] # th = 0 dB
        th = 0 # dB
        drop_prob = 0
        max_drop = 0
        dropout = "PerChDropout"
        checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2025-12-06_22-04-29.pt"
    if t == 14:
        drop_prob = 0.4
        max_drop = 7
        drop_probs = []
        th = 0
        dropout = "SHChannelDropout"
        checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2025-12-21_17-36-40.pt"
    if t == 15:
        drop_prob = 0.3
        max_drop = 6
        drop_probs = []
        th = 0
        dropout = "SHChannelDropout"
        checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2025-12-22_14-53-29.pt"
    if t == 16:
        drop_prob = 0.7
        max_drop = 3
        drop_probs = []
        th = 0
        dropout = "SHChannelDropout"
        checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2025-12-23_05-06-54.pt"
    if t == 17:
        drop_prob = 0
        max_drop = 0
        drop_probs = []
        th = 0
        dropout = "SHChannelDropout"
        checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2025-12-23_13-38-21.pt"

    smallnet = True

    if smallnet:
        net = FT_JNF(
            input_dim=18,
            hidden1_dim=64,
            hidden2_dim=64,
            output_dim=2,
            drop_prob=drop_prob,          # your original
            max_drop=max_drop,
            drop_probs=drop_probs,
            dropout=dropout
        ).to(device)
    else:
        net = FT_JNF(
            input_dim=18,
            hidden1_dim=256,
            hidden2_dim=128,
            output_dim=2,
            drop_prob=drop_prob,          # your original
            max_drop=max_drop,
            drop_probs=drop_probs,
            dropout=dropout
        ).to(device)

    model_type = "_dropout_fullanm"  # regular / dropout
    # if model_type == "_regular":
    #     checkpoint = torch.load("/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints_old/SH_FT_JNF_partial_paper.pt")
    # if model_type == "_dropout":
    #     checkpoint = torch.load("/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2025-08-09_11-00-08.pt")
    # if model_type == "_dropout_fullanm":
    #     if smallnet:
    #         checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2025-12-01_10-08-18.pt"
    #     else:   
    #         checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2025-12-01_09-21-33.pt"
    #     chosen_epoch = load_checkpoint(checkpoint_path, target_epoch=300, net=net)
    #     print(chosen_epoch)

    # net.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # data_dir = '/gpfs0/bgu-br/users/tatarjit/speech-enhancement/datasets/experiment_full_anm/test_of_train_ds_preprocessed_swap' 
    # test_type = 'front hemisphere1 (rigid) radius = 0.1_preprocessed'
    # # test_ds = SimDS(data_dir,'test_data_SH')
    # # max_len_train = find_max_length('/gpfs0/bgu-br/users/tatarjit/speech-enhancement', 'train_data_SH', ambisonics=True)
    # # test_ds = SimDS_preprocessed(data_dir, 'si_et_05_preprocessed_full')
    # test_ds = SimDS_preprocessed(data_dir, test_type)
    # # test_ds = PreprocessedSHDataset(data_dir, "test_data_SH_STFT")
    # testloader = DataLoader(test_ds, batch_size=1, shuffle=False)
    # if len(test_ds) == 0:
    #     print("⚠️ testloader is empty!")
    # else:
    #     print(f"✅ testloader contains {len(test_ds)} samples.")
    # # for i,data in enumerate(trainloader, 0):
    # #     print(i)

    chosen_epoch = load_checkpoint(checkpoint_path, target_epoch=200, net=net)
    print(chosen_epoch)

    plot_snr_dist = False
    wandb_active = False

    # Matrix to store SI-SDR: Rows = test_types, Cols = examples
    num_test_types = 21
    num_examples = 300
    master_si_sdr_noisy = np.zeros((num_test_types, num_examples))
    master_si_sdr_enhanced = np.zeros((num_test_types, num_examples))

    for j in range(1,2):
        # if t == 4 and i == 1:å
        #     continue   # skip i=1 when t=4
        if j == 1:
            data_dir = '/gpfs0/bgu-br/users/tatarjit/speech-enhancement/datasets/experiment_full_anm/test_of_train_ds_preprocessed'
        else:
            data_dir = '/gpfs0/bgu-br/users/tatarjit/speech-enhancement/datasets/experiment_full_anm/test_of_test_ds_preprocessed'

        all_test_types = sorted(os.listdir(data_dir))

        for test_idx, test_type in enumerate(all_test_types):   
            # test_type = "Aria on rigid sphere (simulated)_preprocessed"
            # test_idx = 1

            if test_type == "front hemisphere1 (rigid) radius = 0.1_preprocessed":
                ref_idx = 1

            if test_type == "full circle (rigid) radius = 0.1_preprocessed":
                ref_idx = 1

            if test_type == "planar_preprocessed":
                ref_idx = 6

            if test_type == "random 2D array1 radius = 0.1_preprocessed":
                ref_idx = 6

            if test_type == "random sphere1 radius = 0.1_preprocessed":
                ref_idx = 7

            if test_type == "random sphere3 (rigid) radius = 0.1_preprocessed":
                ref_idx = 4

            if test_type == "random sphere5 (rigid) radius = 0.05_preprocessed":
                ref_idx = 2

            if test_type == "semi circle planar radius = 0.05_preprocessed":
                ref_idx = 6

            if test_type == "ULA along X-axis_preprocessed":
                ref_idx = 7

            if test_type == "uniform sphere (rigid) radius = 0.1_preprocessed":
                ref_idx = 2

            if test_type == "front hemisphere2 (rigid) radius = 0.1_preprocessed":
                ref_idx = 1

            if test_type == "planar (rot=45deg)_preprocessed":
                ref_idx = 5

            if test_type == "random 2D array2 radius = 0.1_preprocessed":
                ref_idx = 2

            if test_type == "random sphere2 radius = 0.1_preprocessed":
                ref_idx = 2

            if test_type == "random sphere4 (rigid) radius = 0.1_preprocessed":
                ref_idx = 7

            if test_type == "random sphere6 (rigid) radius = 0.05_preprocessed":
                ref_idx = 4

            if test_type == "semi circle (rigid) radius = 0.1_preprocessed":
                ref_idx = 4

            if test_type == "ULA along Z-axis_preprocessed":
                ref_idx = 4

            if test_type == "uniform sphere (rigid) radius = 0.05_preprocessed":
                ref_idx = 2

            if test_type == "semi circle planar radius = 0.1_preprocessed":
                ref_idx = 6

            if test_type == "Aria on rigid sphere (simulated)_preprocessed":
                ref_idx = 3

            if test_type == "ULA along Y-axis (tilt=30deg)_preprocessed":
                ref_idx = 4
                
            if test_type == "ULA along x-axis (rot=30deg)_preprocessed":
                ref_idx = 7

            if test_type == "ULA along y-axis_preprocessed":
                ref_idx = 4

            if test_type == "ULA along X-axis (tilt=20)_preprocessed":
                ref_idx = 7

            ref_id = ref_idx - 1

            test_ds = SimDS_preprocessed(data_dir, test_type)
            # num_examples = 2
            # indices = list(range(num_examples))
            # test_ds_subset = Subset(test_ds, indices)
            # test_ds = test_ds_subset
            testloader = DataLoader(test_ds, batch_size=1, shuffle=False)

            stoi_noisy = np.array([])
            si_sdr_noisy = np.array([])
            pesq_noisy = np.array([])
            stoi_enhanced = np.array([])
            si_sdr_enhanced = np.array([])
            pesq_enhanced = np.array([])

            net.eval()
            if smallnet:
                name = f"{test_type}_smallnet_{dropout}_{t}"
            else:
                name = test_type + model_type

            if wandb_active:
                wandb.init(
                    project="AmbiDrop_test",
                    entity="tatarjit-ben-gurion-university-of-the-negev", 
                    name=name,
                    config={
                        "batch_size": batch_size,
                        "learning_rate": lr,
                        "weight_decay": weight_decay,
                        "drop_prob": drop_prob,
                        "max_drop": max_drop,
                        "drop_probs": drop_probs,
                        "th": th,
                        "dropout": dropout,
                        "chosen_epoch": chosen_epoch,
                    }
                )
            # print(chosen_epoch)

            # test_ds = CHiME3(data_dir,'test')
            # testloader = DataLoader(test_ds, batch_size=16, shuffle=False)
            for i, data in enumerate(testloader, 0):
                noisy_mic, clean_mic, x, s = data
                s1 = s.to(device)      # move clean to GPU
                # s1 = s1.unsqueeze(0)
                # print("before", x.shape, s1.shape)
                x = x.to(device) 
                x = zero_random_channels(x, n=0)
                M = net(x)
                Ms = M[:,:,:,0] + 1j*M[:,:,:,1]
                Ms = Ms.squeeze()

                # mask = Ms.detach().cpu()
                # mask_dir = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/models_interp/FT_JNF"
                # save_path = os.path.join(mask_dir, f"mask_{test_type}_{i}.pt")
                # torch.save({'mask': mask}, save_path)

                ref_ch = x[:,:,:,0] + 1j*x[:,:,:,9]
                Y = ref_ch.squeeze(0)
                S_hat = Ms*Y
                y = torch.istft(Y.T, n_fft=512, hop_length=256, win_length=512, window=torch.hamming_window(window_length=512), center=True, normalized=False, onesided=True, return_complex=False, length=s1.shape[1])
                s_hat = torch.istft(S_hat.T, n_fft=512, hop_length=256, win_length=512, window=torch.hamming_window(window_length=512), center=True, normalized=False, onesided=True, return_complex=False,length = s1.shape[1]) 
                s_hat = s_hat/s_hat.max()
                s1 = s1/s1.max()
                y = y/y.max()

                s1 = s1.squeeze(0)
                s1 = s1.detach().cpu()
                s_hat = s_hat.detach().cpu()
                y = y.detach().cpu()
                # print("after", y.shape, s_hat.shape, s1.shape)

                # --- for noisy si-sdr calculation:
                noisy_mic = noisy_mic.to(device)
                clean_mic = clean_mic.to(device)
                s1_mic = clean_mic[:, 0+ref_id, :].squeeze(1)
                s1_mic = s1_mic/s1_mic.max()
                ref_ch = noisy_mic[:,:,:,0+ref_id] + 1j*noisy_mic[:,:,:,7+ref_id]
                Y = ref_ch.squeeze(0)
                y_mic = torch.istft(Y.T, n_fft=512, hop_length=256, win_length=512, window=torch.hamming_window(window_length=512), center=True, normalized=False, onesided=True, return_complex=False, length=s1_mic.shape[1])
                y_mic = y_mic/y_mic.max()
                s1_mic = s1_mic.squeeze(0)
                s1_mic = s1_mic.detach().cpu()
                y_mic = y_mic.detach().cpu()

                # print(f"before: {y.shape} , {s1.shape} , {s_hat.shape}")
                # y, temp = process_segment(y, s1)
                # s_hat, s1 = process_segment(s_hat, s1)
                # print(f"after: {y.shape} , {s1.shape} , {s_hat.shape}")

                stoi_noisy = np.append(stoi_noisy, stoi(s1_mic, y_mic, 16000, extended=False))
                si_sdr_noisy = np.append(si_sdr_noisy, si_snr(y_mic.unsqueeze(0), s1_mic.unsqueeze(0), debug = False))
                pesq_noisy = np.append(pesq_noisy, pesq(16000, s1_mic.numpy(), y_mic.numpy(), mode="wb"))

                # stoi_noisy = np.append(stoi_noisy, stoi(s1, y, 16000, extended=False))
                # si_sdr_noisy = np.append(si_sdr_noisy, si_snr(y.unsqueeze(0), s1.unsqueeze(0), debug = False))
                # pesq_noisy = np.append(pesq_noisy, pesq(16000, s1.numpy(), y.numpy(), mode="wb"))

                stoi_enhanced = np.append(stoi_enhanced, stoi(s1, s_hat, 16000, extended=False))
                si_sdr_enhanced = np.append(si_sdr_enhanced, si_snr(s_hat.unsqueeze(0), s1.unsqueeze(0), debug = False))
                pesq_enhanced = np.append(pesq_enhanced, pesq(16000, s1.numpy(), s_hat.numpy(), mode="wb"))

                val_noisy = si_snr(y_mic.unsqueeze(0), s1_mic.unsqueeze(0), debug=False)
                val_enhanced = si_snr(s_hat.unsqueeze(0), s1.unsqueeze(0), debug=False)

                # Store in the matrix instead of appending to a flat list
                if j == 1:
                    master_si_sdr_noisy[test_idx, i] = val_noisy.item() if torch.is_tensor(val_noisy) else val_noisy
                    master_si_sdr_enhanced[test_idx, i] = val_enhanced.item() if torch.is_tensor(val_enhanced) else val_enhanced
                else:
                    master_si_sdr_noisy[test_idx + 10, i] = val_noisy.item() if torch.is_tensor(val_noisy) else val_noisy
                    master_si_sdr_enhanced[test_idx + 10, i] = val_enhanced.item() if torch.is_tensor(val_enhanced) else val_enhanced

                break
            break
            
            if wandb_active:
                wandb.log({
                    "test/stoi_noisy": float(stoi_noisy.mean()),
                    "test/pesq_noisy": float(pesq_noisy.mean()),
                    "test/si_sdr_noisy": float(si_sdr_noisy.mean()),
                    "test/stoi_enhanced": float(stoi_enhanced.mean()),
                    "test/pesq_enhanced": float(pesq_enhanced.mean()),
                    "test/si_sdr_enhanced": float(si_sdr_enhanced.mean())
                })
                
                wandb.log({
                    "audio/clean": wandb.Audio(s1.numpy(), sample_rate=16000),
                    "audio/enhanced": wandb.Audio(s_hat.numpy(), sample_rate=16000),
                    "audio/noisy": wandb.Audio(y.numpy(), sample_rate=16000),
                })
                wandb.finish()

            # print(f"Reference metrics for distorted speech at {snr_dbs[0]}dB are\n")
            print(f"STOI: {stoi_noisy.mean()}")
            print(f"PESQ: {pesq_noisy.mean()}")
            print(f"SI-SDR: {si_sdr_noisy.mean()}")
            print(f"STOI: {stoi_enhanced.mean()}")
            print(f"PESQ: {pesq_enhanced.mean()}")
            print(f"SI-SDR: {si_sdr_enhanced.mean()}")
            plt.show()

            # break
        # break

print(f"Shape of master_si_sdr_noisy: {master_si_sdr_noisy.shape}")

base_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/snr_distribution_fixed2"
if plot_snr_dist:
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    save_path = os.path.join(base_path, "master_si_sdr_noisy.npy")
    np.save(save_path, master_si_sdr_noisy)
    print(f"Data saved successfully to {save_path}")

    save_path = os.path.join(base_path, "master_si_sdr_enhanced.npy")
    np.save(save_path, master_si_sdr_enhanced)
    print(f"Data saved successfully to {save_path}")

    def csv_to_filtered_table_image(csv_path, output_png_path, title_text):
        # 1. Load the data
        df = pd.read_csv(csv_path)
        
        # 2. Filter columns and round to 1 decimal place
        # Note: Ensure these names match your CSV header exactly
        columns_to_keep = [
            'Bin [dB]', 
            'Sample Count', 
            'Mean Enhanced SI-SDR [dB]', 
            'Mean Improvement [dB]'
        ]
        
        # Keep only requested columns
        df_filtered = df[columns_to_keep].copy()
        
        # Round numerical columns to 1 decimal point
        # Sample Count is kept as int for looks
        df_filtered['Mean Enhanced SI-SDR [dB]'] = df_filtered['Mean Enhanced SI-SDR [dB]'].round(1)
        df_filtered['Mean Improvement [dB]'] = df_filtered['Mean Improvement [dB]'].round(1)

        # 3. Create a figure
        fig, ax = plt.subplots(figsize=(12, len(df_filtered) * 0.5 + 1)) 
        ax.axis('off') 

        # 4. Create the table
        tbl = ax.table(
            cellText=df_filtered.values, 
            colLabels=df_filtered.columns, 
            cellLoc='center', 
            loc='center'
        )

        # 5. Style the table
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.scale(1.2, 1.5) # Increased row height for better readability

        plt.title(title_text, fontsize=16, fontweight='bold', pad=20)

        # Style the header
        for (row, col), cell in tbl.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#2c3e50') # Slightly different dark grey/blue
            
        # 6. Save the image
        plt.savefig(output_png_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Filtered Table image saved to: {output_png_path}")
        
    import matplotlib
    matplotlib.use('Agg')

    # ---- si-sdr distribution across all examples and all arrays ----
    # 1. Convert lists of tensors/values to clean numpy arrays
    noisy_arr = np.array([v.item() if torch.is_tensor(v) else v for v in si_sdr_noisy])
    enhanced_arr = np.array([v.item() if torch.is_tensor(v) else v for v in si_sdr_enhanced])

    noisy_arr = master_si_sdr_noisy.flatten()
    enhanced_arr = master_si_sdr_enhanced.flatten()

    # 2. Create a DataFrame for easy manipulation
    df = pd.DataFrame({
        'Noisy_SI_SDR': noisy_arr,
        'Enhanced_SI_SDR': enhanced_arr,
        'Improvement': enhanced_arr - noisy_arr
    })

    # 3. Define the bins: [i, i+1)
    # We find the min and max integer range
    min_bin = int(np.floor(noisy_arr.min()))
    max_bin = int(np.ceil(noisy_arr.max()))
    bins = np.arange(min_bin, max_bin + 1, 1)

    # 4. Group by Noisy SI-SDR bins
    df['Bin'] = pd.cut(df['Noisy_SI_SDR'], bins=bins, right=False)

    # 5. Calculate statistics for each group
    summary = df.groupby('Bin', observed=True).agg({
        'Noisy_SI_SDR': ['count', 'mean'],
        'Enhanced_SI_SDR': 'mean',
        'Improvement': 'mean'
    }).reset_index()

    # Rename columns for clarity
    summary.columns = [
        'Bin [dB]', 
        'Sample Count', 
        'Mean Noisy SI-SDR [dB]', 
        'Mean Enhanced SI-SDR [dB]', 
        'Mean Improvement [dB]'
    ]

    # 6. Save and Display
    save_file = os.path.join(base_path, 'si-sdr distribution across examples and arrays.csv')
    summary.to_csv(save_file, index=False)
    print("Performance summary saved to 'si-sdr distribution across examples and arrays.csv'")
    print("\n--- Performance Summary Table ---")
    print(summary.to_string(index=False))

    # Example Usage:
    csv_file = os.path.join(base_path, 'si-sdr distribution across examples and arrays.csv')
    png_file = os.path.join(base_path, 'si-sdr distribution across examples and arrays.png')

    if os.path.exists(csv_file):
        csv_to_filtered_table_image(csv_file, png_file, title_text = 'si-sdr distribution (ASM to network) across examples and arrays')

    # ---- si-sdr distribution across all examples and avareged across arrays ----

    final_mean_si_sdr_noisy = np.mean(master_si_sdr_noisy, axis=0)      # Shape: (300,)
    final_mean_si_sdr_enhanced = np.mean(master_si_sdr_enhanced, axis=0)  # Shape: (300,)

    noisy_arr = final_mean_si_sdr_noisy
    enhanced_arr = final_mean_si_sdr_enhanced

    # 2. Create a DataFrame for easy manipulation
    df = pd.DataFrame({
        'Noisy_SI_SDR': noisy_arr,
        'Enhanced_SI_SDR': enhanced_arr,
        'Improvement': enhanced_arr - noisy_arr
    })

    # 3. Define the bins: [i, i+1)
    # We find the min and max integer range
    min_bin = int(np.floor(noisy_arr.min()))
    max_bin = int(np.ceil(noisy_arr.max()))
    bins = np.arange(min_bin, max_bin + 1, 1)

    # 4. Group by Noisy SI-SDR bins
    df['Bin'] = pd.cut(df['Noisy_SI_SDR'], bins=bins, right=False)

    # 5. Calculate statistics for each group
    summary = df.groupby('Bin', observed=True).agg({
        'Noisy_SI_SDR': ['count', 'mean'],
        'Enhanced_SI_SDR': 'mean',
        'Improvement': 'mean'
    }).reset_index()

    # Rename columns for clarity
    summary.columns = [
        'Bin [dB]', 
        'Sample Count', 
        'Mean Noisy SI-SDR [dB]', 
        'Mean Enhanced SI-SDR [dB]', 
        'Mean Improvement [dB]'
    ]

    # 6. Save and Display
    save_file = os.path.join(base_path, 'si-sdr distribution across examples and averaged over arrays.csv')
    summary.to_csv(save_file, index=False)
    print("Performance summary saved to 'si-sdr distribution across examples and averaged over arrays.csv'")
    print("\n--- Performance Summary Table ---")
    print(summary.to_string(index=False))

    # Example Usage:
    csv_file = os.path.join(base_path, 'si-sdr distribution across examples and averaged over arrays.csv')
    png_file = os.path.join(base_path, 'si-sdr distribution across examples and averaged over arrays.png')

    if os.path.exists(csv_file):
        csv_to_filtered_table_image(csv_file, png_file, title_text = 'si-sdr distribution (ASM to network) across examples and averaged over arrays')

# scipy.io.savemat("result1.mat", {"noisy": y, "enhanced": s_hat, "clean": s1})
# 
# save_dir = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/recordings/"  # or any desired folder path
# save_dir = os.path.join("/gpfs0/bgu-br/users/tatarjit/speech-enhancement/recordings/", test_type)
# os.makedirs(save_dir, exist_ok=True)  # make sure the folder exists

# scipy.io.wavfile.write(os.path.join(save_dir, "clean.wav"), 16000, s1.cpu().numpy().astype('float32'))
# scipy.io.wavfile.write(os.path.join(save_dir, "enhanced.wav"), 16000, s_hat.cpu().numpy().astype('float32'))
# scipy.io.wavfile.write(os.path.join(save_dir, "noisy.wav"), 16000, y.cpu().numpy().astype('float32'))

#runai-cmd --name test-dropout  -g 0.3 --cpu-limit 10 -- "conda activate venv && python /gpfs0/bgu-br/users/tatarjit/speech-enhancement/test_SH_FT_JNF.py"

#runai-bgu submit python -n ambidrop-test -c 20 -m 40G -g 1 --conda venv -- "python /gpfs0/bgu-br/users/tatarjit/speech-enhancement/test_SH_FT_JNF.py"
