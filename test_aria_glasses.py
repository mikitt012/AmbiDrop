# =========================
# Standard library imports
# =========================
import logging
import os
from datetime import datetime
from math import factorial, pi, sqrt

# =========================
# Third-party imports
# =========================
import h5py
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from pesq import pesq
from pystoi import stoi
from scipy.io import loadmat, savemat
from scipy.linalg import svd
from scipy.signal import fftconvolve
from scipy.special import lpmv
import soundfile as sf
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import scipy.io
from scipy.signal import resample_poly
import pandas as pd
import re

# from ASM.asm import asm as ASM
from scipy.special import sph_harm, spherical_jn, spherical_yn

import pickle
import sofar
from scipy.signal import butter, lfilter
from scipy import signal as sig

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

def pad_to_length(signal, target_length):
    """
    Pads a complex tensor (C, T) along time dim to target_length with zeros.
    """
    C, T = signal.shape
    if T >= target_length:
        return signal[:, :target_length]
    else:
        pad_size = target_length - T
        pad = torch.zeros(C, pad_size, dtype=signal.dtype, device=signal.device)
        return torch.cat([signal, pad], dim=1)

def preprocess_single_example(
    folder_path,
    V,
    th,
    ph,
    nfft,
    fs,
    cancel_nose,
    train: bool,
    max_length_sec: int = 6,
    sample_rate: int = 16000,
    device: str = "cpu",
    ):
    """
    Process one example folder (ex_XXX).

    Expected files inside folder_path:
        - p.wav
        - pDirect.wav
        - anm.mat (with keys: 'anmt_array', 'anmtDirect')

    Returns:
        noisy_tf_mic   : (T, F, 2*C_mic)   complex STFT split into real/imag
        clean_time_mic : (T,) or (C_mic, T) depending on squeeze
        noisy_tf_anm   : (T, F, 2*C_anm)
        clean_time_anm : (T,) or (C_anm, T)
    """

    array_file = os.path.join(folder_path, "p.wav")
    direct_file = os.path.join(folder_path, "s.wav")

    with open(os.path.join(folder_path, "best_shift.txt"), "r") as f:
        best_shift = int(f.read())

    noisy_speech_mic, fs_rec = sf.read(array_file)   # (T, C_mic)
    noisy_speech_mic = resample_poly(noisy_speech_mic, up=1, down=3, axis=0)
    noisy_speech_mic = noisy_speech_mic.T       # (C_mic, T)
    # noisy_speech_mic = noisy_speech_mic / np.max(np.abs(noisy_speech_mic))

    def lowpass_single_channel(data, cutoff_hz, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff_hz / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return lfilter(b, a, data)

    # Assume your audio is: (Channels, Samples) -> (7, 16000)
    # Let's say we want to "muffle" Channel 0 at 4kHz
    # fs = 16000
    # cutoff = 1500
    # noisy_speech_mic[1, :] = lowpass_single_channel(noisy_speech_mic[1, :], cutoff, fs)
    # noisy_speech_mic[1, :] = 0
    # all_indices = np.arange(noisy_speech_mic.shape[0])
    # keep_indices = all_indices[all_indices != 1]
    # mic_to_ASM = noisy_speech_mic[keep_indices, :]
    # V_to_ASM = V[keep_indices, :, :]
    # V = V_to_ASM

    # Load Clean (16kHz)
    clean_speech_mic, fs_clean = sf.read(direct_file)  # (T, C_mic)
    # # UPSAMPLE clean from 16kHz to 48kHz to match the rest of the pipeline
    # clean_speech_mic = resample_poly(clean_speech_mic, up=fs_rec, down=fs_clean, axis=0)
    clean_speech_mic = clean_speech_mic.T              # (C_mic, T)

    if cancel_nose:
        num_total_ch = noisy_speech_mic.shape[0]
        target_indices = np.array([1])
        noisy_speech_mic_zeroed = noisy_speech_mic.copy()
        noisy_speech_mic_zeroed[target_indices, :] = 0
        V_to_ASM = V.copy()

        keep_indices = np.setdiff1d(np.arange(num_total_ch), target_indices)
        mic_to_ASM = noisy_speech_mic[np.sort(keep_indices), :]
        V_to_ASM = V[keep_indices, :, :]
        V = V_to_ASM

        noisy_speech_anm = array_ambisonics_time(mic_to_ASM, V, th, ph, nfft, fs, N=2)
        noisy_speech_mic = noisy_speech_mic_zeroed

    else:
        noisy_speech_anm = array_ambisonics_time(noisy_speech_mic, V, th, ph, nfft, fs, N=2)

    # noisy_speech_mic = resample_poly(noisy_speech_mic, up=1, down=3, axis=1)

    # ---- Convert to torch complex ----
    noisy_speech_mic = torch.from_numpy(noisy_speech_mic).to(torch.complex64).to(device)
    clean_speech_mic = torch.from_numpy(clean_speech_mic).to(torch.complex64).to(device)
    noisy_speech_anm = torch.from_numpy(noisy_speech_anm).to(torch.complex64).to(device)

    # ---- STFT parameters ----
    n_fft = 512
    hop = 256
    win_len = 512

    # n_fft = 662
    # hop = 331
    # win_len = 662

    win = torch.hamming_window(window_length=win_len, device=noisy_speech_mic.device)

    # STFT shape: (C, F, T) -> transpose to (T, F, C)
    noisy_tf_mic = torch.stft(
        noisy_speech_mic,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win_len,
        window=win,
        center=True,
        normalized=False,
        return_complex=True,
    ).transpose(0, 2)

    noisy_tf_anm = torch.stft(
        noisy_speech_anm,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win_len,
        window=win,
        center=True,
        normalized=False,
        return_complex=True,
    ).transpose(0, 2)

    # Keep only positive freqs
    noisy_tf_mic = noisy_tf_mic[:, : (n_fft // 2 + 1), :]
    noisy_tf_anm = noisy_tf_anm[:, : (n_fft // 2 + 1), :]

    # ---- Normalize magnitudes separately for mic and anm ----
    max_val_mic = noisy_tf_mic.abs().max().item()
    if max_val_mic == 0:
        max_val_mic = 1.0
    noisy_tf_mic = noisy_tf_mic / max_val_mic
    clean_speech_mic = clean_speech_mic / max_val_mic

    max_val_anm = noisy_tf_anm.abs().max().item()
    if max_val_anm == 0:
        max_val_anm = 1.0
    noisy_tf_anm = noisy_tf_anm / max_val_anm

    # ---- Split real/imag along channel dimension ----
    noisy_tf_mic = torch.cat((noisy_tf_mic.real, noisy_tf_mic.imag), dim=2)  # (T, F, 2*C_mic)
    noisy_tf_anm = torch.cat((noisy_tf_anm.real, noisy_tf_anm.imag), dim=2)  # (T, F, 2*C_anm)

    # ---- Time-domain clean (float) ----
    clean_time_mic = clean_speech_mic.float()

    # If you really want 1-D time for a single channel:
    # clean_time_mic = clean_time_mic.squeeze(0)
    # clean_time_anm = clean_time_anm.squeeze(0)

    return noisy_tf_anm, noisy_tf_mic, clean_time_mic, best_shift

def load_noisy_only(folder_path: str):
    """
    Load only the noisy time-domain microphone recording (p.wav)
    from a single example folder ex_XXX.

    Returns:
        noisy (Tensor or ndarray): shape (C, T)
                                   (C = number of channels)
    """

    array_file = os.path.join(folder_path, "p.wav")
    if not os.path.exists(array_file):
        raise FileNotFoundError(f"Missing noisy WAV file: {array_file}")

    # ---- Load audio from p.wav ----
    noisy, sr = sf.read(array_file)      # shape (T, C) OR (T,) if mono

    # Convert mono -> (1, T)
    if noisy.ndim == 1:
        noisy = noisy[None, :]           # (1, T)
    else:
        noisy = noisy.T                  # (C, T)

    return noisy, sr

def swap_noisy_channels(noisy: torch.Tensor, clean_channel_idx: int = 0) -> torch.Tensor:
    """
    Given a noisy STFT tensor (T, F, 2*C) with real/imag concatenated along the last dim,
    swap channel 0 with channel `clean_channel_idx` in both real and imaginary parts.

    Args:
        noisy (torch.Tensor): shape (T, F, 2*C)
        clean_channel_idx (int): index of channel to swap with channel 0

    Returns:
        torch.Tensor: noisy with swapped channels, same shape as input
    """
    # Ensure float tensor (not strictly required but often useful)
    noisy = noisy.float()

    T, F, total_C = noisy.shape
    C = total_C // 2  # number of complex channels

    if clean_channel_idx == 0:
        # nothing to do
        return noisy

    # Split real and imag
    noisy_real = noisy[..., :C].clone()
    noisy_imag = noisy[..., C:].clone()

    # Swap real channels
    noisy_real[..., 0], noisy_real[..., clean_channel_idx] = (
        noisy_real[..., clean_channel_idx].clone(),
        noisy_real[..., 0].clone(),
    )

    # Swap imag channels
    noisy_imag[..., 0], noisy_imag[..., clean_channel_idx] = (
        noisy_imag[..., clean_channel_idx].clone(),
        noisy_imag[..., 0].clone(),
    )

    # Concatenate back
    noisy_swapped = torch.cat([noisy_real, noisy_imag], dim=-1)

    return noisy_swapped

def select_clean_channel(clean_time_mic: torch.Tensor, idx: int = 0) -> torch.Tensor:
    """
    Select a target channel from clean_time_mic.
    
    Args:
        clean_time_mic (Tensor): shape (C, T), (1, T), or (T,)
        idx (int): which channel to select (only used if multichannel)
    
    Returns:
        Tensor: shape (T,) or (1, T) depending on original shape
    """
    # If 1D -> already single channel
    if clean_time_mic.ndim == 1:
        return clean_time_mic.unsqueeze(0)
    
    # If shape is (1, T)
    if clean_time_mic.ndim == 2 and clean_time_mic.shape[0] == 1:
        return clean_time_mic.unsqueeze(0)   # or return as is
    
    # Multichannel (C, T)
    C, T = clean_time_mic.shape
    if idx < 0 or idx >= C:
        raise IndexError(f"idx={idx} is out of range for clean_time_mic with C={C}")

    # Return only that channel
    return clean_time_mic[idx, :].unsqueeze(0)

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

def plot_ir_diagnostics(ir, fs=48000):
    """
    ir: numpy array or torch tensor of shape (CH, T) or (CH, T, Q)
    If (CH, T, Q), we'll squeeze Q or pick the first source position.
    """
    if torch.is_tensor(ir):
        ir = ir.detach().cpu().numpy()
    
    # Handle (CH, T, Q) by picking first source position
    if ir.ndim == 3:
        ir = ir[:, :, 0]
        
    num_channels = ir.shape[0]
    t = np.arange(ir.shape[1]) / fs * 1000  # Time in ms
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # --- 1. Time Domain Plot ---
    for ch in range(num_channels):
        axes[0].plot(t, ir[ch, :], label=f'Ch {ch+1}')
    
    axes[0].set_title("Impulse Response - Time Domain")
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right', ncol=2)

    # --- 2. Frequency Domain Plot (Magnitude) ---
    n_fft = 512
    freqs = np.linspace(0, fs/2, n_fft//2 + 1)
    
    for ch in range(num_channels):
        # Compute FFT and get first 257 bins
        V_ch = np.fft.fft(ir[ch, :], n=n_fft)
        mag_db = 20 * np.log10(np.abs(V_ch[:n_fft//2 + 1]) + 1e-12)
        axes[1].plot(freqs, mag_db, label=f'Ch {ch+1}')
    
    axes[1].set_title("Frequency Response - Magnitude (dB)")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Magnitude (dB)")
    axes[1].set_xscale('log') # Log scale helps see the 200Hz-1kHz range
    axes[1].set_ylim([-60, 10]) # Standard range for audio IRs
    axes[1].grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("ir_diagnostic_plot.png")
    plt.show()
    
def sh2(N, theta, phi):
    """
    Compute spherical harmonics up to order N.

    Args:
        N (int): maximum order
        theta (array): colatitude angles in radians (0 at north pole)
        phi (array): azimuth angles in radians

    Returns:
        Y (np.ndarray): shape ((N+1)**2, len(theta)), complex values
    """
    theta = np.atleast_1d(theta)
    phi = np.atleast_1d(phi)
    
    if len(theta) != len(phi):
        raise ValueError("Lengths of theta and phi must be equal!")
    
    L = len(theta)
    Y = [np.sqrt(1/(4*pi)) * np.ones(L, dtype=complex)]  # n=0 term

    j = 1j  # complex constant

    for n in range(1, N+1):
        # positive m
        Y1 = []
        for m in range(0, n+1):
            # normalization
            a = sqrt((2*n+1)/(4*pi) * factorial(n-m)/factorial(n+m))
            Pnm = lpmv(m, n, np.cos(theta))  # associated Legendre
            Ynm = a * Pnm * np.exp(j*m*phi)
            Y1.append(Ynm)
        Y1 = np.vstack(Y1)  # shape (n+1, L)
        
        # negative m
        Y2 = []
        for m in range(-n, 0):
            # (-1)^m * conjugate of positive m
            Ynm = (-1)**m * np.conj(Y1[-m-1, :])
            Y2.append(Ynm)
        if Y2:
            Y2 = np.vstack(Y2)
            Y_stack = np.vstack([Y2, Y1])
        else:
            Y_stack = Y1

        # append to Y
        Y.append(Y_stack)

    # stack all n
    Y = np.vstack(Y)
    return Y

def compute_spherical_harmonics_matrix(N, theta, phi):
    # Computes a spherical harmonics matrix.

    # Parameters:
    #  N (int):         Maximum degree of spherical harmonics
    #  theta (ndarray): Polar angles (in radians), shape (num_samples,)
    #  phi (ndarray):   Azimuthal angles (in radians), shape (num_samples,)

    # Returns:
    #   Y_matrix (ndarray): A complex matrix of shape (num_samples, (N+1)^2),
    #                         where each column corresponds to a spherical harmonic.

    # make sure phi in range [0,2pi] and theta in range [0, pi]
    assert(phi.min() >= 0.0)

    num_samples = phi.size  # Number of samples
    num_harmonics = (N + 1) ** 2  # Total number of harmonics

    # Initialize the spherical harmonics matrix
    Y_matrix = np.zeros((num_harmonics, num_samples), dtype=complex)

    index = 0
    for n in range(N + 1):
        for m in range(-n, n + 1):
            # Scipy's sph_harm is already orthonormal (matches MATLAB 0.282)
            Y_matrix[index, :] = sph_harm(m, n, phi, theta)
            index += 1

    # REMOVE OR COMMENT OUT THIS BLOCK:
    # for nm in range (Y_matrix.shape[0]):
    #    Y_matrix[nm, :] /= 4.37 
    
    return Y_matrix

def svd_inversion(A, b, snr_lin=1000):
    """
    Performs the SVD-based inversion matching the MATLAB logic.
    A: Steering matrix (v_k), shape (M, Q)
    b: Target vector (Ynm), shape (Q,)
    """
    # 1. Form the normal matrix (M x M)
    lam = 1.0 / snr_lin
    # lam = 10

    # s = np.linalg.svd(A, compute_uv=False)
    # sigma_max = s[0] if s.size > 0 else 1.0
    # lam = 0.1 * sigma_max

    # cond_number = np.linalg.cond(A @ A.conj().T)
    # print(f"Condition Number: {cond_number:.2f}")
    # lam = 0.000001 * cond_number

    mat_to_inv = (A @ A.conj().T) + lam * np.eye(A.shape[0])
    
    # 2. SVD
    U, s, Vh = np.linalg.svd(mat_to_inv)
    
    # 3. Thresholding (MATLAB logic)
    # tol = 1 + M * eps * norm
    tol = 1.0 + A.shape[0] * np.finfo(float).eps * s[0]
    
    # 4. Invert
    s_inv = np.zeros_like(s)
    mask = s > tol
    s_inv[mask] = 1.0 / s[mask]
    
    # 5. Reconstruct Inverse and solve
    inv_mat = Vh.conj().T @ np.diag(s_inv) @ U.conj().T
    return inv_mat @ A @ b

def array_ambisonics_time_domain(p, c_ASM, filt_samp=512):
    """
    p  : (M, T) microphone signals
    c_ASM : (harmonics, F, M) - assumed positive frequencies only
    filt_samp : length of the resulting time-domain filter
    """
    T = p.shape[1]
    N_mic = p.shape[0]
    num_harmonics = c_ASM.shape[0]
    anmt_array = np.zeros((num_harmonics, T), dtype=np.float32)

    for j_idx in range(num_harmonics):
        # 1. Extract and transpose to (M, F) to match MATLAB 'c_f = squeeze(c_ASM(:, j, :))'
        # MATLAB indices are (M, harmonics, F), Python is (harmonics, F, M)
        c_f = c_ASM[j_idx, :, :].T
        # if j_idx == 0:
        #     mic_idx = 0 
        #     freq_axis = np.arange(c_f.shape[1])

        #     plt.figure(figsize=(12, 5))

        #     # Subplot 1: Magnitude
        #     plt.subplot(1, 2, 1)
        #     plt.plot(freq_axis, np.abs(c_f[mic_idx, :]), label=f'Mic {mic_idx+1}')
        #     plt.title(f'Magnitude of c_f (Harmonic {j_idx+1})')
        #     plt.xlabel('Frequency Bin')
        #     plt.ylabel('|c_f|')
        #     plt.grid(True)
        #     plt.legend()
        #     plt.savefig("c_f.png", dpi=250)

        # 2. Replicate MATLAB padding: c_f(:, end+1 : filt_samp) = 0
        # Note: In irfft, the output length is usually 2*(F-1). 
        # To get exactly filt_samp, we specify it in the function call.
        
        # 3. IFFT with Symmetry
        # np.fft.irfft is the direct equivalent of MATLAB's 'symmetric' flag 
        # when provided with only the positive half of the spectrum.
        c_time = np.fft.irfft(c_f, n=filt_samp, axis=1) # (M, filt_samp)

        # 4. Circular Shift: circshift(c_time, filt_samp / 2, 2)
        c_time_cs = np.roll(c_time, filt_samp // 2, axis=1)

        # 5. Mirror the tail: [c_time_cs(:, 1), c_time_cs(:, end:-1:2)]
        # MATLAB index 1 is Python index 0.
        first_col = c_time_cs[:, [0]]
        tail_reversed = c_time_cs[:, :0:-1] 
        c_time_filter = np.concatenate([first_col, tail_reversed], axis=1)

        # 6. Summed Convolution
        tmp = np.zeros(T, dtype=np.float64)
        for m in range(N_mic):
            full_conv = np.convolve(p[m, :].astype(np.float64), 
                                        c_time_filter[m, :].astype(np.float64), 
                                        mode='full')
                
            start_idx = filt_samp // 2 
            tmp += full_conv[start_idx : start_idx + T]

        anmt_array[j_idx, :] = tmp

    return anmt_array

def array_ambisonics_time(p, V, th, ph, nfft, fs, N):
    """
    Python version of the MATLAB array->Ambisonics routine.
    p  : (M, T) microphone signals
    V  : (M, F, Q) steering vectors
    th : (Q,) or (1, Q)
    ph : (Q,) or (1, Q)
    harmonics: 0-based indices of SH coefficients to compute
    """

    def _calculate_coefficients(V,N,th,ph,plot):
        V = V.T
        Y = compute_spherical_harmonics_matrix(N, th, ph)
        # Y_yo = np.load('/gpfs0/bgu-br/users/tatarjit/speech-enhancement/datasets/aria_ds/Y.npy')
        # Y_yo = Y_yo.T
        # Y = Y_yo
        cnm = np.zeros(((N+1)**2, V.shape[1], V.shape[2]), dtype=np.complex128)
        from ASM.tikhonov import tikhonov
        # from ASM.utils import reconstruct_frequency_sh_spectrum_full
        for nm in range((N+1)**2):
            for f in range(V.shape[1]):
                # cnm[nm,f] = tikhonov(A=V[:, f, :].conj(), b=Y[nm, :], lam=1e-3)
                # cnm[nm,:] = np.array([np.linalg.lstsq(V[:, f, :].conj(), Y[nm, :], rcond=None)[0] for f in range(V.shape[1])])

                cnm[nm,f] = tikhonov(A=V[:, f, :].conj(), b=Y[nm, :])
                # cnm[nm, f, :] = svd_inversion(A=V[:, f, :].T, b=Y[nm, :], snr_lin=1000)
        # cnm = reconstruct_frequency_sh_spectrum_full(cnm, freq_axis=1, nm_axis=0, n_fft=2*(V.shape[1] - 1))

        # 1. Handle DC (Frequency index 0) for all channels
        # cnm[1:, 0, :] = 0.0                # Zero out higher orders for all channels
        # cnm[0, 0, :] = cnm[0, 0, :].real    # Force omni (n=0, m=0) to be real for all channels
        # # 2. Handle Nyquist (Frequency index F//2) for all channels
        # nyq_idx = cnm.shape[1] - 1
        # cnm[1:, nyq_idx, :] = 0.0
        # cnm[0, nyq_idx, :] = cnm[0, nyq_idx, :].real

        # from ASM.validate import is_signal_frequency_sh_valid
        # assert is_signal_frequency_sh_valid(cnm, freq_axis=1, sh_axis=0)


        # cnm = np.load('/gpfs0/bgu-br/users/tatarjit/speech-enhancement/datasets/aria_ds/cnm.npy')
        # cnm = cnm.transpose(1, 2, 0)
        # nfft = 332
        # cnm = cnm[:, :nfft//2 + 1, :]
        
        if plot:
            mse = calculate_error(cnm, Y, V)
            n_fft = 512
            pos_freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
            plot_nmse(mse, pos_freqs, save_path="nmse_channels_measuredV.png")

        return cnm

    def calc_ambisonics(mic_signals, Cnm=None, Cnm_domain='frequency'):
        if Cnm_domain == 'frequency':
            # convert to time
            Cnm = np.fft.ifft(np.conj(Cnm), axis=1) # nm x T x ch
        from ASM.utils import convolve_and_sum

        x = mic_signals[np.newaxis, :, :]
        rec_amb = convolve_and_sum(x, Cnm.transpose(0, 2, 1), "time", "time") # input is nm x ch x T
        # output is 1 x nm x T
        anmt_array = np.squeeze(rec_amb, axis=0)

        return anmt_array

    def calculate_error(c, Y, V):
        # c - nm x pos_F x M
        # V - Q x pos_F x M
        # Y - nm x Q

        # mse = np.zeros((c.shape[0], V.shape[1]))
        # for nm in range(mse.shape[0]):
        #     for f in range(V.shape[1]):
        #         tmp = np.linalg.norm(np.conj(c[nm, f, :].T) @ V[:, f, :].T - Y[nm, :].conj())
        #         mse[nm, f] = np.square(tmp / np.linalg.norm(Y[nm, :]))
        #         # mse[nm, f] = tmp
        
        cnm = c.transpose(2, 0, 1)
        sm = V.transpose(2, 0, 1)

        cnm = cnm.transpose(2, 1, 0)  # (F, nm, M)
        sm = sm.transpose(2, 0, 1)    # (F, M, Q)

        raw_err = cnm.conj() @ sm - Y[np.newaxis, ...].conj()  # (F, nm, Q)
        nominator = np.square(np.linalg.norm(raw_err, ord=2, axis=2))  # (F, nm)
        denominator = np.square(np.linalg.norm(Y, ord=2, axis=1))       # (nm,)
        mse = nominator.T / denominator[..., np.newaxis]               # (nm, F)
        return mse

    def _calculate_coefficients_like_matlab(V, N, th, ph, SNR_lin):
        """
        Port of MATLAB:
            c_ASM(:, j, f) = inv(V_f V_f^H + lambda I) V_f Ynm
        with SVD thresholding.

        Args:
            V: np.ndarray with shape (M, F, Q)  (preferred)
            N: SH order
            th, ph: direction grids (length Q)
            SNR_lin: linear SNR used in lambda = 1/SNR_lin

        Returns:
            cnm: np.ndarray with shape (H, F, M)  where H=(N+1)^2
                (this matches your later usage cnm[nm, f] giving length M)
        """
        V = np.asarray(V)
        M, F, Q = V.shape
        H = (N + 1) ** 2

        # MATLAB: Y = sh2(N,th,ph).'   => Q x H
        Y = compute_spherical_harmonics_matrix(N, th, ph)
        Y = np.asarray(Y)

        # Make Y shape = (Q, H)
        if Y.shape == (H, Q):
            Y = Y.T
        elif Y.shape != (Q, H):
            raise ValueError(f"Unexpected Y shape {Y.shape}. Expected (Q,H)=({Q},{H}) or (H,Q)=({H},{Q}).")

        lam = 1.0 / float(SNR_lin)  # lambda = 1/SNR_lin
        eps = np.finfo(np.float64).eps

        # We'll store cnm as (H, F, M) so cnm[nm, f, :] is length-M (mic weights)
        cnm = np.zeros((H, F, M), dtype=np.complex128)

        I_M = np.eye(M, dtype=np.complex128)

        for nm in range(H):
            Ynm = Y[:, nm]  # (Q,)

            for f in range(F):
                v_k = V[:, f, :]  # (M, Q)

                # MATLAB: mat_to_inv = (v_k*v_k') + lambda*eye(M)
                mat_to_inv = (v_k @ v_k.conj().T) + lam * I_M  # (M, M)

                # MATLAB tol:
                # tol_inv = 1 + max(size(mat))*eps(norm(mat))
                # (eps(norm(mat)) ~ eps*norm(mat) in MATLAB double)
                maxdim = max(mat_to_inv.shape)
                tol_inv = 1.0 + (maxdim * eps * np.linalg.norm(mat_to_inv))

                # MATLAB: [U,S,V] = svd(mat_to_inv)
                U, s, Vh = np.linalg.svd(mat_to_inv, full_matrices=False)

                # MATLAB:
                # Sig(Sig <= tol_inv)=0; Sig(Sig ~= 0)=1./Sig(Sig ~= 0)
                s_inv = np.zeros_like(s)
                keep = s > tol_inv
                s_inv[keep] = 1.0 / s[keep]

                # inv_mat = V * diag(s_inv) * U'
                inv_mat = (Vh.conj().T @ (s_inv[:, None] * U.conj().T))  # (M, M)

                # c_ASM(:, j, f) = inv_mat * v_k * Ynm
                # => (M,M)(M,Q)(Q,) -> (M,)
                cnm[nm, f, :] = inv_mat @ (v_k @ Ynm)

        return cnm

    plot = False

    cnm = _calculate_coefficients(V,N,th,ph,plot)
    nfft = 512

    # cnm = np.load('/gpfs0/bgu-br/users/tatarjit/speech-enhancement/datasets/aria_ds/cnm.npy')
    # cnm = cnm.transpose(1, 2, 0) # nm x F x ch
    # nfft = 332
    # cnm = cnm[:, :nfft//2 + 1, :]

    anmt_array = array_ambisonics_time_domain(p, cnm, filt_samp = nfft)
    # anmt_array = calc_ambisonics(p, Cnm=cnm, Cnm_domain='frequency')

    # anmt_array = resample_poly(anmt_array, up=1, down=3, axis=1)

    return anmt_array

import matplotlib
matplotlib.use("Agg")  # important for cluster (no display)
import matplotlib.pyplot as plt

def plot_nmse(mse, freqs, save_path="nmse_plot.png"):
    nmse_db = 10 * np.log10(mse + 1e-12)

    # Distinct style set
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728',
              '#17becf','#8c564b','#e377c2','#7f7f7f','#bcbd22']
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
    markers = ['o','s','d','^','v','<','>','*','x','+']

    plt.figure(figsize=(9,4))
    for ch in range(nmse_db.shape[0]):
        plt.plot(
            freqs,
            nmse_db[ch],
            color=colors[ch % len(colors)],
            linestyle=line_styles[ch % len(line_styles)],
            marker=markers[ch % len(markers)],
            markevery=3,         # show marker every N points
            linewidth=2.0,
            markersize=6,
            label=f"Channel {ch+1}"
        )

    plt.xscale("log")
    plt.xlabel("Frequency (Hz)", fontsize=18)
    plt.ylabel(r'$\varepsilon_{\mathrm{Amb}}$ [dB]', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(bottom=-60)
    # plt.title("NMSE per Channel Across Frequency")
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.legend(loc="lower right", fontsize=12, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250)
    print(f"Saved plot to: {save_path}")

def sweep_alignment_sisdr(s1, s_hat, fs, t0, ranges, save_path="/gpfs0/bgu-br/users/tatarjit/speech-enhancement/sisdr_vs_shift.png"):
    """
    s1  : reference clean signal (1D)
    s_hat : enhanced signal (1D)
    fs  : sampling rate (Hz)
    t0  : initial coarse shift in *samples* (applied to s1)
    d   : sweep range in *samples* around t0 (from -d to +d)
    """

    s1 = np.asarray(s1, dtype=np.float64)
    s_hat = np.asarray(s_hat, dtype=np.float64)

    # 1) coarse alignment: shift s1 by t0 to the right (delay)
    if t0 > 0:
        s1_init = s1[t0:]
    elif t0 < 0:
        s1_init = s1[:t0]  # t0 negative
    else:
        s1_init = s1.copy()

    # Make lengths comparable
    T = min(len(s1_init), len(s_hat))
    s1_init = s1_init[:T]
    s_hat = s_hat[:T]

    # 2) sweep fine shifts from -d to +d
    shifts = np.concatenate([np.arange(start, stop + 1) for start, stop in ranges])
    # Ensure shifts are unique and sorted (optional but recommended)
    shifts = np.unique(shifts)

    sisdr_vals = []

    for i, k in enumerate(shifts):
        # Print k every 100 iterations (i = 0, 100, 200, ...)
        if i % 100 == 0:
            print(f"Processing shift index {i}: k = {k}")

        s1_k, s_hat_k = shifted_overlap(s1_init, s_hat, k)
        if len(s1_k) < 10:
            sisdr_vals.append(np.nan)
            continue
        # print(type(s1_k), getattr(s1_k, 'dtype', None))
        s1_k = torch.from_numpy(s1_k).to(torch.complex64).to(device)
        s_hat_k = torch.from_numpy(s_hat_k).to(torch.complex64).to(device)
        val = si_snr(s_hat_k.unsqueeze(0), s1_k.unsqueeze(0), debug = False)
        sisdr_vals.append(val.detach().cpu().item())
        
    sisdr_vals = np.array(sisdr_vals)

    # # 3) plot SI-SDR vs shift
    # plt.figure(figsize=(8, 4))
    # plt.plot(shifts, sisdr_vals, marker='o')
    # plt.axvline(0, color='k', linestyle='--', alpha=0.5)
    # plt.xlabel("Additional shift (samples) relative to t0")
    # plt.ylabel("SI-SDR (dB)")
    # plt.title("SI-SDR vs relative shift of s1")
    # plt.grid(True)

    # # secondary x-axis in milliseconds
    # ax = plt.gca()
    # ax2 = ax.secondary_xaxis('top', functions=(
    #     lambda k: k * 1000 / fs,          # samples -> ms
    #     lambda ms: ms * fs / 1000         # ms -> samples
    # ))
    # ax2.set_xlabel("Additional shift (ms)")

    # plt.tight_layout()
    # plt.savefig(save_path, dpi=200)
    # print(f"Saved SI-SDR sweep plot to {save_path}")

    # Also return best alignment info
    best_idx = np.nanargmax(sisdr_vals)
    best_shift = shifts[best_idx]
    best_sisdr = sisdr_vals[best_idx]
    print(f"Best extra shift: {best_shift} samples ({best_shift * 1000 / fs:.2f} ms), "
          f"SI-SDR = {best_sisdr:.2f} dB")

    return shifts, sisdr_vals, best_shift, best_sisdr

def align_with_best_shift(s1, y, s_hat, best_shift):
    """
    Align s1 to y and s_hat using best_shift (in samples),
    and return aligned s1, y, s_hat all with the same length.

    Convention:
      best_shift > 0 : s1 is delayed (shifted right)
      best_shift < 0 : s1 is advanced (shifted left)
    """
    s1 = np.asarray(s1)
    y = np.asarray(y)
    s_hat = np.asarray(s_hat)

    # 1) Make them the same initial length
    T = min(len(s1), len(y), len(s_hat))
    s1 = s1[:T]
    y = y[:T]
    s_hat = s_hat[:T]

    k = int(best_shift)

    if k > 0:
        # Delay s1: drop first k samples from s1
        # To keep alignment, drop last k samples from y and s_hat
        s1_new = s1[k:]
        y_new = y[:-k]
        s_hat_new = s_hat[:-k]

    elif k < 0:
        # Advance s1: negative shift
        k = -k
        # Drop last k samples from s1
        # Drop first k samples from y and s_hat
        s1_new = s1[:-k]
        y_new = y[k:]
        s_hat_new = s_hat[k:]

    else:  # k == 0
        s1_new = s1
        y_new = y
        s_hat_new = s_hat

    # 2) Final safety: enforce exactly same length
    L = min(len(s1_new), len(y_new), len(s_hat_new))
    s1_new = s1_new[:L]
    y_new = y_new[:L]
    s_hat_new = s_hat_new[:L]

    return s1_new, y_new, s_hat_new

def shifted_overlap(s1, s_hat, k):
    """
    Shift s1 by k samples (positive = delay s1),
    then return overlapping parts of s1_shifted and s_hat.
    """
    T = min(len(s1), len(s_hat))

    s1 = s1[:T]
    s_hat = s_hat[:T]

    if k > 0:
        # s1 delayed: compare s1[k:] with s_hat[:-k]
        s1_seg = s1[k:]
        s_hat_seg = s_hat[:-k]
    elif k < 0:
        k = -k
        # s1 advanced: compare s1[:-k] with s_hat[k:]
        s1_seg = s1[:-k]
        s_hat_seg = s_hat[k:]
    else:
        s1_seg = s1
        s_hat_seg = s_hat

    # Just in case k is too big
    L = min(len(s1_seg), len(s_hat_seg))
    return s1_seg[:L], s_hat_seg[:L]

class FT_JNF(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim):
        super().__init__()
        num_sh_channels = input_dim // 2

        #LSTM model
        self.LSTM1 = nn.LSTM(input_size=input_dim,hidden_size=hidden1_dim,num_layers=1,batch_first=True,bidirectional=True)
        self.LSTM2 = nn.LSTM(input_size=2*hidden1_dim,hidden_size=hidden2_dim,num_layers=1,batch_first=True,bidirectional=True)
        self.linear = nn.Linear(2*hidden2_dim,output_dim)

        self.hamming_window = torch.hamming_window(512, device=device)

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

    def forward(self,x):
        # print('before reshaping', x.shape)
        x, hidden = self.LSTM1(x.view(x.shape[0],x.shape[1],x.shape[2])) #TxFx512 
        # print('after reshaping', x.shape)
        x, hidden = self.LSTM2(x.permute(1,0,2)) #FxTx256
        x = self.linear(x) #FxTx2
        return x.permute(1,0,2) #TxFx2


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

class SH_FT_JNF(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim, 
                 drop_prob, max_drop, drop_probs, dropout):
        super().__init__()
        num_sh_channels = input_dim // 2
        if dropout == "SHChannelDropout":
            self.channel_dropout = SHChannelDropout(drop_prob=drop_prob, max_drop=max_drop)  # adjust max_drop depending on array type
        else:
            self.channel_dropout = PerChDropout(drop_probs=drop_probs)

        #LSTM model
        self.LSTM1 = nn.LSTM(input_size=input_dim,hidden_size=hidden1_dim,num_layers=1,batch_first=True,bidirectional=True)
        self.LSTM2 = nn.LSTM(input_size=2*hidden1_dim,hidden_size=hidden2_dim,num_layers=1,batch_first=True,bidirectional=True)
        self.linear = nn.Linear(2*hidden2_dim,output_dim)

        self.hamming_window = torch.hamming_window(512, device=device)

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

    def forward(self,x):
        # print('before reshaping', x.shape)
        x, hidden = self.LSTM1(x.view(x.shape[0],x.shape[1],x.shape[2])) #TxFx512 
        # print('after reshaping', x.shape)
        x, hidden = self.LSTM2(x.permute(1,0,2)) #FxTx256
        x = self.linear(x) #FxTx2
        return x.permute(1,0,2) #TxFx2


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

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    wandb.login()
    writer = SummaryWriter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    # ----- CONFIG -----
    AmbiDrop = True
    # data_root = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/datasets/aria_ds"
    # datas = ["mixed_data_1_5int"]
    data_root = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/datasets/nose_bridge_experiment/dataset"
    datas = ["not_blocked", "blocked"]
    scenarios = ["not_blocked", "blocked", "blocked_canceled", "blocked_hard", "blocked_hard_canceled"]
    # scenarios = ["not_blocked"]
    # cancel_nose = False
    # for data_type in datas:
    for scenario in scenarios:
        if scenario == "blocked_canceled":
            data_type = "blocked"
            cancel_nose = True
        elif scenario == "blocked_hard_canceled":
            data_type = "blocked_hard"
            cancel_nose = True
        else:
            data_type = scenario
            cancel_nose = False
        train = False
        max_length_sec = 6
        sample_rate = 16000
        closest_mic_idx = 3
        simulated_steering = True

        wandb_active = True

        base_dir = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/datasets/experiment_full_anm/utils"
        # --- 1. Load steering matrix: CH x F x Q ---
        if simulated_steering:
            steering_name = "simV"
            steer_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/datasets/experiment_full_anm/steering/Aria on rigid sphere (simulated).mat"
            steer_mat = loadmat(steer_path)
            V = steer_mat["V"]          # numpy array, shape (CH, F, Q)

            # --- 2. Load grid (theta, phi): 1 x Q ---
            grid_path = os.path.join(base_dir, "Lebvedev2702.mat")
            grid_mat = loadmat(grid_path)
            th = grid_mat["th"].squeeze()    # shape (Q,)
            ph = grid_mat["ph"].squeeze()    # shape (Q,)
            fs = 16000

            print("---- simulated ATF is chosen ----")

        else:
            steering_name = "measuredV"
            file_path = '/gpfs0/bgu-br/users/tatarjit/speech-enhancement/datasets/aria_ds/aria_atfs.pkl'

            with open(file_path, 'rb') as file:
                data = pickle.load(file)

            # Extract the requested items
            far_field = data['far_field']
            directions = data['directions']  # Contains (azimuth, elevation, distance)

            # Extract azimuth and elevation from the directions triplets
            # Assuming directions is a numpy array or list of triplets
            ph = directions[0, :]
            th = directions[1, :]
            ir = far_field.transpose(1, 0, 2) # shape (CH, T, Q)

            sofa = sofar.read_sofa("/gpfs0/bgu-br/users/tatarjit/speech-enhancement/datasets/aria_ds/aria_atfs_fixed.sofa")
            ir = sofa.Data_IR
            fs = sofa.Data_SamplingRate
            ir = ir.transpose(1, 2, 0) # shape (CH, T, Q)

            directions = sofa.SourcePosition.T
            ph = directions[0, :]
            th = directions[1, :]

            # ir = resample_poly(ir, up=1, down=3, axis=1)
            ir = torch.from_numpy(ir)

            n_fft = 332
            fs = 48000

            # plot_ir_diagnostics(ir)
            
            # 2. Define FFT size to get 257 bins
            ir_fft_full = torch.fft.fft(ir, n=n_fft, dim=1)

            V = ir_fft_full[:, :n_fft//2 + 1, :]

            freqs     = np.fft.fftfreq(n_fft, 1.0 / fs)
            pos_freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)

            V = V.detach().cpu().numpy()

            V_yo = np.load('/gpfs0/bgu-br/users/tatarjit/speech-enhancement/datasets/aria_ds/ATF.npy')
            V_yo = V_yo.transpose(0, 2, 1) # shape (CH, F, Q)
            V_yo = V_yo[:, :n_fft//2 + 1, :]

            V = V_yo

            print("---- measured ATF is chosen ----")

        # --- different dropout ---
        for t in range(0,1):
            if t == 0:
                drop_prob = 0.4
                max_drop = 3
                drop_probs = []
                dropout = "SHChannelDropout"
                checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2025-12-01_10-08-18.pt"
            if t == 1:
                drop_prob = 0.7
                max_drop = 7
                drop_probs = []
                dropout = "SHChannelDropout"
                checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2025-12-04_15-20-29.pt"
            if t == 2:
                drop_prob = 0.6
                max_drop = 7
                drop_probs = []
                dropout = "SHChannelDropout"
                checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2025-12-04_21-18-51.pt"
            if t == 3:
                drop_prob = 0.5
                max_drop = 6
                drop_probs = []
                dropout = "SHChannelDropout"
                checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2025-12-05_03-31-01.pt"
            if t == 4:
                drop_probs = [0, 0.55, 0.9, 0.6, 1, 1, 1, 1, 1] # th = -10 dB
                drop_prob = 0
                max_drop = 0
                dropout = "PerChDropout"
                checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2025-12-04_15-45-32.pt"
            if t == 5:
                drop_probs = [0, 0.35, 0.75, 0.2, 1, 1, 1, 1, 1] # th = -8.2 dB
                drop_prob = 0
                max_drop = 0
                dropout = "PerChDropout"
                checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2025-12-04_21-44-11.pt"
            if t == 6:
                drop_probs = [0, 0.15, 0.6, 0.2, 0.95, 1, 0.95, 1, 0.95] # th = -7 dB
                drop_prob = 0
                max_drop = 0
                dropout = "PerChDropout"
                checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2025-12-05_03-53-11.pt"
            if t == 7:
                drop_probs = [0, 0.15, 0.55, 0.15, 0.8, 1, 0.95, 1, 0.85] # th = -5.7 dB
                drop_prob = 0
                max_drop = 0
                dropout = "PerChDropout"
                checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2025-12-05_09-45-29.pt"
            if t == 8:
                drop_probs = [0, 0.1, 0.45, 0.15, 0.7, 1, 0.85, 1, 0.65] # th = -5 dB
                drop_prob = 0
                max_drop = 0
                dropout = "PerChDropout"
                checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2025-12-05_15-44-31.pt"
            if t == 9:
                drop_probs = [0, 0.1, 0.45, 0.1, 0.55, 1, 0.85, 1, 0.55] # th = -4.2 dB
                drop_prob = 0
                max_drop = 0
                dropout = "PerChDropout"
                checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2025-12-05_21-56-06.pt"
            if t == 10:
                drop_probs = [0, 0.1, 0.45, 0.1, 0.45, 1, 0.75, 1, 0.45] # th = -3.4 dB
                drop_prob = 0
                max_drop = 0
                dropout = "PerChDropout"
                checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2025-12-06_04-07-37.pt"
            if t == 11:
                drop_probs = [0, 0.05, 0.45, 0.05, 0.4, 0.95, 0.5, 0.95, 0.4] # th = -2.4 dB
                drop_prob = 0
                max_drop = 0
                dropout = "PerChDropout"
                checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2025-12-06_09-56-40.pt"
            if t == 12:
                drop_probs = [0, 0.05, 0.45, 0.05, 0.1, 0.75, 0.4, 0.75, 0.1] # th = -1.4 dB
                drop_prob = 0
                max_drop = 0
                dropout = "PerChDropout"
                checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2025-12-06_15-58-32.pt"
            if t == 13:
                drop_probs = [0, 0.05, 0.35, 0.05, 0, 0.4, 0, 0.3, 0] # th = 0 dB
                drop_prob = 0
                max_drop = 0
                dropout = "PerChDropout"
                checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2025-12-06_22-04-29.pt"
            if t == 14:
                drop_prob = 0.4
                max_drop = 7
                drop_probs = []
                dropout = "SHChannelDropout"
                checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2025-12-21_17-36-40.pt"
            if t == 15:
                drop_prob = 0.3
                max_drop = 6
                drop_probs = []
                dropout = "SHChannelDropout"
                checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2025-12-22_14-53-29.pt"
            if t == 16:
                drop_prob = 0.7
                max_drop = 3
                drop_probs = []
                dropout = "SHChannelDropout"
                checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2025-12-23_05-06-54.pt"
            if t == 17:
                drop_prob = 0
                max_drop = 0
                drop_probs = []
                dropout = "SHChannelDropout"
                checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2025-12-23_13-38-21.pt"


            if AmbiDrop:
                model_type = "AmbiDrop"
                ch_num = 18
                # checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2025-12-01_10-08-18.pt"
            else:
                model_type = "baseline"
                ch_num = 14
                checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/FT_JNF,2025-12-29_14-41-04.pt"
            
            smallnet = True

            if smallnet:
                if AmbiDrop:
                    # net = SH_FT_JNF(
                    #     input_dim=ch_num,
                    #     hidden1_dim=64,
                    #     hidden2_dim=64,
                    #     output_dim=2,
                    #     drop_prob=drop_prob,          # your original
                    #     max_drop=max_drop,
                    #     drop_probs=drop_probs,
                    #     dropout=dropout
                    # ).to(device)

                    net = SH_FT_JNF(
                        input_dim=ch_num,
                        hidden1_dim=16,
                        hidden2_dim=16,
                        output_dim=2,
                        drop_prob=drop_prob,          # your original
                        max_drop=max_drop,
                        drop_probs=drop_probs,
                        dropout=dropout
                    ).to(device)
                    checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/checkpoint_size_11074.pt"
                else:
                    net = FT_JNF(
                        input_dim=ch_num,
                        hidden1_dim=64,
                        hidden2_dim=64,
                        output_dim=2
                    ).to(device)
            else:
                net = FT_JNF(
                    input_dim=ch_num,
                    hidden1_dim=256,
                    hidden2_dim=128,
                    output_dim=2
                ).to(device)
            
            chosen_epoch = load_checkpoint(checkpoint_path, target_epoch=200, net=net)
            print(chosen_epoch)

            stoi_noisy = np.array([])
            si_sdr_noisy = np.array([])
            pesq_noisy = np.array([])
            stoi_enhanced = np.array([])
            si_sdr_enhanced = np.array([])
            pesq_enhanced = np.array([])

            net.eval()
            if smallnet:
                # name = f"{data_type}_smallnet_{model_type}_{steering_name}_{t}"
                name = f"{scenario}_11074net_{model_type}_{steering_name}_{t}"
            else:
                name = data_type

            if wandb_active:
                wandb.init(
                    # project=f"Lab_Experiment_{data_type}_{steering_name}",
                    project=f"Lab_Experiment_nosebridge",
                    entity="tatarjit-ben-gurion-university-of-the-negev", 
                    name=name,
                    config={
                        "chosen_epoch": chosen_epoch,
                    }
                )

            # ----- Paths -----
            input_path = os.path.join(data_root, data_type)

            # all subfolders like ex_1, ex_2, ...
            subfolders = [
                d for d in os.listdir(input_path)
                if os.path.isdir(os.path.join(input_path, d)) and d.startswith("ex_")
            ]
            subfolders.sort(key=lambda f: int(re.sub('\D', '', f)))

            # subfolders = subfolders[:1]

            n_fft = 512
            hop = 256
            win_len = 512

            # n_fft = 662
            # hop = 331
            # win_len = 662

            for folder in tqdm(subfolders, desc=f"Processing {data_type}"):
                folder_path = os.path.join(input_path, folder)

                noisy_tf_anm, noisy_tf_mic, clean_time_mic, measured_shift = preprocess_single_example(
                    folder_path=folder_path,
                    V = V,
                    th = th,
                    ph = ph,
                    nfft = n_fft,
                    fs = fs,
                    cancel_nose = cancel_nose,
                    train=train,
                    max_length_sec=max_length_sec,
                    sample_rate=sample_rate,
                    device=device,
                )

                # p, sr = load_noisy_only(folder_path)

                # noisy_tf_mic = swap_noisy_channels(noisy_tf_mic, clean_channel_idx=idx-1)
                clean_signal = select_clean_channel(clean_time_mic, idx=closest_mic_idx-1)

                # --- begin inference ---
                if AmbiDrop:
                    x, s = noisy_tf_anm, clean_signal
                else:
                    x, s = noisy_tf_mic, clean_signal

                s1 = s.to(device)      # move clean to GPU
                x = x.to(device) 
                M = net(x)
                Ms = M[:,:,0] + 1j*M[:,:,1]
                Ms = Ms.squeeze()
                C = ch_num // 2
                if AmbiDrop:
                    ref_ch = x[:,:,0] + 1j*x[:,:,C]
                else:
                    ref_ch = x[:,:,0+closest_mic_idx-1] + 1j*x[:,:,C+closest_mic_idx-1]
                Y = ref_ch.squeeze(0)
                S_hat = Ms*Y
                y = torch.istft(Y.T, n_fft=n_fft, hop_length=hop, win_length=win_len, window=torch.hamming_window(window_length=win_len), center=True, normalized=False, onesided=True, return_complex=False, length=s1.shape[1])
                s_hat = torch.istft(S_hat.T, n_fft=n_fft, hop_length=hop, win_length=win_len, window=torch.hamming_window(window_length=win_len), center=True, normalized=False, onesided=True, return_complex=False,length = s1.shape[1]) 
                s_hat = s_hat/s_hat.max()
                s1 = s1/s1.max()
                y = y/y.max()

                s1 = s1.squeeze(0)
                s1 = s1.detach().cpu()
                s_hat = s_hat.detach().cpu()
                y = y.detach().cpu()
                # print("after", y.shape, s_hat.shape, s1.shape)

                clean_mic = s1

                fs = 16000
                # t0 = 1440     # initial shift in samples
                # d = 1440       # +- range for fine search
                t0 = 0    # initial shift in samples
                ranges = [(measured_shift - 50, measured_shift + 50)]

                # shifts, sisdr_vals, best_shift, best_sisdr = sweep_alignment_sisdr(
                #     s1=s1,
                #     s_hat=s_hat,
                #     fs=fs,
                #     t0=t0,
                #     ranges=ranges,
                #     save_path="sisdr_alignment_sweep.png",
                # )

                # find best shift by correlation
                ref_sig = s1.flatten()
                est_sig = s_hat.flatten()
                correlation = sig.correlate(ref_sig, est_sig, mode='full')
                lags = sig.correlation_lags(len(ref_sig), len(est_sig), mode='full')
                # 2. Find the index of the maximum correlation
                best_lag_idx = np.argmax(np.abs(correlation))
                best_shift = lags[best_lag_idx]
                # print(best_shift)

                # best_shift = -1766
                # best_shift = -1885
                s1, y, s_hat = align_with_best_shift(s1, y, s_hat, best_shift)

                y  = torch.from_numpy(y).to(device)
                s1 = torch.from_numpy(s1).to(device)
                s_hat = torch.from_numpy(s_hat).to(device)

                y = y.detach().cpu()
                s1 = s1.detach().cpu()
                s_hat = s_hat.detach().cpu()

                # --- for noisy si-sdr calculation:
                noisy_mic = noisy_tf_mic.to(device)
                ref_ch = noisy_mic[:,:,0+closest_mic_idx-1] + 1j*noisy_mic[:,:,7+closest_mic_idx-1]
                Y = ref_ch.squeeze(0)
                y_mic = torch.istft(Y.T, n_fft=512, hop_length=256, win_length=512, window=torch.hamming_window(window_length=512), center=True, normalized=False, onesided=True, return_complex=False, length=clean_mic.shape[0])
                y_mic = y_mic/y_mic.max()
                y_mic = y_mic.detach().cpu()

                # s1_mic, y_mic, _ = align_with_best_shift(clean_mic, y_mic, y_mic, measured_shift)

                # find best shift by correlation
                ref_sig = clean_mic.flatten()
                est_sig = y_mic.flatten()
                correlation = sig.correlate(ref_sig, est_sig, mode='full')
                lags = sig.correlation_lags(len(ref_sig), len(est_sig), mode='full')
                best_lag_idx = np.argmax(np.abs(correlation))
                best_shift = lags[best_lag_idx]
                # print(best_shift)
                s1_mic, y_mic, _ = align_with_best_shift(clean_mic, y_mic, y_mic, best_shift)

                y_mic  = torch.from_numpy(y_mic).to(device)
                s1_mic = torch.from_numpy(s1_mic).to(device)
                y_mic = y_mic.detach().cpu()
                s1_mic = s1_mic.detach().cpu()

                stoi_noisy = np.append(stoi_noisy, stoi(s1_mic, y_mic, sample_rate, extended=False))
                si_sdr_noisy = np.append(si_sdr_noisy, si_snr(y_mic.unsqueeze(0), s1_mic.unsqueeze(0), debug = False))
                pesq_noisy = np.append(pesq_noisy, pesq(sample_rate, s1_mic.numpy(), y_mic.numpy(), mode="wb"))

                # stoi_noisy = np.append(stoi_noisy, stoi(s1, y, 16000, extended=False))
                # si_sdr_noisy = np.append(si_sdr_noisy, si_snr(y.unsqueeze(0), s1.unsqueeze(0), debug = False))
                # pesq_noisy = np.append(pesq_noisy, pesq(16000, s1.numpy(), y.numpy(), mode="wb"))

                stoi_enhanced = np.append(stoi_enhanced, stoi(s1, s_hat, sample_rate, extended=False))
                si_sdr_enhanced = np.append(si_sdr_enhanced, si_snr(s_hat.unsqueeze(0), s1.unsqueeze(0), debug = False))
                pesq_enhanced = np.append(pesq_enhanced, pesq(sample_rate, s1.numpy(), s_hat.numpy(), mode="wb"))

                # break

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
                    "audio/clean": wandb.Audio(s1.numpy(), sample_rate=sample_rate),
                    "audio/enhanced": wandb.Audio(s_hat.numpy(), sample_rate=sample_rate),
                    "audio/noisy": wandb.Audio(y.numpy(), sample_rate=sample_rate),
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

            plot_snr_dist = False

            if plot_snr_dist:
                base_path_temp = os.path.join("/gpfs0/bgu-br/users/tatarjit/speech-enhancement/datasets/aria_ds", data_type)
                base_path = base_path_temp + "_plots"
                os.makedirs(base_path, exist_ok=True)

                save_path = os.path.join(base_path, "si_sdr_noisy_05.npy")
                np.save(save_path, si_sdr_noisy)
                print(f"Data saved successfully to {save_path}")

                save_path = os.path.join(base_path, "si_sdr_enhanced_05.npy")
                np.save(save_path, si_sdr_enhanced)
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

                noisy_arr = si_sdr_noisy
                enhanced_arr = si_sdr_enhanced

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
                save_file = os.path.join(base_path, 'si-sdr distribution across examples of aria.csv')
                summary.to_csv(save_file, index=False)
                print("Performance summary saved to 'si-sdr distribution across examples of aria.csv'")
                print("\n--- Performance Summary Table ---")
                print(summary.to_string(index=False))

                # Example Usage:
                base_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/datasets/aria_ds"
                csv_file = os.path.join(base_path, 'si-sdr distribution across examples of aria.csv')
                png_file = os.path.join(base_path, 'si-sdr distribution across examples of aria.png')

                if os.path.exists(csv_file):
                    csv_to_filtered_table_image(csv_file, png_file, title_text = 'si-sdr distribution (ASM to network) across examples and arrays')

#rsync -avh --progress "/Users/mikitatarjitzky/Documents/aria/projectaria_client_sdk_samples/mixed_data/" tatarjit@bhn20:/gpfs0/bgu-br/users/tatarjit/speech-enhancement/datasets/aria_ds/mixed_data

#runai-bgu submit python -n aria-test-nosebridge-11074 -c 20 -m 40G -g 0.5 --conda venv -- "python /gpfs0/bgu-br/users/tatarjit/speech-enhancement/test_aria_glasses.py"
