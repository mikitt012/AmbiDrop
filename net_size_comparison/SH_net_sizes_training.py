
import os
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
import torch.nn as nn
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import ipdb
from pesq import pesq
from pystoi import stoi
import wandb
print(torch.version.cuda)  # Should print the CUDA version
print(torch.backends.cudnn.enabled)  # Should be True if cuDNN is installed
writer = SummaryWriter()
print('device cuda: ', torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
wandb.login()

def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model

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

def channel_dropout(x, drop_prob=0.2, training=True):
    if not training or drop_prob == 0.0:
        return x

    B, T, F, C = x.shape
    # Sample dropout mask per channel: (1, 1, 1, C)
    mask = torch.rand(1, 1, 1, C, device=x.device) > drop_prob
    return x * mask  # Broadcasts to (B, T, F, C)

class SHChannelDropout(nn.Module):
    def __init__(self, drop_prob=0.5, max_drop=4):
        super().__init__()
        self.drop_prob = drop_prob
        self.max_drop = max_drop

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x

        B, T, F, C = x.shape
        assert C == 18, "Expected 10 channels (5 SH real + 5 imag)"
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

class SHChannelDropout(nn.Module):
    def __init__(self, drop_prob=0.5, max_drop=4):
        super().__init__()
        self.drop_prob = drop_prob
        self.max_drop = max_drop

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x

        B, T, F, C = x.shape
        assert C == 18, "Expected 10 channels (5 SH real + 5 imag)"
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

    def visualize_masks(self, save_path=None):
        """
        Plot smooth dropout probability curve.
        """
        freqs = np.linspace(0, self.fs / 2, self.num_freq_bins)
        probs_np = self.dropout_probs.cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(freqs, probs_np, label="Dropout probability (sigmoid)")
        ax.axvline(self.cutoff_freq_hz, color='r', linestyle='--', label=f"Cutoff ≈ {self.cutoff_freq_hz} Hz")
        ax.set_title("Smooth LPF Dropout Probability")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Dropout Probability")
        ax.grid(True)
        ax.legend()

        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()

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
    def __init__(self, data_dir, data_type):
        self.data_type = data_type
        self.data_dir = os.path.join(data_dir, data_type)
        files_list = os.listdir(self.data_dir)
        self.ex_list = [file_name for file_name in files_list if 'ex_' in file_name]

    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, idx):
        try:
            file_path = os.path.join(self.data_dir, self.ex_list[idx])
            ex = scipy.io.loadmat(file_path)
        except:
            print(self.ex_list[idx])
            return self.__getitem__(idx + 1)

        # change to pt, pDirect and pTarget 
        # noisy_speech = ex['anmt'][:,[0, 1, 3, 4, 8]].transpose()
        # clean_speech = ex['anmtDirect'][:,[0, 1, 3, 4, 8]].transpose()  
        noisy_speech = ex['anmt'].transpose()
        clean_speech = ex['anmtDirect'].transpose()  
        # noise = noisy_speech - ex['anmtTarget'][:,[0, 1, 3, 4, 8]].transpose()

        # change the casting to to.float() 
        noisy_speech = torch.from_numpy(noisy_speech).to(torch.complex64).to(device).requires_grad_(False)
        clean_speech = torch.from_numpy(clean_speech).to(torch.complex64).to(device).requires_grad_(False)
        # noise = torch.from_numpy(noise).to(torch.complex64).to(device).requires_grad_(False)


        # here you can trim long signals to avoid out of memory error by setting max_length
        if self.data_type == 'si_tr_s':
            max_length = 6  # in seconds
            segment_length = max_length * 16000  # in samples

            # Find first non-silent frame in clean_speech[0]
            first = (clean_speech[0, :].abs() > 1e-3).nonzero()[0][0].item()
            last = first + segment_length

            # Slice signals
            noisy_segment = noisy_speech[:, first:last]
            clean_segment = clean_speech[:, first:last]
            # noise_segment = noise[:, first:last]

            # Pad if needed
            noisy_speech = pad_to_length(noisy_segment, segment_length)
            clean_speech = pad_to_length(clean_segment, segment_length)
            # noise = pad_to_length(noise_segment, segment_length)

        win = torch.hamming_window(window_length=512)
        # clean_speech_tf = torch.stft(clean_speech, n_fft=512, hop_length=256, win_length=512, window=win, center=True, normalized=False, return_complex=True).transpose(0, 2)
        noisy_speech_tf = torch.stft(noisy_speech, n_fft=512, hop_length=256, win_length=512, window=win, center=True, normalized=False, return_complex=True).transpose(0, 2)
        # noise_tf        = torch.stft(noise,        n_fft=512, hop_length=256, win_length=512, window=win, center=True, normalized=False, return_complex=True).transpose(0, 2)

        # this 3 lines remove negative frequencies from the representation, you dont have to change them.
        # clean_speech_tf = clean_speech_tf[:,0:257,:]
        noisy_speech_tf = noisy_speech_tf[:,0:257,:]
        # noise_tf = noise_tf[:,0:257,:]
        
        # normalization
        max_val = noisy_speech_tf.abs().max().item()
        # clean_speech_tf = clean_speech_tf / max_val
        noisy_speech_tf = noisy_speech_tf / max_val
        # noise_tf = noise_tf / max_val
        clean_speech = clean_speech / max_val
        # noise = noise / max_val

        noisy_speech_tf = torch.cat((noisy_speech_tf.real, noisy_speech_tf.imag), 2)
        
        # return noisy_speech_tf, clean_speech_tf[:, :, 0], noise_tf[:, :, 0], clean_speech[0, :].float(), noise[0, :].float()
        return noisy_speech_tf, clean_speech[0, :].float()

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

        try:
            data = torch.load(file_path, map_location='cpu')  # Load from .pt file
            noisy_speech_tf, clean_speech_1ch = data
            # noisy_speech_tf = noisy_speech_tf[:, :, [0, 1, 3, 4, 8, 9, 10, 12, 13, 17]]
        except Exception as e:
            print(f"[Error loading] {file_name}: {e}")
            return self.__getitem__((idx + 1) % len(self))  # try next valid sample

        return noisy_speech_tf, clean_speech_1ch.float()

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

print('loading data', datetime.now())
torch.manual_seed(0)
data_dir = '/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop'
# train_ds = SimDS(data_dir, 'si_tr_s')
# val_ds = SimDS(data_dir, 'si_dt_05')
train_ds = SimDS_preprocessed(data_dir, 'si_tr_s_preprocessed_full')
val_ds = SimDS_preprocessed(data_dir, 'si_dt_05_preprocessed_full')
generator = torch.Generator(device=device)
train_sampler = RandomSampler(train_ds, generator=generator)
val_sampler = RandomSampler(val_ds, generator=generator)

batch_size = 8
lr = 0.001
weight_decay=1e-6
ephocs = 300

drop_prob = 0.4
max_drop = 3
drop_probs = []
dropout = "SHChannelDropout"
# dropout = "PerChDropout"

trainloader = DataLoader(train_ds, sampler=train_sampler, batch_size=batch_size, drop_last=True)
valloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)
current_time = datetime.now()

model_sizes = [
    # (256, 128), # 1,223,170
    # (128, 128), # 547,330
    # (128, 64),  # 316,674
    # (64, 64),   # 142,594
    # (64, 32),   # 84,610
    # (32, 32),   # 38,530
    # (32, 16),   # 23,874
    # (16, 16),   # 11,074
    # (16, 8),    # 7,330
    (8, 8)        # 3,490
]

for h1, h2 in model_sizes:
    # Initialize a temporary model to count parameters
    net = FT_JNF(
        input_dim=18,
        hidden1_dim=h1,
        hidden2_dim=h2,
        output_dim=2,
        drop_prob=drop_prob,          # your original
        max_drop=max_drop,
        drop_probs=drop_probs,
        dropout=dropout
    ).to(device)
    
    total_params = sum(p.numel() for p in net.parameters())
    print(f"({h1:>3}, {h2:>3}) network {'':<8} | {total_params:>15,}")

    wandb.init(
        project="speech-enhancement",
        entity="tatarjit-ben-gurion-university-of-the-negev",  # 👈 your preferred account name here
        name=f"SH_FT_JNF_train_full_{total_params}_{current_time:%Y%m%d_%H%M%S}",
        config={
            "batch_size": batch_size,
            "epochs": ephocs,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "drop_prob": drop_prob,
            "max_drop": max_drop,
            "drop_probs": drop_probs,
            "dropout": dropout,
            "hidden1_dim": h1,
            "hidden2_dim": h2,
            "net_size": total_params,
        }
    )
    wandb.watch(net, log="all", log_freq=100)  # logs weights, gradients, and parameter histograms

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    prev_loss = float('inf')

    print('finish loading model', datetime.now())
    print('start training', datetime.now())

    max_files_to_process = 6000  # Limit for both training and validation

    # checkpoint_dir = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop"
    # checkpoint_path = max(
    #     [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.startswith("SH_FT_JNF") and f.endswith(".pt")],
    #     default=None,
    #     key=os.path.getctime
    # )

    start_epoch = 0
    # if checkpoint_path and os.path.isfile(checkpoint_path):
    #     checkpoint = torch.load(checkpoint_path, map_location=device)
    #     net.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     start_epoch = checkpoint['epoch'] + 1
    #     prev_loss = checkpoint['loss']
    #     print(f"Resumed from checkpoint: {checkpoint_path} (epoch {start_epoch})")
    # else:
    #     print("No checkpoint found, training from scratch")

    for epoch in range(start_epoch, ephocs):
        # net.channel_dropout.set_epoch(epoch)
        net.train()
        train_loss = 0.0

        # Training Loop
        for i, data in enumerate(trainloader, 0):
            if i >= max_files_to_process:  # Limit to 100 files
                break
                
            #with torch.no_grad():
            #    norms=data[0].abs().max(dim=-1,keepdim=True)[0]
            #    x_normalized = data[0] / (norms + 1e-12)         # avoid division by zero
            #    freq_index = torch.linspace(0, 1, steps=257).float()  # shape: [257]
            #    freq_index = freq_index.view(1, 1, 257, 1)  # shape: [1, 1, 257, 1]
            #    freq_index = freq_index.expand(x_normalized.shape[0], x_normalized.shape[1], 257, 1)  # [1, 544, 257, 1]
            #    x_normalized = torch.cat((x_normalized, freq_index,norms), dim=-1)  # [1, 544, 257, 19]
            #    data.append(spikegen.rate(x_normalized, snn_net.num_steps))
            
            optimizer.zero_grad()
            loss = net.training_step(data, i)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if i % 100 == 99: # print every 100 iterations
                print('epoch: %d, iteration: %d, loss: %.3f' % (epoch + 1, i + 1, train_loss / i), datetime.now())
        torch.cuda.empty_cache()

        total_train_loss = train_loss / i
        writer.add_scalar("Loss", total_train_loss, epoch)

        val_loss = 0.0

        # Validation Loop
        net.eval()
        with torch.no_grad():
            for i, data in enumerate(valloader, 0):
                #norms=data[0].abs().max(dim=-1,keepdim=True)[0]
                #x_normalized = data[0] / (norms + 1e-12)         # avoid division by zero
                #freq_index = torch.linspace(0, 1, steps=257).float()  # shape: [257]
                #freq_index = freq_index.view(1, 1, 257, 1)  # shape: [1, 1, 257, 1]
                #freq_index = freq_index.expand(x_normalized.shape[0], x_normalized.shape[1], 257, 1)  # [1, 544, 257, 1]
                #x_normalized = torch.cat((x_normalized, freq_index,norms), dim=-1)  # [1, 544, 257, 19]
                #data.append(spikegen.rate(x_normalized, snn_net.num_steps))
                loss = net.training_step(data, i)
                val_loss += loss.item()

            total_val_loss = val_loss / i
            writer.add_scalar("val_loss", total_val_loss, epoch)
            # scheduler.step(total_val_loss)
            print('epoch: %d, val_loss: %.3f' % (epoch + 1, total_val_loss), datetime.now())

            if total_val_loss < prev_loss:
                prev_loss = total_val_loss
                save_dir = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/checkpoints"
                os.makedirs(save_dir, exist_ok=True)
                filename = 'SH_FT_JNF,{date:%Y-%m-%d_%H-%M-%S}.pt'.format(date=current_time)
                # filename = total_params
                checkpoint_path = os.path.join(save_dir, filename)
                if os.path.exists(checkpoint_path):
                    checkpoint_list = torch.load(checkpoint_path, map_location="cpu")  # Load existing checkpoints
                else:
                    checkpoint_list = []  # Start with an empty list if no checkpoint exists
                checkpoint_data = {
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_val_loss,
                    # 'scheduler_state_dict': scheduler.state_dict(),  # Save scheduler state
                    'learning_rate': optimizer.param_groups[0]['lr']  # Save current learning rate
                }
                checkpoint_list.append(checkpoint_data)
                torch.save(checkpoint_list, checkpoint_path)
                print(f'Model saved as {filename}')
            print(f"Epoch {epoch + 1}: Training Loss = {total_train_loss:.4f}, Validation Loss = {total_val_loss:.4f}")
            # if (epoch + 1) % 20 == 0:
            #     save_dir = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/dropout_plot"
            #     save_path = f"{save_dir}/dropout_masks_epoch_{epoch+1}.png"
            #     net.channel_dropout.visualize_masks(save_path=save_path)
        wandb.log({"train_loss": total_train_loss, "val_loss": total_val_loss}, step=epoch+1)

    print('Finished Training')
    wandb.finish()
    writer.close()
    
    del net
    # 2. Force Python's Garbage Collector to find the unreferenced object
    # gc.collect()

    # 3. Clear the PyTorch CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

#runai-cmd --name train-SH  -g 1 --cpu-limit 30 -- "conda activate venv && python /Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/train_hanan.py"

#runai-bgu submit python -n full-train -c 40 -m 60G -g 1 --conda venv -- "python /Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/SH_net_sizes_training.py"
