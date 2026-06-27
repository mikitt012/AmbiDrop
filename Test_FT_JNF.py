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
wandb.login()
# import re  
writer = SummaryWriter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# def si_snr(estimate, reference, epsilon=1e-8):
#     estimate = estimate - estimate.mean(dim=-1, keepdim=True)
#     reference = reference - reference.mean(dim=-1, keepdim=True)
#     reference_pow = reference.pow(2).mean(dim=-1, keepdim=True)
#     mix_pow = (estimate * reference).mean(dim=-1, keepdim=True)
#     scale = mix_pow / (reference_pow + epsilon)

#     scaled_reference = scale * reference
#     error = estimate - scaled_reference

#     reference_pow = scaled_reference.pow(2).mean(dim=-1)
#     error_pow = error.pow(2).mean(dim=-1)

#     si_snr = 10 * torch.log10(reference_pow / (error_pow + epsilon))
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

def pad_or_truncate(signal: torch.Tensor, target_length: int = 120000) -> torch.Tensor:
    """
    Pads or truncates a multichannel signal (C, T) to (C, target_length).
    """
    C, T = signal.shape
    if T > target_length:
        print(f"[Warning] Signal truncated from {T} to {target_length} samples.")
        return signal[:, :target_length]
    elif T < target_length:
        pad_len = target_length - T
        pad_tensor = torch.zeros(C, pad_len, device=signal.device, dtype=signal.dtype)
        return torch.cat([signal, pad_tensor], dim=1)
    else:
        return signal

def add_white_noise(signal: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    Adds white Gaussian noise to a multichannel signal tensor (C, T).
    
    Parameters:
    - signal: torch.Tensor, shape (C, T)
    - snr_db: float, desired SNR in decibels

    Returns:
    - torch.Tensor: noisy signal of shape (C, T)
    """
    # Ensure float tensor
    signal = signal.float()

    if signal.ndim == 1:
        signal = signal.unsqueeze(0)  # convert to (1, T)

    # Compute signal power per channel
    signal_power = signal.pow(2).mean(dim=1, keepdim=True)  # shape (C, 1)
    snr_linear = 10 ** (snr_db / 10)

    # Compute noise power and generate noise
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(signal) * torch.sqrt(noise_power)

    return signal + noise


def load_checkpoint(checkpoint_path, target_epoch=None, net=None, optimizer=None, scheduler=None):
    """
    Load the checkpoint for a specific epoch or the latest checkpoint if no epoch is specified.
    Also loads learning rate and scheduler state.
    """
    checkpoint_list = torch.load(checkpoint_path, map_location="cpu")
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

class SimDS(Dataset):
    def __init__(self, data_dir, data_type):
        self.data_dir = os.path.join(data_dir, data_type)

        # Match folders like "sample_00010"
        self.sample_dirs = sorted([
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d)) and d.startswith("sample_")
        ])

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_folder = self.sample_dirs[idx]  # e.g., "sample_1883"
        
        # Extract raw number (e.g., "1883") and zero-pad to 5 digits → "01883"
        number_raw = sample_folder.split("_")[-1]  # "1883"
        sample_number = f"{int(number_raw):05d}"   # → "01883"
        
        sample_path = os.path.join(self.data_dir, sample_folder)

        # Correctly formatted filenames
        noisy_path = os.path.join(sample_path, f"noisy_speech_{sample_number}.wav")
        direct_path = os.path.join(sample_path, f"direct_speech_{sample_number}.wav")

        try:
            # Load audio
            noisy_speech, _ = torchaudio.load(noisy_path)
            clean_speech, _ = torchaudio.load(direct_path)
        except Exception as e:
            print(f"[Error loading {sample_folder}]: {e}")
            return self.__getitem__((idx + 1) % len(self))  # skip invalid

        # adding sensor noise
        noisy_speech = add_white_noise(noisy_speech, 30)
        noisy_speech = pad_or_truncate(noisy_speech, 120000)
        clean_speech = pad_or_truncate(clean_speech, 120000)

        # Convert arrays to tensors and move to the specified device
        noisy_speech = noisy_speech.to(device).float()
        clean_speech = clean_speech.to(device).float()

        # clean_speech_tf  =  torch.stft(clean_speech, n_fft=512, hop_length=256, win_length=512, window=torch.hamming_window(window_length=512), center=True, normalized=False, onesided=True, return_complex=True).transpose(0,2)
        noisy_speech_tf  =  torch.stft(noisy_speech, n_fft=512, hop_length=256, win_length=512, window=torch.hamming_window(window_length=512), center=True, normalized=False, onesided=True, return_complex=True).transpose(0,2)

        max_val = noisy_speech_tf.abs().max().item()
        noisy_speech_tf = noisy_speech_tf/max_val
        clean_speech = clean_speech/max_val
        noisy_speech_tf = torch.cat((noisy_speech_tf.real, noisy_speech_tf.imag),2)

        # print("Clean shape:", clean_speech.shape)
        # print("Noisy shape:", noisy_speech_tf.shape)

        return  clean_speech[0,:], noisy_speech_tf
print('finish section', datetime.now())


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
            # noisy_speech_tf, clean_speech_1ch, ref_id = data
            # noisy_speech_tf, clean_speech_1ch = data
            noisy_speech_tf, clean_speech_1ch, _, _ = data
        except Exception as e:
            print(f"[Error loading] {file_name}: {e}")
            return self.__getitem__((idx + 1) % len(self))  # try next valid sample

        # return noisy_speech_tf, clean_speech_1ch.float(), ref_id
        return noisy_speech_tf, clean_speech_1ch.float()

print('start section', datetime.now())

class FT_JNF(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim):
        super().__init__()
        self.LSTM1 = nn.LSTM(input_size=input_dim,hidden_size=hidden1_dim,num_layers=1,batch_first=True,bidirectional=True)
        self.LSTM2 = nn.LSTM(input_size=2*hidden1_dim,hidden_size=hidden2_dim,num_layers=1,batch_first=True,bidirectional=True)
        self.linear = nn.Linear(2*hidden2_dim,output_dim)
        self.tanh = nn.Tanh()
        self.hamming_window = torch.hamming_window(512, device=device)
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
        x, hidden = self.LSTM1(x.view(B * T, F, C)) #BTxFxC
        x = x.view(B, T, F, -1)     # (B, T, F, hidden1*2)
        x = x.permute(0, 2, 1, 3)   # → (B, F, T, hidden1*2)
        x = x.reshape(B * F, T, -1) # Each frequency bin across time is one sequence
        x, hidden = self.LSTM2(x) #FxTx256
        x = x.view(B, F, T, -1).permute(0, 2, 1, 3)  # → (B, T, F, hidden2*2)
        x = self.linear(x)
        # x = self.tanh(x)
        return x

    def training_step(self, batch, batch_idx):
        #x, S, V, s, v, x_normalized = batch
        x, s = batch
        # x, s, ref_id = batch
        x = x.to(device)
        s = s.to(device)

        # B, T, F, _ = x.shape
        # C = s.shape[1]
        # # Convert ref_id to shape [B, 1, 1, 1] for broadcasting
        # ref_idx = ref_id.view(B, 1, 1, 1).expand(-1, T, F, 1)  # [B, T, F, 1]
        # # Split x into real and imaginary parts
        # x_real = x[..., :C]  # [B, T, F, C]
        # x_imag = x[..., C:]  # [B, T, F, C]
        # # Gather reference channel
        # x_ref_real = torch.gather(x_real, dim=-1, index=ref_idx)  # [B, T, F, 1]
        # x_ref_imag = torch.gather(x_imag, dim=-1, index=ref_idx)  # [B, T, F, 1]
        # # Concatenate real+imag to get complex STFT
        # Y = x_ref_real + 1j*x_ref_imag
        # Y = Y.squeeze()
        # # --- For s ---
        # # ref_idx for s shape [B, 1, 1] to gather along dim=1 (channels)
        # T = s.shape[2]
        # ref_idx_s = ref_id.view(B, 1, 1).expand(-1, 1, T)  # [B, 1, T]
        # s = torch.gather(s, dim=1, index=ref_idx_s).squeeze(1)  # [B, T]

        s = s[:, 0, :].squeeze(1) # first channel
        # Model prediction
        M = self(x)
        # Ms = M[:, :, 0] + 1j * M[:, :, 1]
        Ms = M[:,:,:,0] + 1j*M[:,:,:,1] # BxTxF
        Ms = Ms.squeeze()
        # Mv = 1 - M[:, :, 0] - 1j * M[:, :, 1]
        # Y = x[0, :, :, 0] + 1j * x[0, :, :, 5]
        Y = x[:,:,:,0] + 1j*x[:,:,:,7]

        # Reconstruct signals
        S_hat = Ms * Y
        s_hat = torch.istft(S_hat.permute(0, 2, 1), n_fft=512, hop_length=256,
                            win_length=512, window=self.hamming_window, center=True, normalized=False, 
                            onesided=True, return_complex=False, length=s.shape[1])

        if torch.isnan(s_hat).any() or torch.isnan(s).any():
            print(f"[NaN detected] s_hat or s is NaN at batch {batch_idx}")
            return torch.tensor(0.0, requires_grad=True).to(device)

        if s_hat.abs().max() < 1e-6 or s.abs().max() < 1e-6:
            print(f"[Warning] Very low energy in s_hat or s at batch {batch_idx}")

        # print(x.shape, s.shape, S_hat.shape, s_hat.shape)
        # loss = -si_snr(s_hat.unsqueeze(0),s)
        # loss = -si_snr(s_hat,s)
        loss = -si_snr(s_hat, s, debug=False)
        return loss.mean()

print('finish section', datetime.now())

smallnet = True

if smallnet:
    net = FT_JNF(
        input_dim=14,
        hidden1_dim=64,
        hidden2_dim=64,
        output_dim=2
    ).to(device)
else:
    net = FT_JNF(
        input_dim=14,
        hidden1_dim=256,
        hidden2_dim=128,
        output_dim=2
    ).to(device)

# Load the checkpoint
# checkpoint_path = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/checkpoints/FT_JNF,2025-08-07_08-22-02.pt"
# checkpoint = torch.load(checkpoint_path, map_location="cpu")
# net.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

if smallnet:
    # checkpoint_path = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/checkpoints/FT_JNF,2025-12-01_09-21-58.pt"
    # checkpoint_path = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/checkpoints/FT_JNF,2025-12-29_14-41-04.pt"
    checkpoint_path = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/checkpoints/FT_JNF,2026-03-25_13-37-42.pt"
else:
    checkpoint_path = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/checkpoints/FT_JNF,2025-11-30_14-41-59.pt"
chosen_epoch = load_checkpoint(checkpoint_path, target_epoch=300, net=net)
print(chosen_epoch)

# data_dir = '/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment_full_anm/test_of_test_ds_preprocessed_swap' 
# test_type = 'uniform sphere (rigid) radius = 0.05_preprocessed'
# test_ds = SimDS_preprocessed(data_dir, test_type)
# testloader = DataLoader(test_ds, batch_size=1, shuffle=False)

for j in range(2,3):
    if j == 1:
        data_dir = '/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment_full_anm/test_of_train_ds_preprocessed'
    else:
        data_dir = '/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment_full_anm/test_of_test_ds_preprocessed'
    
    # test_type = "ULA along X-axis_preprocessed"
    for test_type in sorted(os.listdir(data_dir)):
        # test_type = "planar_preprocessed"
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
        # num_examples = 30
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
            name = test_type + "_base-model" + "_smallnet"
        else:
            name = test_type + "_base-model"
        wandb.init(project="FT_JNF_experiment", entity="tatarjit-ben-gurion-university-of-the-negev",name=name)
        print(chosen_epoch)
        
        # test_ds = CHiME3(data_dir,'test')
        # testloader = DataLoader(test_ds, batch_size=16, shuffle=False)
        for i, data in enumerate(testloader, 0):
            x, s = data
            # x, s, ref_id = data
            s1 = s.to(device)      # move clean to GPU
            x = x.to(device)
            # s1 = s1.unsqueeze(0)
            s1 = s1[:, 0+ref_id, :].squeeze(1)

            # --- zero a channel ---
            # n = 3
            # if n != ref_id:
            #     x[:, :, :, n] = 0 
            #     x[:, :, :, n + 7] = 0

            # n = 0
            # if n != ref_id:
            #     x[:, :, :, n] = 0 
            #     x[:, :, :, n + 7] = 0

            # B, T, F, _ = x.shape
            # C = s1.shape[1]
            # # Convert ref_id to shape [B, 1, 1, 1] for broadcasting
            # ref_idx = ref_id.view(B, 1, 1, 1).expand(-1, T, F, 1)  # [B, T, F, 1]
            # # Split x into real and imaginary parts
            # x_real = x[..., :C]  # [B, T, F, C]
            # x_imag = x[..., C:]  # [B, T, F, C]
            # # Gather reference channel
            # x_ref_real = torch.gather(x_real, dim=-1, index=ref_idx)  # [B, T, F, 1]
            # x_ref_imag = torch.gather(x_imag, dim=-1, index=ref_idx)  # [B, T, F, 1]
            # # Concatenate real+imag to get complex STFT
            # Y = x_ref_real + 1j*x_ref_imag
            # Y = Y.squeeze()
            # # --- For s ---
            # # ref_idx for s shape [B, 1, 1] to gather along dim=1 (channels)
            # T = s1.shape[2]
            # ref_idx_s = ref_id.view(B, 1, 1).expand(-1, 1, T)  # [B, 1, T]
            # s1 = torch.gather(s1, dim=1, index=ref_idx_s).squeeze(1)  # [B, T]
        
            M = net(x)
            Ms = M[:,:,:,0] + 1j*M[:,:,:,1]
            Ms = Ms.squeeze()
            ref_ch = x[:,:,:,0+ref_id] + 1j*x[:,:,:,7+ref_id]
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

            stoi_noisy = np.append(stoi_noisy, stoi(s1, y, 16000, extended=False))
            si_sdr_noisy = np.append(si_sdr_noisy, si_snr(y.unsqueeze(0), s1.unsqueeze(0), debug = False))
            pesq_noisy = np.append(pesq_noisy, pesq(16000, s1.numpy(), y.numpy(), mode="wb"))

            stoi_enhanced = np.append(stoi_enhanced, stoi(s1, s_hat, 16000, extended=False))
            si_sdr_enhanced = np.append(si_sdr_enhanced, si_snr(s_hat.unsqueeze(0), s1.unsqueeze(0), debug = False))
            pesq_enhanced = np.append(pesq_enhanced, pesq(16000, s1.numpy(), s_hat.numpy(), mode="wb"))


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

#runai-cmd --name test-regular  -g 0.3 --cpu-limit 10 -- "conda activate venv && python /Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/Test_FT_JNF.py"
#runai-bgu submit python -n baseline-test -c 20 -m 80G -g 0.2 --conda venv -- "python /Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/Test_FT_JNF.py"

