#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch.nn.functional as F
import torch.nn as nn
import torchaudio
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import ipdb
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pesq import pesq
from pystoi import stoi
import logging
import wandb
from torch.optim.lr_scheduler import LinearLR, StepLR, SequentialLR
writer = SummaryWriter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

wandb.login()

# def si_snr(estimate, reference, epsilon=1e-8):
#     # print(estimate.shape, reference.shape)
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
    checkpoint_list = torch.load(checkpoint_path)

    if target_epoch is None:
        # Load the latest checkpoint (i.e., the one with the highest epoch)
        target_epoch = max(checkpoint['epoch'] for checkpoint in checkpoint_list)
        print(f"No epoch specified. Loading the latest checkpoint from epoch {target_epoch}")

    # Find the checkpoint for the target epoch
    checkpoint_to_load = None
    for checkpoint in checkpoint_list:
        if checkpoint['epoch'] == target_epoch:
            checkpoint_to_load = checkpoint
            break

    if checkpoint_to_load is None:
        print(f"Checkpoint for epoch {target_epoch} not found.")
        return None
    else:
        # Load the model and optimizer state
        if net is not None:
            net.load_state_dict(checkpoint_to_load['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint_to_load['optimizer_state_dict'])

        # Load the learning rate (if you want to log or use it later)
        if optimizer is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = checkpoint_to_load['learning_rate']
        
        # Print the learning rate
        print(f"Loaded learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Load the scheduler state (if provided)
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint_to_load['scheduler_state_dict'])

        # Extract the loss or any other metrics you want
        prev_loss = checkpoint_to_load['loss']
        print(f"Loaded checkpoint from epoch {target_epoch}, loss: {prev_loss:.4f}")

        return prev_loss

def get_lr(opt): return opt.param_groups[0]['lr']

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
            # noisy_speech_tf, clean_speech, ref_id = data
            # data['noisy'], data['clean'].float(), data['ref_id'], data['array_name'], data['ex_id']
        except Exception as e:
            print(f"[Error loading] {file_name}: {e}")
            return self.__getitem__((idx + 1) % len(self))  # try next valid sample

        # return noisy_speech_tf, clean_speech_1ch.float(), ref_id
        return data['noisy'], data['clean'].float(), data['ref_id'], data['array_name'], data['ex_id']


print('start section', datetime.now())

class FT_JNF(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim):
        super().__init__()
        # self.channel_dropout = SHChannelDropout(drop_prob=0.4, max_drop=3)
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
        # x = self.channel_dropout(x)
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
        # x, s, ref_id = batch
        # x, s, idx = batch
        x, s, idx, _, _ = batch
        x = x.to(device)
        s = s.to(device)

        # print(x.shape)
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
        # print(Y.shape, s.shape)

        # s = s[:, 0, :].squeeze(1) # first channel
        # # Model prediction
        # M = self(x)
        # # Ms = M[:, :, 0] + 1j * M[:, :, 1]
        # Ms = M[:,:,:,0] + 1j*M[:,:,:,1] # BxTxF
        # Ms = Ms.squeeze()
        # # Mv = 1 - M[:, :, 0] - 1j * M[:, :, 1]
        # # Y = x[0, :, :, 0] + 1j * x[0, :, :, 5]
        # Y = x[:,:,:,0] + 1j*x[:,:,:,7]

        B = x.shape[0]
        batch_idx = torch.arange(B, device=device)

        # Select clean reference per example
        # s shape assumed (B, C, T)
        s = s[batch_idx, idx, :]   # (B, T)

        # Model prediction
        M = self(x)                # (B, T, F, 2)
        Ms = M[...,0] + 1j*M[...,1]   # (B, T, F)

        # Build mixture Y
        real = x[batch_idx, :, :, idx]          # (B, T, F)
        imag = x[batch_idx, :, :, 7 + idx]      # (B, T, F)
        Y = real + 1j * imag

        # print(x.shape, Y.shape, Ms.shape, s.shape)
        # print(idx)

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


print('loading data', datetime.now())
torch.manual_seed(0)
data_dir = '/gpfs0/bgu-br/users/tatarjit/speech-enhancement/datasets/experiment_full_anm'
# train_ds = SimDS(data_dir, 'si_tr_s')
# val_ds = SimDS(data_dir, 'si_dt_05')
# train_ds = SimDS_preprocessed(data_dir, 'train_circle_preprocessed_merged')
# val_ds = SimDS_preprocessed(data_dir, 'val_circle_preprocessed_merged')
train_ds = SimDS_preprocessed(data_dir, 'mic_train_ds_preprocessed_merged')
val_ds = SimDS_preprocessed(data_dir, 'mic_val_ds_preprocessed_merged')
# train_ds = SimDS_preprocessed(data_dir, 'train_1array_preprocessed')
# val_ds = SimDS_preprocessed(data_dir, 'val_1array_preprocessed')
# train_ds = SimDS_preprocessed(data_dir, 'train_ds_preprocessed')
# val_ds = SimDS_preprocessed(data_dir, 'val_ds_preprocessed')
generator = torch.Generator(device=device)
train_sampler = RandomSampler(train_ds, generator=generator)
val_sampler = RandomSampler(val_ds, generator=generator)

batch_size = 8
lr = 0.001
weight_decay=1e-6
ephocs = 300

trainloader = DataLoader(train_ds, sampler=train_sampler, batch_size=batch_size, drop_last=True)
valloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)
current_time = datetime.now()

net = FT_JNF(
    input_dim=14,
    hidden1_dim=64,
    hidden2_dim=64,
    output_dim=2
).to(device)

wandb.init(
    project="speech-enhancement",
    entity="tatarjit-ben-gurion-university-of-the-negev",  # 👈 your preferred account name here
    name=f"FT_JNF_train_smallnet_{current_time:%Y%m%d_%H%M%S}",
    config={
        "batch_size": batch_size,
        "epochs": ephocs,
        "learning_rate": lr,
        "weight_decay": weight_decay,
    }
)
wandb.watch(net, log="all", log_freq=100)  # logs weights, gradients, and parameter histograms

optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

warmup_epochs = 10                  # e.g., first 2 epochs linearly ramp LR up to `lr`
step_size_epochs = 10              # decay every 10 epochs after warmup
gamma = 0.8                        # multiply LR by this factor each step
# warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
# decay  = StepLR(optimizer, step_size=step_size_epochs, gamma=gamma)
# scheduler = SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[warmup_epochs])
# scheduler = StepLR(optimizer, step_size=step_size_epochs, gamma=gamma)

prev_loss = float('inf')

total_params = sum(p.numel() for p in net.parameters())
print(f"Number of parameters: {total_params}")

print('finish loading model', datetime.now())
print('start training', datetime.now())

# checkpoint = torch.load("/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/FT_JNF,2025-08-14_10-26-56.pt")
# net.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# prev_loss = checkpoint['loss']

# target_epoch_to_load = 100

# checkpoint_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/FT_JNF,2025-08-24_08-33-26.pt"
# load_checkpoint(checkpoint_path, target_epoch = None, net=net, optimizer=optimizer)

# load_checkpoint(checkpoint_path, target_epoch=None, net=net, optimizer=optimizer, scheduler=scheduler)

# generator.manual_seed(0)
for epoch in range(ephocs):
    net.train()
    train_loss = 0.0

    # Training Loop
    for i, data in enumerate(trainloader, 0):
        optimizer.zero_grad()
        loss = net.training_step(data, i)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if i % 100 == 99: # print every 100 iterations
            print('epoch: %d, iteration: %d, loss: %.3f' % (epoch + 1, i + 1, train_loss / i), datetime.now())
    # scheduler.step()
    torch.cuda.empty_cache()

    total_train_loss = train_loss / i
    writer.add_scalar("Loss", total_train_loss, epoch)

    val_loss = 0.0

    # Validation Loop
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(valloader, 0):
            loss = net.training_step(data, i)
            val_loss += loss.item()

        total_val_loss = val_loss / i
        writer.add_scalar("val_loss", total_val_loss, epoch)
        print('epoch: %d, val_loss: %.3f' % (epoch + 1, total_val_loss), datetime.now())

        if total_val_loss < prev_loss:
            prev_loss = total_val_loss
            save_dir = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints"
            os.makedirs(save_dir, exist_ok=True)
            filename = 'FT_JNF,{date:%Y-%m-%d_%H-%M-%S}.pt'.format(date=current_time)
            checkpoint_path = os.path.join(save_dir, filename)
            if os.path.exists(checkpoint_path):
                checkpoint_list = torch.load(checkpoint_path)  # Load existing checkpoints
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
    wandb.log({"lr": get_lr(optimizer), "train_loss": total_train_loss, "val_loss": total_val_loss}, step=epoch+1)

print('Finished Training')
wandb.finish()
writer.close()

#runai-cmd --name train-mic-model  -g 0.7 --cpu-limit 40 --memory-limit 40Gi -- "conda activate venv && python /gpfs0/bgu-br/users/tatarjit/speech-enhancement/train_FT_JNF.py"

#runai-bgu submit python -n baseline -c 40 -m 80G -g 1 --conda venv -- "python /gpfs0/bgu-br/users/tatarjit/speech-enhancement/train_FT_JNF.py"




