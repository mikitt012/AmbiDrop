import torch
from torch.autograd import Variable

import models_attention_set

import torch
import torch.nn as nn
import torch.nn.functional as F

class SHChannelDropout(nn.Module):
    def __init__(self, drop_prob=0.5, max_drop=4):
        super().__init__()
        self.drop_prob = drop_prob
        self.max_drop = max_drop

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x

        B, C, T = x.shape
        SH_C = C

        mask = torch.ones(B, C, T, device=x.device)

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
                    mask[b, i, :] = 0.0

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
        x: Tensor of shape (B, C, T)
        """
        if not self.training:
            return x

        B, C, T = x.shape
        SH_C = C

        if self.drop_probs.numel() != SH_C:
            raise ValueError(
                f"drop_probs length ({self.drop_probs.numel()}) must match SH_C ({SH_C})."
            )

        # Broadcast drop_probs to device
        drop_probs = self.drop_probs.to(x.device)

        # Create mask of ones
        mask = torch.ones(B, C, T, device=x.device)

        # For each batch element, sample independent dropout decisions per SH channel
        for b in range(B):
            # Sample Bernoulli per SH channel (0..SH_C-1)
            # Channel 0 has p=0.0, so it will never drop.
            rand_vals = torch.rand(SH_C, device=x.device)
            drop_mask = rand_vals < drop_probs   # shape: (SH_C,)

            # Apply to real & imag together
            for i, drop in enumerate(drop_mask):
                if drop:
                    mask[b, i, :] = 0.0           # real part
                    # mask[b, i + SH_C, :] = 0.0    # imagenary part

        return x * mask

class TasNet(torch.nn.Module):
    def __init__(self, mic_num, num_spk, enc_dim, feature_dim, ch_dim, sample_rate, win, layer, stack, kernel, causal, drop_prob=0.4, max_drop=3, drop_probs=[], dropout="dropout"):
        super(TasNet, self).__init__()
        # hyper parameters
        self.mic_num = mic_num
        self.num_spk = num_spk

        # increased enc dim
        self.enc_dim = enc_dim
        self.feature_dim = feature_dim

        self.ch_dim = ch_dim

        self.win = int(sample_rate * win / 1000)
        self.stride = self.win // 2

        self.layer = layer
        self.stack = stack
        self.kernel = kernel

        self.causal = causal

        self.drop_prob = drop_prob
        self.max_drop = max_drop
        self.drop_probs = drop_probs
        self.dropout = dropout

        # input encoder
        self.encoder = torch.nn.Conv1d(1, self.enc_dim, self.win, bias=False, stride=self.stride)

        # TCN separator
        self.TCN = models_attention_set.TCN(self.mic_num, self.ch_dim, self.enc_dim, self.enc_dim * self.num_spk,
                                            self.feature_dim, self.feature_dim * 4,  # single modified
                                            self.layer, self.stack, self.kernel, causal=self.causal)

        self.receptive_field = self.TCN.receptive_field

        # output decoder
        self.decoder = torch.nn.ConvTranspose1d(self.enc_dim, 1, self.win, bias=False, stride=self.stride)

        if self.dropout == "SHChannelDropout":
            self.channel_dropout = SHChannelDropout(drop_prob=self.drop_prob, max_drop=self.max_drop)  # adjust max_drop depending on array type
        else:
            self.channel_dropout = PerChDropout(drop_probs=self.drop_probs)

    def pad_signal(self, input):
        # input is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")

        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        nchannel = input.size(1)
        nsample = input.size(2)
        rest = self.win - (self.stride + nsample % self.win) % self.win
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, nchannel, rest)).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, nchannel, self.stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

    def forward(self, input):
        # Padding
        input = self.channel_dropout(input)

        output, rest = self.pad_signal(input)

        batch_size = output.size(0)
        num_ch = output.size(1)
        enc_output = self.encoder(output.view(batch_size * num_ch, 1, -1)).view(batch_size, num_ch, self.enc_dim, -1)  # B, C, N, L

        # generate masks
        masks = torch.sigmoid(self.TCN(enc_output)).view(batch_size, self.num_spk, self.enc_dim, -1)  # B, C, N, L

        # reference ch = a00
        masked_output = enc_output[:, 0:1] * masks

        # waveform decoder
        output = self.decoder(masked_output.view(batch_size * self.num_spk, self.enc_dim, -1))  # B*C, 1, L
        output = output[:, :, self.stride:-(rest + self.stride)].contiguous()  # B*C, 1, L

        output = output.view(batch_size, self.num_spk, -1)  # B, C, T

        return output

    def serialize(self, model, optimizer, epoch, tr_loss=None, cv_loss=None):
        """Serialize model and optimizer state, and include hyperparameters."""
        package = {
            # Standard model state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch,
            # Hyperparameters (The important part for the solver)
            'model_mic_num': self.mic_num,
            'model_enc_dim': self.enc_dim,
            'model_feature_dim': self.feature_dim,
            'model_win': self.win, # Use the ms value from config
            'model_layer': self.layer,
            'model_stack': self.stack,
            'model_kernel': self.kernel,
            'model_num_spk': self.num_spk,
            'model_causal': self.causal,
        }
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
        if cv_loss is not None:
            package['cv_loss'] = cv_loss
        return package

    @classmethod
    def load_model_from_package(cls, package, dropout):
        # 1. Instantiate the model using the individual keys from the package
        # We match the keys saved in serialize() to the arguments expected by __init__
        if dropout == "SHChannelDropout":
            drop_prob = 0.4
            max_drop =3
            drop_probs = []
            th = 0

        if dropout == "PerChDropout":
            drop_prob = 0
            max_drop = 0
            drop_probs = [0, 0.1, 0.45, 0.1, 0.45, 1, 0.75, 1, 0.45]
            th = -3.4

        win = package['model_win'] * 1000 / 16000

        model = cls(
            mic_num=package['model_mic_num'],
            num_spk=package['model_num_spk'],
            enc_dim=package['model_enc_dim'],
            feature_dim=package['model_feature_dim'],
            # If ch_dim wasn't in serialize, we provide a default (e.g., 8)
            ch_dim=package.get('model_ch_dim', 8), 
            sample_rate=package.get('sample_rate', 16000),
            win=win,
            layer=package['model_layer'],
            stack=package['model_stack'],
            kernel=package['model_kernel'],
            causal=package['model_causal'],
            drop_prob=drop_prob, max_drop=max_drop, drop_probs=drop_probs, dropout=dropout
        )
        # 2. Load the weights
        model.load_state_dict(package['state_dict'])
        return model

    @classmethod
    def load_model(cls, path, dropout):
        # Load to CPU
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls.load_model_from_package(package, dropout)
        return model