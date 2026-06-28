import torch
import torch.nn as nn


class SHChannelDropout(nn.Module):
    """
    Channel-wise dropout for Spherical Harmonic (Ambisonics) input.

    Randomly zeros out entire SH channels (both real and imaginary parts)
    during training. Channel 0 (a00, omnidirectional) is never dropped
    since it serves as the reference.

    Args:
        drop_prob: Probability of dropping each channel
        max_drop: Maximum number of channels to drop per sample
    """

    def __init__(self, drop_prob=0.5, max_drop=4):
        super().__init__()
        self.drop_prob = drop_prob
        self.max_drop = max_drop

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x

        B, T, F, C = x.shape
        assert C % 2 == 0, f"Expected even number of channels (real + imag), got {C}"
        SH_C = C // 2

        mask = torch.ones(B, T, F, C, device=x.device)

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
                    mask[b, :, :, i] = 0.0
                    mask[b, :, :, i + SH_C] = 0.0

        return x * mask


class SHChannelDropout1D(nn.Module):
    """
    Channel-wise dropout for time-domain SH input with shape (B, C, T).
    Used by Conv-TasNet architecture.

    Args:
        drop_prob: Probability of dropping each channel
        max_drop: Maximum number of channels to drop per sample
    """

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
                    mask[b, i, :] = 0.0

        return x * mask


class PerChDropout(nn.Module):
    """
    Per-channel dropout with individually specified probabilities.

    Each SH channel has its own dropout probability, derived from ASM
    encoding error analysis. Channel 0 (a00) is forced to probability 0
    (never dropped).

    Args:
        drop_probs: 1D list/tensor of length SH_C with per-channel drop probabilities
    """

    def __init__(self, drop_probs):
        super().__init__()

        drop_probs = torch.as_tensor(drop_probs, dtype=torch.float32)
        if drop_probs.numel() < 1:
            raise ValueError("drop_probs must have at least one entry (for a00).")
        drop_probs = drop_probs.clone()
        drop_probs[0] = 0.0

        self.register_buffer("drop_probs", drop_probs)

    def forward(self, x):
        if not self.training:
            return x

        B, T, F, C = x.shape
        SH_C = C // 2

        if self.drop_probs.numel() != SH_C:
            raise ValueError(
                f"drop_probs length ({self.drop_probs.numel()}) must match SH_C ({SH_C})."
            )

        drop_probs = self.drop_probs.to(x.device)

        mask = torch.ones(B, T, F, C, device=x.device)

        for b in range(B):
            rand_vals = torch.rand(SH_C, device=x.device)
            drop_mask = rand_vals < drop_probs

            for i, drop in enumerate(drop_mask):
                if drop:
                    mask[b, :, :, i] = 0.0
                    mask[b, :, :, i + SH_C] = 0.0

        return x * mask


class PerChDropout1D(nn.Module):
    """
    Per-channel dropout for time-domain SH input with shape (B, C, T).
    Used by Conv-TasNet architecture.

    Args:
        drop_probs: 1D list/tensor of length C with per-channel drop probabilities
    """

    def __init__(self, drop_probs):
        super().__init__()

        drop_probs = torch.as_tensor(drop_probs, dtype=torch.float32)
        if drop_probs.numel() < 1:
            raise ValueError("drop_probs must have at least one entry (for a00).")
        drop_probs = drop_probs.clone()
        drop_probs[0] = 0.0

        self.register_buffer("drop_probs", drop_probs)

    def forward(self, x):
        if not self.training:
            return x

        B, C, T = x.shape
        SH_C = C

        if self.drop_probs.numel() != SH_C:
            raise ValueError(
                f"drop_probs length ({self.drop_probs.numel()}) must match SH_C ({SH_C})."
            )

        drop_probs = self.drop_probs.to(x.device)

        mask = torch.ones(B, C, T, device=x.device)

        for b in range(B):
            rand_vals = torch.rand(SH_C, device=x.device)
            drop_mask = rand_vals < drop_probs

            for i, drop in enumerate(drop_mask):
                if drop:
                    mask[b, i, :] = 0.0

        return x * mask
