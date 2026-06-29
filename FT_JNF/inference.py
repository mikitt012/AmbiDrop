import torch
import numpy as np
from pesq import pesq
from pystoi import stoi

from ambidrop.losses import si_snr
from ambidrop.constants import N_FFT, HOP_LENGTH, WIN_LENGTH, SAMPLE_RATE


def reconstruct_signal(net, x, num_channels, ref_id=0, signal_length=None):
    """
    Run model forward pass and reconstruct time-domain enhanced signal.

    Args:
        net: FT_JNF model (in eval mode)
        x: Input STFT tensor, shape (1, T, F, 2C) or (B, T, F, 2C)
        num_channels: Number of channels per part (C in real or imag)
        ref_id: Reference channel index (0 for a00 in AmbiDrop mode)
        signal_length: Target length for iSTFT output

    Returns:
        s_hat: Enhanced time-domain signal
        y: Noisy reference time-domain signal
        Ms: Complex mask
    """
    device = x.device
    M = net(x)
    Ms = M[..., 0] + 1j * M[..., 1]
    Ms = Ms.squeeze()

    ref_real = x[:, :, :, ref_id] if x.dim() == 4 else x[:, :, ref_id]
    ref_imag = x[:, :, :, num_channels + ref_id] if x.dim() == 4 else x[:, :, num_channels + ref_id]
    Y = (ref_real + 1j * ref_imag).squeeze(0)

    S_hat = Ms * Y

    win = torch.hamming_window(WIN_LENGTH, device=device)
    y = torch.istft(Y.T, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                    window=win, center=True, normalized=False,
                    onesided=True, return_complex=False, length=signal_length)
    s_hat = torch.istft(S_hat.T, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                        window=win, center=True, normalized=False,
                        onesided=True, return_complex=False, length=signal_length)

    s_hat = s_hat / s_hat.max()
    y = y / y.max()

    return s_hat, y, Ms


def evaluate_sample(s_hat, s_clean, y_noisy, sr=SAMPLE_RATE):
    """
    Compute SI-SDR, PESQ, and STOI for a single sample.

    All inputs should be 1D numpy arrays or 1D torch tensors (CPU).

    Returns:
        dict with keys: si_sdr_noisy, si_sdr_enhanced, pesq_noisy,
                        pesq_enhanced, stoi_noisy, stoi_enhanced
    """
    if torch.is_tensor(s_hat):
        s_hat = s_hat.detach().cpu()
    if torch.is_tensor(s_clean):
        s_clean = s_clean.detach().cpu()
    if torch.is_tensor(y_noisy):
        y_noisy = y_noisy.detach().cpu()

    s_hat_np = s_hat.numpy() if torch.is_tensor(s_hat) else s_hat
    s_clean_np = s_clean.numpy() if torch.is_tensor(s_clean) else s_clean
    y_noisy_np = y_noisy.numpy() if torch.is_tensor(y_noisy) else y_noisy

    s_hat_t = s_hat if torch.is_tensor(s_hat) else torch.from_numpy(s_hat)
    s_clean_t = s_clean if torch.is_tensor(s_clean) else torch.from_numpy(s_clean)
    y_noisy_t = y_noisy if torch.is_tensor(y_noisy) else torch.from_numpy(y_noisy)

    return {
        'si_sdr_noisy': si_snr(y_noisy_t.unsqueeze(0), s_clean_t.unsqueeze(0)).item(),
        'si_sdr_enhanced': si_snr(s_hat_t.unsqueeze(0), s_clean_t.unsqueeze(0)).item(),
        'pesq_noisy': pesq(sr, s_clean_np, y_noisy_np, mode="wb"),
        'pesq_enhanced': pesq(sr, s_clean_np, s_hat_np, mode="wb"),
        'stoi_noisy': stoi(s_clean_np, y_noisy_np, sr, extended=False),
        'stoi_enhanced': stoi(s_clean_np, s_hat_np, sr, extended=False),
    }
