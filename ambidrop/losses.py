"""
SI-SNR loss functions for speech enhancement training and evaluation.

Public interface:
    si_snr — scale-invariant SNR for real-valued time-domain signals, shape [B, T]
    complex_si_snr — SI-SNR for complex-valued signals (e.g. STFT bins)
"""
import torch


def si_snr(estimate: torch.Tensor, reference: torch.Tensor, epsilon=1e-8, debug=False):
    """
    Compute Scale-Invariant Signal-to-Noise Ratio (SI-SNR).

    Args:
        estimate: Estimated signal, shape [B, T]
        reference: Ground truth signal, shape [B, T]
        epsilon: Small value to avoid division by zero
        debug: If True, print internal debugging info

    Returns:
        SI-SNR per sample, shape [B]
    """
    if debug:
        print(f"[DEBUG] estimate shape: {estimate.shape}, reference shape: {reference.shape}")

    estimate = estimate - estimate.mean(dim=1, keepdim=True)
    reference = reference - reference.mean(dim=1, keepdim=True)

    if debug:
        print(f"[DEBUG] After zero-mean -> estimate: {estimate.shape}, reference: {reference.shape}")

    dot = (estimate * reference).sum(dim=1, keepdim=True)
    ref_energy = (reference ** 2).sum(dim=1, keepdim=True) + epsilon

    scale = dot / ref_energy
    projection = scale * reference

    noise = estimate - projection

    target_power = (projection ** 2).sum(dim=1)
    noise_power = (noise ** 2).sum(dim=1) + epsilon

    si_snr_value = 10 * torch.log10(target_power / noise_power)

    if debug:
        print(f"[DEBUG] target_power: {target_power.mean().item():.4f}, noise_power: {noise_power.mean().item():.4f}")
        print(f"[DEBUG] SI-SNR mean: {si_snr_value.mean().item():.4f} dB")

    return si_snr_value


def complex_si_snr(estimate, reference, epsilon=1e-8):
    """
    Compute SI-SNR for complex-valued signals.

    Args:
        estimate: Complex estimated signal
        reference: Complex ground truth signal
        epsilon: Small value to avoid division by zero

    Returns:
        SI-SNR per sample
    """
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    reference = reference - reference.mean(dim=-1, keepdim=True)

    dot = torch.sum(estimate * reference.conj(), dim=-1, keepdim=True)
    norm = torch.sum(reference * reference.conj(), dim=-1, keepdim=True) + epsilon

    scale = dot / norm
    scaled_reference = scale * reference

    error = estimate - scaled_reference

    reference_power = torch.sum(torch.abs(scaled_reference) ** 2, dim=-1)
    error_power = torch.sum(torch.abs(error) ** 2, dim=-1)

    return 10 * torch.log10((reference_power + epsilon) / (error_power + epsilon))
