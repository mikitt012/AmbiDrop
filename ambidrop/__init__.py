"""
Public re-export surface for the ambidrop package.

Public interface:
    si_snr — scale-invariant SNR loss (time domain)
    complex_si_snr — SI-SNR for complex STFT signals
    SHChannelDropout — random SH channel zeroing for FT-JNF training
    PerChDropout — per-channel dropout with individually specified probabilities
    find_ref_mic — return 0-based index of mic closest to azimuth 0
    encode_ambisonics — unified ASM entry point (compute coefficients + encode)
    compute_asm_coefficients — compute Tikhonov/SVD ASM filter coefficients
    apply_asm_filters — apply precomputed ASM filters to mic signals
    preprocess_mic — preprocess mic-domain data to .pt format
    preprocess_sh_stft — preprocess Ambisonics STFT data to .pt format
    preprocess_sh_time — preprocess Ambisonics time-domain data to .pt format
    preprocess_dataset — batch-preprocess a folder of ex_N/ subfolders
    preprocess_dataset_multi — merge multiple array directories into one dataset
"""
from ambidrop.losses import si_snr, complex_si_snr
from ambidrop.dropouts import SHChannelDropout, PerChDropout
from ambidrop.signal_utils import find_ref_mic
from ambidrop.asm import encode_ambisonics, compute_asm_coefficients, apply_asm_filters
from ambidrop.preprocess import (
    preprocess_mic,
    preprocess_sh_stft,
    preprocess_sh_time,
    preprocess_dataset,
    preprocess_dataset_multi,
)
