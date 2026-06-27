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

import matplotlib
matplotlib.use("Agg")  # important for cluster (no display)
import matplotlib.pyplot as plt

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

def _calculate_coefficients(V,N,th,ph,plot):
    V = V.T
    Y = compute_spherical_harmonics_matrix(N, th, ph)
    # Y_yo = np.load('/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/aria_ds/Y.npy')
    # Y_yo = Y_yo.T
    # Y = Y_yo
    cnm = np.zeros(((N+1)**2, V.shape[1], V.shape[2]), dtype=np.complex128)
    from ASM.tikhonov import tikhonov
    # from ASM.utils import reconstruct_frequency_sh_spectrum_full
    for nm in range((N+1)**2):
        for f in range(V.shape[1]):
            # cnm[nm,f] = tikhonov(A=V[:, f, :].conj(), b=Y[nm, :], lam=1e-3)
            # cnm[nm,:] = np.array([np.linalg.lstsq(V[:, f, :].conj(), Y[nm, :], rcond=None)[0] for f in range(V.shape[1])])

            # cnm[nm,f] = tikhonov(A=V[:, f, :].conj(), b=Y[nm, :])
            cnm[nm, f, :] = svd_inversion(A=V[:, f, :].T, b=Y[nm, :], snr_lin=1000)
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


    # cnm = np.load('/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/aria_ds/cnm.npy')
    # cnm = cnm.transpose(1, 2, 0)
    # nfft = 332
    # cnm = cnm[:, :nfft//2 + 1, :]
    
    if plot:
        mse = calculate_error(cnm, Y, V)
        n_fft = 512
        pos_freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
        plot_nmse(mse, pos_freqs, save_path="nmse_channels_measuredV.png")

    return cnm

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

steer_path = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/ATF_mismatch_ds/rigid sphere.mat"
steer_mat = loadmat(steer_path)
V = steer_mat["V"]          # numpy array, shape (CH, F, Q)

# --- 2. Load grid (theta, phi): 1 x Q ---
grid_path = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment_full_anm/utils/Lebvedev2702.mat"
grid_mat = loadmat(grid_path)
th = grid_mat["th"].squeeze()    # shape (Q,)
ph = grid_mat["ph"].squeeze()    # shape (Q,)
fs = 16000
N=2

plot = True
cnm = _calculate_coefficients(V,N,th,ph,plot)