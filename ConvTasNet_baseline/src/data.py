# Created on 2018/12
# Author: Kaituo XU
"""
Logic:
1. AudioDataLoader generate a minibatch from AudioDataset, the size of this
   minibatch is AudioDataLoader's batchsize. For now, we always set
   AudioDataLoader's batchsize as 1. The real minibatch size we care about is
   set in AudioDataset's __init__(...). So actually, we generate the
   information of one minibatch in AudioDataset.
2. After AudioDataLoader getting one minibatch from AudioDataset,
   AudioDataLoader calls its collate_fn(batch) to process this minibatch.

Input:
    Mixtured WJS0 tr, cv and tt path
Output:
    One batch at a time.
    Each inputs's shape is B x T
    Each targets's shape is B x C x T
"""
import sys
import os
project_root = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement"
if project_root not in sys.path:
    sys.path.append(project_root)

import json
import math

import numpy as np
import torch
import torch.utils.data as data

import librosa

import numpy as np
import librosa
import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import scipy
import torch.nn.functional as F
import scipy.io
from scipy.special import sph_harm, spherical_jn, spherical_yn
from scipy.linalg import svd
from scipy.signal import fftconvolve
from scipy.special import lpmv
import soundfile as sf

def process_segment(noisy, clean, target_samples=96000, threshold=1e-3):
    """
    Finds the first occurrence of speech, slices target_samples from there,
    and pads if the remaining signal is too short.
    tensor shape: [Channels, Time]
    """
    C, T = noisy.shape
    
    # 1. Find the first index where any channel exceeds the threshold
    ref_channel = clean.unsqueeze(0)
    mask = (clean.abs() > threshold).nonzero()
    
    if mask.numel() > 0:
        first = mask[0].item()
    else:
        first = 0 # Fallback if speech never hits threshold
    
    last = first + target_samples

    # Slice both signals using the SAME 'first' point found on ref mic
    noisy_processed = noisy[:, first:last]
    clean_processed = ref_channel[:, first:last]

    # --- STANDARDIZE LENGTH (Padding/Truncating) ---
    curr_len = noisy_processed.shape[1]
    if curr_len < target_samples:
        padding_needed = target_samples - curr_len
        noisy_processed = F.pad(noisy_processed, (0, padding_needed), "constant", 0)
        clean_processed = F.pad(clean_processed, (0, padding_needed), "constant", 0)
    else:
        noisy_processed = noisy_processed[:, :target_samples]
        clean_processed = clean_processed[:, :target_samples]
        
    return noisy_processed, clean_processed.squeeze()

def tensor_to_istft(stft_tensor, length):
    """
    stft_tensor: [T, F, 2C] (where 2C is Real_Ch1, ..., Real_ChC, Imag_Ch1, ..., Imag_ChC)
    Returns: [C, T_samples]
    """
    T, F, two_C = stft_tensor.shape
    C = two_C // 2
    
    # 1. Split into Real and Imaginary parts
    # Assuming the first C are real, next C are imaginary
    real = stft_tensor[:, :, :C] # [T, F, C]
    imag = stft_tensor[:, :, C:] # [T, F, C]
    
    # 2. Create a complex tensor
    # Shape becomes [T, F, C] complex
    stft_complex = torch.complex(real, imag)
    
    # 3. Permute to [C, F, T] 
    # torch.istft expects (Channel, Freq, Time) or (Batch, Freq, Time)
    stft_complex = stft_complex.permute(2, 1, 0)
    
    # 4. Perform iSTFT
    # window should be on the same device as the tensor
    waveform = torch.istft(stft_complex, n_fft=512, hop_length=256, win_length=512, window=torch.hamming_window(window_length=512), center=True, normalized=False, onesided=True, return_complex=False, length=length)
    
    return waveform # Result is [C, T_samples]

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

import numpy as np

def complex_acn_to_real_acn(complex_acn_buffer, max_order, sn3d):
    """
    Converts a complex ACN buffer (N3D) to a real ACN buffer (SN3D).
    
    Args:
        complex_acn_buffer: 2D numpy array [channels, samples]
        max_order: The maximum Ambisonic order L
        
    Returns:
        real_acn_buffer: 2D numpy array [channels, samples] (Real-valued)
    """
    num_channels, num_samples = complex_acn_buffer.shape
    real_acn_buffer = np.zeros((num_channels, num_samples))
    
    for l in range(max_order + 1):
        if sn3d:
            # SN3D normalization factor (AmbiX standard)
            sn3d_scale = 1.0 / np.sqrt(2 * l + 1)
        else:
            sn3d_scale = 1.0
        
        # Center index for m=0 at this order
        n_mid = l**2 + l 
        
        # 1. Process m = 0
        real_acn_buffer[n_mid, :] = complex_acn_buffer[n_mid, :].real * sn3d_scale
        
        # 2. Process pairs of m and -m
        for m in range(1, l + 1):
            n_pos = l**2 + l + m     # Index for +m
            n_neg = l**2 + l - m     # Index for -m
            
            # Complex coefficients
            Y_pos = complex_acn_buffer[n_pos, :]
            Y_neg = complex_acn_buffer[n_neg, :]
            
            # --- Calculate Real SH Components ---
            
            # Real m > 0 (Cosine-like): 
            # S_l^m = 1/sqrt(2) * (Y_l^{-m} + (-1)^m * Y_l^m)
            real_pos_m = (Y_neg + ((-1)**m) * Y_pos) / np.sqrt(2)
            real_acn_buffer[n_pos, :] = real_pos_m.real * sn3d_scale
            
            # Real m < 0 (Sine-like):
            # S_l^-m = i/sqrt(2) * (Y_l^m - (-1)^m * Y_l^{-m})
            real_neg_m = 1j / np.sqrt(2) * (Y_pos - ((-1)**m) * Y_neg)
            real_acn_buffer[n_neg, :] = real_neg_m.real * sn3d_scale
            
    return real_acn_buffer

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

    # fs_target = 16000
    # fs_rec = 48000
    # anmt_array = resample_poly(anmt_array, up=fs_target, down=fs_rec, axis=1)

    return anmt_array

def array_ambisonics_time(p, V, th, ph, N):
    """
    Python version of the MATLAB array->Ambisonics routine.
    p  : (M, T) microphone signals
    V  : (M, F, Q) steering vectors
    th : (Q,) or (1, Q)
    ph : (Q,) or (1, Q)
    harmonics: 0-based indices of SH coefficients to compute
    """

    def _calculate_coefficients(V,N,th,ph):
        V = V.T
        Y = compute_spherical_harmonics_matrix(N, th, ph)
        Y_real = complex_acn_to_real_acn(Y, 2, sn3d=False)
        Y = Y_real

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

        # # 1. Handle DC (Frequency index 0) for all channels
        # cnm[1:, 0, :] = 0.0                # Zero out higher orders for all channels
        # cnm[0, 0, :] = cnm[0, 0, :].real    # Force omni (n=0, m=0) to be real for all channels
        # # 2. Handle Nyquist (Frequency index F//2) for all channels
        # F = cnm.shape[1]
        # if F % 2 == 0:
        #     nyq_idx = F // 2
        #     cnm[1:, nyq_idx, :] = 0.0
        #     cnm[0, nyq_idx, :] = cnm[0, nyq_idx, :].real

        # from ASM.validate import is_signal_frequency_sh_valid
        # assert is_signal_frequency_sh_valid(cnm, freq_axis=1, sh_axis=0)
        
        mse = calculate_error(cnm, Y, V)
        n_fft = 512
        fs = 16000
        pos_freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
        # plot_nmse(mse, pos_freqs, save_path="nmse_channels.png")

        return cnm

    def calc_ambisonics(mic_signals, Cnm=None, Cnm_domain='frequency'):
        if Cnm_domain == 'frequency':
            # convert to time
            Cnm = np.fft.ifft(np.conj(Cnm), axis=1)
        from ASM.utils import convolve_and_sum_any
        rec_amb = convolve_and_sum_any(
            signal1=Cnm,
            signal2=mic_signals,
            signal1_conj=False,
            time_dim1=1,
            time_dim2=1,
            channel_dims1=2,
            channel_dims2=0,
        )

        # from ASM.utils import convolve_and_sum
        # rec_amb = convolve_and_sum(
        #     signal1=Cnm,
        #     signal2=mic_signals,
        #     time_dim1=1,
        #     time_dim2=1,
        #     channel_dims1=2,
        #     channel_dims2=0,
        # )

        return rec_amb.T

    def calculate_error(c, Y, V):
        mse = np.zeros((c.shape[0], V.shape[1]))
        for nm in range(mse.shape[0]):
            for f in range(V.shape[1]):
                tmp = np.linalg.norm(np.conj(c[nm, f, :].T) @ V[:, f, :].T - Y[nm, :])
                mse[nm, f] = np.square(tmp / np.linalg.norm(Y[nm, :]))
                # mse[nm, f] = tmp
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

    cnm = _calculate_coefficients(V,N,th,ph)
    # cnm = _calculate_coefficients_like_matlab(V, N, th, ph, SNR_lin=10)

    anmt_array = array_ambisonics_time_domain(p, cnm)
    # anmt_array = calc_ambisonics(p, Cnm=cnm, Cnm_domain='frequency')

    return anmt_array

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

class MergedDataset(Dataset):
    def __init__(self, processed_dir):
        # Load the list of all .pt file paths
        self.file_list = torch.load(os.path.join(processed_dir, "metadata_list.pt"))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Load the pre-processed tensor
        data = torch.load(self.file_list[idx])
        
        # Return noisy, clean, and any ref info you need
        return data['noisy'], data['clean'], data['ref_id'], data['array_name'], data['ex_id']

class MatDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        # Get all .mat files in the directory
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
        
        # Optional: Sort them numerically so ex_1 is before ex_2
        self.file_list.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Construct full path using os.path.join
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        
        # Load the .mat file
        mat_data = scipy.io.loadmat(file_path)
        
        # Extract fields
        # Note: we use .copy() because scipy arrays can be read-only
        anmt = mat_data['anmt'].astype('complex64')
        anmt_direct = mat_data['anmtDirect'].astype('float32')
        
        # Convert to PyTorch Tensors
        # Usually shapes are [Samples, Channels] - T, C
        anmt_tensor = torch.from_numpy(anmt)
        anmt_direct_tensor = torch.from_numpy(anmt_direct)

        anmt, anmtDirect = process_segment(anmt_tensor.T, anmt_direct_tensor[:,0])
        anmt_real = complex_acn_to_real_acn(anmt, 2, sn3d=False)
        anmt_real = torch.from_numpy(anmt_real)
        
        return anmt_real.float().squeeze(), anmtDirect.squeeze()

class MatDatasetTest(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
        self.folder_list = [d for d in os.listdir(data_dir) 
                            if os.path.isdir(os.path.join(data_dir, d)) 
                            and d.startswith('ex_')]
        
        # 2. Sort them numerically so ex_1, ex_2, ... ex_10 are in order
        self.folder_list.sort(key=lambda x: int(x.split('_')[1]))

    def __len__(self):
        return len(self.folder_list)

    def __getitem__(self, idx):
        # 3. Construct path to the nested anm.mat file
        # Path becomes: data_dir / ex_X / anm.mat
        folder_name = self.folder_list[idx]
        file_path = os.path.join(self.data_dir, folder_name, 'anm.mat')
        
        # Load the .mat file
        mat_data = scipy.io.loadmat(file_path)
        
        # 4. Extract fields (using .copy() or .astype() to ensure they are writable)
        # Note: Ensure these keys match exactly what's inside your anm.mat
        anmt = mat_data['anmt_array'].astype('float32')
        anmt_direct = mat_data['anmtDirect'].astype('float32')
        
        # Convert to PyTorch Tensors
        anmt_tensor = torch.from_numpy(anmt)
        anmt_direct_tensor = torch.from_numpy(anmt_direct)
        
        anmt, anmtDirect = process_segment(anmt_tensor.T, anmt_direct_tensor[:,0])
        
        return anmt, anmtDirect

class MatDatasetTest_ASM(Dataset):
    def __init__(self, data_dir, V, th, ph):
        self.data_dir = data_dir
        self.V = V
        self.th = th
        self.ph = ph

        self.folder_list = [d for d in os.listdir(data_dir) 
                            if os.path.isdir(os.path.join(data_dir, d)) 
                            and d.startswith('ex_')]
        
        # 2. Sort them numerically so ex_1, ex_2, ... ex_10 are in order
        self.folder_list.sort(key=lambda x: int(x.split('_')[1]))

    def __len__(self):
        return len(self.folder_list)

    def __getitem__(self, idx):
        # 3. Construct path to the nested anm.mat file
        # Path becomes: data_dir / ex_X / anm.mat
        folder_name = self.folder_list[idx]
        file_path = os.path.join(self.data_dir, folder_name, 'anm.mat')
        array_file = os.path.join(self.data_dir, folder_name, "p.wav")
        direct_file = os.path.join(self.data_dir, folder_name, "pDirect.wav")
        
        # Load the .mat file
        mat_data = scipy.io.loadmat(file_path)
        noisy_mic, fs_rec = sf.read(array_file)   # (T, C_mic)
        noisy_mic = noisy_mic.T       # (C_mic, T)
        clean_mic, fs_rec = sf.read(direct_file)   # (T, C_mic)
        clean_mic = clean_mic.T       # (C_mic, T)
        
        # 4. Extract fields (using .copy() or .astype() to ensure they are writable)
        # Note: Ensure these keys match exactly what's inside your anm.mat
        # anmt = mat_data['anmt_array'].astype('complex64')

        anmt = array_ambisonics_time(noisy_mic, self.V, self.th, self.ph, N=2)
        # anmt = mat_data['anmt'].astype('complex64')
        anmt_direct = mat_data['anmtDirect'].astype('float32')

        # Convert to PyTorch Tensors
        anmt_tensor = torch.from_numpy(anmt)
        anmt_direct_tensor = torch.from_numpy(anmt_direct)

        # anmt, anmtDirect = process_segment(anmt_tensor.T, anmt_direct_tensor[:,0])
        # anmt_real = complex_acn_to_real_acn(anmt, 2, sn3d=True)
        # anmt_real = torch.from_numpy(anmt_real)
        
        return noisy_mic, clean_mic, anmt_tensor, anmt_direct_tensor[:,0]
        # return anmt_real.float(), anmtDirect

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

        # try:
        #     data = torch.load(file_path, map_location='cpu')  # Load from .pt file
        #     noisy_speech_tf, clean_speech_1ch = data
        #     noisy_speech_tf = noisy_speech_tf[:, :, [0, 1, 3, 4, 8, 9, 10, 12, 13, 17]]
        # except Exception as e:
        #     print(f"[Error loading] {file_name}: {e}")
        #     return self.__getitem__((idx + 1) % len(self))  # try next valid sample

        # return noisy_speech_tf, clean_speech_1ch.float()

        try:
            data = torch.load(file_path, map_location='cpu')  # Load from .pt file
            noisy_tf_mic, clean_mic, noisy_tf_anm, clean_anm = data
        except Exception as e:
            print(f"[Error loading] {file_name}: {e}")
            return self.__getitem__((idx + 1) % len(self))  # try next valid sample
        
        # clean_mic = clean_mic.unsqueeze(0)
        noisy = tensor_to_istft(noisy_tf_mic, clean_mic.shape[1])

        return noisy, clean_mic.float()

class SimDS(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

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

        clean_anm = clean_speech_1ch.unsqueeze(0)
        # print(clean_anm.shape)
        noisy = tensor_to_istft(noisy_speech_tf, clean_anm.shape[1])

        return noisy, clean_anm.float()


class AudioDataset(data.Dataset):

    def __init__(self, json_dir, batch_size, sample_rate=8000, segment=4.0, cv_maxlen=8.0):
        """
        Args:
            json_dir: directory including mix.json, s1.json and s2.json
            segment: duration of audio segment, when set to -1, use full audio

        xxx_infos is a list and each item is a tuple (wav_file, #samples)
        """
        super(AudioDataset, self).__init__()
        mix_json = os.path.join(json_dir, 'mix.json')
        s1_json = os.path.join(json_dir, 's1.json')
        s2_json = os.path.join(json_dir, 's2.json')
        with open(mix_json, 'r') as f:
            mix_infos = json.load(f)
        with open(s1_json, 'r') as f:
            s1_infos = json.load(f)
        with open(s2_json, 'r') as f:
            s2_infos = json.load(f)
        # sort it by #samples (impl bucket)
        def sort(infos): return sorted(
            infos, key=lambda info: int(info[1]), reverse=True)
        sorted_mix_infos = sort(mix_infos)
        sorted_s1_infos = sort(s1_infos)
        sorted_s2_infos = sort(s2_infos)
        if segment >= 0.0:
            # segment length and count dropped utts
            segment_len = int(segment * sample_rate)  # 4s * 8000/s = 32000 samples
            drop_utt, drop_len = 0, 0
            for _, sample in sorted_mix_infos:
                if sample < segment_len:
                    drop_utt += 1
                    drop_len += sample
            print("Drop {} utts({:.2f} h) which is short than {} samples".format(
                drop_utt, drop_len/sample_rate/36000, segment_len))
            # generate minibach infomations
            minibatch = []
            start = 0
            while True:
                num_segments = 0
                end = start
                part_mix, part_s1, part_s2 = [], [], []
                while num_segments < batch_size and end < len(sorted_mix_infos):
                    utt_len = int(sorted_mix_infos[end][1])
                    if utt_len >= segment_len:  # skip too short utt
                        num_segments += math.ceil(utt_len / segment_len)
                        # Ensure num_segments is less than batch_size
                        if num_segments > batch_size:
                            # if num_segments of 1st audio > batch_size, skip it
                            if start == end: end += 1
                            break
                        part_mix.append(sorted_mix_infos[end])
                        part_s1.append(sorted_s1_infos[end])
                        part_s2.append(sorted_s2_infos[end])
                    end += 1
                if len(part_mix) > 0:
                    minibatch.append([part_mix, part_s1, part_s2,
                                      sample_rate, segment_len])
                if end == len(sorted_mix_infos):
                    break
                start = end
            self.minibatch = minibatch
        else:  # Load full utterance but not segment
            # generate minibach infomations
            minibatch = []
            start = 0
            while True:
                end = min(len(sorted_mix_infos), start + batch_size)
                # Skip long audio to avoid out-of-memory issue
                if int(sorted_mix_infos[start][1]) > cv_maxlen * sample_rate:
                    start = end
                    continue
                minibatch.append([sorted_mix_infos[start:end],
                                  sorted_s1_infos[start:end],
                                  sorted_s2_infos[start:end],
                                  sample_rate, segment])
                if end == len(sorted_mix_infos):
                    break
                start = end
            self.minibatch = minibatch

    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)


class AudioDataLoader(data.DataLoader):
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """

    def __init__(self, *args, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


def _collate_fn(batch):
    """
    Args:
        batch: list, len(batch) = 1. See AudioDataset.__getitem__()
    Returns:
        mixtures_pad: B x T, torch.Tensor
        ilens : B, torch.Tentor
        sources_pad: B x C x T, torch.Tensor
    """
    # batch should be located in list
    assert len(batch) == 1
    mixtures, sources = load_mixtures_and_sources(batch[0])

    # get batch of lengths of input sequences
    ilens = np.array([mix.shape[0] for mix in mixtures])

    # perform padding and convert to tensor
    pad_value = 0
    mixtures_pad = pad_list([torch.from_numpy(mix).float()
                             for mix in mixtures], pad_value)
    ilens = torch.from_numpy(ilens)
    sources_pad = pad_list([torch.from_numpy(s).float()
                            for s in sources], pad_value)
    # N x T x C -> N x C x T
    sources_pad = sources_pad.permute((0, 2, 1)).contiguous()
    return mixtures_pad, ilens, sources_pad


# Eval data part
from preprocess import preprocess_one_dir

class EvalDataset(data.Dataset):

    def __init__(self, mix_dir, mix_json, batch_size, sample_rate=8000):
        """
        Args:
            mix_dir: directory including mixture wav files
            mix_json: json file including mixture wav files
        """
        super(EvalDataset, self).__init__()
        assert mix_dir != None or mix_json != None
        if mix_dir is not None:
            # Generate mix.json given mix_dir
            preprocess_one_dir(mix_dir, mix_dir, 'mix',
                               sample_rate=sample_rate)
            mix_json = os.path.join(mix_dir, 'mix.json')
        with open(mix_json, 'r') as f:
            mix_infos = json.load(f)
        # sort it by #samples (impl bucket)
        def sort(infos): return sorted(
            infos, key=lambda info: int(info[1]), reverse=True)
        sorted_mix_infos = sort(mix_infos)
        # generate minibach infomations
        minibatch = []
        start = 0
        while True:
            end = min(len(sorted_mix_infos), start + batch_size)
            minibatch.append([sorted_mix_infos[start:end],
                              sample_rate])
            if end == len(sorted_mix_infos):
                break
            start = end
        self.minibatch = minibatch

    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)


class EvalDataLoader(data.DataLoader):
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """

    def __init__(self, *args, **kwargs):
        super(EvalDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn_eval


def _collate_fn_eval(batch):
    """
    Args:
        batch: list, len(batch) = 1. See AudioDataset.__getitem__()
    Returns:
        mixtures_pad: B x T, torch.Tensor
        ilens : B, torch.Tentor
        filenames: a list contain B strings
    """
    # batch should be located in list
    assert len(batch) == 1
    mixtures, filenames = load_mixtures(batch[0])

    # get batch of lengths of input sequences
    ilens = np.array([mix.shape[0] for mix in mixtures])

    # perform padding and convert to tensor
    pad_value = 0
    mixtures_pad = pad_list([torch.from_numpy(mix).float()
                             for mix in mixtures], pad_value)
    ilens = torch.from_numpy(ilens)
    return mixtures_pad, ilens, filenames


# ------------------------------ utils ------------------------------------
def load_mixtures_and_sources(batch):
    """
    Each info include wav path and wav duration.
    Returns:
        mixtures: a list containing B items, each item is T np.ndarray
        sources: a list containing B items, each item is T x C np.ndarray
        T varies from item to item.
    """
    mixtures, sources = [], []
    mix_infos, s1_infos, s2_infos, sample_rate, segment_len = batch
    # for each utterance
    for mix_info, s1_info, s2_info in zip(mix_infos, s1_infos, s2_infos):
        mix_path = mix_info[0]
        s1_path = s1_info[0]
        s2_path = s2_info[0]
        assert mix_info[1] == s1_info[1] and s1_info[1] == s2_info[1]
        # read wav file
        mix, _ = librosa.load(mix_path, sr=sample_rate)
        s1, _ = librosa.load(s1_path, sr=sample_rate)
        s2, _ = librosa.load(s2_path, sr=sample_rate)
        # merge s1 and s2
        s = np.dstack((s1, s2))[0]  # T x C, C = 2
        utt_len = mix.shape[-1]
        if segment_len >= 0:
            # segment
            for i in range(0, utt_len - segment_len + 1, segment_len):
                mixtures.append(mix[i:i+segment_len])
                sources.append(s[i:i+segment_len])
            if utt_len % segment_len != 0:
                mixtures.append(mix[-segment_len:])
                sources.append(s[-segment_len:])
        else:  # full utterance
            mixtures.append(mix)
            sources.append(s)
    return mixtures, sources


def load_mixtures(batch):
    """
    Returns:
        mixtures: a list containing B items, each item is T np.ndarray
        filenames: a list containing B strings
        T varies from item to item.
    """
    mixtures, filenames = [], []
    mix_infos, sample_rate = batch
    # for each utterance
    for mix_info in mix_infos:
        mix_path = mix_info[0]
        # read wav file
        mix, _ = librosa.load(mix_path, sr=sample_rate)
        mixtures.append(mix)
        filenames.append(mix_path)
    return mixtures, filenames


def pad_list(xs, pad_value):
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, * xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad


if __name__ == "__main__":
    import sys
    json_dir, batch_size = sys.argv[1:3]
    dataset = AudioDataset(json_dir, int(batch_size))
    data_loader = AudioDataLoader(dataset, batch_size=1,
                                  num_workers=4)
    for i, batch in enumerate(data_loader):
        mixtures, lens, sources = batch
        print(i)
        print(mixtures.size())
        print(sources.size())
        print(lens)
        if i < 10:
            print(mixtures)
            print(sources)
