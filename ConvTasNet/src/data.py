"""
Dataset classes and signal processing utilities for IC Conv-TasNet.
"""
import sys
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import scipy.io
from scipy.linalg import svd

try:
    from scipy.special import sph_harm
except ImportError:
    from scipy.special import sph_harm_y
    def sph_harm(m, n, phi, theta):
        return sph_harm_y(n, m, theta, phi)

project_root = os.path.join(os.path.dirname(__file__), '..', '..')
if project_root not in sys.path:
    sys.path.append(project_root)


# ── Signal processing helpers ────────────────────────────────────────────────

def process_segment(noisy, clean, target_samples=96000, threshold=1e-3):
    """Find first speech onset, slice target_samples, pad if needed."""
    C, T = noisy.shape
    ref_channel = clean.unsqueeze(0)
    mask = (clean.abs() > threshold).nonzero()
    first = mask[0].item() if mask.numel() > 0 else 0
    last = first + target_samples

    noisy_processed = noisy[:, first:last]
    clean_processed = ref_channel[:, first:last]

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
    """Convert STFT tensor [T, F, 2C] back to time-domain [C, T_samples]."""
    T, F, two_C = stft_tensor.shape
    C = two_C // 2
    real = stft_tensor[:, :, :C]
    imag = stft_tensor[:, :, C:]
    stft_complex = torch.complex(real, imag)
    stft_complex = stft_complex.permute(2, 1, 0)
    waveform = torch.istft(stft_complex, n_fft=512, hop_length=256, win_length=512,
                           window=torch.hamming_window(window_length=512),
                           center=True, normalized=False, onesided=True,
                           return_complex=False, length=length)
    return waveform


def compute_spherical_harmonics_matrix(N, theta, phi):
    """Compute SH matrix of shape ((N+1)^2, num_samples)."""
    assert(phi.min() >= 0.0)
    num_samples = phi.size
    num_harmonics = (N + 1) ** 2
    Y_matrix = np.zeros((num_harmonics, num_samples), dtype=complex)
    index = 0
    for n in range(N + 1):
        for m in range(-n, n + 1):
            Y_matrix[index, :] = sph_harm(m, n, phi, theta)
            index += 1
    return Y_matrix


def complex_acn_to_real_acn(complex_acn_buffer, max_order, sn3d):
    """Convert complex ACN buffer to real ACN buffer."""
    num_channels, num_samples = complex_acn_buffer.shape
    real_acn_buffer = np.zeros((num_channels, num_samples))
    for l in range(max_order + 1):
        sn3d_scale = 1.0 / np.sqrt(2 * l + 1) if sn3d else 1.0
        n_mid = l**2 + l
        real_acn_buffer[n_mid, :] = complex_acn_buffer[n_mid, :].real * sn3d_scale
        for m in range(1, l + 1):
            n_pos = l**2 + l + m
            n_neg = l**2 + l - m
            Y_pos = complex_acn_buffer[n_pos, :]
            Y_neg = complex_acn_buffer[n_neg, :]
            real_pos_m = (Y_neg + ((-1)**m) * Y_pos) / np.sqrt(2)
            real_acn_buffer[n_pos, :] = real_pos_m.real * sn3d_scale
            real_neg_m = 1j / np.sqrt(2) * (Y_pos - ((-1)**m) * Y_neg)
            real_acn_buffer[n_neg, :] = real_neg_m.real * sn3d_scale
    return real_acn_buffer


def array_ambisonics_time_domain(p, c_ASM, filt_samp=512):
    """Convert microphone signals to Ambisonics via time-domain convolution."""
    T = p.shape[1]
    N_mic = p.shape[0]
    num_harmonics = c_ASM.shape[0]
    anmt_array = np.zeros((num_harmonics, T), dtype=np.float32)
    for j_idx in range(num_harmonics):
        c_f = c_ASM[j_idx, :, :].T
        c_time = np.fft.irfft(c_f, n=filt_samp, axis=1)
        c_time_cs = np.roll(c_time, filt_samp // 2, axis=1)
        first_col = c_time_cs[:, [0]]
        tail_reversed = c_time_cs[:, :0:-1]
        c_time_filter = np.concatenate([first_col, tail_reversed], axis=1)
        tmp = np.zeros(T, dtype=np.float64)
        for m in range(N_mic):
            full_conv = np.convolve(p[m, :].astype(np.float64),
                                    c_time_filter[m, :].astype(np.float64), mode='full')
            start_idx = filt_samp // 2
            tmp += full_conv[start_idx : start_idx + T]
        anmt_array[j_idx, :] = tmp
    return anmt_array


def array_ambisonics_time(p, V, th, ph, N):
    """Full ASM pipeline: compute coefficients with Tikhonov and encode."""
    from ASM.tikhonov import tikhonov

    V_t = V.T
    Y = compute_spherical_harmonics_matrix(N, th, ph)
    Y_real = complex_acn_to_real_acn(Y, 2, sn3d=False)
    Y = Y_real

    cnm = np.zeros(((N+1)**2, V_t.shape[1], V_t.shape[2]), dtype=np.complex128)
    for nm in range((N+1)**2):
        for f in range(V_t.shape[1]):
            cnm[nm, f] = tikhonov(A=V_t[:, f, :].conj(), b=Y[nm, :])

    return array_ambisonics_time_domain(p, cnm)


# ── Dataset classes ──────────────────────────────────────────────────────────

class MergedDataset(Dataset):
    """Loads preprocessed .pt files listed in a metadata file."""
    def __init__(self, processed_dir):
        self.file_list = torch.load(os.path.join(processed_dir, "metadata_list.pt"), map_location="cpu")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = torch.load(self.file_list[idx], map_location="cpu")
        return data['noisy'], data['clean'], data['ref_id'], data['array_name'], data['ex_id']


class MatDataset(Dataset):
    """Loads raw .mat files with Ambisonics signals, converts to real-valued SH."""
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
        self.file_list.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        mat_data = scipy.io.loadmat(file_path)
        anmt = mat_data['anmt'].astype('complex64')
        anmt_direct = mat_data['anmtDirect'].astype('float32')
        anmt_tensor = torch.from_numpy(anmt)
        anmt_direct_tensor = torch.from_numpy(anmt_direct)
        anmt, anmtDirect = process_segment(anmt_tensor.T, anmt_direct_tensor[:, 0])
        anmt_real = complex_acn_to_real_acn(anmt, 2, sn3d=False)
        anmt_real = torch.from_numpy(anmt_real)
        return anmt_real.float().squeeze(), anmtDirect.squeeze()


class MatDatasetTest(Dataset):
    """Loads test .mat files from folder structure (ex_*/anm.mat + p.wav)."""
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.folder_list = sorted(
            [d for d in os.listdir(data_dir)
             if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("ex_")],
            key=lambda f: int(f.split('_')[1])
        )

    def __len__(self):
        return len(self.folder_list)

    def __getitem__(self, idx):
        import soundfile as sf
        folder_path = os.path.join(self.data_dir, self.folder_list[idx])
        mat_file = os.path.join(folder_path, "anm.mat")
        array_file = os.path.join(folder_path, "p.wav")
        direct_file = os.path.join(folder_path, "pDirect.wav")

        mat_data = scipy.io.loadmat(mat_file)
        noisy_mic, _ = sf.read(array_file)
        noisy_mic = noisy_mic.T
        clean_mic, _ = sf.read(direct_file)
        clean_mic = clean_mic.T

        anmt = mat_data['anmt'].astype('complex64')
        anmt_direct = mat_data['anmtDirect'].astype('float32')
        anmt_tensor = torch.from_numpy(anmt)
        anmt_direct_tensor = torch.from_numpy(anmt_direct)

        anmt, anmtDirect = process_segment(anmt_tensor.T, anmt_direct_tensor[:, 0])
        anmt_real = complex_acn_to_real_acn(anmt, 2, sn3d=False)
        anmt_real = torch.from_numpy(anmt_real)

        noisy_mic = torch.from_numpy(noisy_mic).float()
        clean_mic = torch.from_numpy(clean_mic).float()

        return noisy_mic, clean_mic, anmt_real.float().squeeze(), anmtDirect.squeeze()


class MatDatasetTest_ASM(Dataset):
    """Loads test data and computes ASM on-the-fly from steering matrix."""
    def __init__(self, data_dir, V, th, ph):
        self.data_dir = data_dir
        self.V = V
        self.th = th
        self.ph = ph
        self.folder_list = sorted(
            [d for d in os.listdir(data_dir)
             if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("ex_")],
            key=lambda f: int(f.split('_')[1])
        )

    def __len__(self):
        return len(self.folder_list)

    def __getitem__(self, idx):
        import soundfile as sf
        folder_path = os.path.join(self.data_dir, self.folder_list[idx])
        array_file = os.path.join(folder_path, "p.wav")
        direct_file = os.path.join(folder_path, "pDirect.wav")
        mat_file = os.path.join(folder_path, "anm.mat")

        noisy_mic, _ = sf.read(array_file)
        noisy_mic = noisy_mic.T
        clean_mic, _ = sf.read(direct_file)
        clean_mic = clean_mic.T

        anmt = array_ambisonics_time(noisy_mic, self.V, self.th, self.ph, N=2)
        anmt_tensor = torch.from_numpy(anmt).float()
        noisy_mic_tensor = torch.from_numpy(noisy_mic).float()
        clean_mic_tensor = torch.from_numpy(clean_mic).float()

        # a00 clean from .mat (what the AmbiDrop model was trained against)
        mat_data = scipy.io.loadmat(mat_file)
        anmt_direct = mat_data['anmtDirect'].astype('float32')
        clean_anm = torch.from_numpy(anmt_direct[:, 0]).float()

        return noisy_mic_tensor, clean_mic_tensor, anmt_tensor, clean_anm


class SimDS_preprocessed(Dataset):
    """Loads preprocessed .pt files, converts STFT back to time-domain."""
    def __init__(self, data_dir, data_type='.', mode='ambidrop'):
        self.data_dir = os.path.join(data_dir, data_type) if data_type != '.' else data_dir
        self.mode = mode
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
            data = torch.load(file_path, map_location='cpu')
        except Exception as e:
            print(f"[Error loading] {file_name}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        if isinstance(data, dict):
            noisy_tf = data['noisy']
            clean = data['clean'].float()
            noisy = tensor_to_istft(noisy_tf, clean.shape[-1])
            if clean.dim() == 2:
                ref_id = data.get('ref_id', 0)
                return noisy, clean, ref_id, data.get('array_name', ''), data.get('ex_id', '')
            return noisy, clean
        elif isinstance(data, (tuple, list)):
            if len(data) == 4:
                noisy_tf_mic, clean_mic, noisy_tf_anm, clean_anm = data
                if self.mode == 'ambidrop':
                    clean_anm = clean_anm.unsqueeze(0) if clean_anm.dim() == 1 else clean_anm
                    noisy = tensor_to_istft(noisy_tf_anm, clean_anm.shape[-1])
                    return noisy, clean_anm.float()
                else:
                    noisy = tensor_to_istft(noisy_tf_mic, clean_mic.shape[-1])
                    return noisy, clean_mic.float()
            elif len(data) == 2:
                noisy_tf, clean = data
                noisy = tensor_to_istft(noisy_tf, clean.shape[-1])
                return noisy, clean.float()
            else:
                raise ValueError(f"Unexpected tuple length {len(data)} in {file_name}")
        else:
            raise ValueError(f"Unexpected data type {type(data)} in {file_name}")
