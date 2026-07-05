"""
Dataset classes for IC Conv-TasNet covering training, validation, and inference.

Public interface:
    MergedDataset — loads preprocessed .pt files listed in a metadata_list.pt file
    MatDataset — loads raw .mat Ambisonics files and converts to real-valued ACN
    PrecomputedASMDataset — reads precomputed real-ACN Ambisonics from anmt_array in anm.mat
    MicToRealAmbisonicsDataset — loads raw p.wav and computes real-ACN Ambisonics on-the-fly via ASM
    SimDS_preprocessed — loads preprocessed .pt files and converts STFT back to time domain
"""
import sys
import os

import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io
from scipy.linalg import svd

project_root = os.path.join(os.path.dirname(__file__), '..')
if project_root not in sys.path:
    sys.path.append(project_root)

from ambidrop.signal_utils import process_segment, tensor_to_istft, complex_acn_to_real_acn
from ambidrop.asm import encode_ambisonics


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



class PrecomputedASMDataset(Dataset):
    """
    Reads precomputed real-ACN Ambisonics from anm.mat (anmt_array field)
    instead of computing it on-the-fly from a steering matrix.

    Mirrors MicToRealAmbisonicsDataset: returns full-length signals with no
    onset-based windowing so results are directly comparable to RESULTS.md.

    Returns the same 4-tuple as MicToRealAmbisonicsDataset:
        (noisy_mic, clean_mic, anmt_array_tensor, clean_anm)
    """
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
        mat_data = scipy.io.loadmat(os.path.join(folder_path, "anm.mat"))

        noisy_mic, _ = sf.read(os.path.join(folder_path, "p.wav"))
        clean_mic, _ = sf.read(os.path.join(folder_path, "pDirect.wav"))
        noisy_mic = noisy_mic.T   # (M, T)
        clean_mic = clean_mic.T   # (M, T)

        anmt_array  = mat_data["anmt_array"].T.astype(np.float32)       # (9, T) real-ACN
        anmt_direct = mat_data["anmtDirect"][:, 0].real.astype(np.float32)  # (T,) a00

        return (
            torch.from_numpy(noisy_mic).float(),   # (M, T)
            torch.from_numpy(clean_mic).float(),   # (M, T)
            torch.from_numpy(anmt_array),          # (9, T)
            torch.from_numpy(anmt_direct),         # (T,)
        )


class MicToRealAmbisonicsDataset(Dataset):
    """Loads test data and computes real-ACN Ambisonics on-the-fly from steering matrix."""
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

        anmt, _ = encode_ambisonics(noisy_mic, self.V, sh_order=2, th=self.th, ph=self.ph, sh_type="real")
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
                    return noisy, clean_mic.float(), 0, '', ''
            elif len(data) == 2:
                noisy, clean = data
                if noisy.dim() == 3:  # STFT format (T_frames, F, 2C) → convert to time domain
                    noisy = tensor_to_istft(noisy, clean.shape[-1])
                # else: already time domain (C, T) from preprocess_sh_time
                return noisy, clean.float()
            else:
                raise ValueError(f"Unexpected tuple length {len(data)} in {file_name}")
        else:
            raise ValueError(f"Unexpected data type {type(data)} in {file_name}")
