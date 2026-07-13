"""
Dataset classes for IC Conv-TasNet covering training, validation, and inference.

Public interface:
    SimDS_preprocessed — loads preprocessed .pt files produced by preprocess_mic_time,
                         preprocess_sh_time, or preprocess_ambisonics_time and returns
                         the appropriate tuple for each pipeline stage.

Data formats (determined by the 'format' key in saved dicts, or tuple length):

    format='time'          (preprocess_mic_time)
        → 5-tuple: (noisy (M,T), clean (M,T), ref_id, array_name, ex_id)
        → used by Solver (baseline training) and baseline test evaluation

    format='ambidrop_test' (preprocess_ambisonics_time)
        → 4-tuple: (noisy_mic (M,T), clean_mic (M,T), anmt (9,T), clean_anm (T,))
        → used by AmbiDrop test evaluation only (never by Solver)

    2-tuple               (preprocess_sh_time)
        → 2-tuple: (real_acn (9,T), clean_a00 (T,))
        → used by Solver (AmbiDrop training)
"""
import sys
import os

import torch
from torch.utils.data import Dataset

project_root = os.path.join(os.path.dirname(__file__), '..')
if project_root not in sys.path:
    sys.path.append(project_root)


class SimDS_preprocessed(Dataset):
    """Loads preprocessed .pt files for all ConvTasNet pipeline stages."""
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
            fmt = data.get('format', '')
            if fmt == 'ambidrop_test':
                # AmbiDrop test: preprocess_ambisonics_time
                return (
                    data['noisy_mic'].float(),   # (M, T)
                    data['clean_mic'].float(),   # (M, T)
                    data['anmt'].float(),        # (9, T)
                    data['clean_anm'].float(),   # (T,)
                    data.get('ref_id', 0),       # int — reference mic for noisy SI-SDR
                )
            else:
                # Baseline: preprocess_mic_time — time-domain mic signals
                return (
                    data['noisy'].float(),              # (M, T)
                    data['clean'].float(),              # (M, T)
                    data.get('ref_id', 0),
                    data.get('array_name', ''),
                    data.get('ex_id', ''),
                )

        elif isinstance(data, (tuple, list)) and len(data) == 2:
            # AmbiDrop train: preprocess_sh_time — (real-ACN (9,T), a00 (T,))
            noisy, clean = data
            return noisy.float(), clean.float()

        else:
            raise ValueError(f"Unexpected data format in {file_name}")
