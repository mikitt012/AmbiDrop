"""
PyTorch Dataset for FT-JNF that loads preprocessed .pt example files.

Public interface:
    SimDS_preprocessed — Dataset that auto-normalizes 2-tuple / 4-tuple / 5-tuple / dict on-disk formats to a unified dict with keys: noisy, clean, ref_id, array_name, ex_id
"""
import os
import torch
from torch.utils.data import Dataset


class SimDS_preprocessed(Dataset):
    """
    Loads preprocessed .pt files. Auto-detects format and normalizes to a dict.

    Handles three on-disk formats:
      - 2-tuple: (noisy_tf, clean_1ch)
      - 4-tuple: (noisy_tf_mic, clean_mic, noisy_tf_anm, clean_anm)
      - dict:    {'noisy': ..., 'clean': ..., 'ref_id': ..., ...}
    """

    def __init__(self, data_dir, data_type=None):
        if data_type is not None:
            self.data_dir = os.path.join(data_dir, data_type)
        else:
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
            data = torch.load(file_path, map_location='cpu')
        except Exception as e:
            print(f"[Error loading] {file_name}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        if isinstance(data, dict):
            result = {
                'noisy': data['noisy'],
                'clean': data['clean'].float(),
                'ref_id': data.get('ref_id', None),
                'array_name': data.get('array_name', ''),
                'ex_id': data.get('ex_id', ''),
            }
            if 'noisy_mic' in data:
                result['noisy_mic'] = data['noisy_mic']
                result['clean_mic'] = data['clean_mic'].float()
            return result
        elif isinstance(data, (tuple, list)):
            if len(data) == 2:
                noisy_tf, clean_1ch = data
                return {
                    'noisy': noisy_tf,
                    'clean': clean_1ch.float(),
                    'ref_id': -1,
                    'array_name': '',
                    'ex_id': file_name,
                }
            elif len(data) == 4:
                noisy_tf_mic, clean_mic, noisy_tf_anm, clean_anm = data
                return {
                    'noisy_mic': noisy_tf_mic,
                    'clean_mic': clean_mic.float(),
                    'noisy': noisy_tf_anm,
                    'clean': clean_anm.float(),
                    'ref_id': -1,
                    'array_name': '',
                    'ex_id': file_name,
                }
            elif len(data) == 5:
                noisy_tf, clean, ref_id, array_name, ex_id = data
                return {
                    'noisy': noisy_tf,
                    'clean': clean.float(),
                    'ref_id': ref_id,
                    'array_name': array_name,
                    'ex_id': ex_id,
                }
            else:
                raise ValueError(f"Unexpected tuple length {len(data)} in {file_name}")
        else:
            raise ValueError(f"Unexpected data type {type(data)} in {file_name}")

