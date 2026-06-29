import os
import torch
from torch.utils.data import Dataset
import scipy.io

from ambidrop.signal_utils import add_white_noise_numpy, pad_or_truncate_numpy
from ambidrop.constants import get_device


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
            return {
                'noisy': data['noisy'],
                'clean': data['clean'].float(),
                'ref_id': data.get('ref_id', None),
                'array_name': data.get('array_name', ''),
                'ex_id': data.get('ex_id', ''),
            }
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


class SimDS(Dataset):
    """
    Loads raw .mat files and converts to STFT representation.

    Args:
        data_dir: Base directory
        data_type: Subdirectory name
        ambisonics: If True, loads 'anm_t'/'anm_clean_rir_t' fields;
                    otherwise loads 'p'/'pDirect' fields
    """

    def __init__(self, data_dir, data_type, ambisonics=False):
        self.data_dir = os.path.join(data_dir, data_type)
        self.ambisonics = ambisonics

        files_list = os.listdir(self.data_dir)
        if ambisonics:
            self.ex_list = [f for f in files_list if f.startswith("Ambisonics_") and f.endswith(".mat")]
        else:
            self.ex_list = [f for f in files_list if 'ex' in f and f.endswith(".mat")]

    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, idx):
        device = get_device()
        file_path = os.path.join(self.data_dir, self.ex_list[idx])
        ex = scipy.io.loadmat(file_path)

        if self.ambisonics:
            noisy_speech = ex['anm_t']
            clean_speech = ex['anm_clean_rir_t']
        else:
            noisy_speech = ex['p']
            clean_speech = ex['pDirect']

        noisy_speech = add_white_noise_numpy(noisy_speech, 30)

        noisy_speech = noisy_speech.T
        clean_speech = clean_speech.T

        noisy_speech = pad_or_truncate_numpy(noisy_speech, 120000)
        clean_speech = pad_or_truncate_numpy(clean_speech, 120000)

        noisy_real = torch.from_numpy(noisy_speech.real).float().to(device)
        noisy_imag = torch.from_numpy(noisy_speech.imag).float().to(device)
        clean = torch.from_numpy(clean_speech).to(torch.complex64).to(device)

        win = torch.hamming_window(512).to(device)
        noisy_real_tf = torch.stft(noisy_real, n_fft=512, hop_length=256, win_length=512,
                                   window=win, center=True, normalized=False,
                                   onesided=True, return_complex=True)
        noisy_imag_tf = torch.stft(noisy_imag, n_fft=512, hop_length=256, win_length=512,
                                   window=win, center=True, normalized=False,
                                   onesided=True, return_complex=True)
        noisy_speech_tf = noisy_real_tf + 1j * noisy_imag_tf
        noisy_speech_tf = noisy_speech_tf.transpose(0, 2)
        noisy_speech_tf = torch.cat((noisy_speech_tf.real, noisy_speech_tf.imag), dim=2)

        clean_ch0 = clean[0]

        return {
            'noisy': noisy_speech_tf,
            'clean': clean_ch0,
        }


class PreprocessedSHDataset(Dataset):
    """
    Loads preprocessed .pt files with normalization applied at load time.
    Each .pt contains a tuple (clean_ch0, noisy_speech_tf).
    """

    def __init__(self, root, split):
        self.folder = os.path.join(root, split)
        self.samples = sorted(
            os.path.join(self.folder, fname)
            for fname in os.listdir(self.folder)
            if fname.endswith(".pt")
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        clean_real_ch0, noisy_speech_tf = torch.load(self.samples[idx], map_location="cpu")
        max_val = noisy_speech_tf.abs().max().item()
        noisy_speech_tf = noisy_speech_tf / max_val
        clean_real_ch0 = clean_real_ch0 / max_val
        return {
            'noisy': noisy_speech_tf,
            'clean': clean_real_ch0,
        }
