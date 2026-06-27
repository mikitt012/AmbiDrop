import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch.nn.functional as F
import torch.nn as nn
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import ipdb
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pesq import pesq
from pystoi import stoi
import logging
import h5py
import sounddevice as sd  # pip install sounddevice
import soundfile as sf
import re  
writer = SummaryWriter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

import os
import scipy.io
import numpy as np
import torch

def add_white_noise(signal, snr_db):
    """
    Add white noise to a signal based on a desired SNR in dB.

    Parameters:
    - signal: np.array, the original signal. Can be multi-channel (channels as rows).
    - snr_db: float, the desired signal-to-noise ratio in dB.

    Returns:
    - np.array, the signal with added white noise.
    """
    # Calculate signal power and convert SNR from dB to linear
    is_complex = np.iscomplexobj(signal)

    signal_power = np.mean(np.abs(signal)**2, axis=1, keepdims=True)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    if is_complex:
        noise_real = np.random.normal(0, np.sqrt(noise_power / 2), signal.shape)
        noise_imag = np.random.normal(0, np.sqrt(noise_power / 2), signal.shape)
        noise = noise_real + 1j * noise_imag
    else:
        noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)

    return signal + noise

def pad_or_truncate(signal, target_length=120000):
    """
    Pads or truncates a multichannel signal (C, T) to (C, target_length).
    """
    C, T = signal.shape
    if T > target_length:
        print(f"[Warning] Signal truncated from {T} to {target_length} samples.")
        return signal[:, :target_length]
    elif T < target_length:
        pad_width = ((0, 0), (0, target_length - T))
        return np.pad(signal, pad_width, mode='constant')
    else:
        return signal

def pad_to_length(signal, target_length):
    """
    Pads a complex tensor (C, T) along time dim to target_length with zeros.
    """
    C, T = signal.shape
    if T >= target_length:
        return signal[:, :target_length]
    else:
        pad_size = target_length - T
        pad = torch.zeros(C, pad_size, dtype=signal.dtype, device=signal.device)
        return torch.cat([signal, pad], dim=1)

def preprocess_and_save(
    data_dir: str,
    data_type: str,
    max_len: int,
    ambisonics: bool,
    out_root: str,
    start_idx: int = 0,
    end_idx: int = None     # up to but not including
):
    """
    For every .mat in data_dir/data_type, run the same pad→STFT→normalize→split pipeline
    and save the result as a .pt in out_root/<data_type>_pt/.

    Returns nothing; writes one .pt per input .mat.
    """
    in_folder  = os.path.join(data_dir, data_type)
    out_folder = os.path.join(out_root, data_type + "_STFT")
    os.makedirs(out_folder, exist_ok=True)

    # choose your file-pattern
    if ambisonics:
        mat_files = sorted(f for f in os.listdir(in_folder)
                           if f.startswith("Ambisonics_") and f.endswith(".mat"))
    else:
        mat_files = sorted(f for f in os.listdir(in_folder)
                           if "ex" in f and f.endswith(".mat"))

    mat_files = mat_files[start_idx:end_idx]
    for fname in mat_files:
        # 1) load
        data = scipy.io.loadmat(os.path.join(in_folder, fname))
        if ambisonics:
            noisy = data["anm_t"]
            clean = data["anm_clean_rir_t"]
        else:
            noisy = data["p"]
            clean = data["pDirect"]

        # 2) add noise if desired
        noisy = add_white_noise(noisy, 30)

        # 3) transpose → (C, N)
        noisy_speech = noisy.T
        clean_speech = clean.T

        noisy_speech = pad_or_truncate(noisy_speech, 120000)
        clean_speech = pad_or_truncate(clean_speech, 120000)

        # 5) to torch
        noisy_real = torch.from_numpy(noisy_speech.real).float().to(device)
        noisy_imag = torch.from_numpy(noisy_speech.imag).float().to(device)
        clean = torch.from_numpy(clean_speech).float().to(device)

        # 6) STFT on real & imag parts separately
        win = torch.hamming_window(512).to(device)
        noisy_real_tf = torch.stft(noisy_real, n_fft=512, hop_length=256, win_length=512,
                                    window=win, center=True, normalized=False, onesided=True, return_complex=True)
        noisy_imag_tf = torch.stft(noisy_imag, n_fft=512, hop_length=256, win_length=512,
                                    window=win, center=True, normalized=False, onesided=True, return_complex=True)
        noisy_speech_tf = noisy_real_tf + 1j * noisy_imag_tf

        # 7) transpose → T×F×C
        noisy_speech_tf = noisy_speech_tf.transpose(0, 2)

        # Split complex STFT into real and imaginary parts for model input
        noisy_speech_tf = torch.cat((noisy_speech_tf.real, noisy_speech_tf.imag), dim=2)  # T x F x 2C

        # 10) select first clean channel
        clean_ch0 = clean[0]

        # 11) save
        out_path = os.path.join(out_folder, fname.replace(".mat", ".pt"))
        torch.save((clean_ch0, noisy_speech_tf), out_path)
        os.remove(os.path.join(in_folder, fname))

    print(f"Saved {len(mat_files)} preprocessed files to {out_folder}")

# preprocess_and_save(
#     data_dir   = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop",
#     data_type  = "test_data_SH",
#     max_len    = 120000,
#     ambisonics = True,
#     out_root   = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop",
#     start_idx=0,
#     end_idx=None
# )

# clean_ch0, _ = torch.load("/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/train_data_SH_STFT/Ambisonics_06310.pt", map_location="cpu")
# sr = 16000  # your sampling rate
# scipy.io.wavfile.write("clean_ch0_.wav", 16000, clean_ch0.cpu().numpy().astype('float32'))

import os
import torch
import scipy.io
from tqdm import tqdm

ARRAY_REF_MAPPING = {
    "front hemisphere1 (rigid) radius = 0.1": 1,
    "full circle (rigid) radius = 0.1d": 1,
    "planar": 6,
    "random 2D array1 radius = 0.1": 6,
    "random sphere1 radius = 0.1": 7,
    "random sphere3 (rigid) radius = 0.1": 4,
    "random sphere5 (rigid) radius = 0.05": 2,
    "semi circle planar radius = 0.05": 6,
    "ULA along X-axis": 7,
    "uniform sphere (rigid) radius = 0.1": 2,
    "front hemisphere2 (rigid) radius = 0.1": 1,
    "planar (rot=45deg)": 5,
    "random 2D array2 radius = 0.1": 2,
    "random sphere2 radius = 0.1": 2,
    "random sphere4 (rigid) radius = 0.1": 7,
    "random sphere6 (rigid) radius = 0.05": 4,
    "semi circle (rigid) radius = 0.1": 4,
    "ULA along Z-axis": 4,
    "uniform sphere (rigid) radius = 0.05": 2,
    "semi circle planar radius = 0.1": 6,
    "Aria on rigid sphere (simulated)": 3,
    "ULA along Y-axis (tilt=30deg)": 4,
    "ULA along x-axis (rot=30deg)": 7,
    "ULA along y-axis": 4,
    "ULA along X-axis (tilt=20)": 7
}


def preprocess_and_save_to_folder(data_dir, data_type, train, out_root, max_length_sec=6, sample_rate=16000):
    """
    Preprocesses all .mat files in the given directory and saves each output as .pt file in out_root.
    """

    input_path = os.path.join(data_dir, data_type)
    # output_path = os.path.join(out_root, data_type + "_preprocessed_full")
    output_path = os.path.join(out_root, data_type + "_preprocessed")
    os.makedirs(output_path, exist_ok=True)

    # ex_list = [f for f in os.listdir(input_path) if 'ex' in f]

    subfolders = [d for d in os.listdir(input_path)
                  if os.path.isdir(os.path.join(input_path, d)) and d.startswith("ex_")]

    for folder in tqdm(subfolders, desc=f"Processing {data_type}"):
        try:
            out_file = os.path.join(output_path, folder + ".pt")
            save_path = os.path.join(output_path, out_file)
            # 2. Check if it already exists
            if os.path.exists(save_path):
                print(f"Skipping {folder}: File already exists at {save_path}")
                continue

            raw_ref = ARRAY_REF_MAPPING.get(data_type, 1) # Default to 1 if not found
            ref_id = raw_ref - 1  # Apply your -1 offset
            
            folder_path = os.path.join(input_path, folder)
            array_file = os.path.join(folder_path, "p.wav")
            direct_file = os.path.join(folder_path, "pDirect.wav")
            # mat_file = os.path.join(folder_path, "anm.mat")

            # if not os.path.exists(mat_file):
            #     print(f"Skipping {folder}: .mat file missing")
            #     continue

            # mat_data = scipy.io.loadmat(mat_file)

            noisy_speech_mic, _ = sf.read(array_file)  # (T, C)
            noisy_speech_mic = noisy_speech_mic.transpose()
            clean_speech_mic, _ = sf.read(direct_file) # (T, C)
            clean_speech_mic = clean_speech_mic.transpose()
            # noisy_speech_anm = mat_data["anmt_array"].transpose()
            # clean_speech_anm = mat_data["anmtDirect"].transpose()
            # noisy_speech_anm = mat_data["anmt"].transpose()

            # print('noisy_speech anm shape:', noisy_speech_anm.shape)
            # print('noisy_speech mic shape:', noisy_speech_mic.shape)
            # print('clean_speech shape:', clean_speech.shape)

    # for fname in tqdm(ex_list, desc=f"Processing {data_type}"):
    #     try:
    #         file_path = os.path.join(input_path, fname)
    #         ex = scipy.io.loadmat(file_path)

    #         noisy_speech_mic = ex['p']
    #         clean_speech_mic = ex['pDirect']

            noisy_speech_mic = torch.from_numpy(noisy_speech_mic).to(torch.complex64)
            # noisy_speech_anm = torch.from_numpy(noisy_speech_anm).to(torch.complex64)
            clean_speech_mic = torch.from_numpy(clean_speech_mic).to(torch.complex64)
    #         # clean_speech_anm = torch.from_numpy(clean_speech_anm).to(torch.complex64)

            if train:   
                max_length = max_length_sec * sample_rate
                first = (clean_speech_mic[0, :].abs() > 1e-3).nonzero()[0][0].item()
                last = first + max_length

                noisy_speech_mic = pad_to_length(noisy_speech_mic[:, first:last], max_length)
                clean_speech_mic = pad_to_length(clean_speech_mic[:, first:last], max_length)

                # first = (clean_speech_anm[0, :].abs() > 1e-3).nonzero()[0][0].item()
                # last = first + max_length

                # noisy_speech_anm = pad_to_length(noisy_speech_anm[:, first:last], max_length)
                # clean_speech_anm = pad_to_length(clean_speech_anm[:, first:last], max_length)

            # print('noisy_speech shape:', noisy_speech.shape)
            # print('clean_speech shape:', clean_speech[0, :].shape)

            win = torch.hamming_window(window_length=512, device=noisy_speech_mic.device)
            noisy_tf_mic = torch.stft(noisy_speech_mic, n_fft=512, hop_length=256, win_length=512,
                                  window=win, center=True, normalized=False, return_complex=True).transpose(0, 2)
            # noisy_tf_anm = torch.stft(noisy_speech_anm, n_fft=512, hop_length=256, win_length=512,
            #                       window=win, center=True, normalized=False, return_complex=True).transpose(0, 2)

            noisy_tf_mic = noisy_tf_mic[:, 0:257, :]
            max_val = noisy_tf_mic.abs().max().item()
            noisy_tf_mic = noisy_tf_mic / max_val
            clean_speech_mic = clean_speech_mic / max_val

            # noisy_tf_anm = noisy_tf_anm[:, 0:257, :]
            # max_val = noisy_tf_anm.abs().max().item()
            # noisy_tf_anm = noisy_tf_anm / max_val
            # clean_speech_anm = clean_speech_anm / max_val

            noisy_tf_mic = torch.cat((noisy_tf_mic.real, noisy_tf_mic.imag), dim=2)
            # noisy_tf_anm = torch.cat((noisy_tf_anm.real, noisy_tf_anm.imag), dim=2)

            # print('noisy stft shape:', noisy_tf.shape)
            clean_time_mic = clean_speech_mic.float()
            clean_time_mic = clean_time_mic.squeeze(0)
            # clean_time_anm = clean_speech_anm.float()
            # clean_time_anm = clean_time_anm.squeeze(0)

            # print("  noisy_tf_mic:", noisy_tf_mic.shape) # TxFx2C
            # print("  noisy_tf_anm:", noisy_tf_anm.shape) # TxFx2C
            # print("  clean_time_mic:", clean_time_mic.shape) # T
            # print("  clean_time_anm:", clean_time_anm.shape) # T

            # name_only, ext = os.path.splitext(fname)
            # new_name = re.sub(r"([a-zA-Z]+)(\d+)", r"\1_\2", name_only)
            # out_file = new_name + ".pt"
            # out_file = os.path.join(output_path, fname + ".pt")
            # torch.save((noisy_tf_mic, clean_time_mic, ref_id), os.path.join(output_path, out_file))

            # pt_filename = f"{data_type}_{folder}.pt"
            # pt_path = os.path.join(output_path, pt_filename)
            
            # SAVE THE REF_ID IN THE PT FILE
            torch.save({
                'noisy': noisy_tf_mic,
                'clean': clean_time_mic,
                'ref_id': ref_id,   # <--- Added this here
                'array_name': data_type,
                'ex_id': folder
            }, save_path)
                
            # torch.save((noisy_tf_mic, clean_time_mic, noisy_tf_anm, clean_time_anm), os.path.join(output_path, out_file))

        # except Exception as e:
        #     print(f"[Error] {fname}: {e}")

            # print("  noisy_tf_mic:", noisy_tf_mic.shape)
            # print("  clean_time_mic:", clean_time_mic.shape)
            # print("  noisy_tf_anm:", noisy_tf_anm.shape)
            # print("  clean_time_anm:", clean_time_anm.shape)
            # Save as .pt file
            # out_file = os.path.join(output_path, folder + ".pt")
            # torch.save((noisy_tf_mic, clean_time_mic, noisy_tf_anm, clean_time_anm), out_file)

        except Exception as e:
            print(f"[Error] {folder}: {e}")

# preprocess_and_save_to_folder(
#     # data_dir = '/gpfs0/bgu-br/projects/sim_dataset_ambisonics/',
#     data_dir = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment2",
#     data_type = "val_ds",
#     train = True,
#     out_root = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment2"
# )

data_root = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment_full_anm/mic_train_ds"
out_root = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment_full_anm/mic_train_ds_preprocessed"

for data_type in sorted(os.listdir(data_root)):
    input_path = os.path.join(data_root, data_type)
    
    # Skip if not a directory
    if not os.path.isdir(input_path):
        continue

    preprocess_and_save_to_folder(
        data_dir=data_root,
        data_type=data_type,
        train=True,
        out_root=out_root
    )

def check_tensor_shapes(preprocessed_folder):
    """
    Check if all (noisy_tf, clean_time) tensors in the preprocessed folder are of the same length.
    """
    files = sorted([f for f in os.listdir(preprocessed_folder) if f.endswith('.pt')])
    lengths = []

    for f in files:
        try:
            path = os.path.join(preprocessed_folder, f)
            noisy_tf, clean_time = torch.load(path, map_location='cpu')

            lengths.append((noisy_tf.shape[0], clean_time.shape[0]))

        except Exception as e:
            print(f"[Error loading] {f}: {e}")

    # Get unique shapes
    unique_lengths = list(set(lengths))
    print(f"Unique shape pairs (noisy_tf T, clean_time T): {unique_lengths}")

    if len(unique_lengths) == 1:
        print("✅ All samples have consistent time lengths.")
    else:
        print("❌ Inconsistent lengths found!")

    return unique_lengths

# check_tensor_shapes('/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/si_dt_05_preprocessed')


#runai-cmd --name prepro  -g 0.2 --cpu-limit 20 -- "conda activate venv && python /Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/utils/mic_data_process.py"
#runai-bgu submit python -n prepro -c 20 -m 40G -g 0.2 --conda venv -- "python /Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/utils/mic_data_process.py"
