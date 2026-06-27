import os
import torch

def extract_channel_and_save(src_dir, dst_dir, clean_channel_idx=0):
    """
    Extracts noisy and a specific channel from clean samples from .pt files
    and saves them to dst_dir in the same tuple format (noisy, clean_channel).

    Args:
        src_dir (str): Source folder with .pt files (each containing (noisy, clean))
        dst_dir (str): Destination folder where processed files will be saved
        clean_channel_idx (int): Index of the clean channel to keep
    """
    os.makedirs(dst_dir, exist_ok=True)

    # List all .pt files
    sample_files = sorted([f for f in os.listdir(src_dir) if f.endswith('.pt')])

    if not sample_files:
        raise RuntimeError(f"No .pt files found in {src_dir}")

    for idx, file_name in enumerate(sample_files, 1):
        file_path = os.path.join(src_dir, file_name)

        try:
            noisy, clean = torch.load(file_path, map_location='cpu')
            noisy = noisy.float()

        except Exception as e:
            print(f"[Error loading {file_name}]: {e}")
            continue

        # Save as tuple in same format
        dst_file = os.path.join(dst_dir, file_name)
        print(clean_channel_idx)
        torch.save((noisy, clean, clean_channel_idx), dst_file)

    print(f"Processed {len(sample_files)} samples. Saved to {dst_dir}")

def extract_channel_and_swap(src_dir, dst_dir, mic=True, clean_channel_idx=0):
    """
    Extracts noisy and a specific channel from clean samples from .pt files,
    swaps it with the first channel, and saves them to dst_dir in the same tuple format (noisy, clean).

    Args:
        src_dir (str): Source folder with .pt files (each containing (noisy, clean))
        dst_dir (str): Destination folder where processed files will be saved
        clean_channel_idx (int): Index of the clean channel to swap to first
    """
    os.makedirs(dst_dir, exist_ok=True)

    # List all .pt files
    sample_files = sorted([f for f in os.listdir(src_dir) if f.endswith('.pt')])

    if not sample_files:
        raise RuntimeError(f"No .pt files found in {src_dir}")

    for idx, file_name in enumerate(sample_files, 1):
        file_path = os.path.join(src_dir, file_name)

        try:
            if mic:
                noisy, clean = torch.load(file_path, map_location='cpu')
            else:
                noisy, clean, noisy_anm, clean_anm = torch.load(file_path, map_location='cpu')
            noisy = noisy.float()
            clean = clean.float()
            print(noisy.shape, clean.shape)
        except Exception as e:
            print(f"[Error loading {file_name}]: {e}")
            continue

        # --- Swap the noisy STFT channel ---
        T, F, total_C = noisy.shape
        C = total_C // 2
        if clean_channel_idx != 0:
            # real part
            noisy_real = noisy[..., :C].clone()
            noisy_imag = noisy[..., C:].clone()
            # swap channels
            noisy_real[..., 0], noisy_real[..., clean_channel_idx] = noisy_real[..., clean_channel_idx].clone(), noisy_real[..., 0].clone()
            noisy_imag[..., 0], noisy_imag[..., clean_channel_idx] = noisy_imag[..., clean_channel_idx].clone(), noisy_imag[..., 0].clone()
            # concatenate back
            noisy = torch.cat([noisy_real, noisy_imag], dim=-1)

            # --- Swap channels in clean ---
            clean[0, :], clean[clean_channel_idx, :] = clean[clean_channel_idx, :].clone(), clean[0, :].clone()

        # Save tuple in same formatw
        dst_file = os.path.join(dst_dir, file_name)
        if mic:
            torch.save((noisy, clean, clean_channel_idx), dst_file)
        else:
            torch.save((noisy, clean, noisy_anm, clean_anm), dst_file)

    print(f"Processed {len(sample_files)} samples. Saved to {dst_dir}")

# train arrays
# src_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment2/EasyComTest_preprocessed"
# dst_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment2/EasyComTest_preprocessed"
# extract_channel_and_swap(src_folder, dst_folder, clean_channel_idx=1)

# src_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment3/test_for_train_ds_preprocessed/random 1_preprocessed"
# dst_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment2/test_for_train_ds_preprocessed/random 1_preprocessed"
# extract_channel_and_swap(src_folder, dst_folder, clean_channel_idx=4)

# src_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment3/test_for_train_ds_preprocessed/random 2_preprocessed"
# dst_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment2/test_for_train_ds_preprocessed/random 2_preprocessed"
# extract_channel_and_swap(src_folder, dst_folder, clean_channel_idx=1)

# src_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment3/test_for_train_ds_preprocessed/semi circle radius = 0.05_preprocessed"
# dst_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment2/test_for_train_ds_preprocessed/semi circle radius = 0.05_preprocessed"
# extract_channel_and_swap(src_folder, dst_folder, clean_channel_idx=2)

# src_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment3/test_for_train_ds_preprocessed/ULA along Y-axis_preprocessed"
# dst_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment2/test_for_train_ds_preprocessed/ULA along Y-axis_preprocessed"
# extract_channel_and_swap(src_folder, dst_folder, clean_channel_idx=2)

# src_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment3/test_for_train_ds_preprocessed/X-shaped_preprocessed"
# dst_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment2/test_for_train_ds_preprocessed/X-shaped_preprocessed"
# extract_channel_and_swap(src_folder, dst_folder, clean_channel_idx=1)

# # test arrays
# src_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment3/test_for_test_ds_preprocessed/full circle radius = 0.05_preprocessed"
# dst_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment2/test_for_test_ds_preprocessed/full circle radius = 0.05_preprocessed"
# extract_channel_and_swap(src_folder, dst_folder, clean_channel_idx=0)

# src_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment3/test_for_test_ds_preprocessed/plus-shaped_preprocessed"
# dst_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment2/test_for_test_ds_preprocessed/plus-shaped_preprocessed"
# extract_channel_and_swap(src_folder, dst_folder, clean_channel_idx=1)

# src_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment3/test_for_test_ds_preprocessed/random 3_preprocessed"
# dst_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment2/test_for_test_ds_preprocessed/random 3_preprocessed"
# extract_channel_and_swap(src_folder, dst_folder, clean_channel_idx=4)

# src_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment3/test_for_test_ds_preprocessed/random 4_preprocessed"
# dst_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment2/test_for_test_ds_preprocessed/random 4_preprocessed"
# extract_channel_and_swap(src_folder, dst_folder, clean_channel_idx=0)

# src_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment3/test_for_test_ds_preprocessed/semi circle radius = 0.1_preprocessed"
# dst_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment2/test_for_test_ds_preprocessed/semi circle radius = 0.1_preprocessed"
# extract_channel_and_swap(src_folder, dst_folder, clean_channel_idx=2)

# src_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment3/test_for_test_ds_preprocessed/ULA along X-axis_preprocessed"
# dst_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment2/test_for_test_ds_preprocessed/ULA along X-axis_preprocessed"
# extract_channel_and_swap(src_folder, dst_folder, clean_channel_idx=4)

# # --- Train arrays ---
# src_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment1/experiment_train_ds_preprocessed/full_circle_0.1_preprocessed"
# dst_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment2/experiment_train_ds_preprocessed/full_circle_0.1_preprocessed"
# extract_channel_and_swap(src_folder, dst_folder, clean_channel_idx=0)

# src_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment1/experiment_train_ds_preprocessed/random_1_preprocessed"
# dst_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment2/experiment_train_ds_preprocessed/random_1_preprocessed"
# extract_channel_and_swap(src_folder, dst_folder, clean_channel_idx=4)

# src_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment1/experiment_train_ds_preprocessed/random_2_preprocessed"
# dst_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment2/experiment_train_ds_preprocessed/random_2_preprocessed"
# extract_channel_and_swap(src_folder, dst_folder, clean_channel_idx=1)

# src_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment1/experiment_train_ds_preprocessed/semi_circle_0.05_preprocessed"
# dst_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment2/experiment_train_ds_preprocessed/semi_circle_0.05_preprocessed"
# extract_channel_and_swap(src_folder, dst_folder, clean_channel_idx=2)

# src_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment1/experiment_train_ds_preprocessed/ULA_Y_preprocessed"
# dst_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment2/experiment_train_ds_preprocessed/ULA_Y_preprocessed"
# extract_channel_and_swap(src_folder, dst_folder, clean_channel_idx=2)

# src_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment1/experiment_train_ds_preprocessed/X_shaped_preprocessed"
# dst_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment2/experiment_train_ds_preprocessed/X_shaped_preprocessed"
# extract_channel_and_swap(src_folder, dst_folder, clean_channel_idx=1)

# # --- val arrays ---
# src_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment1/experiment_val_ds_preprocessed/full circle radius = 0.1_preprocessed"
# dst_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment2/experiment_val_ds_preprocessed/full circle radius = 0.1_preprocessed"
# extract_channel_and_swap(src_folder, dst_folder, clean_channel_idx=0)

# src_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment1/experiment_val_ds_preprocessed/random 1_preprocessed"
# dst_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment2/experiment_val_ds_preprocessed/random 1_preprocessed"
# extract_channel_and_swap(src_folder, dst_folder, clean_channel_idx=4)

# src_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment1/experiment_val_ds_preprocessed/random 2_preprocessed"
# dst_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment2/experiment_val_ds_preprocessed/random 2_preprocessed"
# extract_channel_and_swap(src_folder, dst_folder, clean_channel_idx=1)

# src_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment1/experiment_val_ds_preprocessed/semi circle radius = 0.05_preprocessed"
# dst_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment2/experiment_val_ds_preprocessed/semi circle radius = 0.05_preprocessed"
# extract_channel_and_swap(src_folder, dst_folder, clean_channel_idx=2)

# src_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment1/experiment_val_ds_preprocessed/ULA along Y-axis_preprocessed"
# dst_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment2/experiment_val_ds_preprocessed/ULA along Y-axis_preprocessed"
# extract_channel_and_swap(src_folder, dst_folder, clean_channel_idx=2)

# src_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment1/experiment_val_ds_preprocessed/X-shaped_preprocessed"
# dst_folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment2/experiment_val_ds_preprocessed/X-shaped_preprocessed"
# extract_channel_and_swap(src_folder, dst_folder, clean_channel_idx=1)

# --------- full anm experiment ---------
# --- test of train arrays ---
# folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment_full_anm/test_of_train_ds_preprocessed_swap"
# mic = False
for i in range(1, 3):
    if i == 1:
        # --- mic training ---
        folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment_full_anm/mic_train_ds_preprocessed_swap"
        mic = True
    else:
        # --- mic validation ---
        folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment_full_anm/mic_val_ds_preprocessed_swap"
        mic = True

    array_type = "front hemisphere1 (rigid) radius = 0.1_preprocessed"
    print(array_type)
    idx = 1
    src_folder = os.path.join(folder, array_type)
    dst_folder = os.path.join(folder + "_swap", array_type)
    extract_channel_and_swap(src_folder, dst_folder, mic=mic, clean_channel_idx=idx-1)

    array_type = "full circle (rigid) radius = 0.1_preprocessed"
    print(array_type)
    idx = 1
    src_folder = os.path.join(folder, array_type)
    dst_folder = os.path.join(folder + "_swap", array_type)
    extract_channel_and_swap(src_folder, dst_folder, mic=mic, clean_channel_idx=idx-1)

    array_type = "planar_preprocessed"
    print(array_type)
    idx = 6
    src_folder = os.path.join(folder, array_type)
    dst_folder = os.path.join(folder + "_swap", array_type)
    extract_channel_and_swap(src_folder, dst_folder, mic=mic, clean_channel_idx=idx-1)

    array_type = "random 2D array1 radius = 0.1_preprocessed"
    print(array_type)
    idx = 6
    src_folder = os.path.join(folder, array_type)
    dst_folder = os.path.join(folder + "_swap", array_type)
    extract_channel_and_swap(src_folder, dst_folder, mic=mic, clean_channel_idx=idx-1)

    array_type = "random sphere1 radius = 0.1_preprocessed"
    print(array_type)
    idx = 7
    src_folder = os.path.join(folder, array_type)
    dst_folder = os.path.join(folder + "_swap", array_type)
    extract_channel_and_swap(src_folder, dst_folder, mic=mic, clean_channel_idx=idx-1)

    array_type = "random sphere3 (rigid) radius = 0.1_preprocessed"
    print(array_type)
    idx = 4
    src_folder = os.path.join(folder, array_type)
    dst_folder = os.path.join(folder + "_swap", array_type)
    extract_channel_and_swap(src_folder, dst_folder, mic=mic, clean_channel_idx=idx-1)

    array_type = "random sphere5 (rigid) radius = 0.05_preprocessed"
    print(array_type)
    idx = 2
    src_folder = os.path.join(folder, array_type)
    dst_folder = os.path.join(folder + "_swap", array_type)
    extract_channel_and_swap(src_folder, dst_folder, mic=mic, clean_channel_idx=idx-1)

    array_type = "semi circle planar radius = 0.05_preprocessed"
    print(array_type)
    idx = 6
    src_folder = os.path.join(folder, array_type)
    dst_folder = os.path.join(folder + "_swap", array_type)
    extract_channel_and_swap(src_folder, dst_folder, mic=mic, clean_channel_idx=idx-1)

    array_type = "ULA along X-axis_preprocessed"
    print(array_type)
    idx = 7
    src_folder = os.path.join(folder, array_type)
    dst_folder = os.path.join(folder + "_swap", array_type)
    extract_channel_and_swap(src_folder, dst_folder, mic=mic, clean_channel_idx=idx-1)

    array_type = "uniform sphere (rigid) radius = 0.1_preprocessed"
    print(array_type)
    idx = 2
    src_folder = os.path.join(folder, array_type)
    dst_folder = os.path.join(folder + "_swap", array_type)
    extract_channel_and_swap(src_folder, dst_folder, mic=mic, clean_channel_idx=idx-1)

# # --- test of test arrays ---
# folder = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment_full_anm/test_of_test_ds_preprocessed_swap"
# mic = False

# array_type = "front hemisphere2 (rigid) radius = 0.1_preprocessed"
# print(array_type)
# idx = 1
# src_folder = os.path.join(folder, array_type)
# dst_folder = os.path.join(folder + "_swap", array_type)
# extract_channel_and_swap(src_folder, dst_folder, mic=mic, clean_channel_idx=idx-1)

# array_type = "planar (rot=45deg)_preprocessed"
# print(array_type)
# idx = 5
# src_folder = os.path.join(folder, array_type)
# dst_folder = os.path.join(folder + "_swap", array_type)
# extract_channel_and_swap(src_folder, dst_folder, mic=mic, clean_channel_idx=idx-1)

# array_type = "random 2D array2 radius = 0.1_preprocessed"
# print(array_type)
# idx = 2
# src_folder = os.path.join(folder, array_type)
# dst_folder = os.path.join(folder + "_swap", array_type)
# extract_channel_and_swap(src_folder, dst_folder, mic=mic, clean_channel_idx=idx-1)

# array_type = "random sphere2 radius = 0.1_preprocessed"
# print(array_type)
# idx = 2
# src_folder = os.path.join(folder, array_type)
# dst_folder = os.path.join(folder + "_swap", array_type)
# extract_channel_and_swap(src_folder, dst_folder, mic=mic, clean_channel_idx=idx-1)

# array_type = "random sphere4 (rigid) radius = 0.1_preprocessed"
# print(array_type)
# idx = 7
# src_folder = os.path.join(folder, array_type)
# dst_folder = os.path.join(folder + "_swap", array_type)
# extract_channel_and_swap(src_folder, dst_folder, mic=mic, clean_channel_idx=idx-1)

# array_type = "random sphere6 (rigid) radius = 0.05_preprocessed"
# print(array_type)
# idx = 4
# src_folder = os.path.join(folder, array_type)
# dst_folder = os.path.join(folder + "_swap", array_type)
# extract_channel_and_swap(src_folder, dst_folder, mic=mic, clean_channel_idx=idx-1)

# array_type = "semi circle (rigid) radius = 0.1_preprocessed"
# print(array_type)
# idx = 4
# src_folder = os.path.join(folder, array_type)
# dst_folder = os.path.join(folder + "_swap", array_type)
# extract_channel_and_swap(src_folder, dst_folder, mic=mic, clean_channel_idx=idx-1)

# array_type = "ULA along Z-axis_preprocessed"
# print(array_type)
# idx = 4
# src_folder = os.path.join(folder, array_type)
# dst_folder = os.path.join(folder + "_swap", array_type)
# extract_channel_and_swap(src_folder, dst_folder, mic=mic, clean_channel_idx=idx-1)

# array_type = "uniform sphere (rigid) radius = 0.05_preprocessed"
# print(array_type)
# idx = 2
# src_folder = os.path.join(folder, array_type)
# dst_folder = os.path.join(folder + "_swap", array_type)
# extract_channel_and_swap(src_folder, dst_folder, mic=mic, clean_channel_idx=idx-1)

# array_type = "semi circle planar radius = 0.1_preprocessed"
# print(array_type)
# idx = 6
# src_folder = os.path.join(folder, array_type)
# dst_folder = os.path.join(folder + "_swap", array_type)
# extract_channel_and_swap(src_folder, dst_folder, mic=mic, clean_channel_idx=idx-1)

# array_type = "Aria on rigid sphere (simulated)_preprocessed"
# print(array_type)
# idx = 3
# src_folder = os.path.join(folder, array_type)
# dst_folder = os.path.join(folder + "_swap", array_type)
# extract_channel_and_swap(src_folder, dst_folder, mic=mic, clean_channel_idx=idx-1)

# array_type = "ULA along Y-axis (tilt=30deg)_preprocessed"
# print(array_type)
# idx = 4
# src_folder = os.path.join(folder, array_type)
# dst_folder = os.path.join(folder + "_swap", array_type)
# extract_channel_and_swap(src_folder, dst_folder, mic=mic, clean_channel_idx=idx-1)

#runai-cmd --name ref-change  -g 0.1 --cpu-limit 20 -- "conda activate venv && python /Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/utils/clean_change.py"
#runai-bgu submit python -n clean-change2 -c 20 -m 40G -g 0.2 --conda venv -- "python /Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/utils/clean_change.py"

