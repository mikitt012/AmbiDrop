import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np

def inspect_checkpoint(checkpoint_path):
    # Load the checkpoint to CPU to avoid memory issues
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 1. Determine the structure (List of epochs vs Single dictionary)
    if isinstance(checkpoint, list):
        num_saves = len(checkpoint)
        print(f"Total epochs/saves found: {num_saves}")
        
        for i, state in enumerate(checkpoint):
            # Calculate total parameters (Net Size)
            # state_dict is usually under 'model_state_dict' or the state itself
            sd = state.get('model_state_dict', state) 
            total_params = sum(p.numel() for p in sd.values())
            
            print(f"Save Index {i}: Size = {total_params:,} parameters")
            
    elif isinstance(checkpoint, dict):
        # Check if epochs are stored as keys like 'epoch_1', 'epoch_2'
        epoch_keys = [k for k in checkpoint.keys() if 'state' in k.lower()]
        print(f"Total save keys found: {len(epoch_keys)}")
        
        for key in epoch_keys:
            sd = checkpoint[key]
            total_params = sum(p.numel() for p in sd.values())
            print(f"Key '{key}': Size = {total_params:,} parameters")
    else:
        print("Unknown checkpoint format.")

# # Usage on your cluster path
# ckpt_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2026-03-04_09-13-31.pt"
# if os.path.exists(ckpt_path):
#     inspect_checkpoint(ckpt_path)
# else:
#     print("File not found.")

def split_checkpoint_by_size(input_path):
    # Load the big checkpoint
    checkpoint = torch.load(input_path, map_location='cpu')
    
    # Dictionary to group saves: {param_count: [list_of_states]}
    grouped_saves = {}

    if isinstance(checkpoint, list):
        for state in checkpoint:
            # Get the state dict (handle different naming conventions)
            sd = state.get('model_state_dict', state)
            param_count = sum(p.numel() for p in sd.values())
            
            if param_count not in grouped_saves:
                grouped_saves[param_count] = []
            
            grouped_saves[param_count].append(state)

        # Save each group into a different file
        for size, states in grouped_saves.items():
            # Create a name based on parameter count (e.g., net_size_1200000.pt)
            new_filename = f"checkpoint_size_{size}.pt"
            save_path = os.path.join(os.path.dirname(input_path), new_filename)
            
            torch.save(states, save_path)
            print(f"Saved {len(states)} iterations to {save_path} (Size: {size:,} params)")
    else:
        print("Checkpoint is not a list. No splitting needed.")

# Run it
input_ckpt = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/SH_FT_JNF,2026-03-09_14-23-53.pt"
# split_checkpoint_by_size(input_ckpt)

def load_checkpoint(checkpoint_path, target_epoch=None, net=None, optimizer=None, scheduler=None):
    """
    Load the checkpoint for a specific epoch or the latest checkpoint if no epoch is specified.
    Also loads learning rate and scheduler state.
    """
    checkpoint_list = torch.load(checkpoint_path)
    available_epochs = [ckpt["epoch"] for ckpt in checkpoint_list]

    # If no epoch specified, pick the latest
    if target_epoch is None:
        target_epoch = max(available_epochs)
        print(f"No epoch specified. Loading the latest checkpoint from epoch {target_epoch}")

    # Find exact match OR closest
    if target_epoch in available_epochs:
        chosen_epoch = target_epoch
    else:
        # Pick epoch with minimum distance to target
        chosen_epoch = min(available_epochs, key=lambda e: abs(e - target_epoch))
        print(f"Epoch {target_epoch} not found. Using closest epoch: {chosen_epoch}")

    # Retrieve the selected checkpoint
    checkpoint_to_load = next(ckpt for ckpt in checkpoint_list if ckpt["epoch"] == chosen_epoch)
    
    # Load the model and optimizer state
    if net is not None:
        net.load_state_dict(checkpoint_to_load['model_state_dict'])
    # if optimizer is not None:
    #     optimizer.load_state_dict(checkpoint_to_load['optimizer_state_dict'])

    # # Load the learning rate (if you want to log or use it later)
    # if optimizer is not None:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = checkpoint_to_load['learning_rate']
    
    # # Print the learning rate
    # print(f"Loaded learning rate: {optimizer.param_groups[0]['lr']:.6f}")

    # Load the scheduler state (if provided)
    # if scheduler is not None:
    #     scheduler.load_state_dict(checkpoint_to_load['scheduler_state_dict'])

    # Extract the loss or any other metrics you want
    # prev_loss = checkpoint_to_load['loss']
    # print(f"Loaded checkpoint from epoch {target_epoch}, loss: {prev_loss:.4f}")

        return chosen_epoch

# checkpoint_list = torch.load("/gpfs0/bgu-br/users/tatarjit/speech-enhancement/checkpoints/checkpoint_size_3490.pt")
# available_epochs = [ckpt["epoch"] for ckpt in checkpoint_list]
# print(available_epochs)

def horizontal_histogram(csv_path):
    df = pd.read_csv(csv_path)
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 14))

    # Horizontal bar chart
    bars = ax.barh(df["Bin [dB]"], df["Sample Count"], color='skyblue', edgecolor='black')

    # Use log scale for X axis because of the massive outlier (3101)
    ax.set_xscale('log')

    # Add labels to the bars
    for i, (bar, mean_val) in enumerate(zip(bars, df["Mean Enhanced SI-SDR [dB]"])):
        width = bar.get_width()
        ax.text(width * 1.05, bar.get_y() + bar.get_height()/2, 
                f'{mean_val:.1f} dB', 
                va='center', fontweight='bold', color='darkblue')

    ax.set_xlabel("Sample Count (Log Scale)", fontsize=12)
    ax.set_ylabel("Bin [dB]", fontsize=12)
    ax.set_title("Distribution of Improvement: Sample Count vs. Mean Enhanced SI-SDR", fontsize=14, pad=20)
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("/gpfs0/bgu-br/users/tatarjit/speech-enhancement/snr_distribution_fixed/si_sdr_test_histogram_array_avg.png", dpi=300)
    plt.show()

csv_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/snr_distribution_fixed/si-sdr distribution across examples and averaged over arrays.csv"
# horizontal_histogram(csv_path)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def horizontal_histogram_5db(csv_path):
    df = pd.read_csv(csv_path)

    # 1. Extract the lower bound of the string bin to make it numeric
    # e.g., "[-10, -9]" -> -10.0
    df['bin_numeric'] = df['Bin [dB]'].str.extract(r'\[(.*?),').astype(float)

    # 2. Define 5dB Buckets
    # Range from -30 to 10 to cover all data points
    bins_5db = np.arange(-30, 7, 2)
    df['5db_group'] = pd.cut(df['bin_numeric'], bins=bins_5db, right=False)

    # 3. Aggregate data
    # Sum the samples, but average the SI-SDR performance
    agg_df = df.groupby('5db_group', observed=True).agg({
        'Sample Count': 'sum',
        'Mean Improvement [dB]': 'mean'
    }).reset_index()

    # Convert groups to readable strings for the Y-axis
    agg_df['bin_label'] = agg_df['5db_group'].astype(str)

    # 4. Plotting
    fig, ax = plt.subplots(figsize=(12, 10))

    # Horizontal bar chart
    bars = ax.barh(agg_df["bin_label"], agg_df["Sample Count"], 
                   color='skyblue', edgecolor='black')

    # Use log scale due to the massive 0-5dB concentration
    ax.set_xscale('log')

    max_count = agg_df["Sample Count"].max()
    # Increase the limit by 3x or 5x to leave room for the text on a log scale
    ax.set_xlim(right=max_count * 3)

    # Add labels to the bars
    for i, (bar, mean_val) in enumerate(zip(bars, agg_df["Mean Improvement [dB]"])):
        width = bar.get_width()
        if width > 0:
            ax.text(width * 1.05, bar.get_y() + bar.get_height()/2, 
                    f'{mean_val:.1f} dB', 
                    va='center', fontweight='bold', color='darkblue', fontsize=20)

    ax.set_xlabel("Sample Count (Log Scale)", fontsize=23, labelpad=15)
    ax.set_ylabel("Input SI-SDR Bin [dB]", fontsize=23, labelpad=15)
    
    # Make tick numbers bigger
    ax.tick_params(axis='both', which='major', labelsize=20)
    # ax.set_title("Distribution: Sample Count vs. Mean Enhanced SI-SDR", 
                #  fontsize=14, pad=20)
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("/gpfs0/bgu-br/users/tatarjit/speech-enhancement/snr_distribution_fixed2/si_sdr_2db_aggregated_histogram_bigger.png", dpi=300)
    plt.show()

# Run with your path
csv_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/snr_distribution_fixed2/si-sdr distribution across examples and arrays.csv"
horizontal_histogram_5db(csv_path)

import sofar as sf

# sofa = sf.read_sofa("/gpfs0/bgu-br/users/tatarjit/speech-enhancement/datasets/aria_ds/aria_atfs_fixed.sofa")

# # This prints a summary of all fields to the console
# sofa.inspect()

# # If you just want a list of the keys (field names) to loop through:
# fields = [key for key in sofa.__dict__.keys() if not key.startswith("_")]
# print(fields)

# ir = sofa.Data_IR
# fs = sofa.Data_SamplingRate
# directions = sofa.SourcePosition
# print(directions)

# ir = ir.transpose(1, 2, 0)

# def plot_ir_diagnostics(ir, fs=fs):
#     """
#     ir: numpy array or torch tensor of shape (CH, T) or (CH, T, Q)
#     If (CH, T, Q), we'll squeeze Q or pick the first source position.
#     """
#     if torch.is_tensor(ir):
#         ir = ir.detach().cpu().numpy()
    
#     # Handle (CH, T, Q) by picking first source position
#     if ir.ndim == 3:
#         ir = ir[:, :, 0]
        
#     num_channels = ir.shape[0]
#     t = np.arange(ir.shape[1]) / fs * 1000  # Time in ms
    
#     fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
#     # --- 1. Time Domain Plot ---
#     for ch in range(num_channels):
#         axes[0].plot(t, ir[ch, :], label=f'Ch {ch+1}')
    
#     axes[0].set_title("Impulse Response - Time Domain")
#     axes[0].set_xlabel("Time (ms)")
#     axes[0].set_ylabel("Amplitude")
#     axes[0].grid(True, alpha=0.3)
#     axes[0].legend(loc='upper right', ncol=2)

#     # --- 2. Frequency Domain Plot (Magnitude) ---
#     n_fft = 512
#     freqs = np.linspace(0, fs/2, n_fft//2 + 1)
    
#     for ch in range(num_channels):
#         # Compute FFT and get first 257 bins
#         V_ch = np.fft.fft(ir[ch, :], n=n_fft)
#         mag_db = 20 * np.log10(np.abs(V_ch[:n_fft//2 + 1]) + 1e-12)
#         axes[1].plot(freqs, mag_db, label=f'Ch {ch+1}')
    
#     axes[1].set_title("Frequency Response - Magnitude (dB)")
#     axes[1].set_xlabel("Frequency (Hz)")
#     axes[1].set_ylabel("Magnitude (dB)")
#     axes[1].set_xscale('log') # Log scale helps see the 200Hz-1kHz range
#     axes[1].set_ylim([-60, 10]) # Standard range for audio IRs
#     axes[1].grid(True, which='both', alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig("ir_diagnostic_plot.png")
#     plt.show()

# # Usage:
# plot_ir_diagnostics(ir)

import pandas as pd
import matplotlib.pyplot as plt

def generate_performance_plots(csv_path):
    df = pd.read_csv(csv_path)
    
    # Clean data
    metrics = ['SI-SDR', 'PESQ', 'STOI']
    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors='coerce')

    # Figure Settings
    models_to_compare = ['Baseline', 'AmbiDrop']
    colors = {'Baseline': '#1f77b4', 'AmbiDrop': '#ff7f0e'}
    
    # --- Figure 1: Enhanced Results Only ---
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))
    fig1.suptitle("Enhanced Signal Comparison (Baseline vs AmbiDrop)", fontsize=16)
    
    df_enh = df[df['type'] == 'Enhanced']
    
    for i, m in enumerate(metrics):
        for model in models_to_compare:
            subset = df_enh[df_enh['model'] == model]
            # Group by number of channels removed and average across arrays
            stats = subset.groupby('num_ch_removed')[m].agg(['mean', 'std']).reset_index()
            
            axes1[i].errorbar(stats['num_ch_removed'], stats['mean'], yerr=stats['std'], 
                             label=model, color=colors[model], marker='o', capsize=5)
            axes1[i].set_title(f'Enhanced {m}')
            axes1[i].legend()

    plt.tight_layout()
    plt.savefig("enhanced_only.png")

    # --- Figure 2: Robustness (Shows Noisy vs Enhanced for each model) ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    fig2.suptitle("Model Robustness: Noisy vs Enhanced", fontsize=16)

    for i, m in enumerate(metrics):
        for model in models_to_compare:
            # Enhanced Line
            enh_stats = df[(df['model'] == model) & (df['type'] == 'Enhanced')].groupby('num_ch_removed')[m].mean()
            axes2[i].plot(enh_stats.index, enh_stats.values, label=f'{model} (Enhanced)', 
                         color=colors[model], marker='o', linewidth=2)
            
            # Noisy Line (Dashed)
            noisy_stats = df[(df['model'] == model) & (df['type'] == 'Noisy')].groupby('num_ch_removed')[m].mean()
            axes2[i].plot(noisy_stats.index, noisy_stats.values, label=f'{model} (Noisy)', 
                         color=colors[model], linestyle='--', alpha=0.6)

        axes2[i].set_title(f'{m} Robustness')
        axes2[i].legend()
        axes2[i].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig("robustness_comparison.png")
    print("Done! Created 'enhanced_only.png' and 'robustness_comparison.png'")

import pandas as pd

# arrays = ["random sphere1 radius = 0.1", "semi circle planar radius = 0.05", "ULA along X-axis", "uniform sphere (rigid) radius = 0.1"]
# ch_removed = [0,1,2,3,4,5]
# models = ["Ambidrop"]
# types = ["noisy", "enhanced"]

# data = {
#     'array': arrays,
#     'num_ch_removed': ch_removed,
#     'model': models,
#     'type': types,
#     'SI-SDR': si_sdr,
#     'PESQ': pesq,
#     'STOI': stoi
# }

# df = pd.DataFrame(data)
# df.to_csv('model_results.csv', index=False)

# import matplotlib.pyplot as plt

# # --- 1. Your Data (Example lists) ---
# # Replace these with your actual lists
# x_axis = [0, 1, 2, 3, 4, 5]  # Number of channels removed
# x_axis = [7,6,5,4,3,2]

# # SI-SDR lists
# sdr_baseline = [5.3, -2.975, -10.875, -16.175, -18.95, -19.725]
# sdr_ambidrop = [4.85, 4.125, 3.225, 1.65, -2.35, -12.025]
# sdr_noisy    = [-6.725, -6.725, -6.725, -6.725, -6.725, -6.725]

# # PESQ lists
# pesq_baseline = [1.66, 1.32, 1.23, 1.2175, 1.2175, 1.22]
# pesq_ambidrop = [1.735, 1.635, 1.5725, 1.505, 1.3925, 1.2775]
# pesq_noisy    = [1.1725, 1.1725, 1.1725, 1.1725, 1.1725, 1.1725]

# # STOI lists
# stoi_baseline = [0.8425, 0.685, 0.565, 0.4975, 0.4425, 0.395]
# stoi_ambidrop = [0.855, 0.8275, 0.805, 0.77, 0.7075, 0.5925]
# stoi_noisy    = [0.59, 0.59, 0.59, 0.59, 0.59, 0.59]

# # --- 2. Plotting Logic ---
# metrics = ['SI-SDR', 'PESQ', 'STOI']
# baseline_data = [sdr_baseline, pesq_baseline, stoi_baseline]
# ambidrop_data = [sdr_ambidrop, pesq_ambidrop, stoi_ambidrop]
# noisy_data    = [sdr_noisy, pesq_noisy, stoi_noisy]

# fig, axes = plt.subplots(1, 1, figsize=(18, 5))
# fig.suptitle('Mean Model Performance vs. Number of Channels', fontsize=16, fontweight='bold')

# for i in range(1):
#     # Plot AmbiDrop
#     axes[i].plot(x_axis, ambidrop_data[i], label='AmbiDrop', 
#                  color='#ff7f0e', marker='s', linewidth=2)
    
#     # Plot Baseline
#     axes[i].plot(x_axis, baseline_data[i], label='Baseline', 
#                  color='#1f77b4', marker='o', linewidth=2)
    
#     # Plot Noisy (Dashed reference)
#     axes[i].plot(x_axis, noisy_data[i], label='Noisy Input', 
#                  color='#7f7f7f', linestyle='--', linewidth=1.5)

#     # Formatting
#     axes[i].set_title(metrics[i], fontsize=14)
#     axes[i].set_xlabel('Number of Input Channels')
#     axes[i].set_ylabel('Score')
#     axes[i].grid(True, alpha=0.3)
#     axes[i].legend()
#     axes[i].set_xlim(7.5, 1.5)

# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()
# plt.savefig('my_metric_comparison.png') # Uncomment to save

#runai-bgu submit python -n ahh -c 20 -m 40G -g 1 --conda venv -- "python /gpfs0/bgu-br/users/tatarjit/speech-enhancement/a.py"

# file_path = "/gpfs0/bgu-br/projects/sim_dataset_ambisonics/si_tr_s/ex_1.mat"
# import scipy.io

# # Load the file
# mat_contents = scipy.io.loadmat(file_path)

# # Filter out the 'internal' metadata fields that start with __
# fields = {k: v for k, v in mat_contents.items() if not k.startswith('__')}

# print("Fields found in .mat file:")
# for key, value in fields.items():
#     # If it's a numpy array (common for MAT files), show the shape
#     try:
#         print(f" - {key}: shape {value.shape}, type {value.dtype}")
#     except AttributeError:
#         print(f" - {key}: value {value}")

import torch
from pprint import pprint

# 1. Load the dictionary
# data = torch.load('/gpfs0/bgu-br/users/tatarjit/speech-enhancement/datasets/experiment_full_anm/mic_val_ds_preprocessed_merged/ex_400.pt', map_location='cpu')

# print(f"--- Dictionary Contents of {type(data)} ---")

# # 2. Iterate and inspect
# for key, value in data.items():
#     print(f"\nKey: {key}")
    
#     # Check if it's a Tensor (don't print the whole thing if it's huge)
#     if torch.is_tensor(value):
#         print(f"  [Tensor] Shape: {list(value.shape)}, Dtype: {value.dtype}")
#         # Print just a small sample if you want:
#         # print(f"  Sample: {value.flatten()[:5]}") 

#     # Check if it's a List or another Dict
#     elif isinstance(value, (list, dict)):
#         print(f"  [{type(value).__name__}] Length: {len(value)}")
#         # Show a small preview of the first item
#         preview = str(value)[:100] + "..." if len(str(value)) > 100 else value
#         print(f"  Preview: {preview}")

#     # Otherwise, it's likely a string, int, or float
#     else:
#         print(f"  [Value]: {value}")

# print("\n--- End of File ---")


# import soundfile as sf
# import numpy as np
# import os

# def check_wavs_identical(path1, path2):
#     # 1. Check if files exist
#     if not os.path.exists(path1) or not os.path.exists(path2):
#         print("One or both files do not exist.")
#         return False

#     # 2. Load the audio data and sample rates
#     data1, sr1 = sf.read(path1)
#     data2, sr2 = sf.read(path2)

#     # 3. Quick checks: Sample Rate and Shape
#     if sr1 != sr2:
#         print(f"Different sample rates: {sr1} vs {sr2}")
#         return False
    
#     if data1.shape != data2.shape:
#         print(f"Different shapes: {data1.shape} vs {data2.shape}")
#         return False

#     # 4. Deep check: Compare all values
#     # np.array_equal returns True only if every single sample is identical
#     is_identical = np.array_equal(data1, data2)
    
#     if is_identical:
#         print("Files are identical.")
#     else:
#         # Check if they are nearly identical (e.g., due to tiny precision errors)
#         if np.allclose(data1, data2, atol=1e-7):
#             print("Files are NOT bit-exact, but are numerically identical (within 1e-7).")
#         else:
#             print("Files are different.")
            
#     return is_identical

# # --- Example Usage ---
# folder1_p = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/datasets/experiment_full_anm/test_of_train_ds/front hemisphere1 (rigid) radius = 0.1/ex_1/p.wav"
# folder2_p = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/datasets/experiment_full_anm/test_of_train_ds/planar/ex_1/p.wav"
# check_wavs_identical(folder1_p, folder2_p)

# import scipy.io
# import numpy as np
# import os

# def check_mats_identical(path1, path2):
#     # 1. בדיקה אם הקבצים קיימים
#     if not os.path.exists(path1) or not os.path.exists(path2):
#         print("One or both .mat files do not exist.")
#         return False

#     # 2. טעינת הנתונים
#     mat1 = scipy.io.loadmat(path1)
#     mat2 = scipy.io.loadmat(path2)

#     # 3. סינון מפתחות המערכת של MATLAB (כמו __header__, __version__)
#     keys1 = {k for k in mat1.keys() if not k.startswith('__')}
#     keys2 = {k for k in mat2.keys() if not k.startswith('__')}

#     # בדיקה אם יש את אותם שמות משתנים בשני הקבצים
#     if keys1 != keys2:
#         print(f"Different variables found! File1 keys: {keys1}, File2 keys: {keys2}")
#         return False

#     # 4. השוואה עמוקה של כל משתנה
#     all_identical = True
#     for key in keys1:
#         print(key)
#         val1 = mat1[key]
#         val2 = mat2[key]

#         # בדיקה אם אלו מערכים של numpy
#         if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
#             if val1.shape != val2.shape:
#                 print(f"Variable '{key}' has different shapes: {val1.shape} vs {val2.shape}")
#                 all_identical = False
#                 continue
            
#             # בדיקת זהות מוחלטת
#             if not np.array_equal(val1, val2):
#                 # בדיקה אם זה רק הבדל זניח של דיוק (float precision)
#                 if np.allclose(val1, val2, atol=1e-7):
#                     print(f"Variable '{key}' is numerically identical but not bit-exact.")
#                 else:
#                     print(f"Variable '{key}' is DIFFERENT.")
#                     all_identical = False
#         else:
#             # השוואה רגילה למשתנים שהם לא מערכים (סטרינגים, מספרים בודדים)
#             if val1 != val2:
#                 print(f"Variable '{key}' is DIFFERENT.")
#                 all_identical = False

#     if all_identical:
#         print("Success: Both .mat files contain identical data.")
    
#     return all_identical

# # --- הרצה על הקבצים שלך ---
# mat_path1 = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/datasets/experiment_full_anm/test_of_train_ds/front hemisphere1 (rigid) radius = 0.1/ex_1/anm.mat"
# mat_path2 = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/datasets/experiment_full_anm/test_of_train_ds/planar/ex_1/anm.mat"

# check_mats_identical(mat_path1, mat_path2)

# data = np.load('/gpfs0/bgu-br/users/tatarjit/speech-enhancement/FaSNet_with_AmbiDrop/master_si_sdr_enhanced.npy')

# # 2. הדפסת העמודה הראשונה
# # הסימן ':' אומר "קח את כל השורות", והמספר '0' אומר "קח את העמודה הראשונה"
# first_col = data[:, 0]

# print("First column shape:", first_col.shape)
# print("First column values:")
# print(first_col)

import torch
import numpy as np

import torch

def compare_two_pt_tuples(path1, path2):
    # 1. טעינת הקבצים
    data1 = torch.load(path1, map_location='cpu', weights_only=False)
    data2 = torch.load(path2, map_location='cpu', weights_only=False)

    # בדיקה ששניהם אכן tuples (או lists)
    if not isinstance(data1, (tuple, list)) or not isinstance(data2, (tuple, list)):
        print(f"Error: One of the files is not a tuple. Type1: {type(data1)}, Type2: {type(data2)}")
        return

    # 2. בדיקת אורך ה-tuple
    if len(data1) != len(data2):
        print(f"Mismatch in tuple length: {len(data1)} vs {len(data2)}")
        # נמשיך להשוות עד האורך המינימלי ביניהם
    
    min_len = min(len(data1), len(data2))
    print(f"Comparing {min_len} elements...\n")

    all_identical = True

    # 3. לולאת השוואה
    for i in range(min_len):
        item1 = data1[i]
        item2 = data2[i]

        # בדיקה אם האיבר הוא Tensor
        if torch.is_tensor(item1) and torch.is_tensor(item2):
            if item1.shape != item2.shape:
                print(f"Element {i} (Tensor): Different shapes {item1.shape} vs {item2.shape}")
                all_identical = False
            elif not torch.allclose(item1.abs(), item2.abs(), atol=1e-7):
                print(f"Element {i} (Tensor): DIFFERENT values.")
                all_identical = False
            else:
                print(f"Element {i} (Tensor): Identical.")

        # בדיקה אם האיבר הוא מילון (אם "One contain dict" הכוונה לאיבר בתוך ה-tuple)
        elif isinstance(item1, dict) and isinstance(item2, dict):
            if item1.keys() != item2.keys():
                print(f"Element {i} (Dict): Different keys.")
                all_identical = False
            else:
                print(f"Element {i} (Dict): Keys match, checking values...")
                # כאן אפשר להוסיף השוואה עמוקה של המילון אם צריך
        
        # השוואה רגילה (עבור ref_id, strings, etc)
        else:
            if item1 == item2:
                print(f"Element {i} ({type(item1).__name__}): Identical.")
            else:
                print(f"Element {i} ({type(item1).__name__}): DIFFERENT ({item1} vs {item2})")
                all_identical = False

    if all_identical and len(data1) == len(data2):
        print("\nConclusion: Files are identical.")
    else:
        print("\nConclusion: Files are DIFFERENT.")

# # --- שימוש ---

# # --- הרצה ---
# # path1 = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/datasets/experiment_full_anm/test_of_train_ds_preprocessed/front hemisphere1 (rigid) radius = 0.1_preprocessed/ex_1.pt"
# # path2 = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/datasets/experiment_full_anm/test_of_train_ds_preprocessed_swap_swap/front hemisphere1 (rigid) radius = 0.1_preprocessed/ex_1.pt"
# # compare_two_pt_tuples(path1, path2)

# import numpy as np
# from scipy.io import loadmat, savemat

# IR = np.load("/gpfs0/bgu-br/users/tatarjit/speech-enhancement/datasets/ATF_mismatch_ds/Aria_shroom/IR.npy")
# ATF = np.load("/gpfs0/bgu-br/users/tatarjit/speech-enhancement/datasets/ATF_mismatch_ds/Aria_shroom/ATF.npy")

# savemat('Aria_ATF_16khz.mat', {'ATF': ATF})
# savemat('Aria_IR_16khz.mat', {'IR': IR})