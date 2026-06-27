import pyroomacoustics as pra
import numpy as np
# import easyCom_loc
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import sounddevice as sd
import random
import soundfile as sf
from scipy.io import savemat
import os
# from zero_pad import zero_pad_signals

# def zero_pad_signals(signals, target_length):
#     """
#     Zero-pad a list of 1D numpy arrays to the same target length.
#     """
#     padded = []
#     for s in signals:
#         pad_len = target_length - len(s)
#         if pad_len < 0:
#             raise ValueError("Signal is longer than target length")
#         padded.append(np.pad(s, (0, pad_len)))
#     return np.stack(padded)

def zero_pad_signals(signals, target_length, idx):
    """
    Truncate or zero-pad 1D numpy arrays to a fixed target length.
    Prints a warning if a signal is truncated.
    """
    padded = []
    for s in signals:
        if len(s) > target_length:
            print(f"[Warning] Signal at index {idx} truncated from {len(s)} to {target_length} samples.")
            s = s[:target_length]  # Truncate
        else:
            s = np.pad(s, (0, target_length - len(s)))  # Zero-pad
        padded.append(s)
    return np.stack(padded)

# Constants
Fs = 16000  # Sampling frequency
c = 343     # Speed of sound
nfft = 512  # FFT length
SNR = float('inf')  # Signal-to-Noise Ratio in dB (equivalent to infinity)
# array_version = 7
# Assuming `easyCom_loc.get_microphone_locations()` provides relative mic locations as a 2D array
# Each row represents [x, y, z] coordinates relative to the array center
mic_relative_loc = np.array([
    [ 82,  -5, -29],  # Mic 1
    [ -1,  -1,  30],  # Mic 2
    [-77,  -2,  11],  # Mic 3
    [-83,  -5, -60],  # Mic 4
]) / 1000.0

folder_path = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/datasets/experiment2/val_ds"
os.makedirs(folder_path, exist_ok=True)
for ex in range(167):
    file_name = f"ex{ex}.mat"
    file_path = os.path.join(folder_path, file_name)
    
    if os.path.exists(file_path):
        print(f"{file_name} already exists")
        continue

    dataset_type = "si_dt_05"  # Options: "si_tr_s", "si_et_05", "si_dt_05"
    folder_name = "/gpfs0/bgu-br/users/tatarjit/speech-enhancement/wsj0/si_dt_05"

    # Reverberation time
    T60 = 0.2 + 0.3 * np.random.rand()

    # Room dimensions
    room_dim = [
        2.5 + 2.5 * np.random.rand(),  # Length of the room
        3 + 6 * np.random.rand(),      # Width of the room
        2.2 + 1.3 * np.random.rand()   # Height of the room
    ]

    # We invert Sabine's formula to obtain the parameters for the ISM simulator
    r_absorption, max_order = pra.inverse_sabine(T60, room_dim)

    # Create the room
    room = pra.ShoeBox(
        room_dim, fs=16000, materials=pra.Material(r_absorption), max_order=max_order,
        use_rand_ism=True, max_rand_disp=0.05 )

    # Array position (at least 1 m away from walls)
    array_center = np.array([
        1 + (room_dim[0] - 2) * np.random.rand(),  # x-coordinate
        1 + (room_dim[1] - 2) * np.random.rand(),  # y-coordinate
        1.5   ])                                   # z-coordinate (fixed at 1.5 m)
    # Array rotation
    phs = 2 * np.pi * np.random.rand()  # Random rotation angle in radians
    # Calculate absolute microphone positions
    rotation_matrix = np.array([
        [np.cos(phs), -np.sin(phs), 0],
        [np.sin(phs), np.cos(phs), 0],
        [0, 0, 1]
    ])

    # Apply rotation and translation to the relative microphone positions
    mic_absolute_loc = (rotation_matrix @ mic_relative_loc.T).T + array_center
    #pertube the mics locations
    # easyCom_loc.perturb_array(mic_absolute_loc,p_val= 0.03,perturbation_flag=0)

    # Add microphones to the room
    room.add_microphone_array(mic_absolute_loc.T)

    # Source position
    rs = 0.3 + 0.7 * np.random.rand()  # r source
    Xs = array_center + np.array([rs * np.cos(phs), rs * np.sin(phs), 0])  # Source position

    # Interference angles
    ph_segments = phs + np.linspace(np.deg2rad(20), np.deg2rad(340), 6)
    phi = ph_segments[:5] + (np.deg2rad(320 / 5) * np.random.rand(5))  # phi interference

    # Interference sources
    while True:
        ri = 1 + 7 * np.random.rand(5)  # r interference
        Xi = array_center + np.column_stack([
            ri * np.cos(phi),  # x-coordinates
            ri * np.sin(phi),  # y-coordinates
            0.1 + np.sqrt(0.08) * np.random.randn(5)  # z-coordinates
        ])

        # Check if all interference sources are within bounds
        if np.all((Xi >= [0, 0, 0]) & (Xi <= room_dim), axis=(0, 1)):
            break
            
    # mic_absolute_loc = clamp_to_room(mic_absolute_loc, room_dim)
    # Xs = clamp_to_room(Xs, room_dim)
    # Xi = clamp_to_room(Xi, room_dim)

    # # Ensure every mic is at least X meters away from the source
    # min_dist = 0.1  # Minimum 10 cm to avoid zero delay
    # for mic in mic_absolute_loc:
    #     dist = np.linalg.norm(mic - Xs)
    #     if dist < min_dist:
    #         print(f"⚠️ Source too close to mic (dist={dist:.3f} m), repositioning...")
    #         shift = (mic - Xs)
    #         shift = shift / (np.linalg.norm(shift) + 1e-6) * (min_dist - dist + 1e-3)
    #         Xs = clamp_to_room(Xs - shift, room_dim)


    ''' # Create a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot source position
    ax.scatter(Xs[0], Xs[1], Xs[2], marker='o', s=100, linewidths=2, label='Source')
    
    # Plot microphone position
    ax.scatter(array_center[0], array_center[1], array_center[2], marker='x', s=50, linewidths=2, label='Microphone')
    
    # Plot interference positions
    ax.scatter(Xi[:, 0], Xi[:, 1], Xi[:, 2], marker='D', s=50, linewidths=2, label='Interference')
    
    # Set axis limits
    ax.set_xlim([0, room_dim[0]])
    ax.set_ylim([0, room_dim[1]])
    ax.set_zlim([0, room_dim[2]])
    
    # Add labels and grid
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.grid(True)
    
    # Define direction vectors for X, Y, and Z axes
    Xdir = [np.cos(phs), np.sin(phs), 0]
    Ydir = [np.cos(phs + np.pi / 2), np.sin(phs + np.pi / 2), 0]
    Zdir = [0, 0, 1]
    
    # Add quivers for direction vectors
    ax.quiver(array_center[0], array_center[1], array_center[2], Xdir[0], Xdir[1], Xdir[2], length=0.5, color='r', label='X Direction')
    ax.quiver(array_center[0], array_center[1], array_center[2], Ydir[0], Ydir[1], Ydir[2], length=0.5, color='g', label='Y Direction')
    ax.quiver(array_center[0], array_center[1], array_center[2], Zdir[0], Zdir[1], Zdir[2], length=0.5, color='b', label='Z Direction')
    
    # Add legend
    ax.legend(loc='best')
    
    # Set 2D view
    # ax.view_init(elev=90, azim=-90)
    
    # Show plot
    plt.show()
    '''

    # Load speech signals
    # -------------------
    # Get list of subfolders (ignoring '.' and '..')
    folders = [f for f in os.listdir(folder_name) if os.path.isdir(os.path.join(folder_name, f))]
    random_indices = random.sample(range(len(folders)), 6)

    s = []         # List to store signals
    names = []     # List to store file names
    len_signals = []  # List to store signal lengths

    for i in random_indices:
        # Get all .wav files in the selected folder
        utts = [os.path.join(folder_name, folders[i], f) for f in os.listdir(os.path.join(folder_name, folders[i])) if f.endswith('.wav')]
        if utts:  # Ensure there are .wav files
            selected_file = random.choice(utts)  # Randomly select a .wav file
            signal, sr = sf.read(selected_file)  # Read the audio file
            s.append(signal)
            names.append(os.path.basename(selected_file))  # Store the file name
            len_signals.append(len(signal))

    # Pad signals to match the length of the longest signal
    max_len = max(len_signals)
    for i in range(6):
        n_zeros = max_len - len_signals[i]
        if n_zeros > 0:
            # Pad with zeros
            s[i] = np.pad(s[i], (0, n_zeros), 'constant', constant_values=0)
            # Randomly circular shift the signal
            shift_amount = random.randint(0, n_zeros)
            s[i] = np.roll(s[i], shift_amount)

    # Add the first source (desired source) to the room
    room.add_source(Xs, signal=s[0])

    # Add the interference sources to the room
    for i in range(Xi.shape[0]):  # Loop over each interference source
        room.add_source(Xi[i, :], signal=s[i + 1])  # Add interference source and its corresponding signal

    room.compute_rir()

    # Simulate the propagation of sources in the room
    room.simulate()
    # Retrieve microphone signals
    mic_signals = room.mic_array.signals  # Shape: (num_mics, num_samples)

    '''   listen to mixture of signals
    
    # Play each microphone signal
    for i, mic_signal in enumerate(mic_signals):
        print(f"Playing signal from Microphone {i + 1}...")
        sd.play(mic_signal, samplerate=Fs)  # Play the signal with the sampling rate
        sd.wait()  # Wait until the playback finishes
        print(f"Finished playing signal from Microphone {i + 1}")
    '''

    # Simulate only the target source (no interference)
    room_target_only = pra.ShoeBox(
        room_dim, fs=Fs, materials=pra.Material(r_absorption), max_order=max_order,
        use_rand_ism=True, max_rand_disp=0.05
    )
    room_target_only.add_microphone_array(mic_absolute_loc.T)
    room_target_only.add_source(Xs, signal=s[0])  # Only the target source
    room_target_only.compute_rir()
    room_target_only.simulate()

    # Retrieve microphone signals for target source only
    pTarget = room_target_only.mic_array.signals  # Signals of the target source only

    # Simulate only the direct path for the target source (no reflections)
    room_direct_only = pra.ShoeBox(
        room_dim, fs=Fs, max_order=0  # Set max_order=0 to exclude reflections
    )
    room_direct_only.add_microphone_array(mic_absolute_loc.T)
    room_direct_only.add_source(Xs, signal=s[0])  # Only the target source
    room_direct_only.compute_rir()
    room_direct_only.simulate()

    # Retrieve microphone signals for direct path only
    pDirect = room_direct_only.mic_array.signals  # Direct path signals of the target source

    # Get the number of samples in `p` (assumed to be the longest signal)
    max_len = mic_signals.shape[1]  # Length of signals in `p`
    print(mic_signals.shape)

    # Zero-pad pTarget and pDirect
    pTarget = zero_pad_signals(pTarget, max_len, ex)
    pDirect = zero_pad_signals(pDirect, max_len, ex)

    # Save data to a .mat file
    data_to_save = {
        "p": mic_signals,                # Microphone signals (with interference and reverberation)
        # "pTarget": pTarget,              # Microphone signals of the target source only
        "pDirect": pDirect              # Direct path signals of the target source
        # "L": room_dim,                   # Room dimensions
        # "T60": T60,                      # Reverberation time
        # "Xm": array_center,              # Array center location
        # "Xs": Xs,                        # Desired source location
        # "phs": phs,                      # Rotation angle of the array
        # "array_ver" : array_version ,     # array version
        # "mic_rel_loc": mic_relative_loc,  # Microphone reletive location
        # "names": names                   # Names of the sources
    }

    # Save to .mat file
    savemat(file_path, data_to_save)

    print(f"Data saved to {file_path}")

    '''   listen
    #Select the first microphone's signals
    mic_p = (mic_signals[0])       # First mic of p (with interference and reverberation)
    mic_pTarget = (pTarget[0])    # First mic of pTarget (only target source with reverberation)
    mic_pDirect = (pDirect[0])    # First mic of pDirect (only target source, direct path)
    
    # Play each signal
    print("Playing mic_p (with interference and reverberation)...")
    sd.play(mic_p, samplerate=Fs)
    sd.wait()  # Wait for playback to finish
    print("Finished playing mic_p.")
    
    print("Playing mic_pTarget (target source with reverberation)...")
    sd.play(mic_pTarget, samplerate=Fs)
    sd.wait()  # Wait for playback to finish
    print("Finished playing mic_pTarget.")
    
    print("Playing mic_pDirect (target source, direct path only)...")
    sd.play(mic_pDirect, samplerate=Fs)
    sd.wait()  # Wait for playback to finish
    print("Finished playing mic_pDirect.")
    '''

#runai-cmd --name datagen  -g 0.1 --cpu-limit 30 -- "conda activate venv && python /gpfs0/bgu-br/users/tatarjit/speech-enhancement/utils/data_gen.py"
