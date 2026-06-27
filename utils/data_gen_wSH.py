import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt
import random
import soundfile as sf
from scipy.io import savemat
import os
from scipy.special import lpmv
from math import factorial, sqrt, pi
from scipy.linalg import svd
from scipy.signal import fftconvolve

def sh2(N, theta, phi):
    """
    Compute spherical harmonics up to order N.

    Args:
        N (int): maximum order
        theta (array): colatitude angles in radians (0 at north pole)
        phi (array): azimuth angles in radians

    Returns:
        Y (np.ndarray): shape ((N+1)**2, len(theta)), complex values
    """
    theta = np.atleast_1d(theta)
    phi = np.atleast_1d(phi)
    
    if len(theta) != len(phi):
        raise ValueError("Lengths of theta and phi must be equal!")
    
    L = len(theta)
    Y = [np.sqrt(1/(4*pi)) * np.ones(L, dtype=complex)]  # n=0 term

    j = 1j  # complex constant

    for n in range(1, N+1):
        # positive m
        Y1 = []
        for m in range(0, n+1):
            # normalization
            a = sqrt((2*n+1)/(4*pi) * factorial(n-m)/factorial(n+m))
            Pnm = lpmv(m, n, np.cos(theta))  # associated Legendre
            Ynm = a * Pnm * np.exp(j*m*phi)
            Y1.append(Ynm)
        Y1 = np.vstack(Y1)  # shape (n+1, L)
        
        # negative m
        Y2 = []
        for m in range(-n, 0):
            # (-1)^m * conjugate of positive m
            Ynm = (-1)**m * np.conj(Y1[-m-1, :])
            Y2.append(Ynm)
        if Y2:
            Y2 = np.vstack(Y2)
            Y_stack = np.vstack([Y2, Y1])
        else:
            Y_stack = Y1

        # append to Y
        Y.append(Y_stack)

    # stack all n
    Y = np.vstack(Y)
    return Y

def array_ambisonics_time_domain(p, V, th, ph, N, harmonics, fVec, SNR_lin, filt_samp):
    """
    Generate time-domain Ambisonics array signals (Python version).

    Args:
        p (np.ndarray): M x T microphone signals
        V (np.ndarray): M x F x Q steering vectors (frequency domain)
        th (np.ndarray): theta angles of source positions
        ph (np.ndarray): phi angles of source positions
        N (int): max SH order
        harmonics (list or np.ndarray): indices of SH harmonics to process
        fVec (np.ndarray): frequency vector (length F)
        SNR_lin (float): linear SNR
        filt_samp (int): length of time-domain filter

    Returns:
        anmt_array (np.ndarray): harmonics x T, time-domain Ambisonics signals
    """

    N_mic, T = p.shape
    Q = (N+1)**2
    F = len(fVec)

    # Allocate ASM coefficients
    c_ASM = np.zeros((N_mic, len(harmonics), F), dtype=np.complex64)

    # Spherical harmonics matrix
    Y = sh2(N, th, ph).T  # shape Q x len(th)

    # ----- frequency-domain ASM -----
    for j_idx, j in enumerate(harmonics):
        Ynm = Y[:, j]  # Q x 1

        for f in range(F):
            v_k = V[:, f, :]  # M x Q
            lambda_reg = 1 / SNR_lin

            mat_to_inv = v_k @ v_k.conj().T + lambda_reg * np.eye(N_mic)

            # tolerance for numerical stability
            tol_inv = 1 + max(mat_to_inv.shape) * np.finfo(mat_to_inv.dtype).eps * np.linalg.norm(mat_to_inv)

            # SVD-based pseudo-inverse
            U, S, Vh = svd(mat_to_inv)
            S_inv = np.zeros_like(S)
            S_inv[S > tol_inv] = 1.0 / S[S > tol_inv]
            inv_mat = Vh.conj().T @ np.diag(S_inv) @ U.conj().T

            # ASM coefficient
            c_ASM[:, j_idx, f] = inv_mat @ v_k @ Ynm

    # ----- time-domain Ambisonics -----
    anmt_array = np.zeros((len(harmonics), T))

    for j_idx in range(len(harmonics)):
        c_f = c_ASM[:, j_idx, :]  # M x F

        # Zero-padding to filt_samp
        if c_f.shape[1] < filt_samp:
            c_f = np.pad(c_f, ((0, 0), (0, filt_samp - c_f.shape[1])), mode='constant')

        # IFFT to time domain
        c_time = np.fft.ifft(c_f, axis=1).real

        # Circular shift by half filter length
        c_time_cs = np.roll(c_time, filt_samp//2, axis=1)

        # Flip time-reversed component
        c_time_cs = np.concatenate([c_time_cs[:, [0]], c_time_cs[:, 1:][:, ::-1]], axis=1)

        # Convolve with microphone signals
        anmt_array_temp = np.zeros(T)
        for i in range(N_mic):
            anmt_array_temp += fftconvolve(p[i, :], c_time_cs[i, :], mode='same')

        anmt_array[j_idx, :] = anmt_array_temp

    return anmt_array

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

def generate_uniform_circle_array(N_mic=5, radius=0.1):
    """
    Generate 3D positions of N microphones in a uniform circular array on XY plane.
    
    Args:
        N_mic (int): Number of microphones
        radius (float): Radius of the circle (meters)
    
    Returns:
        positions (np.ndarray): Array of shape (N_mic, 3) with x, y, z coordinates
    """
    # Uniform angles
    angles = np.linspace(0, 2*np.pi, N_mic, endpoint=False)
    
    # Compute positions
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    z = np.zeros(N_mic)
    
    positions = np.stack([x, y, z], axis=1)
    return positions

    import numpy as np

def generate_microphone_arrays(N_mic, radius):
    """
    Returns a dict `arrays` with 8 different [N_mic x 3] microphone arrays.
    Random arrays are now fixed with given coordinates scaled by 0.1.
    """
    arrays = {}

    # 1. Uniform Circle (radius)
    phi = np.linspace(0, 2*np.pi, N_mic, endpoint=False)
    x = radius * np.cos(phi)
    y = radius * np.sin(phi)
    arrays['uniform_circle_r01'] = np.column_stack([x, y, np.zeros(N_mic)])

    # 2. Uniform Circle (radius 0.05)
    phi = np.linspace(0, 2*np.pi, N_mic, endpoint=False)
    x = 0.05 * np.cos(phi)
    y = 0.05 * np.sin(phi)
    arrays['uniform_circle_r005'] = np.column_stack([x, y, np.zeros(N_mic)])

    # 3. Semi-Uniform Circle (radius)
    phi = np.linspace(-np.pi/2, np.pi/2, N_mic)
    x = radius * np.cos(phi)
    y = radius * np.sin(phi)
    arrays['semi_uniform_circle_r01'] = np.column_stack([x, y, np.zeros(N_mic)])

    # 4. Semi-Uniform Circle (radius 0.05)
    phi = np.linspace(-np.pi/2, np.pi/2, N_mic)
    x = 0.05 * np.cos(phi)
    y = 0.05 * np.sin(phi)
    arrays['semi_uniform_circle_r005'] = np.column_stack([x, y, np.zeros(N_mic)])

    # 5. ULA along Y-axis
    y = np.linspace(-radius, radius, N_mic)
    x = np.zeros_like(y)
    arrays['ula_y_axis'] = np.column_stack([x, y, np.zeros(N_mic)])

    # 6. ULA along X-axis
    x = np.linspace(-radius, radius, N_mic)
    y = np.zeros_like(x)
    arrays['ula_x_axis'] = np.column_stack([x, y, np.zeros(N_mic)])

    # 7. X-shaped array (center + 4 corners)
    center = np.array([0,0])
    d = radius / np.sqrt(2)
    corner_offsets = np.array([[d,d],[d,-d],[-d,d],[-d,-d]])
    x = np.concatenate([[center[0]], corner_offsets[:,0]])
    y = np.concatenate([[center[1]], corner_offsets[:,1]])
    arrays['x_shape'] = np.column_stack([x, y, np.zeros(len(x))])

    # 8. Plus-shaped array (center + 4 on axes)
    corner_offsets = np.array([[radius,0],[-radius,0],[0,radius],[0,-radius]])
    x = np.concatenate([[center[0]], corner_offsets[:,0]])
    y = np.concatenate([[center[1]], corner_offsets[:,1]])
    arrays['plus_shape'] = np.column_stack([x, y, np.zeros(len(x))])

    # Random arrays (fixed and scaled by 0.1)
    arrays['random1'] = 0.1 * np.array([
        [0.454,-0.096,0],
        [-0.363,-0.354,0],
        [-0.167,0.299,0],
        [0.452,0.478,0],
        [0.740,0.362,0]
    ])

    arrays['random2'] = 0.1 * np.array([
        [0.093,0.805,0],
        [0.726,-0.539,0],
        [-0.4,0.793,0],
        [0.496,0.402,0],
        [-0.187,0.327,0]
    ])

    arrays['random3'] = 0.1 * np.array([
        [0.222,0.542,0],
        [-0.916,0.012,0],
        [-0.329,0.023,0],
        [-0.082,0.411,0],
        [0.610,-0.247,0]
    ])

    arrays['random4'] = 0.1 * np.array([
        [0.906,-0.037,0],
        [0.757,-0.342,0],
        [-0.825,-0.061,0],
        [-0.609,-0.635,0],
        [0.398,0.087,0]
    ])

    return arrays

# Function to circularly shift coordinates
def shift_closest_to_phi0(r_m, radius):
    """
    r_m: np.array of shape (N_mic, 3)
    radius: float, distance along phi=0
    """
    ref_point = np.array([radius, 0, 0])
    distances = np.linalg.norm(r_m - ref_point, axis=1)
    idx_closest = np.argmin(distances)
    
    # Circularly shift rows so closest mic comes first
    r_m_shifted = np.roll(r_m, -idx_closest, axis=0)
    return r_m_shifted

# Constants
Fs = 16000  # Sampling frequency
c = 343     # Speed of sound
nfft = 512  # FFT length
# SNR = float('inf')  # Signal-to-Noise Ratio in dB (equivalent to infinity)
SNR_dB = 30           # assumed sensor SNR in dB
sig_n = 0.1           # noise standard deviation
sig_s = 10**(SNR_dB / 10) * sig_n   # signal variance corresponding to SNR
SNR_lin = sig_s / sig_n             # linear SNR
# array_version = 7
# Assuming `easyCom_loc.get_microphone_locations()` provides relative mic locations as a 2D array
# Each row represents [x, y, z] coordinates relative to the array center
# mic_relative_loc = np.array([
#     [ 82,  -5, -29],  # Mic 1
#     [ -1,  -1,  30],  # Mic 2
#     [-77,  -2,  11],  # Mic 3
#     [-83,  -5, -60],  # Mic 4
# ]) / 1000.0

# mic_relative_loc = generate_uniform_circle_array(N_mic=5, radius=0.1)

arrays = generate_microphone_arrays(5, 0.1)
train_id = [1, 4, 5, 7, 9, 10]
test_id  = [2, 3, 6, 8, 11, 12]

arrays = generate_microphone_arrays(N_mic=5, radius=0.1)
global_idx = 1  # start global index

for array_idx in train_id:
    print(f"Processing train array {array_idx}...")
    # Your code here, e.g., shift closest mic
    key = list(arrays.keys())[(array_idx - 1) % len(arrays)]
    r_m = arrays[key]
    mic_relative_loc = shift_closest_to_phi0(r_m, radius=0.1)

    folder_path = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/datasets/experiment2/val_ds"
    os.makedirs(folder_path, exist_ok=True)
    for ex in range(167):
        file_name = f"ex{global_idx}.mat"
        file_path = os.path.join(folder_path, file_name)
        
        if os.path.exists(file_path):
            print(f"{file_name} already exists")
            global_idx += 1  # increment global index
            continue

        dataset_type = "si_dt_05"  # Options: "si_tr_s", "si_et_05", "si_dt_05"
        folder_name = "/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/wsj0/si_dt_05"

        global_idx += 1  # increment global index
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

        # ASM calculation
        # harmonics = [1, 2, 4, 5, 9]
        # filt_samp = 512
        # F = 256
        # N = 2
        # fVec = np.linspace(0, 8000, F)
        # anmt_array = array_ambisonics_time_domain(mic_signals, V, th, ph, N, harmonics, fVec, SNR_lin, filt_samp)

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

#runai-cmd --name datagen2  -g 0.1 --cpu-limit 10 -- "conda activate venv && python /Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/utils/data_gen.py"
