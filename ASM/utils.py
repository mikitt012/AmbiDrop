from scipy.io import wavfile
from typing import Optional
from scipy import fft as spfft  # threaded FFTs
import matplotlib.pyplot as plt
import os
from ASM.validate import is_signal_frequency_symmetric
from itertools import cycle
from scipy.fft import fft, ifft, next_fast_len
import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

# def safe_mkdir(path):
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     return path

# def band_mask(freqs, f_min=None, f_max=None):
#     freqs = np.asarray(freqs)
#     mask = np.ones_like(freqs, dtype=bool)
#     if f_min is not None:
#         mask &= freqs >= f_min
#     if f_max is not None:
#         mask &= freqs <= f_max
#     return mask

# def aggregate_error(freqs, err_l, err_r, f_min=None, f_max=None):
#     """
#     Aggregate per-frequency, per-ear errors into a single band-RMS error.

#     freqs : (F,)
#     err_l, err_r : (F,)
#         Error per frequency for left/right ear.

#     Returns:
#         E_rms : scalar, RMS of ear-averaged error in [f_min, f_max]
#     """
#     freqs = np.asarray(freqs)
#     err_l = np.asarray(err_l)
#     err_r = np.asarray(err_r)

#     mask = band_mask(freqs, f_min=f_min, f_max=f_max)
#     if not np.any(mask):
#         raise ValueError("No frequencies in the selected band.")

#     # ear-averaged error (you can switch to RMS across ears if you prefer)
#     e = 0.5 * (err_l[mask] + err_r[mask])

#     # band RMS
#     E_rms = float(np.sqrt(np.mean(e**2)))
#     return E_rms

# # def ifft(x, axis=0, positive_values_only=False, even_length=False):
# #     """
# #     Compute the inverse FFT of a real-valued array in the frequency domain.
# #     :param x: 2D or 3D complex-valued array in the frequency domain
# #     :param axis: axis of the frequency dimension
# #     :param positive_values_only: if True compute symmetric fft
# #     :param even_length: if True, assume the input has an even length along the frequency axis
# #     :return: Inverse FFT of the input array in the time domain
# #     """
# #     if positive_values_only:
# #         if even_length:
# #             n = x.shape[axis] * 2 - 2
# #         else:
# #             n = x.shape[axis] * 2 - 1
# #         return np.fft.irfft(x, axis=axis, n=n)
# #     else:
# #         return np.fft.ifft(x, axis=axis)

# def soundsc(p, fs, save_path=None, play=True):
#     import sounddevice as sd
#     # normalize so that abs max is 0.3
#     p =(p / np.max(np.abs(p))) * 0.3

#     if play:
#         sd.play(p, samplerate=fs)
#         sd.wait()

#     if save_path is not None:
#         # ensure float32 (standard for WAV)
#         wavfile.write(save_path, fs, p.astype(np.float32))
#         print(f"Saved to: {save_path}")

# def load_file(file_path):
#     """
#     Load a text file and return its contents as a list of lines.
#     """
#     path_suffix = os.path.splitext(file_path)[1].lower()
#     if path_suffix in ['.yaml', '.yml']:
#         import yaml
#         with open(file_path, 'r') as f:
#             data = yaml.safe_load(f)
#         return data

# def parse_array_from_config(config):
#     array_type = config['type']
#     x_label = config['x_label'] if 'x_label' in config else None
#     mics_r = config['mics_r'] if 'mics_r' in config else None
#     if config['mics_grid']:
#         mics_grid = config['mics_grid']
#         if list(mics_grid.keys()) == ['fibonacci']:
#             from utils.acl_utils import fibonacci_sphere_angles
#             mics_grid = fibonacci_sphere_angles(config['mics_grid']['fibonacci'], return_degrees=True)
#     else:
#         raise ValueError('No microphone grid specified in the configuration file.')
#     return {
#         'array_type': array_type,
#         'mics_r': mics_r,
#         'mics_grid': mics_grid,
#         'x_label': x_label if x_label is not None else f"{array_type} array",
#     }


# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from itertools import cycle


# def plot_error(freqs, errors: dict, figsize=(10, 5), title=None, save_path=None, show=True, styles: dict = None,
#                ylabel='Error (dB)', ylim=None, beta=0.1):
#     """
#     Plot error curves. Automatically changes line style if a curve overlaps
#     significantly with a previously plotted one.

#     Parameters
#     ----------
#     beta : float, optional (default=0.1)
#         Overlap threshold in dB. If the maximum difference between a new curve
#         and any previous curve is less than 'beta', the new curve is considered
#         "on top" and its style is changed.
#     """
#     plt.figure(figsize=figsize)

#     # Store the dB values of curves we have already plotted to check for overlaps
#     history_db = []

#     # Cycle of styles to use ONLY when overlap is detected
#     # (Dashed, Dotted, Dash-Dot)
#     overlap_style_cycler = cycle(['--', ':', '-.'])

#     for label, err in errors.items():
#         # 1. Convert to dB for plotting and comparison
#         curr_db = 10 * np.log10(err)

#         # 2. Determine Style
#         # Priority 1: User manual styles
#         if styles is not None and label in styles:
#             line_style = styles[label]

#         else:
#             # Priority 2: Check for overlap with ANY previous curve
#             is_overlapping = False
#             for prev_db in history_db:
#                 # Check if the curves are "on top of each other" (max diff < beta)
#                 # You can change np.max to np.mean if you want a looser 'average' check
#                 if np.max(np.abs(curr_db - prev_db)) < beta:
#                     is_overlapping = True
#                     break

#             if is_overlapping:
#                 line_style = next(overlap_style_cycler)
#             else:
#                 line_style = '-'  # Default solid for unique curves

#         # 3. Plot
#         plt.plot(freqs, curr_db, label=label, linestyle=line_style)

#         # 4. Save to history
#         history_db.append(curr_db)

#     if ylim is not None:
#         plt.ylim(ylim)

#     plt.xlabel('Frequency (Hz)')
#     plt.ylabel(ylabel)
#     if title is not None:
#         plt.title(title)
#     plt.legend()
#     plt.grid(True, which="both", ls="-", alpha=0.5)
#     plt.xscale('log')
#     plt.tight_layout()

#     if save_path is not None:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         plt.savefig(save_path, dpi=300)
#         print(f"✅ Plot saved to: {save_path}")

#     if show:
#         plt.show()
#     else:
#         plt.close()

# def plot_metric(names, values, title, ylabel="values", figsize=(10, 10), save_path=None, show=True):
#     """
#     Plots numerical values against string configuration names using subplots.
#     Each metric gets its own subplot with dynamic scaling and 20% padding.

#     Parameters:
#     - names: List of strings (x-axis labels)
#     - values: Dictionary {label: list_of_numbers}
#     - title: String for the chart title
#     - figsize: Tuple (width, height). Note: Height might need to be larger for multiple subplots.
#     """
#     num_metrics = len(values)

#     # Create a vertical stack of subplots sharing the same X-axis
#     fig, axes = plt.subplots(num_metrics, 1, figsize=figsize, sharex=True)

#     # Ensure axes is always a list (handles the case of a single metric)
#     if num_metrics == 1:
#         axes = [axes]

#     # Iterate through each metric and its corresponding subplot axis
#     for ax, (label, vals) in zip(axes, values.items()):
#         # 1. Plot the data
#         ax.plot(names, vals, label=label, marker='o', linewidth=2, markersize=6, color='tab:blue')

#         # 2. Calculate Dynamic Limits with 20% Padding
#         v_min = min(vals)
#         v_max = max(vals)
#         v_range = v_max - v_min

#         # Handle case where all values are the same (flat line)
#         if v_range == 0:
#             padding = abs(v_min) * 0.2 if v_min != 0 else 1.0
#         else:
#             padding = v_range * 0.2

#         # Set the custom limits
#         ax.set_ylim(v_min - padding, v_max + padding)

#         # 3. Styling per subplot
#         ax.set_ylabel(label, fontsize=10, fontweight='bold')
#         ax.grid(True, linestyle='--', alpha=0.6)
#         ax.legend(loc='upper right', frameon=True)

#     # Global Styling (X-axis labels only on the bottom plot)
#     plt.xticks(rotation=45, ha='right')
#     plt.xlabel("Configuration")

#     # Main Title
#     fig.suptitle(title, fontsize=16, fontweight='bold')

#     # Adjust layout to prevent overlap
#     plt.tight_layout()

#     # Save Logic
#     if save_path is not None:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         plt.savefig(save_path, dpi=300)
#         print(f"✅ Plot saved to: {save_path}")

#     if show:
#         plt.show()
#     else:
#         plt.close()


# def reconstruct_frequency_sh_spectrum_full(H_pos, n_fft=None, nm_axis=0, freq_axis=1):
#     """
#     Reconstruct the full complex frequency spectrum for SH-domain signals.
#     Handles both EVEN and ODD FFT lengths.

#     Parameters
#     ----------
#     H_pos : ndarray
#         One-sided spectrum (rfft output).
#     n_fft : int, optional
#         The length of the original time-domain signal (FFT size).
#         If None, assumes Even length: 2 * (K_pos - 1).
#         **Must be provided if the original length was Odd.**
#     """
#     N = int(np.sqrt(H_pos.shape[nm_axis]) - 1)
#     nm_list = []
#     for n in range(N + 1):
#         for m in range(-n, n + 1):
#             nm_list.append((n, m))

#     # Move frequency axis to 1
#     H_pos = np.moveaxis(H_pos, freq_axis, 1)
#     nm, K_pos = H_pos.shape[:2]

#     # 1. Determine Full FFT Size
#     if n_fft is None:
#         F = 2 * (K_pos - 1)  # Default to Even assumption
#     else:
#         F = n_fft

#     # Validate input shape matches expected rfft size
#     expected_K = (F // 2) + 1
#     if K_pos != expected_K:
#         raise ValueError(f"Input freq size {K_pos} does not match n_fft={F} (expected {expected_K})")

#     H_full = np.zeros((nm, F) + H_pos.shape[2:], dtype=complex)

#     # Copy positive frequencies (0 to Nyquist-ish)
#     H_full[:, :K_pos, ...] = H_pos

#     nm_to_index = {nm_list[i]: i for i in range(len(nm_list))}

#     # 2. Define Indices based on Even/Odd
#     if F % 2 == 0:
#         # EVEN Case: Has Nyquist at index F/2
#         idx_nyq = K_pos - 1
#         k_pos = np.arange(1, idx_nyq)  # Exclude DC and Nyquist
#         k_neg = F - k_pos  # Map to upper half
#         has_nyquist = True
#     else:
#         # ODD Case: No Nyquist bin
#         k_pos = np.arange(1, K_pos)  # Exclude DC only
#         k_neg = F - k_pos
#         has_nyquist = False

#     for idx, (n, m) in enumerate(nm_list):
#         if m == 0:
#             # Standard Hermitian (Real in time)
#             H_full[idx, k_neg, ...] = H_pos[idx, k_pos, ...].conj()
#             if has_nyquist:
#                 H_full[idx, idx_nyq, ...].imag = 0.0

#         else:
#             idx_minus = nm_to_index[(n, -m)]
#             parity = (-1) ** m

#             # Fill Negative Frequencies
#             H_full[idx_minus, k_neg, ...] = parity * H_pos[idx, k_pos, ...].conj()

#             # Enforce Nyquist Consistency (Only for Even)
#             if has_nyquist:
#                 val_nyq = H_pos[idx, idx_nyq, ...]
#                 H_full[idx_minus, idx_nyq, ...] = parity * val_nyq.conj()

#     # Restore axis
#     H_full = np.moveaxis(H_full, 1, freq_axis)

#     return H_full

# def reconstruct_frequency_spectrum_full(s, freq_axis=0):
#     """
#     Construct a Hermitian-symmetric full spectrum from the positive-frequency side.

#     Parameters
#     ----------
#     s : ndarray
#         Array containing the DC + positive frequency bins.
#         Must include:
#             - index 0: DC
#             - index -1: Nyquist (for even-length FFT)
#             - columns 1..Npos-2: positive frequencies
#     freq_axis : int, optional
#         Axis along which the concatenation is applied.
#         Default is 0.

#     Returns
#     -------
#     out : ndarray
#         Hermitian-symmetric full FFT spectrum suitable for np.fft.ifft.

#     Notes
#     -----
#     If s has shape (..., Fpos), the output will have shape (..., 2*Fpos - 2),
#     matching numpy's Hermitian symmetry for a real IFFT:
#         [0, +1, +2, ..., +N/2,  -N/2+1, ..., -1]
#     """
#     s = np.asarray(s)

#     # Move target axis to front for simplicity
#     s_T = np.moveaxis(s, freq_axis, 0)     # shape: (Fpos, ...)

#     if s_T.shape[0] < 2:
#         raise ValueError(
#             "Input must contain at least DC and one positive frequency bin."
#         )

#     # Extract:
#     #   - s_T[0]   : DC
#     #   - s_T[-1]  : Nyquist
#     #   - s_T[1:-1]: positive frequencies
#     pos = s_T
#     neg = pos[-2:0:-1].conj()    # reversed 1..Npos-2, conjugated

#     # Concatenate: [DC, positives, Nyquist, negative frequencies]
#     out_T = np.concatenate([pos, neg], axis=0)

#     # Move back to original axis placement
#     out = np.moveaxis(out_T, 0, freq_axis)
#     return out

# def convolve_and_sum_any(
#     signal1,
#     signal2,
#     *,
#     time_dim1: int,
#     time_dim2: int,
#     channel_dims1: int,
#     channel_dims2: int,
#     mode: str = 'full',
#     domain1: str = 'time',   # 'time' or 'freq'
#     domain2: str = 'time',   # 'time' or 'freq'
#     output_domain: str = 'time',  # 'time' or 'freq'
#     signal1_conj: bool = False,
#     nfft: Optional[int] = None,
#     real_signals: Optional[bool] = None,  # if None, auto-detect from dtype & domain
#     workers: int = 1  # >1 = multithreaded FFT via SciPy
# ):
#     """
#     Channel-wise convolution sum: sum_c (x_c * h_c).
#     Handles time/frequency inputs and can return time or frequency output.

#     Shapes:
#       - If domainX='time': moveaxis will put (C, T, ...).
#       - If domainX='freq': moveaxis will put (C, K, ...) where K = nfft//2+1 (real) or nfft (complex).
#     """
#     # Convert to numpy arrays
#     x = np.asarray(signal1)
#     h = np.asarray(signal2)

#     # Move (C, T-or-K) to front
#     x = np.moveaxis(x, [channel_dims1, time_dim1], [0, 1])
#     h = np.moveaxis(h, [channel_dims2, time_dim2], [0, 1])

#     Cx, Tx_or_Kx, *extra1 = x.shape
#     Ch, Th_or_Kh, *extra2 = h.shape
#     if Cx != Ch:
#         raise ValueError(f"Channel mismatch: {Cx} vs {Ch}")
#     C = Cx

#     # Determine real/complex pipeline
#     if real_signals is None:
#         if domain1 == 'freq' or domain2 == 'freq':
#             # If already in freq, infer from dtype (complex spectrum -> complex path)
#             real_signals = not (np.iscomplexobj(x) or np.iscomplexobj(h))
#         else:
#             # time inputs: infer from dtype (int/float -> real; complex -> complex)
#             real_signals = not (np.iscomplexobj(x) or np.iscomplexobj(h))

#     # Output length & nfft
#     if domain1 == 'time' and domain2 == 'time':
#         T1, T2 = Tx_or_Kx, Th_or_Kh
#         full_L = T1 + T2 - 1
#         if mode == 'full':
#             L = full_L
#         elif mode == 'same':
#             L = max(T1, T2)
#         elif mode == 'valid':
#             L = max(T1, T2) - min(T1, T2) + 1
#         else:
#             raise ValueError("mode must be 'full'|'same'|'valid'")
#         if nfft is None:
#             nfft = full_L
#     else:
#         # At least one is already in frequency domain: you must provide nfft or we infer from K
#         if nfft is None:
#             # Infer from the freq axis sizes
#             Kx = Tx_or_Kx
#             Kh = Th_or_Kh
#             if real_signals:
#                 # K = nfft//2 + 1
#                 nx = 2 * (Kx - 1)
#                 nh = 2 * (Kh - 1)
#             else:
#                 nx = Kx if domain1 == 'freq' else None
#                 nh = Kh if domain2 == 'freq' else None
#             # pick the defined one (or ensure both match)
#             candidates = [v for v in (nx, nh) if v is not None]
#             if not candidates:
#                 raise ValueError("Provide nfft when giving time/freq mix without enough info to infer it.")
#             if any(v != candidates[0] for v in candidates):
#                 raise ValueError("Inconsistent frequency sizes; please provide nfft explicitly.")
#             nfft = candidates[0]

#         # If any input is time-domain, we still need L/mode to trim if returning time
#         if domain1 == 'time' and domain2 == 'freq':
#             T1 = Tx_or_Kx
#             T2 = None
#             full_L = T1 + (nfft if real_signals else nfft) - 1  # effective impulse length unknown; treat as full nfft
#             # In practice when mixing, use mode='full' or set L explicitly; here we mirror 'full'
#             L = T1 + (nfft if real_signals else nfft) - 1
#             mode = 'full'
#         elif domain1 == 'freq' and domain2 == 'time':
#             T2 = Th_or_Kh
#             L = (nfft if real_signals else nfft) + T2 - 1
#             full_L = L
#             mode = 'full'
#         else:
#             # both freq domain: if returning time we need L; choose full_L = nfft for a circular -> linear via zero-pad expectation
#             full_L = nfft
#             L = nfft

#     # Reshape to broadcast extras:
#     # x: (C, T/K, *extra1, [1...])
#     # h: (C, T/K, [1...], *extra2)
#     x = x.reshape((C, Tx_or_Kx, *extra1, *([1] * len(extra2))))
#     h = h.reshape((C, Th_or_Kh, *([1] * len(extra1)), *extra2))

#     # Prepare frequency-domain tensors
#     if domain1 == 'time':
#         if real_signals:
#             X = spfft.rfft(x, n=nfft, axis=1, workers=workers)
#         else:
#             X = spfft.fft(x, n=nfft, axis=1, workers=workers)
#         if signal1_conj:
#             X = np.conj(X)
#     else:
#         X = x  # already freq

#     if domain2 == 'time':
#         if real_signals:
#             H = spfft.rfft(h, n=nfft, axis=1, workers=workers)
#         else:
#             H = spfft.fft(h, n=nfft, axis=1, workers=workers)
#     else:
#         H = h  # already freq

#     # Multiply and sum over channels: S = sum_c X_c * H_c
#     # Do it with minimal temporaries:
#     #   tmp = X * H  (in-place on a copy of X to keep inputs intact)
#     tmp = X * H
#     S = np.add.reduce(tmp, axis=0)  # shape: (K, *extra1, *extra2)
#     real_signals = is_signal_frequency_symmetric(S, freq_axis=0, is_print=False)

#     if output_domain == 'freq':
#         return S, nfft  # caller can reuse this directly

#     # Back to time domain
#     if real_signals:
#         y_full = spfft.irfft(S, n=nfft, axis=0, workers=workers)  # (nfft, *extra1, *extra2)
#     else:
#         y_full = spfft.ifft(S, n=nfft, axis=0, workers=workers)

#     # Trim per mode
#     if mode == 'full':
#         y = y_full
#     elif mode == 'same':
#         start = (y_full.shape[0] - L) // 2
#         y = y_full[start:start + L]
#     else:  # 'valid'
#         start = (y_full.shape[0] - L) // 2
#         y = y_full[start:start + L]

#     return y

# import numpy as np

# def acn_to_nm(l):
#     """
#     ACN index -> (n, m) with n>=0, m in [-n..n]
#     """
#     n = int(np.floor(np.sqrt(l)))
#     while (n + 1) ** 2 <= l:
#         n += 1
#     m = l - n * (n + 1)
#     return n, m

# def check_sh_hrtf(h, N, verbose=True):
#     """
#     h: ndarray, shape (L, F, E)
#        SH-domain HRTF in frequency domain (full complex FFT).
#     N: int
#        Maximum SH order.
#     """
#     L, F, E = h.shape
#     assert L == (N + 1) ** 2, f"L={L} but (N+1)^2={(N+1)**2}"
#     nyq = F // 2  # assuming even F

#     # ---- 1. m=0 channels: DC, Nyquist, Hermitian ----
#     m0_indices = []
#     for l in range(L):
#         n, m = acn_to_nm(l)
#         if m == 0:
#             m0_indices.append(l)

#     dc_imags = []
#     nyq_imags = []
#     hermitian_errors = []

#     for l in m0_indices:
#         # DC & Nyquist imaginary parts
#         dc_imags.append(np.max(np.abs(h[l, 0, :].imag)))
#         nyq_imags.append(np.max(np.abs(h[l, nyq, :].imag)))

#         # Hermitian symmetry across freq for m=0 channel
#         pos = slice(1, nyq)                # 1..nyq-1
#         neg = slice(-1, -nyq, -1)          # -1..-nyq+1

#         diff = h[l, pos, :] - np.conj(h[l, neg, :])
#         hermitian_errors.append(np.max(np.abs(diff)))

#     dc_imag_max = max(dc_imags) if dc_imags else 0.0
#     nyq_imag_max = max(nyq_imags) if nyq_imags else 0.0
#     hermitian_max = max(hermitian_errors) if hermitian_errors else 0.0

#     # ---- 2. ±m conjugation: h_{n,-m} ≈ (-1)^m conj(h_{n,m}) ----
#     pm_errors = []
#     for l in range(L):
#         n, m = acn_to_nm(l)
#         if m <= 0:
#             continue  # only handle m>0, pair with -m
#         l_pos = l
#         l_neg = n * (n + 1) + (-m)   # ACN index for (n,-m)

#         # expected: h[n,-m] = (-1)^m * conj(h[n,m])
#         lhs = h[l_neg, :, :]                     # (F,E)
#         rhs = ((-1) ** m) * np.conj(h[l_pos, :, :])
#         pm_errors.append(np.max(np.abs(lhs - rhs)))

#     pm_error_max = max(pm_errors) if pm_errors else 0.0

#     # ---- 3. Low-frequency energy ratios (monopole vs first order) ----
#     # Use a few lowest non-DC bins
#     low_bins = range(1, min(6, nyq))  # f=1..5 or up to nyq-1
#     ratios = []  # (f, n, m, ratio)

#     # locate (0,0) channel
#     l_00 = 0  # in ACN, l=0 is always (0,0)

#     for f_idx in low_bins:
#         mag_00 = np.linalg.norm(h[l_00, f_idx, :])
#         for l in range(L):
#             n, m = acn_to_nm(l)
#             if n == 0:
#                 continue
#             mag_nm = np.linalg.norm(h[l, f_idx, :])
#             r = mag_nm / (mag_00 + 1e-12)
#             ratios.append((f_idx, n, m, r))

#     if verbose:
#         print("=== m=0 channels (n=0..N, m=0) ===")
#         print("max |Im(DC)|      :", dc_imag_max)
#         print("max |Im(Nyquist)| :", nyq_imag_max)
#         print("max Hermitian err :", hermitian_max)
#         print()
#         print("=== ±m conjugation: h_{n,-m} ≈ (-1)^m conj(h_{n,m}) ===")
#         print("max ±m error      :", pm_error_max)
#         print()
#         print("=== Low-frequency energy ratios |h_{n,m}| / |h_{0,0}| ===")
#         for (f_idx, n, m, r) in ratios:
#             if n == 1:  # typically interesting for N=1
#                 print(f"f={f_idx:3d}, (n,m)=({n},{m}): ratio={r:.3e}")

#     return {
#         "dc_imag_max": dc_imag_max,
#         "nyq_imag_max": nyq_imag_max,
#         "hermitian_m0_max": hermitian_max,
#         "pm_error_max": pm_error_max,
#         "low_freq_ratios": ratios,
#     }

from scipy.signal import fftconvolve

def convolve_and_sum(
    signal1: np.ndarray, signal2: np.ndarray, signal1_domain: str, signal2_domain: str
) -> np.ndarray:
    """
    Parameters:
    ----------
    signal1 : np.ndarray, shape (N1, ch, T1) - Ambisonics
    signal2 : np.ndarray, shape (N2, ch, T2) - HRTFs or RIRs
    signal1_domain: str, 'time' or 'freq' represents signal1 domain.
    signal2_domain: str, 'time' or 'freq' represents signal2 domain.

    Returns:
    -------
    output : np.ndarray, shape (N1, N2, T1 + T2 - 1)
    """
    # 1. Strict Domain Enforcement
    if signal1_domain != "time" or signal2_domain != "time":
        raise ValueError(
            f"Domain Mismatch: Both signals must be in 'time' domain. "
            f"Received: signal1={signal1_domain}, signal2={signal2_domain}. "
            f"Please convert signals to time domain before calling this function "
            f"to ensure correct linear convolution padding."
        )
    # 2. Input Validation
    assert signal1.ndim == 3, "signal1 must be 3D: (N1, ch, T1)"
    assert signal2.ndim == 3, "signal2 must be 3D: (N2, ch, T2)"

    N1, ch1, T1 = signal1.shape
    N2, ch2, T2 = signal2.shape

    if ch1 != ch2:
        raise ValueError(
            f"signal1 number of channels ({ch1}) must match signal2 number of channels ({ch2})."
        )

    L_out   = T1 + T2 - 1
    T_short = min(T1, T2)
    T_long  = max(T1, T2)

    # --- Choose strategy ---
    # OLA is beneficial when one signal is >> 8× longer than the other,
    # because it avoids padding the short filter to the full signal length.
    use_ola = (T_long > 8 * T_short) and (T_long > 1000)

    if not use_ola:
        # --- Full-FFT path (original algorithm) ---
        fft_len  = next_fast_len(L_out)
        S1       = fft(signal1[:, np.newaxis, :, :], n=fft_len, axis=-1)
        S2       = fft(signal2[np.newaxis, :, :, :], n=fft_len, axis=-1)
        S_result = np.sum(S1 * S2, axis=2)
        output   = ifft(S_result, n=fft_len, axis=-1)
        return output[..., :L_out]

    # --- OLA path ---
    # Block the longer signal; treat the shorter one as the "filter" whose
    # FFT is pre-computed once. Each block generates partial output that is
    # overlap-added into the result array.
    out_dtype = np.result_type(signal1.dtype, signal2.dtype, np.complex64)
    output    = np.zeros((N1, N2, L_out), dtype=out_dtype)

    block = next_fast_len(T_short * 8)
    step  = block - T_short + 1          # non-overlapping input step per block

    if T1 <= T2:
        # signal1 is the short "filter"; block-process signal2
        H_pre    = fft(signal1[:, np.newaxis, :, :], n=block, axis=-1, workers=-1)  # (N1, 1, ch, block)
        n_blocks = int(np.ceil(T2 / step))
        for k in range(n_blocks):
            start   = k * step
            end     = min(start + step, T2)
            chunk   = signal2[np.newaxis, :, :, start:end]              # (1, N2, ch, chunk_len)
            S_block = fft(chunk, n=block, axis=-1, workers=-1)          # (1, N2, ch, block)
            Y_block = np.sum(H_pre * S_block, axis=2)                   # (N1, N2, block)
            y_block = ifft(Y_block, n=block, axis=-1, workers=-1)       # (N1, N2, block)
            out_end = min(start + block, L_out)
            output[:, :, start:out_end] += y_block[:, :, :out_end - start]
    else:
        # signal2 is the short "filter"; block-process signal1
        H_pre    = fft(signal2[np.newaxis, :, :, :], n=block, axis=-1, workers=-1)  # (1, N2, ch, block)
        n_blocks = int(np.ceil(T1 / step))
        for k in range(n_blocks):
            start   = k * step
            end     = min(start + step, T1)
            chunk   = signal1[:, np.newaxis, :, start:end]              # (N1, 1, ch, chunk_len)
            S_block = fft(chunk, n=block, axis=-1, workers=-1)          # (N1, 1, ch, block)
            Y_block = np.sum(S_block * H_pre, axis=2)                   # (N1, N2, block)
            y_block = ifft(Y_block, n=block, axis=-1, workers=-1)       # (N1, N2, block)
            out_end = min(start + block, L_out)
            output[:, :, start:out_end] += y_block[:, :, :out_end - start]

    return output