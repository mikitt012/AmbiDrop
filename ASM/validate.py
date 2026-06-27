import numpy as np
import warnings

def is_signal_frequency_symmetric(s, freq_axis=0, atol=1e-9, rtol=1e-7, is_print=True):
    """
    Check if a real-valued signal is symmetric (Hermitian) along a given axis.
    """
    s = np.asarray(s)
    N = s.shape[freq_axis]
    j = np.arange(1, N//2)
    s_pos = np.take(s, j, axis=freq_axis)
    s_neg = np.take(s, -j, axis=freq_axis).conj()
    pair_ok = np.allclose(s_pos, s_neg, atol=atol, rtol=rtol)

    # DC must be real
    dc = np.take(s, 0, axis=freq_axis)
    dc_ok = np.allclose(dc.imag, 0, atol=atol)

    # Nyquist (only if even N) must be real
    if N % 2 == 0:
        nyq = np.take(s, N // 2, axis=freq_axis)
        nyq_ok = np.allclose(nyq.imag, 0, atol=atol)
    else:
        nyq_ok = True

    ok = pair_ok and dc_ok and nyq_ok

    if not ok and is_print:
        if not pair_ok:
            warnings.warn("Signal is not Hermitian symmetric: frequency pairs do not match.")
        if not dc_ok:
            warnings.warn("Signal is not Hermitian symmetric: DC component is not real.")
        if not nyq_ok:
            warnings.warn("Signal is not Hermitian symmetric: Nyquist component is not real.")
    return pair_ok and dc_ok and nyq_ok

def is_signal_frequency_space_valid(s, freq_axis=1):
    # check if signal for signal at f==0 is uniform
    dc = np.take(s, 0, axis=freq_axis)
    dc = np.abs(dc)
    if np.isclose(dc.mean(axis=0), dc.min(axis=0)).all():
        freq0_ok = True
    else:
        freq0_ok = False
        print('frequencyXspace domain signal is not constant across f=0!')

    if is_signal_frequency_symmetric(s, freq_axis=freq_axis):
        symmetric_ok = True
    else:
        symmetric_ok = False
    return symmetric_ok and freq0_ok

def is_signal_frequency_sh_valid(s, freq_axis=1, sh_axis=0, atol=1e-9, rtol=1e-7):
    """
    Check if a complex SH frequency spectrum satisfies the reality condition:
    A_{n,-m}[F-k] = (-1)^m * A_{n,m}[k]*.

    Parameters
    ----------
    s : ndarray, shape (nm, F, ...)
        The full complex SH frequency spectrum.
    freq_axis : int
        The axis corresponding to the full frequency dimension (F). Default is 1.
    sh_axis : int
        The axis corresponding to the SH channel dimension (nm). Default is 0.
    atol : float
        Absolute tolerance for np.allclose comparison.
    rtol : float
        Relative tolerance for np.allclose comparison.

    Returns
    -------
    ok : bool
        True if the spectrum satisfies the complex SH reality condition, False otherwise.
    """
    s = np.asarray(s)
    nm = s.shape[sh_axis]
    F = s.shape[freq_axis]

    # --- 1. Utility Setup ---
    # Move axes to a canonical order (nm, freq, ...)
    s_canonical = np.moveaxis(s, sh_axis, 0)
    s_canonical = np.moveaxis(s_canonical, freq_axis, 1)

    # Calculate max degree N from number of SH channels (nm = (N+1)^2)
    N = int(np.sqrt(nm) - 1)

    # Generate the (n, m) list corresponding to the channel indices
    nm_list = []
    for n in range(N + 1):
        for m in range(-n, n + 1):
            nm_list.append((n, m))
    nm_to_index = {nm_list[i]: i for i in range(len(nm_list))}

    # Indices for interior positive frequencies (1 to F/2 - 1)
    k_pos = np.arange(1, F // 2)

    # Flags to track failure reasons
    ok_pairs = True
    ok_dc_nyq = True

    # --- 2. Check SH Symmetry for all m ---
    for idx, (n, m) in enumerate(nm_list):
        # We only need to check channels where m >= 0. The symmetry definition
        # (A_{n,-m} defined by A_{n,m}) automatically covers m < 0.
        if m < 0:
            continue

        # Extract positive frequencies (k=1 to F/2 - 1) for A_{n,m}
        A_pos = s_canonical[idx, k_pos, ...]

        if m == 0:
            # --- m=0 Case: Standard Hermitian Symmetry A_{n,0}[F-k] = A_{n,0}[k]* ---
            # Extract corresponding negative frequencies (F-k) for A_{n,0}
            A_neg = s_canonical[idx, F - k_pos, ...]
            A_pos_conj = A_pos.conj()

            # Check frequency pairs
            pair_ok = np.allclose(A_neg, A_pos_conj, atol=atol, rtol=rtol)
            if not pair_ok:
                print(f"SH Symmetry Failed: Channel (n={n}, m={m}) frequency pairs do not match standard Hermitian.")
                ok_pairs = False

            # Check DC (k=0) must be real
            dc = s_canonical[idx, 0, ...]
            dc_ok = np.allclose(dc.imag, 0, atol=atol)
            if not dc_ok:
                print(f"SH Symmetry Failed: Channel (n={n}, m={m}) DC component (k=0) is not real.")
                ok_dc_nyq = False

            # Check Nyquist (k=F/2) must be real (only if F is even)
            if F % 2 == 0:
                nyq = s_canonical[idx, F // 2, ...]
                nyq_ok = np.allclose(nyq.imag, 0, atol=atol)
                if not nyq_ok:
                    print(f"SH Symmetry Failed: Channel (n={n}, m={m}) Nyquist component (k=F/2) is not real.")
                    ok_dc_nyq = False

        else:
            # --- m > 0 Case: Complex SH Symmetry A_{n,-m}[F-k] = (-1)^m * A_{n,m}[k]* ---

            idx_minus = nm_to_index[(n, -m)]

            # Extract negative frequencies (F-k) for the mirror channel A_{n,-m}
            A_neg_mirror = s_canonical[idx_minus, F - k_pos, ...]

            # Calculate the expected value: (-1)^m * A_{n,m}[k]*
            expected_val = ((-1) ** m) * A_pos.conj()

            # Check frequency pairs
            pair_ok = np.allclose(A_neg_mirror, expected_val, atol=atol, rtol=rtol)
            if not pair_ok:
                print(
                    f"SH Symmetry Failed: Channel (n={n}, m={m}) and its mirror (-m={-m}) fail the complex SH symmetry.")
                ok_pairs = False

    # ---- check f=0 validity
    ok_n0 = True
    if not np.allclose(s_canonical[0, 0, ...].imag, 0, atol=atol, rtol=rtol):
        print("SH Symmetry Failed: Channel (n=0, m=0) DC component (k=0) imaginary part is not zero.")
        ok_n0 = False
    if ok_n0 and not np.allclose(s_canonical[1:, 0, ...], 0, atol=atol, rtol=rtol):
        print("SH Symmetry Failed: Channel (n!=0, ) (k=0) are not zero.")
        ok_n0 = False


    # --- 3. Final Result ---
    ok = ok_pairs and ok_dc_nyq and ok_n0

    return ok