"""
ambidrop/asm.py — Unified ASM (Ambisonics Signal Matching) encoding.

Replaces the duplicated inline implementations that previously existed in:
  - datagenerator/generate_inference_ds.py  (Type C dataset generation)
  - FT_JNF/test_real.py                     (real-world Aria inference)
  - ConvTasNet/datasets.py                   (on-the-fly time-domain encoding)

Public interface:
    encode_ambisonics(mic_signals, V, ...)   — unified entry point; returns (encoded, cnm)
    compute_asm_coefficients(V, ...)          — compute filter coefficients only
    apply_asm_filters(mic_signals, cnm, ...)  — apply precomputed filters

Usage:
    from ambidrop.asm import encode_ambisonics

    # Compute and apply:
    encoded, cnm = encode_ambisonics(mic_signals, V, sh_order=2, th=th, ph=ph)

    # Reuse precomputed coefficients (e.g. test_real.py precomputed mode):
    encoded, cnm = encode_ambisonics(mic_signals, V, cnm=precomputed_cnm)

    # Conv-TasNet variant (real-valued Y matrix):
    encoded, cnm = encode_ambisonics(mic_signals, V, sh_type="real", th=th, ph=ph)
"""

import numpy as np
try:
    from scipy.special import sph_harm
except ImportError:
    from scipy.special import sph_harm_y
    def sph_harm(m, n, phi, theta):   # noqa: E302
        return sph_harm_y(n, m, theta, phi)

from ambidrop.signal_utils import complex_acn_to_real_acn


# ── Spherical harmonics matrix ───────────────────────────────────────────────

def _compute_sh_matrix(sh_order: int, th: np.ndarray, ph: np.ndarray) -> np.ndarray:
    """Complex SH matrix Y, shape ((N+1)^2, Q). ACN order, sph_harm convention."""
    Q = len(th)
    H = (sh_order + 1) ** 2
    Y = np.zeros((H, Q), dtype=complex)
    idx = 0
    for n in range(sh_order + 1):
        for m in range(-n, n + 1):
            Y[idx, :] = sph_harm(m, n, ph, th)   # scipy: sph_harm(m, n, phi, theta)
            idx += 1
    return Y


# ── Core functions ───────────────────────────────────────────────────────────

def compute_asm_coefficients(
    V: np.ndarray,
    sh_order: int,
    th: np.ndarray,
    ph: np.ndarray,
    method: str = "tikhonov",
    sh_type: str = "complex",
    svd_snr_lin: float = 1000.0,
) -> np.ndarray:
    """
    Compute ASM filter coefficients.

    Parameters
    ----------
    V          : steering matrix (M, F_pos, Q) — complex, positive frequencies
    sh_order   : Ambisonics order (2 → 9 channels)
    th, ph     : source-grid colatitude / azimuth arrays, length Q
    method     : "tikhonov" (default) or "svd"
    sh_type    : "complex" (standard, for FT-JNF) or "real" (real ACN, for Conv-TasNet)
    svd_snr_lin: regularisation SNR for SVD method (default 1000)

    Returns
    -------
    cnm : (H, F_pos, M) complex128  where H = (sh_order+1)^2
    """
    H = (sh_order + 1) ** 2
    Y = _compute_sh_matrix(sh_order, np.asarray(th), np.asarray(ph))  # (H, Q)
    if sh_type == "real":
        Y = complex_acn_to_real_acn(Y, sh_order, sn3d=False)          # (H, Q) real
    elif sh_type != "complex":
        raise ValueError(f"sh_type must be 'complex' or 'real', got {sh_type!r}")

    if method == "tikhonov":
        return _asm_tikhonov(V, Y, H)
    elif method == "svd":
        return _asm_svd(V, Y, H, svd_snr_lin)
    else:
        raise ValueError(f"method must be 'tikhonov' or 'svd', got {method!r}")


def apply_asm_filters(
    mic_signals: np.ndarray,
    cnm: np.ndarray,
    filt_samp: int = 512,
) -> np.ndarray:
    """
    Apply ASM filter coefficients to microphone signals via time-domain convolution.

    Parameters
    ----------
    mic_signals : (M, T) real float
    cnm         : ((N+1)^2, F_pos, M) complex — from compute_asm_coefficients
    filt_samp   : IFFT length for time-domain filters (default 512)

    Returns
    -------
    anmt : ((N+1)^2, T) float32 — time-domain Ambisonics
    """
    M, T = mic_signals.shape
    H = cnm.shape[0]
    anmt = np.zeros((H, T), dtype=np.float32)

    for j in range(H):
        c_f = cnm[j, :, :].T                          # (M, F_pos)
        c_time = np.fft.irfft(c_f, n=filt_samp, axis=1)    # (M, filt_samp)
        c_time_cs = np.roll(c_time, filt_samp // 2, axis=1)
        first_col = c_time_cs[:, [0]]
        tail_reversed = c_time_cs[:, :0:-1]
        c_time_filter = np.concatenate([first_col, tail_reversed], axis=1)

        tmp = np.zeros(T, dtype=np.float64)
        for m in range(M):
            full_conv = np.convolve(
                np.real(mic_signals[m, :]).astype(np.float64),
                c_time_filter[m, :].astype(np.float64),
                mode="full",
            )
            tmp += full_conv[filt_samp // 2 : filt_samp // 2 + T]
        anmt[j, :] = tmp

    return anmt


def encode_ambisonics(
    mic_signals: np.ndarray,
    V: np.ndarray,
    sh_order: int = 2,
    th: np.ndarray = None,
    ph: np.ndarray = None,
    method: str = "tikhonov",
    sh_type: str = "complex",
    cnm: np.ndarray = None,
    filt_samp: int = 512,
    svd_snr_lin: float = 1000.0,
):
    """
    Unified ASM entry point: compute coefficients (or reuse precomputed) and encode.

    Parameters
    ----------
    mic_signals : (M, T) time-domain microphone signals
    V           : (M, F_pos, Q) steering matrix
    sh_order    : Ambisonics order (default 2 → 9 channels)
    th, ph      : source-grid angles (required if cnm is None)
    method      : "tikhonov" or "svd" (used only if cnm is None)
    sh_type     : "complex" (FT-JNF) or "real" (Conv-TasNet, real ACN Y matrix)
    cnm         : precomputed filter coefficients — if provided, skip computation
    filt_samp   : IFFT length for time-domain filters (default 512)
    svd_snr_lin : SVD regularisation SNR (only used when method="svd")

    Returns
    -------
    (encoded, cnm)
      encoded : ((N+1)^2, T) float32 — time-domain Ambisonics
      cnm     : ((N+1)^2, F_pos, M) complex — coefficients (useful for caching)
    """
    if cnm is None:
        if th is None or ph is None:
            raise ValueError("th and ph must be provided when cnm is None")
        cnm = compute_asm_coefficients(V, sh_order, th, ph, method, sh_type, svd_snr_lin)
    encoded = apply_asm_filters(mic_signals, cnm, filt_samp)
    return encoded, cnm


# ── Private implementations ──────────────────────────────────────────────────

def tikhonov(A, b, lam=None, L=None, rcond=None):
    """
    Solve (A^H A + lam^2 L^H L) x = A^H b via Tikhonov regularization.

    If lam is None, it is automatically selected as a fraction of the
    largest singular value of A (heuristic).

    Parameters
    ----------
    A : array_like, shape (M, N)
        System matrix.
    b : array_like, shape (M,) or (M, K)
        Right-hand side.
    lam : float or None
        Regularization parameter (lambda). If None, calculated automatically.
    L : array_like or None, shape (P, N), optional
        Regularization matrix. If None, L = I (standard ridge).
    rcond : float or None, optional
        Cutoff for small singular values, passed to np.linalg.lstsq.

    Returns
    -------
    x : ndarray, shape (N,) or (N, K)
        Regularized solution.
    """
    A = np.asarray(A)
    b = np.asarray(b)
    m, n = A.shape

    # 1. Automatic Lambda Selection (Heuristic)
    # ---------------------------------------------------------
    if lam is None:
        # Calculate largest singular value of A to determine scale
        # We use compute_uv=False because we only need the values (faster)
        s = np.linalg.svd(A, compute_uv=False)
        sigma_max = s[0] if s.size > 0 else 1.0

        # Heuristic: Choose lambda as 1% of the spectral norm (max singular value)
        # This acts as a soft floor for singular values < 0.01 * sigma_max
        # lam = 0.12 * sigma_max
        lam = 0.01 * sigma_max
        # print(f"Auto-selected lambda: {lam:.4e} (12% of max singular value)")

    if lam is None:
        # Calculate largest singular value of A to determine scale
        # We use compute_uv=False because we only need the values (faster)
        s = np.linalg.svd(A, compute_uv=False)
        sigma_max = s[0] if s.size > 0 else 1.0
        sigma_min = s[-1] if s.size > 1 else sigma_max
        # Floor sigma_min to prevent inf condition number when A is rank-deficient
        # (e.g. at DC where the steering matrix collapses to rank-1).
        sigma_min = max(sigma_min, 1e-6 * sigma_max)
        condition_number = sigma_max / sigma_min

        lam = max(1e-7 * condition_number, 1e-12)
        # lam = min(lam, 1e-2 * sigma_max)  # cap: never over-regularize beyond 1% of signal scale

    # 2. Setup Regularization Matrix L
    # ---------------------------------------------------------
    if L is None:
        L = np.eye(n, dtype=A.dtype)
    else:
        L = np.asarray(L)
        if L.shape[1] != n:
            raise ValueError(f"L must have shape (P, {n}), got {L.shape}")

    # 3. Build Augmented System
    #    Minimize ||Ax - b||^2 + ||lam * L x||^2
    #    Equivalent to solving:
    #    [A      ] x = [b]
    #    [lam * L]     [0]
    # ---------------------------------------------------------

    # Note: Changed from np.sqrt(lam) to lam to match docstring formula (lam^2 L^H L)
    lamL = lam * L

    if b.ndim == 1:
        A_aug = np.vstack([A, lamL])
        b_aug = np.concatenate([b, np.zeros(L.shape[0], dtype=b.dtype)])
    else:
        A_aug = np.vstack([A, lamL])
        zeros_block = np.zeros((L.shape[0], b.shape[1]), dtype=b.dtype)
        b_aug = np.vstack([b, zeros_block])

    # 4. Solve Least Squares
    x, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=rcond)
    return x


def _asm_tikhonov(V: np.ndarray, Y: np.ndarray, H: int) -> np.ndarray:
    """Tikhonov-regularised ASM coefficients. V: (M, F, Q), Y: (H, Q)."""
    V_t = V.T   # (Q, F, M)
    F = V_t.shape[1]
    M = V_t.shape[2]
    cnm = np.zeros((H, F, M), dtype=np.complex128)
    for nm in range(H):
        for f in range(F):
            cnm[nm, f] = tikhonov(A=V_t[:, f, :].conj(), b=Y[nm, :])
    return cnm


def _asm_svd(V: np.ndarray, Y: np.ndarray, H: int, snr_lin: float) -> np.ndarray:
    """SVD-based ASM coefficients. V: (M, F, Q), Y: (H, Q)."""
    M, F, Q = V.shape
    if Y.shape == (H, Q):
        Y = Y.T                     # → (Q, H)
    lam = 1.0 / float(snr_lin)
    eps = np.finfo(np.float64).eps
    cnm = np.zeros((H, F, M), dtype=np.complex128)
    I_M = np.eye(M, dtype=np.complex128)
    for nm in range(H):
        Ynm = Y[:, nm]              # (Q,)
        for f in range(F):
            v_k = V[:, f, :]        # (M, Q)
            mat_to_inv = (v_k @ v_k.conj().T) + lam * I_M
            maxdim = max(mat_to_inv.shape)
            tol_inv = 1.0 + maxdim * eps * np.linalg.norm(mat_to_inv)
            U, s, Vh = np.linalg.svd(mat_to_inv, full_matrices=False)
            s_inv = np.zeros_like(s)
            keep = s > tol_inv
            s_inv[keep] = 1.0 / s[keep]
            inv_mat = Vh.conj().T @ (s_inv[:, None] * U.conj().T)
            cnm[nm, f, :] = inv_mat @ (v_k @ Ynm)
    return cnm
