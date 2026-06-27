
import numpy as np

import numpy as np


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

