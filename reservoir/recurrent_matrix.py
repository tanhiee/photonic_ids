"""
reservoir/recurrent_matrix.py
=============================
Fixed random recurrent weight matrix W_res for the MRR mesh.

W_res shape: [N_nodes, N_nodes] = [64, 64]
Spectral radius is normalised to target_rho < 1 (echo-state property).
"""
from __future__ import annotations
import numpy as np


def build_recurrent_matrix(
    n_nodes:         int,
    spectral_radius: float = 0.9,
    connectivity:    float = 0.8,
    random_seed:     int   = 42,
    distribution:    str   = 'normal',
    no_self_loops:   bool  = True,
) -> np.ndarray:
    """
    Build the fixed random recurrent weight matrix W_res.

    The matrix is scaled so that its spectral radius (largest absolute
    eigenvalue) equals `spectral_radius`. A spectral radius < 1 guarantees
    the echo-state property: the reservoir state is uniquely determined
    by the input history, with fading memory.

    Parameters
    ----------
    n_nodes : int
        Reservoir size (64 for 8x8 mesh).
    spectral_radius : float
        Target spectral radius. Must be < 1 for ESP.
    connectivity : float
        Fraction of non-zero connections.
    random_seed : int
    distribution : str  'normal' or 'uniform'
    no_self_loops : bool
        If True, zero out the diagonal (no self-coupling).

    Returns
    -------
    np.ndarray  shape [n_nodes, n_nodes]  float64
    """
    rng = np.random.default_rng(random_seed)

    if distribution == 'uniform':
        W = rng.uniform(-1.0, 1.0, size=(n_nodes, n_nodes))
    else:
        W = rng.normal(0.0, 1.0, size=(n_nodes, n_nodes))

    # Sparsity mask
    mask = rng.random((n_nodes, n_nodes)) < connectivity
    if no_self_loops:
        np.fill_diagonal(mask, False)
    W *= mask.astype(np.float64)

    # Normalise to target spectral radius
    eigenvalues = np.linalg.eigvals(W)
    rho_current = np.max(np.abs(eigenvalues))
    if rho_current > 1e-10:
        W *= spectral_radius / rho_current

    return W.astype(np.float64)


def verify_spectral_radius(W: np.ndarray) -> float:
    """Compute and return the actual spectral radius of W."""
    return float(np.max(np.abs(np.linalg.eigvals(W))))
