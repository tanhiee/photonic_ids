"""
reservoir/projection.py
=======================
Static random input mask W_in for projecting WDM channel inputs
onto the 64-node MRR mesh reservoir.

W_in shape: [N_nodes, N_channels] = [64, 16]
Each column (WDM channel) is connected to a random subset of nodes.
"""
from __future__ import annotations
import numpy as np
from typing import Optional


def build_input_mask(
    n_nodes:      int,
    n_channels:   int,
    connectivity: float = 0.8,
    input_scaling: float = 0.5,
    random_seed:  int   = 42,
    distribution: str   = 'normal',
) -> np.ndarray:
    """
    Build the static input projection matrix W_in.

    W_in[i, k] is the coupling weight from WDM channel k to MRR node i.
    Entries are drawn from N(0,1) and then sparsified by a binary mask.

    Parameters
    ----------
    n_nodes : int
        Number of reservoir nodes (64).
    n_channels : int
        Number of WDM input channels (16).
    connectivity : float
        Fraction of non-zero connections (0.8 = 80% dense).
    input_scaling : float
        Global gain applied to all weights.
    random_seed : int
        NumPy random seed.
    distribution : str
        'normal' (N(0,1)) or 'uniform' (U(-1,1)).

    Returns
    -------
    np.ndarray  shape [n_nodes, n_channels]  float64
        Sparse random input projection matrix.

    Notes
    -----
    The matrix is fixed (not trained). Its statistical properties
    (distribution, connectivity, scaling) control how the input
    signal is distributed across the reservoir nodes.
    """
    rng = np.random.default_rng(random_seed)

    if distribution == 'uniform':
        W = rng.uniform(-1.0, 1.0, size=(n_nodes, n_channels))
    else:  # normal
        W = rng.normal(0.0, 1.0, size=(n_nodes, n_channels))

    # Sparsity mask
    mask = rng.random((n_nodes, n_channels)) < connectivity
    W   *= mask.astype(np.float64)
    W   *= input_scaling

    return W.astype(np.float64)
