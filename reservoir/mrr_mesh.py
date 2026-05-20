"""
reservoir/mrr_mesh.py
=====================
8x8 MRR Mesh Photonic Reservoir — main forward-pass class.

Integrates physics/ layer with reservoir/ weight matrices to provide
a clean batch forward-pass interface.
"""
from __future__ import annotations
import logging
import numpy as np
from typing import Optional

from physics.mrr_dynamics import MRRDynamicsIntegrator, IntegrationParams
from physics.constants import DerivedOpticalConstants, SiliconConstants, DERIVED, SI
from reservoir.projection import build_input_mask
from reservoir.recurrent_matrix import build_recurrent_matrix
from config import ReservoirConfig

log = logging.getLogger(__name__)


class PhotonicMesh:
    """
    All-optical 8x8 Microring Resonator mesh reservoir.

    Bundles:
        - Input mask W_in [64, 16]
        - Recurrent matrix W_res [64, 64]
        - Per-node static detuning array [64]
        - MRRDynamicsIntegrator (ODE integration engine)

    Parameters
    ----------
    cfg : ReservoirConfig
        Reservoir topology and operational parameters.
    dc : DerivedOpticalConstants
        Pre-computed optical constants.
    mat : SiliconConstants
        Silicon material constants.
    """

    def __init__(
        self,
        cfg: ReservoirConfig = None,
        dc:  DerivedOpticalConstants = DERIVED,
        mat: SiliconConstants = SI,
    ) -> None:
        from config import ReservoirConfig as RC
        self.cfg = cfg or RC()
        self.dc  = dc
        self.mat = mat

        n_nodes  = self.cfg.n_nodes
        n_ch     = self.cfg.n_input_channels
        seed     = self.cfg.random_seed

        self.W_in = build_input_mask(
            n_nodes=n_nodes, n_channels=n_ch,
            connectivity=self.cfg.mask_connectivity,
            input_scaling=self.cfg.input_scaling,
            random_seed=seed,
        )
        self.W_res = build_recurrent_matrix(
            n_nodes=n_nodes,
            spectral_radius=self.cfg.spectral_radius,
            connectivity=self.cfg.mask_connectivity,
            random_seed=seed,
        )

        # Per-node static detuning: small random offsets for diversity
        rng = np.random.default_rng(seed + 1)
        self.delta_omega_static = rng.normal(
            0.0, dc.kappa_total * 0.1, size=n_nodes
        )

        # Photon scale from input power
        P_W = 1e-3 * 10.0 ** ((self.cfg.input_power_dBm - 30.0) / 10.0)
        self.photon_scale = P_W * dc.T_rt / dc.photon_energy

        params = IntegrationParams(
            n_steps=self.cfg.n_steps,
            scheme=self.cfg.scheme,
        )
        self.integrator = MRRDynamicsIntegrator(dc=dc, mat=mat, params=params)

        log.info(
            'PhotonicMesh: %dx%d nodes, W_in=%s, W_res=%s, rho(W_res)=%.3f',
            self.cfg.grid_rows, self.cfg.grid_cols,
            self.W_in.shape, self.W_res.shape,
            self.cfg.spectral_radius,
        )

    def forward(self, x_optical: np.ndarray) -> np.ndarray:
        """
        Map MZM optical amplitudes to 64-D PD intensity state.

        Parameters
        ----------
        x_optical : np.ndarray  shape [N, 16]

        Returns
        -------
        np.ndarray  shape [N, 64]  normalised PD intensities in [0,1].
        """
        return self.integrator.forward(
            x_optical           = x_optical,
            W_in                = self.W_in,
            W_res               = self.W_res,
            delta_omega_static  = self.delta_omega_static,
            photon_scale        = self.photon_scale,
        )
