"""
E1_single_ring.py — Milestone 1: Single MRR Dynamics Visualization
===================================================================
Simulates and plots the time-domain dynamics of a single MRR node:
  - Intracavity photon number |a|^2(t)
  - Free-carrier density N_c(t)
  - Temperature rise dT(t)
  - Total resonance detuning dw(t)

Usage:  python scripts/E1_single_ring.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from physics import MRRDynamicsIntegrator, IntegrationParams, DERIVED, SI
from reservoir.projection import build_input_mask
from reservoir.recurrent_matrix import build_recurrent_matrix


def main():
    print('=== E1: Single MRR Ring Dynamics ===')

    # Build minimal reservoir (1 channel -> 1 node for single-ring analysis)
    W_in  = build_input_mask(n_nodes=64, n_channels=16, random_seed=42)
    W_res = build_recurrent_matrix(n_nodes=64, spectral_radius=0.9, random_seed=42)

    params = IntegrationParams(n_steps=200, scheme='euler')
    integrator = MRRDynamicsIntegrator(dc=DERIVED, mat=SI, params=params)

    # Input: moderate drive on all WDM channels
    x_optical = np.ones(16) * 0.6   # 60% MZM amplitude

    P_W = 1e-3 * 10 ** ((-10 - 30) / 10)  # -10 dBm
    photon_scale = P_W * DERIVED.T_rt / DERIVED.photon_energy

    trace = integrator.get_state_trace(
        x_optical_single   = x_optical,
        W_in               = W_in,
        W_res              = W_res,
        photon_scale       = photon_scale,
        n_trace_steps      = 200,
    )

    t   = trace['t_norm']           # time in units of tau_FC
    asq = trace['a_sq'][:, 0]      # node 0
    Nc  = trace['N_c'][:, 0]
    dT  = trace['delta_T'][:, 0]
    dw  = trace['delta_omega_total'][:, 0]

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(t, asq, 'b-', linewidth=1.5)
    axes[0].set_ylabel('|a|² [photons]')
    axes[0].set_title('E1: Single MRR Dynamics — Node 0')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, Nc, 'r-', linewidth=1.5)
    axes[1].set_ylabel('N_c [m⁻³]')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, dT, 'g-', linewidth=1.5)
    axes[2].set_ylabel('ΔT [K]')
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(t, dw / 1e9, 'm-', linewidth=1.5)
    axes[3].set_ylabel('Δω_total [GHz]')
    axes[3].set_xlabel('Time [units of τ_FC = 10 ns]')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('E1_single_ring_dynamics.png', dpi=150, bbox_inches='tight')
    print('Plot saved: E1_single_ring_dynamics.png')
    plt.show()


if __name__ == '__main__':
    main()
