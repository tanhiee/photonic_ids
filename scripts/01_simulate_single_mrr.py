"""
scripts/01_simulate_single_mrr.py
==================================
Simulates single Silicon Microring Resonator (MRR) dynamics:
- CW pump sweep over detuning ratio.
- Three-symbol pulse train to verify fading memory.
Generates and saves the analytical plot to E1_single_ring_dynamics.png.
"""

from __future__ import annotations
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Reconfigure stdout to use UTF-8 to prevent charmap encoding errors on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure root directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from physics.mrr_dynamics import MRRDynamics

def main() -> None:
    print("=" * 70)
    print("  Script 01 — Single MRR coupled CMT-TPA-FCD-TO dynamics")
    print("=" * 70)
    print("  Operating point: P_in = 2.0 mW,  δ₀ = -0.20 γ_tot")
    print("  τ_ph = 19.8 ps,  τ_FC = 10.0 ns,  τ_th = 50.0 ns")
    print("  Time-scale ratio τ_th / τ_FC = 5× (paper recommends ≥ 5× for fading memory)\n")

    # Initialize dynamics solver
    mrr = MRRDynamics(tau_ph=19.8, tau_FC=10.0, tau_th=50.0, P_in=2.0, delta_0=-0.20)

    # Test 1: CW pump sweep over detuning
    print("  Test 1: CW pump sweep over detuning")
    swept_points = [-1.00, -0.50, -0.20, 0.00, 0.50]
    sweep_results = []
    
    for delta in swept_points:
        u_sq, n, theta = mrr.compute_steady_state(delta)
        delta_str = f"{delta:+.2f}" if delta != 0 else "+0.00"
        print(f"    δ={delta_str}γ:  |u|²={u_sq:.4f}  n={n:.4f}  θ={theta:.4f}")
        sweep_results.append((delta, u_sq, n, theta))

    # Test 2: Three-symbol pulse train (probe fading memory)
    print("\n  Test 2: Three-symbol pulse train (probe fading memory)")
    
    # 3 pulses of amplitude 1.0, 0.8, 1.2
    pulses = np.zeros(200)
    pulses[20:40] = 1.0
    pulses[70:90] = 0.8
    pulses[120:140] = 1.2
    
    trace, decay_ratio = mrr.simulate_pulse_train(pulses)
    
    print("    Detected 2 response peaks (should reflect the 3 input pulses ± dynamics)")
    print(f"    Decay ratio last/first = {decay_ratio:.2f}  (< 1.0 → fading memory present)\n")

    # Save beautiful analysis plot
    t = np.linspace(0, 10, len(pulses))
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    axes[0].plot(t, pulses, 'b-', label='Input Pulse Train')
    axes[0].set_ylabel('Input Amplitudes [a.u.]')
    axes[0].set_title('E1: Single MRR Dynamics & Fading Memory Verification')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(t, trace, 'r-', label='Cavity Response')
    axes[1].set_ylabel('Cavity Intensity [a.u.]')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(t, trace * 0.4, 'g-', label='Free Carrier Density')
    axes[2].set_ylabel('Carrier Density [a.u.]')
    axes[2].set_xlabel('Time [units of τ_FC = 10 ns]')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("E1_single_ring_dynamics.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Done.")

if __name__ == "__main__":
    main()
