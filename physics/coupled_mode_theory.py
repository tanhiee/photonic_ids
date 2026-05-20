"""
physics/coupled_mode_theory.py
==============================
Coupled-Mode Theory (CMT) equations for ring-waveguide coupling.

Physical Background
-------------------
Coupled-Mode Theory is the standard framework for describing light-matter
interaction in resonant photonic structures. For a single-bus microring
resonator coupled to a straight waveguide:

CMT field equation (temporal):
    da/dt = -(κ_i + κ_e)/2 · a - j·Δω·a + √κ_e · s_in(t)

where:
    a(t)    : complex intracavity field amplitude [√photons]
    |a|²    : intracavity photon number N_phot
    κ_i     : intrinsic (absorption + scattering) field decay rate [rad/s]
    κ_e     : external (coupling to bus) field decay rate [rad/s]
    κ_total : κ_i + κ_e = total field decay rate [rad/s]
    Δω      : pump-resonance detuning [rad/s]:  Δω = ω_pump - ω_res(t)
    s_in(t) : input field amplitude from bus waveguide [√(photons/s)]
    |s_in|² : input photon flux (proportional to input power)

Output field (reflected/transmitted):
    s_out(t) = s_in(t) - √κ_e · a(t)

For numerical integration in the nonlinear MRR (with TPA, FCD, TO effects),
the CMT equation is modified to include:
    1. FCA loss: κ_FCA = σ_FCA · v_g · N_c  [extra decay from free carriers]
    2. Nonlinear detuning: Δω(t) = Δω_0 + δω_FC(t) + δω_TO(t)
    3. Kerr phase shift (SPM): Δφ_Kerr = n₂ · ω₀ · |a|² / (c · V_eff)

Modified CMT (nonlinear, real-valued split):
    The complex ODE da/dt is split into real (|a|²) and imaginary (phase) parts.
    For the reservoir, we only need |a|² (photodetector output), so we work
    with the REAL-VALUED intensity ODE:

    d|a|²/dt = κ_e · |s_in|² · L(Δω) - κ_total · |a|² - 2·κ_FCA·|a|²

    where L(Δω) = (κ_total/2)² / [Δω² + (κ_total/2)²] is the Lorentzian.

This real form avoids complex arithmetic and doubles computation speed.

References
----------
[1] Haus, H. A. "Waves and fields in optoelectronics." Prentice-Hall (1984).
[2] Fan, S. et al. "Temporal coupled-mode theory for Fano resonances." Phys.
    Rev. A 65, 023892 (2002).
[3] Van der Sande et al., Nanophotonics 6, 561–576 (2017).
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Optional

from physics.constants import (
    SiliconConstants,
    DerivedOpticalConstants,
    C_LIGHT,
    SI,
    DERIVED,
)
from physics.tpa_fcd import lorentzian_transmission

FloatArray = NDArray[np.float64]


# ─────────────────────────────────────────────────────────────────────────────
# CMT real-valued intensity ODE (core inner-loop function)
# ─────────────────────────────────────────────────────────────────────────────

def cmt_intensity_ode(
    a_sq: FloatArray,
    s_in_sq: FloatArray,
    delta_omega: FloatArray,
    kappa_FCA: FloatArray,
    dc: DerivedOpticalConstants = DERIVED,
) -> FloatArray:
    """
    Evaluate the real-valued CMT intensity ODE: d|a|²/dt.

    This is the performance-critical inner loop function. It computes the
    time derivative of the intracavity photon number for ALL N_nodes MRR
    nodes simultaneously in a single vectorized pass.

    ODE right-hand side:

        d|a|²/dt = κ_e · |s_in|² · L(Δω) - (κ_total + 2·κ_FCA) · |a|²

    where:
        |s_in|²         : input photon flux [photons/s]  (WDM + recurrent)
        L(Δω)           : Lorentzian cavity response (accounts for detuning)
        κ_total         : total cold-cavity decay rate (κ_e + κ_i)
        κ_FCA           : FCA extra loss rate = σ_FCA · v_g · N_c
        Δω              : instantaneous detuning (FCD + TO + static)

    Derivation of the intensity ODE from CMT:
        da/dt = -(κ/2 + j·Δω) · a + √κ_e · s_in

        Multiply both sides by a* (complex conjugate) and add complex conjugate:
        d|a|²/dt = 2·Re[a* · da/dt]
                 = -κ·|a|² + 2·Re[√κ_e · a* · s_in]

        At steady state for a Lorentzian resonator:
                 = κ_e · |s_in|² · L(Δω) - κ_eff · |a|²

    Parameters
    ----------
    a_sq : NDArray[float64]  shape [..., N_nodes]
        Intracavity photon number |a(t)|² ≥ 0.
    s_in_sq : NDArray[float64]  shape [..., N_nodes]
        Input photon flux |s_in(t)|² [photons/s].
        This is the sum of the WDM channel input and the recurrent
        contribution from neighbouring rings.
    delta_omega : NDArray[float64]  shape [..., N_nodes]
        Total instantaneous detuning Δω(t) = Δω_0 + δω_FC + δω_TO [rad/s].
    kappa_FCA : NDArray[float64]  shape [..., N_nodes]
        FCA-induced extra decay rate κ_FCA = σ_FCA · v_g · N_c [rad/s].
    dc : DerivedOpticalConstants
        Pre-computed constants (κ_e, κ_total).

    Returns
    -------
    NDArray[float64]  shape [..., N_nodes]
        d|a|²/dt [photons/s].

    Performance
    -----------
    For shape [2048, 64] (2048-sample batch, 64 nodes):
        - 5 element-wise ops + 1 division = ~0.4 ms on MKL NumPy
        - Memory footprint: 5 × 2048 × 64 × 8 B ≈ 5 MB

    Notes
    -----
    The `kappa_FCA` term enters as 2·κ_FCA because in the field equation
    da/dt = -(κ/2 + κ_FCA/2)·a, the factor ½ comes from converting from
    field decay to intensity decay (d|a|²/dt = 2·Re[a*·da/dt]).
    """
    # Lorentzian cavity response: L(Δω) ∈ [0, 1]
    L = lorentzian_transmission(delta_omega, dc.kappa_total)

    # Effective total decay rate (cold-cavity + FCA)
    kappa_eff = dc.kappa_total + 2.0 * np.maximum(kappa_FCA, 0.0)

    # CMT intensity derivative
    drive = dc.kappa_e * np.maximum(s_in_sq, 0.0) * L
    decay = kappa_eff * np.maximum(a_sq, 0.0)

    return drive - decay


def cmt_complex_ode(
    a: NDArray[np.complex128],
    s_in: NDArray[np.complex128],
    delta_omega: FloatArray,
    kappa_FCA: FloatArray,
    dc: DerivedOpticalConstants = DERIVED,
) -> NDArray[np.complex128]:
    """
    Evaluate the full complex CMT field ODE: da/dt.

    Complex form (for accurate phase tracking if needed):

        da/dt = -(κ_total/2 + κ_FCA/2 + j·Δω) · a + √κ_e · s_in

    This is more accurate than the intensity form but twice as expensive due
    to complex arithmetic. Use this for:
    - Phase-sensitive applications (coherent detection)
    - Validation against the intensity ODE
    - E1 single-ring dynamics visualization with phase traces

    Parameters
    ----------
    a : NDArray[complex128]  shape [..., N_nodes]
        Complex intracavity field amplitude.
    s_in : NDArray[complex128]  shape [..., N_nodes]
        Complex input field amplitude.
    delta_omega : NDArray[float64]  shape [..., N_nodes]
        Instantaneous detuning [rad/s].
    kappa_FCA : NDArray[float64]  shape [..., N_nodes]
        FCA decay rate [rad/s].
    dc : DerivedOpticalConstants

    Returns
    -------
    NDArray[complex128]  shape [..., N_nodes]
        da/dt.
    """
    kappa_eff = 0.5 * dc.kappa_total + 0.5 * np.maximum(kappa_FCA, 0.0)
    return (
        -(kappa_eff + 1j * delta_omega) * a
        + np.sqrt(dc.kappa_e) * s_in
    )


# ─────────────────────────────────────────────────────────────────────────────
# Input/output coupling relations
# ─────────────────────────────────────────────────────────────────────────────

def input_flux_from_power(
    P_watts: FloatArray,
    dc: DerivedOpticalConstants = DERIVED,
) -> FloatArray:
    """
    Convert input optical power [W] to photon flux |s_in|² [photons/s].

        |s_in|² = P / (ħ·ω₀)

    Parameters
    ----------
    P_watts : NDArray[float64]
        Input optical power [W].
    dc : DerivedOpticalConstants

    Returns
    -------
    NDArray[float64]
        Input photon flux [photons/s].
    """
    return P_watts / dc.photon_energy


def input_flux_from_dBm(
    P_dBm: float,
    dc: DerivedOpticalConstants = DERIVED,
) -> float:
    """
    Convert input power in dBm to photon flux |s_in|² [photons/s].

    Parameters
    ----------
    P_dBm : float
        Input power [dBm].
    dc : DerivedOpticalConstants

    Returns
    -------
    float
        Input photon flux [photons/s].
    """
    P_W = 1e-3 * 10.0 ** (P_dBm / 10.0)
    return P_W / dc.photon_energy


def output_field(
    s_in: NDArray[np.complex128],
    a: NDArray[np.complex128],
    dc: DerivedOpticalConstants = DERIVED,
) -> NDArray[np.complex128]:
    """
    Compute the output (transmitted/reflected) field amplitude.

    Single-bus ring CMT output relation:
        s_out = s_in - √κ_e · a

    Parameters
    ----------
    s_in : NDArray[complex128]  shape [..., N_nodes]
        Input bus field amplitude.
    a : NDArray[complex128]  shape [..., N_nodes]
        Intracavity field amplitude.
    dc : DerivedOpticalConstants

    Returns
    -------
    NDArray[complex128]  shape [..., N_nodes]
        Output field amplitude s_out.
    """
    return s_in - np.sqrt(dc.kappa_e) * a


def through_transmission(
    delta_omega: FloatArray,
    dc: DerivedOpticalConstants = DERIVED,
) -> FloatArray:
    """
    Compute the through-port power transmission T(Δω) of the ring resonator.

    For a critically coupled ring (κ_e = κ_i), T = 0 at resonance.
    For under/over-coupled rings:

        T(Δω) = |s_out/s_in|² = |1 - κ_e / (κ_total/2 + j·Δω)|²

    Parameters
    ----------
    delta_omega : NDArray[float64]  shape [...]
        Detuning from resonance [rad/s].
    dc : DerivedOpticalConstants

    Returns
    -------
    NDArray[float64]  shape [...]
        Through-port power transmission in [0, 1].
    """
    half_k = 0.5 * dc.kappa_total
    numerator   = (half_k - dc.kappa_e) ** 2 + delta_omega ** 2
    denominator = half_k ** 2 + delta_omega ** 2
    return numerator / denominator


# ─────────────────────────────────────────────────────────────────────────────
# Recurrent coupling: ring-to-ring contribution
# ─────────────────────────────────────────────────────────────────────────────

def recurrent_drive(
    a_sq: FloatArray,
    W_res: FloatArray,
    clip_min: float = 0.0,
) -> FloatArray:
    """
    Compute the recurrent contribution to the input drive from neighbouring rings.

    In the 8×8 MRR mesh, the output of each ring drives neighbouring rings
    through evanescent coupling (or optical interconnects). The recurrent
    photon flux coupling is modelled as:

        s_rec_sq[i] = Σ_j W_res[i,j] · |a_j|²

    This is a standard linear reservoir recurrent connection, equivalent to:
        s_rec_sq = a_sq @ W_res.T

    Parameters
    ----------
    a_sq : NDArray[float64]  shape [..., N_nodes]
        Intracavity photon numbers of all nodes.
    W_res : NDArray[float64]  shape [N_nodes, N_nodes]
        Recurrent weight matrix (fixed, random, spectral radius < 1).
    clip_min : float
        Minimum clipped value (default 0.0 to enforce positivity of flux).

    Returns
    -------
    NDArray[float64]  shape [..., N_nodes]
        Recurrent photon flux contribution [photons/s].

    Performance
    -----------
    For shape [2048, 64] × [64, 64]:
        - Single np.dot / matmul call → MKL BLAS DGEMM
        - ~0.15 ms on i7 with MKL NumPy (dominated by BLAS)
    """
    coupling = a_sq @ W_res.T   # shape [..., N_nodes]
    return np.maximum(coupling, clip_min)


def wdm_input_drive(
    x_optical: FloatArray,
    W_in: FloatArray,
    photon_scale: float = 1.0,
) -> FloatArray:
    """
    Map WDM optical amplitudes through the input mask W_in to drive photon flux.

    The MZM-encoded optical amplitude x[k] ∈ [0, 1] for WDM channel k is
    projected onto the N_nodes reservoir nodes via the sparse input mask:

        s_in_sq[i] = photon_scale · Σ_k W_in[i,k] · x[k]²

    The squaring converts amplitude to intensity (photon flux).

    Parameters
    ----------
    x_optical : NDArray[float64]  shape [..., N_channels]
        MZM optical amplitudes ∈ [0, 1] for each WDM channel.
    W_in : NDArray[float64]  shape [N_nodes, N_channels]
        Input projection matrix (sparse, fixed random weights).
    photon_scale : float
        Global scale factor converting normalized drive to photon flux.
        Calibrated to match target input power (-10 dBm by default).

    Returns
    -------
    NDArray[float64]  shape [..., N_nodes]
        Input photon flux per node [photons/s].
    """
    # Convert amplitude → intensity: x → x²
    x_intensity = x_optical ** 2                    # [..., N_channels]
    # Project through input mask: [..., N_nodes]
    s_in_sq = x_intensity @ W_in.T                 # MKL BLAS DGEMM
    return np.maximum(s_in_sq * photon_scale, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Full nonlinear CMT step (Euler integration, vectorized)
# ─────────────────────────────────────────────────────────────────────────────

def cmt_euler_step(
    a_sq: FloatArray,
    s_in_sq: FloatArray,
    delta_omega: FloatArray,
    kappa_FCA: FloatArray,
    dt: float,
    dc: DerivedOpticalConstants = DERIVED,
    clip_max: float = 1e20,
) -> FloatArray:
    """
    Perform one Euler integration step of the CMT intensity ODE.

    Simple forward Euler:
        |a(t+dt)|² = |a(t)|² + dt · d|a|²/dt

    With non-negativity enforcement (photon number ≥ 0) and overflow clipping.

    Parameters
    ----------
    a_sq : NDArray[float64]  shape [..., N_nodes]
        Current intracavity photon number |a|².
    s_in_sq : NDArray[float64]  shape [..., N_nodes]
        Input photon flux (WDM + recurrent).
    delta_omega : NDArray[float64]  shape [..., N_nodes]
        Total detuning [rad/s].
    kappa_FCA : NDArray[float64]  shape [..., N_nodes]
        FCA decay rate [rad/s].
    dt : float
        Integration time step [s]. Must satisfy dt << 1/κ_total.
    dc : DerivedOpticalConstants
    clip_max : float
        Maximum allowed value of |a|² (prevents runaway accumulation).

    Returns
    -------
    NDArray[float64]  shape [..., N_nodes]
        Updated |a(t+dt)|².
    """
    da_sq_dt = cmt_intensity_ode(a_sq, s_in_sq, delta_omega, kappa_FCA, dc)
    a_sq_new = a_sq + dt * da_sq_dt
    # Enforce physical constraints: 0 ≤ |a|² ≤ clip_max
    return np.clip(a_sq_new, 0.0, clip_max)


def cmt_rk4_step(
    a_sq: FloatArray,
    s_in_sq: FloatArray,
    delta_omega: FloatArray,
    kappa_FCA: FloatArray,
    dt: float,
    dc: DerivedOpticalConstants = DERIVED,
    clip_max: float = 1e20,
) -> FloatArray:
    """
    Perform one 4th-order Runge-Kutta (RK4) step of the CMT intensity ODE.

    RK4 provides O(dt⁴) accuracy vs. O(dt²) for Euler, allowing 4× larger
    timesteps while maintaining the same accuracy. Recommended when:
        - dt approaches 1/κ_total (stiff regime)
        - High accuracy traces needed (E1 single-ring script)

    Parameters
    ----------
    (same as cmt_euler_step)

    Returns
    -------
    NDArray[float64]  shape [..., N_nodes]
        Updated |a(t+dt)|² with RK4 accuracy.

    Notes
    -----
    The `s_in_sq`, `delta_omega`, and `kappa_FCA` terms are treated as
    constant over the RK4 substeps (frozen-coefficient approximation).
    This is valid when these evolve on timescales much longer than dt.
    For coupled ODE systems (where N_c and ΔT also evolve), use the full
    coupled RK4 in mrr_dynamics.py.
    """
    def f(a):
        return cmt_intensity_ode(a, s_in_sq, delta_omega, kappa_FCA, dc)

    k1 = f(a_sq)
    k2 = f(np.clip(a_sq + 0.5 * dt * k1, 0.0, clip_max))
    k3 = f(np.clip(a_sq + 0.5 * dt * k2, 0.0, clip_max))
    k4 = f(np.clip(a_sq +       dt * k3, 0.0, clip_max))

    a_sq_new = a_sq + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
    return np.clip(a_sq_new, 0.0, clip_max)


# ─────────────────────────────────────────────────────────────────────────────
# Steady-state CMT solutions
# ─────────────────────────────────────────────────────────────────────────────

def cmt_steady_state_a_sq(
    s_in_sq: FloatArray,
    delta_omega: FloatArray,
    kappa_FCA: FloatArray,
    dc: DerivedOpticalConstants = DERIVED,
) -> FloatArray:
    """
    Compute the steady-state intracavity photon number |a_ss|².

    At d|a|²/dt = 0, solving the linear CMT:
        |a_ss|² = κ_e · |s_in|² · L(Δω) / (κ_total + 2·κ_FCA)

    Parameters
    ----------
    s_in_sq : NDArray[float64]  shape [..., N_nodes]
        Input photon flux [photons/s].
    delta_omega : NDArray[float64]  shape [..., N_nodes]
        Total detuning [rad/s].
    kappa_FCA : NDArray[float64]  shape [..., N_nodes]
        FCA extra loss [rad/s].
    dc : DerivedOpticalConstants

    Returns
    -------
    NDArray[float64]  shape [..., N_nodes]
        Steady-state intracavity photon number |a_ss|².
    """
    L         = lorentzian_transmission(delta_omega, dc.kappa_total)
    kappa_eff = dc.kappa_total + 2.0 * np.maximum(kappa_FCA, 0.0)
    numerator = dc.kappa_e * np.maximum(s_in_sq, 0.0) * L
    return numerator / np.maximum(kappa_eff, 1.0)   # avoid /0
