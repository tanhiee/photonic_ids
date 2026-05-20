"""
physics/constants.py
====================
Silicon-on-Insulator (SOI) Microring Resonator material and optical
constants at λ₀ = 1550 nm (telecom C-band).

All values are in SI units unless explicitly stated otherwise.

References
----------
[1] Borghi, M. et al. "Nonlinear silicon photonics." J. Opt. 19, 093004 (2017).
[2] Soref, R. & Bennett, B. "Electrooptical effects in silicon." IEEE J. Quantum
    Electron. 23, 123–129 (1987).  [Free-Carrier Dispersion model]
[3] Absil, P. P. et al. "Microring resonators for nonlinear optic applications."
    Opt. Lett. 25, 554–556 (2000).
[4] Van Vaerenbergh, T. et al. "Excitability in optically injected microdisk
    lasers with phase-controlled injection." Opt. Express 20, 20292 (2012).
[5] Vandoorne, K. et al. "Experimental demonstration of reservoir computing on
    a silicon photonics chip." Nat. Commun. 5, 3541 (2014).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Final


# ─────────────────────────────────────────────────────────────────────────────
# Fundamental physical constants (CODATA 2018)
# ─────────────────────────────────────────────────────────────────────────────

#: Speed of light in vacuum [m/s]
C_LIGHT: Final[float] = 2.99792458e8

#: Reduced Planck constant [J·s]
HBAR: Final[float] = 1.054571817e-34

#: Planck constant [J·s]
H_PLANCK: Final[float] = 6.62607015e-34

#: Elementary charge [C]
Q_ELECTRON: Final[float] = 1.602176634e-19

#: Boltzmann constant [J/K]
K_BOLTZMANN: Final[float] = 1.380649e-23

#: Permittivity of free space [F/m]
EPSILON_0: Final[float] = 8.8541878128e-12


# ─────────────────────────────────────────────────────────────────────────────
# Operating wavelength and derived optical frequency
# ─────────────────────────────────────────────────────────────────────────────

#: Operating wavelength [m] — ITU-T C-band centre
LAMBDA_0: Final[float] = 1.55e-6

#: Angular carrier frequency [rad/s]:  ω₀ = 2π c / λ₀
OMEGA_0: Final[float] = 2.0 * np.pi * C_LIGHT / LAMBDA_0

#: Linear carrier frequency [Hz]
NU_0: Final[float] = C_LIGHT / LAMBDA_0

#: Photon energy at λ₀ [J]:  E_ph = ħ ω₀
E_PHOTON: Final[float] = HBAR * OMEGA_0


# ─────────────────────────────────────────────────────────────────────────────
# Silicon material constants at λ₀ = 1550 nm, T = 300 K
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SiliconConstants:
    """
    Crystalline silicon optical and thermal material constants at 1550 nm.

    All parameters correspond to room temperature (T = 300 K) and the
    standard (100) crystal orientation used in SOI waveguide platforms.

    Attributes
    ----------
    n0 : float
        Linear (real) refractive index of crystalline Si.
        Value: 3.476 (Palik, Handbook of Optical Constants, 1985).

    n2 : float
        Kerr nonlinear refractive index [m²/W].
        Governs Self-Phase Modulation (SPM) via:
            Δn_Kerr = n₂ · I(t)
        Value: 6.0×10⁻¹⁸ m²/W [Ref 1].

    beta_TPA : float
        Two-Photon Absorption (TPA) coefficient [m/W].
        Rate of two-photon absorption:
            dI/dz|_TPA = -β_TPA · I²
        Value: 5.0×10⁻¹² m/W [Ref 1].
        This generates free electron-hole pairs proportional to I².

    sigma_FCA : float
        Free-Carrier Absorption (FCA) cross-section [m²].
        Extra loss from free carriers:
            Δα_FCA = σ_FCA · N_c
        Value: 1.45×10⁻²¹ m² [Ref 2].

    dn_FC : float
        Free-Carrier Dispersion (FCD) coefficient [m³].
        Refractive index change per carrier density (Soref–Bennett model):
            Δn_FCD = dn_FC · N_c
        Value: -1.73×10⁻²⁷ m³ [Ref 2].
        Negative sign: free electrons DECREASE refractive index (blue-shift).

    tau_FC : float
        Free-carrier lifetime [s].
        Governs recovery of N_c after TPA excitation:
            dN_c/dt|_decay = -N_c / τ_FC
        Value: 10 ns for unprocessed SOI; reducible to <1 ns with p-i-n junction.

    dn_dT : float
        Thermo-optic coefficient [K⁻¹].
        Refractive index change with temperature:
            Δn_TO = (dn/dT) · ΔT
        Value: +1.86×10⁻⁴ K⁻¹ (positive: heating RED-shifts resonance) [Ref 1].

    tau_th : float
        Thermal relaxation time constant [s].
        Governs heat dissipation in the ring waveguide:
            dΔT/dt|_decay = -ΔT / τ_th
        Value: 50 ns for standard SOI oxide cladding.

    rho_cp : float
        Volumetric heat capacity [J/(m³·K)].
        Product of mass density ρ and specific heat capacity c_p.
        Value: 1.63×10⁶ J/(m³·K) for crystalline Si at 300 K.

    n_group : float
        Group index of the SOI waveguide at 1550 nm.
        n_g = n(λ) - λ · dn/dλ ≈ 3.7 for 450 nm × 220 nm strip waveguide.
        Used to compute group velocity: v_g = c / n_g.
    """
    n0:         float = 3.476           # linear refractive index
    n2:         float = 6.0e-18         # Kerr index [m²/W]
    beta_TPA:   float = 5.0e-12         # TPA coefficient [m/W]
    sigma_FCA:  float = 1.45e-21        # FCA cross-section [m²]
    dn_FC:      float = -1.73e-27       # FCD coefficient [m³]  (Soref–Bennett)
    tau_FC:     float = 10.0e-9         # Free-carrier lifetime [s] → 10 ns
    dn_dT:      float = 1.86e-4         # Thermo-optic coeff [K⁻¹]
    tau_th:     float = 50.0e-9         # Thermal time constant [s] → 50 ns
    rho_cp:     float = 1.63e6          # Volumetric heat capacity [J/(m³·K)]
    n_group:    float = 3.7             # Group index (450nm × 220nm strip)


# ─────────────────────────────────────────────────────────────────────────────
# Microring Resonator (MRR) physical geometry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MRRGeometry:
    """
    Physical geometry of a single Silicon Microring Resonator node.

    The ring geometry sets the Free Spectral Range (FSR), round-trip time,
    and effective mode volume — all critical for the nonlinear ODE integration.

    Attributes
    ----------
    radius : float
        Ring outer radius [m].
        r = 7.5 µm → FSR ≈ 1.83 THz for n_g = 3.7.
        FSR = v_g / (2π r) = c / (n_g · 2π r).

    wg_width : float
        Waveguide width [m]. Standard SOI: 450 nm.

    wg_height : float
        Waveguide height (thickness) [m]. Standard SOI: 220 nm.

    coupling_gap : float
        Bus-to-ring coupling gap [m]. Controls κ_e.
        Typical: 150–250 nm.

    kappa_sq : float
        Power coupling coefficient κ² (dimensionless, ∈ [0, 1]).
        Fraction of optical power coupled per round-trip.
        κ² = 0.05 → 5% per-pass coupling.
        Related to κ_e [rad/s]: κ_e = κ² / T_rt.

    loss_dB_per_cm : float
        Propagation loss of the ring waveguide [dB/cm].
        Used to compute intrinsic decay rate κ_i.
        Typical SOI: 1–3 dB/cm.

    Q_factor : float
        Loaded quality factor (dimensionless).
        Q = ω₀ / (κ_e + κ_i) = ω₀ / κ_total.
        Target: Q ~ 10⁴ for broadband IDS operation.
    """
    radius:         float = 7.5e-6      # [m] → 7.5 µm
    wg_width:       float = 450e-9      # [m] → 450 nm
    wg_height:      float = 220e-9      # [m] → 220 nm
    coupling_gap:   float = 200e-9      # [m] → 200 nm
    kappa_sq:       float = 0.05        # power coupling coefficient
    loss_dB_per_cm: float = 2.0         # propagation loss [dB/cm]
    Q_factor:       float = 1.0e4       # loaded Q factor


# ─────────────────────────────────────────────────────────────────────────────
# Derived optical constants (computed from geometry + material)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DerivedOpticalConstants:
    """
    Derived quantities computed from SiliconConstants and MRRGeometry.

    These are pre-computed once and cached for use in the NumPy ODE solvers
    to avoid redundant per-step computation.

    Attributes
    ----------
    L_ring : float
        Ring circumference [m]:  L = 2π · r.

    T_rt : float
        Round-trip time [s]:  T_rt = L / v_g = n_g · 2π · r / c.

    FSR : float
        Free Spectral Range [Hz]:  FSR = 1 / T_rt = v_g / L.

    V_eff : float
        Effective optical mode volume [m³].
        Approximated as:  V_eff = π · r · w · h (cross-sectional area × circumference).
        Used in TPA free-carrier generation rate.

    kappa_e : float
        External (coupling) field decay rate [rad/s]:
            κ_e = κ² / T_rt

    kappa_i : float
        Intrinsic (loss) field decay rate [rad/s]:
            κ_i = α_lin · v_g,   where α_lin = loss[1/m] = loss[dB/cm] × 100 / (20 log₁₀ e)

    kappa_total : float
        Total field decay rate [rad/s]:  κ = κ_e + κ_i.
        Related to Q: Q = ω₀ / κ.

    beta_eff : float
        TPA-driven free-carrier generation coefficient [m⁻³/s per (a.u.)²].
        In CMT photon-number normalisation:
            β_eff = β_TPA · c² / (2 · ħω₀ · n_g² · V_eff)

    dw_FC_coef : float
        FCD resonance shift coefficient [rad/s per m⁻³]:
            δω_FC = (ω₀ / n₀) · dn_FC · N_c

    dw_TO_coef : float
        Thermo-optic resonance shift coefficient [rad/s per K]:
            δω_TO = (ω₀ / n₀) · (dn/dT) · ΔT

    heat_coef : float
        Thermal heating rate coefficient [K/(s · m⁻³)]:
            dΔT/dt|_heat = heat_coef · N_c · |a|²_phot

    photon_energy : float
        Photon energy at λ₀ [J]:  E_ph = ħ · ω₀.
    """
    L_ring:         float = 0.0
    T_rt:           float = 0.0
    FSR:            float = 0.0
    V_eff:          float = 0.0
    kappa_e:        float = 0.0
    kappa_i:        float = 0.0
    kappa_total:    float = 0.0
    beta_eff:       float = 0.0
    dw_FC_coef:     float = 0.0
    dw_TO_coef:     float = 0.0
    heat_coef:      float = 0.0
    photon_energy:  float = 0.0

    @classmethod
    def compute(
        cls,
        mat: SiliconConstants,
        mrr: MRRGeometry,
    ) -> "DerivedOpticalConstants":
        """
        Compute all derived constants from material and geometry parameters.

        Parameters
        ----------
        mat : SiliconConstants
            Silicon material constants.
        mrr : MRRGeometry
            MRR physical geometry.

        Returns
        -------
        DerivedOpticalConstants
            Populated dataclass with all computed fields.
        """
        import math

        v_g     = C_LIGHT / mat.n_group                   # group velocity [m/s]
        L_ring  = 2.0 * math.pi * mrr.radius               # circumference [m]
        T_rt    = L_ring / v_g                             # round-trip time [s]
        FSR     = 1.0 / T_rt                               # FSR [Hz]

        # Effective mode volume [m³]
        V_eff = math.pi * mrr.radius * mrr.wg_width * mrr.wg_height

        # Coupling rates [rad/s]
        kappa_e = mrr.kappa_sq / T_rt

        # Intrinsic loss rate from propagation loss
        # loss [dB/cm] → loss [1/m]:  α = loss[dB/cm] × 100 / (10 × log₁₀(e) × 2)
        # Factor 2 because intensity α = 2 × field α (power vs. amplitude)
        log10e = math.log10(math.e)
        alpha_m = (mrr.loss_dB_per_cm * 100.0) / (20.0 * log10e)  # field loss [1/m]
        kappa_i = alpha_m * v_g                            # intrinsic decay [rad/s]

        kappa_total = kappa_e + kappa_i

        # TPA carrier generation coefficient [m^-3 / s / photon^2]
        # From CMT normalisation where |a|² = photon number in cavity:
        #   G_TPA = β_eff · |a|⁴
        #   β_eff = β_TPA · c² / (2 · ħω₀ · n_g² · V_eff)
        beta_eff = (
            mat.beta_TPA * (C_LIGHT ** 2)
            / (2.0 * HBAR * OMEGA_0 * (mat.n_group ** 2) * V_eff)
        )

        # FCD resonance shift: δω_FC = (ω₀ / n₀) · dn_FC · N_c  [rad/s / m^-3]
        dw_FC_coef = (OMEGA_0 / mat.n0) * mat.dn_FC

        # Thermo-optic shift: δω_TO = (ω₀ / n₀) · (dn/dT) · ΔT  [rad/s / K]
        dw_TO_coef = (OMEGA_0 / mat.n0) * mat.dn_dT

        # Thermal heating: dΔT/dt|_heat = heat_coef · N_c · |a|²
        # P_abs = σ_FCA · N_c · v_g · ħω₀ · |a|²  (absorbed power [W])
        # dΔT/dt = P_abs / (ρ·cp · V_eff)
        heat_coef = (
            mat.sigma_FCA * v_g * HBAR * OMEGA_0
            / (mat.rho_cp * V_eff)
        )

        photon_energy = HBAR * OMEGA_0

        return cls(
            L_ring=L_ring,
            T_rt=T_rt,
            FSR=FSR,
            V_eff=V_eff,
            kappa_e=kappa_e,
            kappa_i=kappa_i,
            kappa_total=kappa_total,
            beta_eff=beta_eff,
            dw_FC_coef=dw_FC_coef,
            dw_TO_coef=dw_TO_coef,
            heat_coef=heat_coef,
            photon_energy=photon_energy,
        )

    def summary(self) -> str:
        """Return a human-readable table of derived constants."""
        lines = [
            "DerivedOpticalConstants",
            "=" * 54,
            f"  L_ring        : {self.L_ring * 1e6:.2f} µm",
            f"  T_rt          : {self.T_rt * 1e12:.3f} ps",
            f"  FSR           : {self.FSR / 1e12:.3f} THz",
            f"  V_eff         : {self.V_eff:.3e} m³",
            f"  κ_e           : {self.kappa_e:.3e} rad/s",
            f"  κ_i           : {self.kappa_i:.3e} rad/s",
            f"  κ_total       : {self.kappa_total:.3e} rad/s",
            f"  β_eff (TPA)   : {self.beta_eff:.3e}",
            f"  dω_FC/dN_c    : {self.dw_FC_coef:.3e} rad/s/m⁻³",
            f"  dω_TO/dT      : {self.dw_TO_coef:.3e} rad/s/K",
            f"  heat_coef     : {self.heat_coef:.3e} K/s/m⁻³",
            f"  E_photon      : {self.photon_energy:.3e} J",
            "=" * 54,
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# WDM Channel Grid (ITU-T C-band)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WDMGrid:
    """
    ITU-T C-band 100 GHz WDM channel grid for 16 input channels.

    16 channels span ≈ 1547–1551 nm, each carrying one preprocessed
    network traffic feature (PCA component) encoded by the MZM.

    Attributes
    ----------
    center_wavelength_nm : float
        Centre wavelength of the WDM comb [nm]. Default: 1550 nm.
    channel_spacing_GHz : float
        Frequency spacing between adjacent channels [GHz]. Default: 100 GHz.
    n_channels : int
        Total WDM channels = number of PCA features after reduction.
    """
    center_wavelength_nm:  float = 1550.0
    channel_spacing_GHz:   float = 100.0
    n_channels:            int   = 16

    def wavelengths_nm(self) -> np.ndarray:
        """
        Compute the centre wavelengths of all 16 WDM channels [nm].

        Uses the ITU-T frequency grid:
            f_k = f_center + (k - (N-1)/2) × Δf,   k = 0, …, N-1

        Returns
        -------
        np.ndarray  shape [16]
            Wavelengths in nm, ascending order.
        """
        f_center_Hz = C_LIGHT / (self.center_wavelength_nm * 1e-9)
        delta_f_Hz  = self.channel_spacing_GHz * 1e9
        half_N      = (self.n_channels - 1) / 2.0
        k           = np.arange(self.n_channels, dtype=np.float64)
        freqs_Hz    = f_center_Hz + (k - half_N) * delta_f_Hz
        return (C_LIGHT / freqs_Hz) * 1e9   # convert m to nm


# ─────────────────────────────────────────────────────────────────────────────
# Default singleton instances
# ─────────────────────────────────────────────────────────────────────────────

#: Default silicon material constants (room temperature, 1550 nm)
SI: Final[SiliconConstants] = SiliconConstants()

#: Default MRR geometry (7.5 µm radius, 450×220 nm waveguide)
MRR: Final[MRRGeometry] = MRRGeometry()

#: Pre-computed derived optical constants from defaults
DERIVED: Final[DerivedOpticalConstants] = DerivedOpticalConstants.compute(SI, MRR)

#: Default WDM grid (16-channel, 100 GHz spacing, C-band)
WDM: Final[WDMGrid] = WDMGrid()
