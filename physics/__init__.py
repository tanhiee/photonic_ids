"""
physics/__init__.py
===================
Public API for the physics layer.

Import hierarchy:
    constants       → tpa_fcd, thermo_optic
    tpa_fcd         → constants
    thermo_optic    → constants, tpa_fcd
    coupled_mode_theory → constants, tpa_fcd
    mrr_dynamics    → all of the above
"""
from physics.constants import (
    SiliconConstants,
    MRRGeometry,
    DerivedOpticalConstants,
    WDMGrid,
    SI,
    MRR,
    DERIVED,
    WDM,
    C_LIGHT,
    HBAR,
    OMEGA_0,
    E_PHOTON,
    LAMBDA_0,
)
from physics.tpa_fcd import (
    tpa_generation_rate,
    fca_loss_rate,
    free_carrier_ode,
    free_carrier_steady_state,
    fcd_resonance_shift,
    lorentzian_transmission,
    compute_tpa_fcd_derivatives,
)
from physics.thermo_optic import (
    fca_heating_rate,
    thermal_ode,
    thermal_steady_state,
    thermo_optic_shift,
    thermo_optic_wavelength_shift,
    total_resonance_detuning,
    compute_thermal_derivatives,
)
from physics.coupled_mode_theory import (
    cmt_intensity_ode,
    cmt_complex_ode,
    cmt_euler_step,
    cmt_rk4_step,
    cmt_steady_state_a_sq,
    input_flux_from_power,
    input_flux_from_dBm,
    recurrent_drive,
    wdm_input_drive,
    through_transmission,
)
from physics.mrr_dynamics import (
    MRRState,
    IntegrationParams,
    MRRDynamicsIntegrator,
    mrr_derivatives,
    euler_step,
    rk4_step,
)

__all__ = [
    # Constants
    "SiliconConstants", "MRRGeometry", "DerivedOpticalConstants",
    "WDMGrid", "SI", "MRR", "DERIVED", "WDM",
    "C_LIGHT", "HBAR", "OMEGA_0", "E_PHOTON", "LAMBDA_0",
    # TPA/FCD
    "tpa_generation_rate", "fca_loss_rate", "free_carrier_ode",
    "free_carrier_steady_state", "fcd_resonance_shift",
    "lorentzian_transmission", "compute_tpa_fcd_derivatives",
    # Thermo-optic
    "fca_heating_rate", "thermal_ode", "thermal_steady_state",
    "thermo_optic_shift", "thermo_optic_wavelength_shift",
    "total_resonance_detuning", "compute_thermal_derivatives",
    # CMT
    "cmt_intensity_ode", "cmt_complex_ode", "cmt_euler_step",
    "cmt_rk4_step", "cmt_steady_state_a_sq",
    "input_flux_from_power", "input_flux_from_dBm",
    "recurrent_drive", "wdm_input_drive", "through_transmission",
    # MRR dynamics
    "MRRState", "IntegrationParams", "MRRDynamicsIntegrator",
    "mrr_derivatives", "euler_step", "rk4_step",
]
