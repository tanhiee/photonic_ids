"""
data/wdm_encoder.py
===================
WDM channel assignment and encoding utilities.
Maps PCA-reduced features to ITU-T C-band WDM channels.
"""
from __future__ import annotations
import numpy as np
from physics.constants import WDM, C_LIGHT


def channel_wavelengths_nm() -> np.ndarray:
    """Return the 16 WDM channel centre wavelengths [nm]."""
    return WDM.wavelengths_nm()


def channel_frequencies_thz() -> np.ndarray:
    """Return the 16 WDM channel centre frequencies [THz]."""
    wl_m = channel_wavelengths_nm() * 1e-9
    return C_LIGHT / wl_m / 1e12


def assign_features_to_channels(x_optical: np.ndarray) -> dict:
    """
    Create a mapping of PCA feature index to WDM channel.

    Parameters
    ----------
    x_optical : np.ndarray  shape [N, 16]
        MZM-encoded optical amplitudes.

    Returns
    -------
    dict  {channel_idx: {'wavelength_nm': float, 'mean_amplitude': float}}
    """
    wl = channel_wavelengths_nm()
    return {
        i: {
            'wavelength_nm': wl[i],
            'mean_amplitude': float(x_optical[:, i].mean()),
            'std_amplitude':  float(x_optical[:, i].std()),
        }
        for i in range(min(x_optical.shape[1], 16))
    }
