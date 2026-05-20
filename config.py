"""
config.py — Master System Configuration
=======================================
Centralised configuration for the All-Optical Photonic RC-IDS v2 framework.
All physical constants live in physics/constants.py.
This file handles high-level system and pipeline settings.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ReservoirConfig:
    """8x8 MRR mesh reservoir configuration."""
    grid_rows:            int   = 8
    grid_cols:            int   = 8
    n_input_channels:     int   = 16    # WDM channels = PCA components
    spectral_radius:      float = 0.9   # W_res spectral radius
    input_scaling:        float = 0.5   # W_in global gain
    mask_connectivity:    float = 0.8   # fraction of active connections
    input_power_dBm:      float = -10.0 # per-channel WDM power
    n_steps:              int   = 8     # ODE steps per sample
    scheme:               str   = 'euler'  # 'euler' or 'rk4'
    random_seed:          int   = 42

    @property
    def n_nodes(self) -> int:
        return self.grid_rows * self.grid_cols


@dataclass
class PreprocessorConfig:
    """Stage 1 preprocessing pipeline configuration."""
    n_raw_features:   int   = 79       # CICIDS2017 feature count
    n_pca_components: int   = 16       # PCA output (= WDM channels)
    pca_whiten:       bool  = True
    scaler_range:     tuple = (0.0, 1.0)


@dataclass
class ReadoutConfig:
    """Ridge regression readout configuration."""
    alpha:                float = 1e-3
    n_classes:            int   = 8    # 7 attacks + Benign
    softmax_temperature:  float = 1.0
    confidence_threshold: float = 0.85


@dataclass
class FeedbackConfig:
    """Adaptive zero-day feedback loop configuration."""
    buffer_max_size:          int   = 500
    dbscan_eps:               float = 0.5
    dbscan_min_samples:       int   = 10
    pca_components_feedback:  int   = 20
    retrain_trigger_size:     int   = 50
    hot_swap_alpha:           float = 1e-3


@dataclass
class DataConfig:
    """Dataset and data pipeline configuration."""
    dataset_path:   Optional[str] = None  # None = use mock data
    train_ratio:    float         = 0.70
    memmap_dtype:   str           = 'float32'  # for np.memmap efficiency
    chunk_size:     int           = 50_000     # rows per CSV chunk
    random_seed:    int           = 42
    n_mock_samples: int           = 5_000


@dataclass
class SystemConfig:
    """
    Master configuration object.

    Usage
    -----
    >>> from config import SystemConfig
    >>> cfg = SystemConfig()
    >>> cfg.reservoir.n_nodes
    64
    """
    reservoir:   ReservoirConfig   = field(default_factory=ReservoirConfig)
    preprocessor: PreprocessorConfig = field(default_factory=PreprocessorConfig)
    readout:     ReadoutConfig     = field(default_factory=ReadoutConfig)
    feedback:    FeedbackConfig    = field(default_factory=FeedbackConfig)
    data:        DataConfig        = field(default_factory=DataConfig)
    verbose:     bool              = True


DEFAULT_CONFIG = SystemConfig()
