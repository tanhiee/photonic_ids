# All-Optical Photonic Reservoir Computing IDS — v2

A modular, high-performance simulation framework for **Terabit/s Zero-Day Attack Detection** using Silicon Microring Resonator (MRR) photonic reservoir computing.

## Quick Start

```bash
# 1. Create and activate Conda environment (MKL-optimized)
conda env create -f conda_env.yaml
conda activate photonic_rc_ids

# 2. Run E1: Single ring dynamics
python scripts/E1_single_ring.py

# 3. Run E3: Full offline training (mock data)
python scripts/E3_offline_training.py

# 4. Run E3: Full offline training (real CICIDS2017)
python scripts/E3_offline_training.py --data path/to/CICIDS2017.csv
```

## Architecture

| Layer | Module | Description |
|-------|--------|-------------|
| **physics/** | constants.py | Si material constants at 1550 nm |
| | tpa_fcd.py | Two-Photon Absorption + Free-Carrier Dispersion |
| | thermo_optic.py | Thermo-optic effect & thermal ODE |
| | coupled_mode_theory.py | CMT ring-waveguide equations |
| | mrr_dynamics.py | Vectorized ODE integrator (Euler/RK4) |
| **data/** | cicids_loader.py | Memory-mapped CICIDS2017 loader |
| | preprocessor.py | MinMaxScaler + PCA + MZM encode |
| **reservoir/** | mrr_mesh.py | 8x8 MRR mesh forward pass |
| | readout.py | Ridge Classifier + Softmax |
| **training_eval/** | offline_trainer.py | Phase 1 batch training |
| | online_streamer.py | Phase 2 streaming inference |
| | adaptive_loop.py | DBSCAN zero-day feedback |

## Scripts (9 Milestones)

| Script | Target |
|--------|--------|
| E1_single_ring.py | Single MRR dynamics traces |
| E2_mesh_projection.py | 64-ring mesh validation |
| E3_offline_training.py | Full CICIDS2017 pipeline |
| E4_full_eval.py | ~94.93% accuracy target |
| E5_ablation.py | Component ablation study |
| E6_zeroday_defense.py | DBSCAN adaptive defense |
| E7_capacity.py | Terabit/s capacity analysis |
| E8_capex_opex.py | 93.9% CapEx savings vs H100 |
| E9_adversarial.py | Adversarial robustness |

Report all issues to the project maintainer.
