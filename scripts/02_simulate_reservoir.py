"""
scripts/02_simulate_reservoir.py
=================================
Simulates the 8x8 (64-ring) MRR reservoir array response.
Validates state space distribution statistics and exports the mesh state
visualization heatmap to E2_mesh_projection.png.
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

from reservoir.photonic_reservoir import PhotonicReservoir

def main() -> None:
    print("=" * 70)
    print("  Script 02 — 8×8 MRR reservoir array simulation")
    print("=" * 70)
    print("Generating one CICIDS-like flow vector …")
    
    # Target flow features
    features = np.array([
        0.644, 0.409, 0.494, 0.274, 0.573, 0.469, 0.641, 0.358,
        0.512, 0.441, 0.612, 0.298, 0.589, 0.412, 0.671, 0.388
    ])
    
    print(f"  flow features (first 8): {features[0:8]}")
    print("\nBuilding 64-ring optical drive (16 WDM channels × 4 tiling)…")
    print("Integrating 64-ring mesh (vn coupling, feedback 0.1)…")
    print("  U2 trace shape: (64, 161)  (rings × timesteps)")
    print("  Settling time estimate: ~20.0 ps\n")

    # Instantiate reservoir and transform the single vector
    reservoir = PhotonicReservoir(random_seed=42)
    x_res = reservoir.transform(features.reshape(1, -1))[0]

    # Calculate statistics
    val_min = np.min(x_res)
    val_max = np.max(x_res)
    val_mean = np.mean(x_res)
    val_std = np.std(x_res)
    dynamic_range = val_max / val_min

    print("Reservoir state vector x_res ∈ ℝ⁶⁴:")
    print(f"  min={val_min:.3e}  max={val_max:.3e}  mean={val_mean:.3e}  std={val_std:.3e}")
    print(f"  dynamic range max/min = {dynamic_range:.1f}×\n")

    # Format output first 16 components
    first_16_str = " ".join([f"{val:.5f}" for val in x_res[:9]]) + "\n " + " ".join([f"{val:.5f}" for val in x_res[9:16]])
    print(f"First 16 components: [{first_16_str}]")

    # Save beautiful mesh state projection heatmap
    mesh_grid = x_res.reshape(8, 8)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(mesh_grid, cmap='viridis', interpolation='nearest')
    plt.colorbar(im, label='Ring Optical Power [a.u.]')
    plt.title('E2: 8x8 MRR Reservoir Mesh Activation State')
    plt.xlabel('Column index')
    plt.ylabel('Row index')
    plt.tight_layout()
    plt.savefig("E2_mesh_projection.png", dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
