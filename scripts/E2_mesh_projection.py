"""
E2_mesh_projection.py — Milestone 2: 64-Ring Mesh Projection Validation
========================================================================
Validates the full 8x8 MRR mesh forward pass:
  - Visualizes W_in and W_res weight matrices
  - Shows per-node photon number distribution (bar chart)
  - Plots 2D heatmap of reservoir state across the 8x8 grid
  - Reports spectral radius verification

Usage:  python scripts/E2_mesh_projection.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import SystemConfig
from data.mock_generator import generate_mock_cicids
from data.preprocessor import NetworkTrafficPreprocessor
from reservoir.mrr_mesh import PhotonicMesh
from reservoir.recurrent_matrix import verify_spectral_radius


def main():
    print("=== E2: 8x8 MRR Mesh Projection Validation ===\n")

    cfg   = SystemConfig()
    mesh  = PhotonicMesh(cfg.reservoir)

    rho_actual = verify_spectral_radius(mesh.W_res)
    print(f"  W_in  shape : {mesh.W_in.shape}   (nodes x channels)")
    print(f"  W_res shape : {mesh.W_res.shape}  (nodes x nodes)")
    print(f"  Spectral radius rho(W_res) = {rho_actual:.4f}  (target={cfg.reservoir.spectral_radius})")
    print(f"  Echo-state property: {'SATISFIED' if rho_actual < 1.0 else 'VIOLATED'}")

    df     = generate_mock_cicids(n_samples=200, random_seed=42)
    prep   = NetworkTrafficPreprocessor(cfg)
    X_raw  = df[[c for c in df.columns if c != 'Label']].values.astype('float32')
    y_raw  = df['Label'].values
    X_tr, X_te, y_tr, y_te, le = prep.fit_transform(X_raw, y_raw)

    print(f"\n  Running reservoir forward pass on {len(X_tr)} samples...")
    states = mesh.forward(X_tr)
    print(f"  Output states shape : {states.shape}")
    print(f"  State range         : [{states.min():.4f}, {states.max():.4f}]")
    print(f"  Mean activity       : {states.mean():.4f}  (std={states.std():.4f})")
    print(f"  Dead nodes (all-0)  : {int((states.std(axis=0) < 1e-6).sum())} / 64")

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(mesh.W_in, aspect='auto', cmap='RdBu_r',
                     vmin=-np.abs(mesh.W_in).max(), vmax=np.abs(mesh.W_in).max())
    ax1.set_title('W_in  [64 nodes x 16 channels]', fontsize=10, fontweight='bold')
    ax1.set_xlabel('WDM Channel'); ax1.set_ylabel('MRR Node')
    plt.colorbar(im1, ax=ax1, shrink=0.85)

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(mesh.W_res, aspect='auto', cmap='RdBu_r',
                     vmin=-np.abs(mesh.W_res).max(), vmax=np.abs(mesh.W_res).max())
    ax2.set_title(f'W_res  [64x64]   rho={rho_actual:.3f}', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Source Node'); ax2.set_ylabel('Target Node')
    plt.colorbar(im2, ax=ax2, shrink=0.85)

    ax3 = fig.add_subplot(gs[0, 2])
    grid = states.mean(axis=0).reshape(8, 8)
    im3  = ax3.imshow(grid, cmap='inferno', aspect='equal')
    ax3.set_title('Mean PD Intensity\n(8x8 MRR grid, batch avg)', fontsize=10, fontweight='bold')
    ax3.set_xticks(range(8)); ax3.set_yticks(range(8))
    ax3.set_xlabel('Column'); ax3.set_ylabel('Row')
    plt.colorbar(im3, ax=ax3, shrink=0.85)

    ax4 = fig.add_subplot(gs[1, :2])
    node_means = states.mean(axis=0)
    node_stds  = states.std(axis=0)
    x = np.arange(64)
    ax4.bar(x, node_means, yerr=node_stds, alpha=0.7, color='steelblue',
            error_kw=dict(ecolor='gray', capsize=2, linewidth=0.8))
    ax4.set_xlabel('Node Index'); ax4.set_ylabel('Normalised PD Intensity')
    ax4.set_title('Per-Node Mean Activity +/- 1 sigma  (64 MRR nodes)', fontsize=10, fontweight='bold')
    ax4.set_xlim(-1, 64)
    ax4.axhline(node_means.mean(), color='red', linestyle='--', linewidth=1.5,
                label=f'Mean = {node_means.mean():.3f}')
    ax4.legend(); ax4.grid(axis='y', alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 2])
    for class_id in range(min(4, len(le.classes_))):
        mask = y_tr == class_id
        if mask.sum() > 5:
            ax5.hist(states[mask, 0], bins=20, alpha=0.5,
                     label=le.classes_[class_id], density=True)
    ax5.set_xlabel('State Value (Node 0)'); ax5.set_ylabel('Density')
    ax5.set_title('Class-conditional State Distribution\n(Node 0)', fontsize=10, fontweight='bold')
    ax5.legend(fontsize=8); ax5.grid(alpha=0.3)

    fig.suptitle('E2: 8x8 Photonic MRR Mesh -- Reservoir Projection Analysis',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.savefig('E2_mesh_projection.png', dpi=150, bbox_inches='tight')
    print("\n  Plot saved: E2_mesh_projection.png")
    plt.show()


if __name__ == '__main__':
    main()
