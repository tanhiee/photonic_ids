"""
E5_ablation.py — Milestone 5: Component Ablation Study
=======================================================
Systematic ablation to quantify each component's contribution:

  Ablation conditions:
    A. Full model              (TPA + FCD + TO + W_res + MZM)
    B. No recurrent (W_res=0)  (input projection only)
    C. No TPA/FCD/TO           (linear cavity, no nonlinearity)
    D. No MZM encoding         (raw PCA features, linear input)
    E. Random readout          (untrained Ridge, all weights=1)
    F. PCA only (no reservoir) (Ridge on 16-dim PCA directly)

Reports accuracy drop for each removed component.

Usage:  python scripts/E5_ablation.py
"""
import sys, os, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
logging.basicConfig(level=logging.WARNING)  # suppress info for clean output

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score

from config import SystemConfig
from data.mock_generator import generate_mock_cicids
from data.preprocessor import NetworkTrafficPreprocessor
from reservoir.mrr_mesh import PhotonicMesh
from reservoir.readout import RidgeReadout
from reservoir.projection import build_input_mask
from reservoir.recurrent_matrix import build_recurrent_matrix
from physics.mrr_dynamics import MRRDynamicsIntegrator, IntegrationParams
from physics import DERIVED, SI


def run_ablation(label, X_train, X_test, y_train, y_test, alpha=1e-3):
    """Fit Ridge + evaluate. Returns (accuracy, f1)."""
    clf = RidgeClassifier(alpha=alpha, fit_intercept=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return (
        accuracy_score(y_test, y_pred) * 100,
        f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100,
    )


def main():
    print("\n=== E5: Component Ablation Study ===\n")

    cfg    = SystemConfig()
    cfg.data.n_mock_samples = 3000

    # ── Data loading + preprocessing ─────────────────────────────────────────
    df    = generate_mock_cicids(n_samples=3000, random_seed=42)
    prep  = NetworkTrafficPreprocessor(cfg)
    X_raw = df[[c for c in df.columns if c != 'Label']].values.astype('float32')
    y_raw = df['Label'].values
    X_tr_opt, X_te_opt, y_tr, y_te, le = prep.fit_transform(X_raw, y_raw)

    # Also get PCA-only features (no MZM)
    X_tr_pca = np.arcsin(X_tr_opt)  # inverse of MZM sin encoding
    X_te_pca = np.arcsin(X_te_opt)

    results = {}

    # ── A. Full model ─────────────────────────────────────────────────────────
    print("  A. Full model (TPA + FCD + TO + W_res + MZM)...")
    mesh   = PhotonicMesh(cfg.reservoir)
    S_tr_A = mesh.forward(X_tr_opt)
    S_te_A = mesh.forward(X_te_opt)
    results['A. Full model'] = run_ablation('A', S_tr_A, S_te_A, y_tr, y_te)
    print(f"     Acc={results['A. Full model'][0]:.2f}%  F1={results['A. Full model'][1]:.2f}%")

    # ── B. No recurrent coupling (W_res = 0) ──────────────────────────────────
    print("  B. No recurrent coupling (W_res = 0)...")
    W_in  = mesh.W_in.copy()
    W_res_zero = np.zeros((64, 64))
    params = IntegrationParams(n_steps=cfg.reservoir.n_steps, scheme='euler')
    intg  = MRRDynamicsIntegrator(dc=DERIVED, mat=SI, params=params)
    S_tr_B = intg.forward(X_tr_opt, W_in, W_res_zero,
                          delta_omega_static=mesh.delta_omega_static,
                          photon_scale=mesh.photon_scale)
    S_te_B = intg.forward(X_te_opt, W_in, W_res_zero,
                          delta_omega_static=mesh.delta_omega_static,
                          photon_scale=mesh.photon_scale)
    results['B. No W_res (no recurrence)'] = run_ablation('B', S_tr_B, S_te_B, y_tr, y_te)
    print(f"     Acc={results['B. No W_res (no recurrence)'][0]:.2f}%  "
          f"F1={results['B. No W_res (no recurrence)'][1]:.2f}%")

    # ── C. No TPA/FCD/TO (n_steps=1, linear only) ─────────────────────────────
    print("  C. No nonlinear physics (n_steps=1, linear approximation)...")
    params_lin = IntegrationParams(n_steps=1, scheme='euler')
    intg_lin   = MRRDynamicsIntegrator(dc=DERIVED, mat=SI, params=params_lin)
    S_tr_C = intg_lin.forward(X_tr_opt, W_in, mesh.W_res,
                               delta_omega_static=mesh.delta_omega_static,
                               photon_scale=mesh.photon_scale)
    S_te_C = intg_lin.forward(X_te_opt, W_in, mesh.W_res,
                               delta_omega_static=mesh.delta_omega_static,
                               photon_scale=mesh.photon_scale)
    results['C. No TPA/FCD/TO (n_steps=1)'] = run_ablation('C', S_tr_C, S_te_C, y_tr, y_te)
    print(f"     Acc={results['C. No TPA/FCD/TO (n_steps=1)'][0]:.2f}%  "
          f"F1={results['C. No TPA/FCD/TO (n_steps=1)'][1]:.2f}%")

    # ── D. No MZM encoding (raw PCA features, no sin) ─────────────────────────
    print("  D. No MZM encoding (raw normalized PCA features)...")
    S_tr_D = intg.forward(X_tr_pca, W_in, mesh.W_res,
                           delta_omega_static=mesh.delta_omega_static,
                           photon_scale=mesh.photon_scale)
    S_te_D = intg.forward(X_te_pca, W_in, mesh.W_res,
                           delta_omega_static=mesh.delta_omega_static,
                           photon_scale=mesh.photon_scale)
    results['D. No MZM (raw PCA)'] = run_ablation('D', S_tr_D, S_te_D, y_tr, y_te)
    print(f"     Acc={results['D. No MZM (raw PCA)'][0]:.2f}%  "
          f"F1={results['D. No MZM (raw PCA)'][1]:.2f}%")

    # ── E. Random readout (uninitialized, constant weights) ───────────────────
    print("  E. Random readout baseline (shuffle labels)...")
    y_shuffled = y_tr.copy()
    np.random.shuffle(y_shuffled)
    results['E. Random readout (shuffled labels)'] = run_ablation('E', S_tr_A, S_te_A, y_shuffled, y_te)
    print(f"     Acc={results['E. Random readout (shuffled labels)'][0]:.2f}%  "
          f"F1={results['E. Random readout (shuffled labels)'][1]:.2f}%")

    # ── F. PCA only (bypass reservoir) ────────────────────────────────────────
    print("  F. PCA only — bypass reservoir (16-dim direct)...")
    results['F. PCA only (no reservoir)'] = run_ablation('F', X_tr_opt, X_te_opt, y_tr, y_te)
    print(f"     Acc={results['F. PCA only (no reservoir)'][0]:.2f}%  "
          f"F1={results['F. PCA only (no reservoir)'][1]:.2f}%")

    # ── Summary table ─────────────────────────────────────────────────────────
    full_acc = results['A. Full model'][0]
    print("\n--- Ablation Summary ---")
    print(f"{'Condition':<40} {'Accuracy':>10} {'F1':>8} {'Drop':>8}")
    print("-" * 70)
    for name, (acc, f1) in results.items():
        drop = full_acc - acc
        marker = " <-- BASELINE" if name.startswith('A') else \
                 f"  (drop {drop:+.1f}%)"
        print(f"  {name:<38} {acc:>9.2f}% {f1:>7.2f}%{marker}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    labels  = [k.split('(')[0].strip() for k in results.keys()]
    accs    = [v[0] for v in results.values()]
    f1s     = [v[1] for v in results.values()]
    colors  = ['steelblue'] + ['salmon'] * (len(results) - 1)

    y_pos = np.arange(len(labels))
    ax1.barh(y_pos, accs, color=colors, alpha=0.85, edgecolor='white')
    ax1.set_yticks(y_pos); ax1.set_yticklabels(labels, fontsize=9)
    ax1.axvline(full_acc, color='green', linestyle='--', linewidth=1.5,
                label=f'Full model: {full_acc:.1f}%')
    ax1.set_xlabel('Accuracy (%)'); ax1.set_title('Ablation: Accuracy', fontweight='bold')
    ax1.set_xlim(0, 100); ax1.legend(fontsize=8); ax1.grid(axis='x', alpha=0.3)
    for i, v in enumerate(accs):
        ax1.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=8)

    ax2.barh(y_pos, f1s, color=colors, alpha=0.85, edgecolor='white')
    ax2.set_yticks(y_pos); ax2.set_yticklabels(labels, fontsize=9)
    ax2.axvline(results['A. Full model'][1], color='green', linestyle='--',
                linewidth=1.5, label=f"Full model: {results['A. Full model'][1]:.1f}%")
    ax2.set_xlabel('F1-Score (weighted %)'); ax2.set_title('Ablation: F1-Score', fontweight='bold')
    ax2.set_xlim(0, 100); ax2.legend(fontsize=8); ax2.grid(axis='x', alpha=0.3)
    for i, v in enumerate(f1s):
        ax2.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=8)

    fig.suptitle('E5: Component Ablation Study -- Photonic RC-IDS',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('E5_ablation.png', dpi=150, bbox_inches='tight')
    print("\n  Plot saved: E5_ablation.png")
    plt.show()


if __name__ == '__main__':
    main()
