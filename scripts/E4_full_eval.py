"""
E4_full_eval.py — Milestone 4: Complete Performance Evaluation
==============================================================
Full evaluation pipeline targeting:
  - Accuracy  ~94.93%  (on real CICIDS2017)
  - Latency   ~105 ps  (optical processing, simulation proxy: <0.1 ms/sample)
  - F1-Score  ~94%     (weighted)
  - ROC-AUC   ~0.99

Reports:
  - Per-class precision, recall, F1
  - Confusion matrix heatmap
  - ROC curve (OvR, each class)
  - Latency distribution histogram
  - Comparison table vs. state-of-the-art baselines

Usage:
    python scripts/E4_full_eval.py
    python scripts/E4_full_eval.py --data path/to/CICIDS2017.csv --alpha 1e-4
"""
import sys, os, argparse, logging, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s', datefmt='%H:%M:%S')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc as sklearn_auc,
    accuracy_score, f1_score
)
from sklearn.preprocessing import label_binarize

from config import SystemConfig
from training_eval.offline_trainer import OfflineTrainer
from training_eval.metrics import ClassificationMetrics


def plot_confusion_matrix(cm, class_names, ax, title='Confusion Matrix'):
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f'{cm[i,j]}', ha='center', va='center', fontsize=7,
                    color='white' if cm[i,j] > thresh else 'black')
    plt.colorbar(im, ax=ax, shrink=0.8)


def plot_roc_curves(y_test, proba, class_names, ax, title='ROC Curves (OvR)'):
    y_bin = label_binarize(y_test, classes=list(range(len(class_names))))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    for i, (cname, col) in enumerate(zip(class_names, colors)):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        roc_auc     = sklearn_auc(fpr, tpr)
        ax.plot(fpr, tpr, color=col, linewidth=1.5,
                label=f'{cname} (AUC={roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(alpha=0.3)


def measure_per_sample_latency(readout, X_test, n_samples=200):
    """Measure per-sample inference latency distribution."""
    latencies = []
    for i in range(min(n_samples, len(X_test))):
        t0 = time.perf_counter()
        _ = readout.predict_proba(X_test[i:i+1])
        latencies.append((time.perf_counter() - t0) * 1000)  # ms
    return np.array(latencies)


def main():
    parser = argparse.ArgumentParser(description='E4: Full Evaluation')
    parser.add_argument('--data',  type=str,   default=None)
    parser.add_argument('--alpha', type=float, default=1e-3)
    parser.add_argument('--seed',  type=int,   default=42)
    args = parser.parse_args()

    print("\n=== E4: Full Performance Evaluation ===\n")

    cfg                  = SystemConfig()
    cfg.data.random_seed = args.seed
    cfg.readout.alpha    = args.alpha

    # ── Train ────────────────────────────────────────────────────────────────
    trainer = OfflineTrainer(cfg)
    metrics, trainer = trainer.run(dataset_path=args.data)

    readout = trainer.readout
    mesh    = trainer.mesh
    le      = trainer.preprocessor.label_encoder
    X_test  = trainer._X_test_states
    y_test  = trainer._y_test
    class_names = list(le.classes_)

    # ── Detailed per-class report ─────────────────────────────────────────────
    y_pred = readout.predict(X_test)
    proba  = readout.predict_proba(X_test)
    cm     = confusion_matrix(y_test, y_pred)

    print("\n--- Per-Class Classification Report ---")
    print(classification_report(y_test, y_pred,
          target_names=class_names, zero_division=0))

    # ── Latency measurement ───────────────────────────────────────────────────
    latencies = measure_per_sample_latency(readout, X_test, n_samples=300)
    print(f"--- Readout Latency (Ridge inference only) ---")
    print(f"  Mean  : {latencies.mean():.4f} ms/sample")
    print(f"  P99   : {np.percentile(latencies, 99):.4f} ms/sample")
    print(f"  Min   : {latencies.min():.4f} ms/sample")
    print(f"  Note  : Optical reservoir latency ≈ T_rt × n_steps = "
          f"{0.582e-12 * cfg.reservoir.n_steps * 1e12:.1f} ps  (photon transit)")

    # ── Baseline comparison table ─────────────────────────────────────────────
    acc = accuracy_score(y_test, y_pred) * 100
    f1  = f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100
    print("\n--- Comparison vs. Published Baselines ---")
    print(f"{'Method':<28} {'Accuracy':>10} {'F1':>8}")
    print("-" * 50)
    print(f"{'Photonic RC-IDS (ours)':<28} {acc:>9.2f}% {f1:>7.2f}%")
    print(f"{'CNN-LSTM (Gao 2023)':<28} {'91.23%':>10} {'90.8%':>8}")
    print(f"{'Random Forest':<28} {'88.50%':>10} {'87.2%':>8}")
    print(f"{'XGBoost':<28} {'90.10%':>10} {'89.5%':>8}")
    print(f"{'Target (paper claim)':<28} {'94.93%':>10} {'~94%':>8}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :2])
    plot_confusion_matrix(cm, class_names, ax1,
                          title=f'Confusion Matrix  (Acc={acc:.2f}%)')

    ax2 = fig.add_subplot(gs[0, 2])
    plot_roc_curves(y_test, proba, class_names, ax2)

    ax3 = fig.add_subplot(gs[1, 0])
    per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
    colors = ['green' if v >= 0.80 else 'orange' if v >= 0.60 else 'red'
              for v in per_class_f1]
    ax3.barh(class_names, per_class_f1, color=colors, alpha=0.8)
    ax3.axvline(0.90, color='green', linestyle='--', linewidth=1.5, label='90% target')
    ax3.set_xlabel('F1-Score'); ax3.set_title('Per-Class F1-Score', fontweight='bold')
    ax3.set_xlim(0, 1); ax3.legend(fontsize=8); ax3.grid(axis='x', alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(latencies, bins=30, color='steelblue', alpha=0.8, edgecolor='white')
    ax4.axvline(latencies.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean={latencies.mean():.4f} ms')
    ax4.axvline(np.percentile(latencies, 99), color='orange', linestyle=':',
                linewidth=2, label=f'P99={np.percentile(latencies,99):.4f} ms')
    ax4.set_xlabel('Latency (ms)'); ax4.set_ylabel('Count')
    ax4.set_title('Ridge Readout Latency Distribution', fontweight='bold')
    ax4.legend(fontsize=8); ax4.grid(alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 2])
    methods = ['Photonic RC\n(ours)', 'CNN-LSTM', 'Random\nForest', 'XGBoost', 'Target']
    acc_vals = [acc, 91.23, 88.50, 90.10, 94.93]
    bar_colors = ['steelblue', 'gray', 'gray', 'gray', 'green']
    bars = ax5.bar(methods, acc_vals, color=bar_colors, alpha=0.8, edgecolor='white')
    ax5.set_ylim(80, 100); ax5.set_ylabel('Accuracy (%)')
    ax5.set_title('Accuracy vs. Baselines', fontweight='bold')
    for bar, val in zip(bars, acc_vals):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax5.axhline(94.93, color='green', linestyle='--', linewidth=1, alpha=0.5)
    ax5.grid(axis='y', alpha=0.3)

    fig.suptitle('E4: All-Optical Photonic RC-IDS -- Full Performance Evaluation',
                 fontsize=13, fontweight='bold')
    plt.savefig('E4_full_eval.png', dpi=150, bbox_inches='tight')
    print("\n  Plot saved: E4_full_eval.png")
    plt.show()


if __name__ == '__main__':
    main()
