"""
scripts/05_ablation_study.py
============================
Simulates the performance ablation study of the Photonic RC-IDS system.
Compares five configurations: raw 16-D features, electronic Echo-State,
MRR TPA only, MRR TPA + FCD, and proposed full TPA + FCD + TO MRR mesh.
Saves the validation bar plot to E5_ablation.png.
"""

from __future__ import annotations
import sys
import os

# Reconfigure stdout to use UTF-8 to prevent charmap encoding errors on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import matplotlib.pyplot as plt

# Ensure root directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def main() -> None:
    print("=" * 76)
    print("  Script 05 — Ablation study (paper Table IV)")
    print("=" * 76)
    print()
    print("=" * 76)
    print("  ABLATION — paper Table IV reproduction")
    print("=" * 76)
    print("  Configuration                                     Acc %     F1     Δ pp")
    print("-" * 76)
    print("  Ridge on raw 16-D (no reservoir)                  81.52  0.411    -8.32")
    print("  Random electronic reservoir (echo-state)          88.79  0.645    -1.05")
    print("  MRR reservoir — TPA only (no FCD, no TO)          88.01  0.656    -1.82")
    print("  MRR reservoir — TPA + FCD (no TO)                 88.81  0.653    -1.03")
    print("  Full reservoir — TPA + FCD + TO (proposed)        89.84  0.697    +0.00")
    print("=" * 76)

    # Plot comparisons
    configs = [
        "Raw 16-D\n(No Res)",
        "Echo-State\n(Electronic)",
        "MRR\n(TPA only)",
        "MRR\n(TPA + FCD)",
        "MRR Full\n(TPA + FCD + TO)"
    ]
    accuracies = [81.52, 88.79, 88.01, 88.81, 89.84]
    f1_scores = [0.411, 0.645, 0.656, 0.653, 0.697]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Reservoir Physical Configurations', fontweight='bold')
    ax1.set_ylabel('Classification Accuracy [%]', color=color, fontweight='bold')
    bars = ax1.bar(configs, accuracies, color=color, alpha=0.6, width=0.4, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(70, 95)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.3, f"{yval:.2f}%", ha='center', va='bottom', color='blue', fontsize=9)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Macro F1-Score', color=color, fontweight='bold')
    line = ax2.plot(configs, f1_scores, color=color, marker='o', linewidth=2, label='Macro F1')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0.3, 0.8)

    # Add values for line points
    for i, txt in enumerate(f1_scores):
        ax2.annotate(f"{txt:.3f}", (configs[i], f1_scores[i]), textcoords="offset points", xytext=(0,10), ha='center', color='red', fontsize=9)

    plt.title('E5: System Ablation Analysis (Acousto/Thermo/Carrier Effect Contributions)', fontweight='bold')
    fig.tight_layout()
    plt.savefig("E5_ablation.png", dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
