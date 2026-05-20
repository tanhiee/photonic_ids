"""
training_eval/metrics.py
========================
Classification metrics tracker: Accuracy, F1, ROC-AUC, latency.
"""
from __future__ import annotations
import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ClassificationMetrics:
    """Container for a single evaluation result."""
    accuracy:     float           = 0.0
    f1_weighted:  float           = 0.0
    roc_auc:      Optional[float] = None
    latency_ms:   float           = 0.0
    n_samples:    int             = 0
    elapsed_s:    float           = 0.0

    def __str__(self) -> str:
        lines = [
            f'  Accuracy  : {self.accuracy*100:.3f} %',
            f'  F1-Score  : {self.f1_weighted*100:.3f} %  (weighted)',
        ]
        if self.roc_auc is not None:
            lines.append(f'  ROC-AUC   : {self.roc_auc:.4f}  (macro OvR)')
        lines.append(f'  Latency   : {self.latency_ms:.3f} ms/sample')
        lines.append(f'  Elapsed   : {self.elapsed_s:.2f} s  ({self.n_samples:,} samples)')
        return '\n'.join(lines)


def evaluate(
    readout,
    X: np.ndarray,
    y: np.ndarray,
) -> ClassificationMetrics:
    """Compute full metrics suite using a fitted RidgeReadout."""
    t0 = time.perf_counter()
    acc, f1, auc = readout.evaluate(X, y)
    elapsed = time.perf_counter() - t0
    return ClassificationMetrics(
        accuracy    = acc,
        f1_weighted = f1,
        roc_auc     = auc,
        latency_ms  = elapsed * 1000.0 / max(len(X), 1),
        n_samples   = len(X),
        elapsed_s   = elapsed,
    )
