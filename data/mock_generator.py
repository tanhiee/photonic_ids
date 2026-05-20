"""
data/mock_generator.py
======================
Generate a synthetic CICIDS2017-like DataFrame for testing and CI.
Produces class-separated Gaussian clusters in 79-dimensional feature space.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional

CANONICAL_CLASSES = [
    'Benign', 'DDoS', 'DoS_Hulk', 'DoS_GoldenEye',
    'PortScan', 'FTP_Patator', 'Web_Attack', 'Bot',
]


def generate_mock_cicids(
    n_samples:  int = 5_000,
    n_features: int = 79,
    n_classes:  int = 8,
    random_seed: int = 42,
    class_imbalance: bool = True,
) -> pd.DataFrame:
    """
    Generate a synthetic CICIDS2017-like DataFrame.

    Parameters
    ----------
    n_samples : int
        Total number of rows.
    n_features : int
        Number of numeric feature columns (default 79 = CICIDS2017).
    n_classes : int
        Number of distinct traffic classes (max 8).
    random_seed : int
        NumPy random seed for reproducibility.
    class_imbalance : bool
        If True, Benign class gets 60% of samples (realistic imbalance).

    Returns
    -------
    pd.DataFrame
        Columns: Feature_0..Feature_{n_features-1}, Label.
    """
    rng = np.random.default_rng(random_seed)
    n_cls = min(n_classes, len(CANONICAL_CLASSES))
    classes = CANONICAL_CLASSES[:n_cls]

    if class_imbalance:
        weights = np.ones(n_cls) * (0.4 / max(n_cls - 1, 1))
        weights[0] = 0.60  # Benign majority
    else:
        weights = np.ones(n_cls) / n_cls

    counts = (weights * n_samples).astype(int)
    counts[-1] += n_samples - counts.sum()  # fix rounding

    rows = []
    for cls_idx, (cls_name, cnt) in enumerate(zip(classes, counts)):
        mean = rng.uniform(0.1, 0.9, size=n_features) * (cls_idx + 1) / n_cls
        cov  = np.eye(n_features) * 0.03
        data = np.clip(
            rng.multivariate_normal(mean, cov, size=cnt), 0.0, None
        )
        for row in data:
            record = {f'Feature_{i}': v for i, v in enumerate(row)}
            record['Label'] = cls_name
            rows.append(record)

    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
