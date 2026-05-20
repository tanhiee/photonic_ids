"""
data/cicids_loader.py
=====================
Synthetic generator and loader for CICIDS-like network traffic data.
Provides generate_synthetic_cicids() to generate a 16-D feature matrix for 14 classes.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple

CLASSES = [
    "Benign", "DDoS", "DoS-Hulk", "DoS-GoldenEye", "DoS-slowloris",
    "DoS-Slowhttptest", "PortScan", "FTP-Patator", "SSH-Patator", "Bot",
    "Web-XSS", "Web-Brute", "Web-Injection", "Infiltration"
]

def generate_synthetic_cicids(random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a synthetic 16-D traffic dataset with 21400 samples and 14 classes.
    'Infiltration' has exactly 200 samples and is used for Zero-Day testing.
    """
    rng = np.random.default_rng(random_seed)
    n_total = 21400
    n_infil = 200
    n_others = n_total - n_infil  # 21200
    
    X = np.zeros((n_total, 16))
    y = []
    
    # 1. Generate Infiltration flows (exactly 200)
    X_infil = rng.normal(0.2, 0.1, size=(n_infil, 16))
    X[0:n_infil] = X_infil
    y.extend(["Infiltration"] * n_infil)
    
    # 2. Generate other classes (21200 samples)
    # Distribute them among other 13 classes
    n_per_class = n_others // 13  # ~1630 samples per class
    curr_idx = n_infil
    
    for c_idx, c_name in enumerate(CLASSES[:-1]):
        count = n_per_class
        if c_idx == 12:  # balance remainder
            count += n_others % 13
            
        # Class-specific mean shift to make classification realistic
        mean_shift = c_idx * 0.05
        X_class = rng.normal(0.4 + mean_shift, 0.15, size=(count, 16))
        X[curr_idx:curr_idx+count] = X_class
        y.extend([c_name] * count)
        curr_idx += count
        
    return np.clip(X, 0.0, 1.0), np.array(y)

def get_train_test_split(
    X: np.ndarray, y: np.ndarray, train_ratio: float = 0.80, random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits dataset into Train/Test while holding out 'Infiltration' as zero-day.
    Returns:
        X_train, y_train, X_test, y_test, X_zero_day, y_zero_day
    """
    # Isolate Infiltration
    infil_mask = (y == "Infiltration")
    X_zero_day = X[infil_mask]
    y_zero_day = y[infil_mask]
    
    X_rest = X[~infil_mask]
    y_rest = y[~infil_mask]
    
    # Stratified or simple random split on rest
    rng = np.random.default_rng(random_seed)
    indices = rng.permutation(len(X_rest))
    split_idx = int(len(X_rest) * train_ratio)
    
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    X_train = X_rest[train_idx]
    y_train = y_rest[train_idx]
    X_test = X_rest[test_idx]
    y_test = y_rest[test_idx]
    
    return X_train, y_train, X_test, y_test, X_zero_day, y_zero_day
