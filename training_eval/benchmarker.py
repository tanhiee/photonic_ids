"""
training_eval/benchmarker.py
============================
Throughput, latency, and memory profiler for the photonic RC-IDS.
"""
from __future__ import annotations
import time
import psutil
import os
import numpy as np
from typing import Callable


def measure_throughput(
    fn: Callable,
    args: tuple,
    n_repeats: int = 10,
) -> dict:
    """
    Measure function throughput and latency.

    Parameters
    ----------
    fn       : callable to benchmark
    args     : positional args to fn
    n_repeats: number of timed calls

    Returns
    -------
    dict with keys: mean_ms, std_ms, min_ms, max_ms, throughput_per_s.
    """
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        result = fn(*args)
        times.append(time.perf_counter() - t0)
    times = np.array(times) * 1000.0  # ms
    n_samples = args[0].shape[0] if hasattr(args[0], 'shape') else 1
    return {
        'mean_ms':          float(np.mean(times)),
        'std_ms':           float(np.std(times)),
        'min_ms':           float(np.min(times)),
        'max_ms':           float(np.max(times)),
        'throughput_per_s': float(n_samples / (np.mean(times) / 1000.0)),
    }


def memory_usage_mb() -> float:
    """Return current process RSS memory usage [MB]."""
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / 1024 ** 2
