"""
training_eval/online_streamer.py
================================
Phase 2: Online sample-by-sample streaming with adaptive feedback.
"""
from __future__ import annotations
import logging
import time
import numpy as np
from typing import Optional

log = logging.getLogger(__name__)


class OnlineStreamer:
    """
    Simulates real-time streaming inference with the photonic RC-IDS.

    Parameters
    ----------
    mesh    : PhotonicMesh  — reservoir forward pass
    readout : RidgeReadout  — inference layer
    feedback: AdaptiveFeedbackLoop — zero-day detection
    cfg     : SystemConfig
    """

    def __init__(self, mesh, readout, feedback, cfg=None) -> None:
        from config import DEFAULT_CONFIG
        self.mesh     = mesh
        self.readout  = readout
        self.feedback = feedback
        self.cfg      = cfg or DEFAULT_CONFIG

    def stream(
        self,
        X_optical: np.ndarray,
        y_true:    np.ndarray,
        n_samples: int = 300,
    ) -> dict:
        """
        Stream n_samples through the full pipeline.

        Returns
        -------
        dict with keys: accuracy, n_buffered, n_retrains, throughput_samples_per_s.
        """
        stream_size = min(n_samples, len(X_optical))
        n_correct  = 0
        n_buffered = 0
        n_retrains = 0
        threshold  = self.cfg.readout.confidence_threshold

        t0 = time.perf_counter()
        for i in range(stream_size):
            x_opt   = X_optical[i:i+1]           # [1, 16]
            state   = self.mesh.forward(x_opt)   # [1, 64]
            probs   = self.readout.predict_proba(state)  # [1, C]
            pred    = np.argmax(probs, axis=1)[0]
            conf    = float(np.max(probs))

            if pred == y_true[i]:
                n_correct += 1

            if conf < threshold:
                n_buffered += 1
                retrained = self.feedback.push_sample(
                    state_vector     = state[0],
                    optical_features = x_opt[0],
                    true_label       = None,
                    confidence       = conf,
                )
                if retrained:
                    n_retrains += 1
                    log.info('Hot-swap at sample %d (retrain #%d)', i+1, n_retrains)

        elapsed    = time.perf_counter() - t0
        throughput = stream_size / elapsed if elapsed > 0 else 0.0
        accuracy   = n_correct / stream_size if stream_size > 0 else 0.0

        return {
            'accuracy':   accuracy,
            'n_buffered': n_buffered,
            'n_retrains': n_retrains,
            'throughput_samples_per_s': throughput,
            'elapsed_s':  elapsed,
        }
