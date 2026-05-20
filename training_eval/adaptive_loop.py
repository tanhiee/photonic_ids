"""
training_eval/adaptive_loop.py
==============================
Adaptive Zero-Day detection feedback loop.
DBSCAN clustering on 64-D reservoir states + Ridge hot-swap retraining.
"""
from __future__ import annotations
import logging
from collections import deque
from typing import Deque, Optional, List
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier

from config import FeedbackConfig

log = logging.getLogger(__name__)


class AdaptiveFeedbackLoop:
    """
    Zero-Day adaptive feedback loop.

    Buffers low-confidence samples, clusters them with DBSCAN in PCA-reduced
    64-D reservoir state space, and hot-swaps the Ridge readout when novel
    attack clusters exceed the trigger size.

    Parameters
    ----------
    cfg      : FeedbackConfig
    readout  : RidgeReadout
    base_X   : np.ndarray or None  — base training reservoir states (anti-forgetting)
    base_y   : np.ndarray or None
    """

    def __init__(
        self,
        cfg:     FeedbackConfig = None,
        readout  = None,
        base_X:  Optional[np.ndarray] = None,
        base_y:  Optional[np.ndarray] = None,
    ) -> None:
        self.cfg    = cfg or FeedbackConfig()
        self.readout = readout
        self.base_X  = base_X
        self.base_y  = base_y
        self.buffer: Deque = deque(maxlen=self.cfg.buffer_max_size)
        self.n_retrains = 0
        self.n_novel_clusters = 0
        self._next_label = (
            len(np.unique(base_y)) if base_y is not None else 8
        )

    def push_sample(
        self,
        state_vector:     np.ndarray,
        optical_features: np.ndarray,
        true_label:       Optional[int],
        confidence:       float = 0.0,
    ) -> bool:
        """Add sample to buffer. Returns True if hot-swap was triggered."""
        self.buffer.append({
            'state':      state_vector.astype(np.float32),
            'optical':    optical_features.astype(np.float32),
            'label':      true_label,
            'confidence': confidence,
        })
        if len(self.buffer) >= self.cfg.buffer_max_size:
            return self._cluster_and_retrain()
        return False

    def _cluster_and_retrain(self) -> bool:
        """Run DBSCAN, detect novel clusters, and retrain if needed."""
        states = np.vstack([s['state'] for s in self.buffer])

        n_comp = min(self.cfg.pca_components_feedback, states.shape[0]-1, states.shape[1])
        pca    = PCA(n_components=n_comp, random_state=42)
        states_r = pca.fit_transform(states)

        db = DBSCAN(eps=self.cfg.dbscan_eps,
                    min_samples=self.cfg.dbscan_min_samples, n_jobs=-1)
        labels = db.fit_predict(states_r)
        unique = set(labels) - {-1}
        log.info('DBSCAN: %d clusters, %d noise, buffer=%d',
                 len(unique), int(np.sum(labels==-1)), len(self.buffer))

        X_novel: List[np.ndarray] = []
        y_novel: List[int]         = []
        for cid in sorted(unique):
            mask = labels == cid
            if mask.sum() >= self.cfg.retrain_trigger_size:
                self.n_novel_clusters += 1
                pseudo = self._next_label
                self._next_label += 1
                X_novel.append(states[mask])
                y_novel.extend([pseudo] * int(mask.sum()))
                log.info('Novel cluster %d: %d samples -> pseudo-label %d',
                         cid, int(mask.sum()), pseudo)

        self.buffer.clear()
        if not X_novel:
            return False

        X_n = np.vstack(X_novel)
        y_n = np.array(y_novel, dtype=np.int64)

        if self.base_X is not None:
            X_aug = np.vstack([self.base_X, X_n])
            y_aug = np.concatenate([self.base_y, y_n])
        else:
            X_aug, y_aug = X_n, y_n

        clf = RidgeClassifier(alpha=self.cfg.hot_swap_alpha, fit_intercept=True)
        clf.fit(X_aug, y_aug)
        if self.readout is not None:
            self.readout.set_weights({'coef': clf.coef_,
                                       'intercept': clf.intercept_,
                                       'classes': clf.classes_})
        self.n_retrains += 1
        log.info('Hot-swap #%d complete. Now %d classes.', self.n_retrains, len(clf.classes_))
        return True

    def get_diagnostics(self) -> dict:
        """Return feedback loop state summary."""
        confs = [s['confidence'] for s in self.buffer]
        return {
            'buffer_size':        len(self.buffer),
            'n_retrains':         self.n_retrains,
            'n_novel_clusters':   self.n_novel_clusters,
            'next_novel_label':   self._next_label,
            'mean_confidence':    float(np.mean(confs)) if confs else 0.0,
        }
