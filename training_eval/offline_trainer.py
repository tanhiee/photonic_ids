"""
training_eval/offline_trainer.py
================================
Phase 1: Offline batch training pipeline.
Loads data -> preprocesses -> runs reservoir -> trains Ridge readout.
"""
from __future__ import annotations
import logging
import time
import numpy as np
from typing import Optional, Tuple

from config import SystemConfig, DEFAULT_CONFIG
from data.cicids_loader import CICIDS2017Loader
from data.preprocessor import NetworkTrafficPreprocessor
from reservoir.mrr_mesh import PhotonicMesh
from reservoir.readout import RidgeReadout
from training_eval.metrics import ClassificationMetrics, evaluate

log = logging.getLogger(__name__)


class OfflineTrainer:
    """
    Orchestrates the full offline training pipeline:
        Load -> Preprocess -> Reservoir -> Ridge Fit -> Evaluate

    Parameters
    ----------
    cfg : SystemConfig
    """

    def __init__(self, cfg: SystemConfig = None) -> None:
        self.cfg          = cfg or DEFAULT_CONFIG
        self.loader       = CICIDS2017Loader(cfg.data)
        self.preprocessor = NetworkTrafficPreprocessor(cfg)
        self.mesh:    Optional[PhotonicMesh] = None
        self.readout: Optional[RidgeReadout] = None
        self._X_train_states: Optional[np.ndarray] = None
        self._X_test_states:  Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._y_test:  Optional[np.ndarray] = None

    def run(
        self,
        dataset_path: Optional[str] = None,
    ) -> Tuple[ClassificationMetrics, 'OfflineTrainer']:
        """
        Execute the full offline training and evaluation pipeline.

        Returns
        -------
        (metrics, self)
        """
        t_start = time.perf_counter()

        # Stage 1: Load
        log.info('--- Stage 1: Loading data ---')
        X_raw, y_raw = self.loader.load(dataset_path)

        # Stage 2: Preprocess
        log.info('--- Stage 2: Preprocessing ---')
        X_tr_opt, X_te_opt, y_tr, y_te, le = self.preprocessor.fit_transform(X_raw, y_raw)
        self._y_train = y_tr
        self._y_test  = y_te

        # Stage 3: Photonic Reservoir
        log.info('--- Stage 3: Photonic Reservoir (8x8 MRR mesh) ---')
        self.mesh = PhotonicMesh(self.cfg.reservoir)
        self._X_train_states = self.mesh.forward(X_tr_opt)
        self._X_test_states  = self.mesh.forward(X_te_opt)
        log.info('Reservoir states: train=%s, test=%s',
                 self._X_train_states.shape, self._X_test_states.shape)

        # Stage 4: Ridge Readout
        log.info('--- Stage 4: Ridge Readout (alpha=%.0e) ---', self.cfg.readout.alpha)
        self.readout = RidgeReadout(
            alpha=self.cfg.readout.alpha,
            temperature=self.cfg.readout.softmax_temperature,
        )
        self.readout.fit(self._X_train_states, y_tr)

        # Evaluation
        log.info('--- Evaluation ---')
        metrics = evaluate(self.readout, self._X_test_states, y_te)
        metrics.elapsed_s = time.perf_counter() - t_start
        log.info('\n%s', metrics)
        return metrics, self
