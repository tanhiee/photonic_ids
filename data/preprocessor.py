"""
data/preprocessor.py
====================
Stage 1 preprocessing: Raw CICIDS2017 -> MZM optical amplitudes.

Pipeline:
    X_raw [N, 79]
      -> MinMaxScaler [0, 1]
      -> PCA (79 -> 16, whiten)
      -> per-component MinMax re-normalise [0, 1]
      -> MZM encode: u = sin(pi/2 * x)
      -> X_optical [N, 16]  in [0, 1]
"""
from __future__ import annotations
import logging
import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from typing import Optional, Tuple

from config import PreprocessorConfig, DataConfig, SystemConfig

log = logging.getLogger(__name__)


class NetworkTrafficPreprocessor:
    """
    Stage 1 preprocessor: CSV/Mock -> Clean -> Scale -> PCA -> MZM -> Split.

    Parameters
    ----------
    cfg : SystemConfig
        Master system configuration.
    """

    def __init__(self, cfg: SystemConfig = None) -> None:
        from config import DEFAULT_CONFIG
        self.cfg = cfg or DEFAULT_CONFIG
        pcfg = self.cfg.preprocessor
        self.scaler        = MinMaxScaler(feature_range=pcfg.scaler_range)
        self.pca           = PCA(
            n_components=pcfg.n_pca_components,
            whiten=pcfg.pca_whiten,
            random_state=self.cfg.data.random_seed,
        )
        self.label_encoder = LabelEncoder()
        self._pca_min: Optional[np.ndarray] = None
        self._pca_max: Optional[np.ndarray] = None
        self._X_train_optical: Optional[np.ndarray] = None
        self._X_test_optical:  Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._y_test:  Optional[np.ndarray] = None

    def fit_transform(
        self,
        X_raw: np.ndarray,
        y_raw: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
        """
        Fit the pipeline and split into train/test optical arrays.

        Returns
        -------
        (X_train_optical, X_test_optical, y_train, y_test, label_encoder)
        """
        dcfg = self.cfg.data
        y_enc = self.label_encoder.fit_transform(y_raw)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_raw, y_enc,
            test_size=(1.0 - dcfg.train_ratio),
            random_state=dcfg.random_seed,
            stratify=y_enc,
        )

        X_tr_scaled = self.scaler.fit_transform(X_tr.astype(np.float64))
        X_te_scaled = self.scaler.transform(X_te.astype(np.float64))

        X_tr_pca = self.pca.fit_transform(X_tr_scaled)
        X_te_pca = self.pca.transform(X_te_scaled)

        X_tr_norm = self._pca_minmax(X_tr_pca, fit=True)
        X_te_norm = self._pca_minmax(X_te_pca, fit=False)

        X_tr_opt = self._mzm_encode(X_tr_norm)
        X_te_opt = self._mzm_encode(X_te_norm)

        self._X_train_optical = X_tr_opt
        self._X_test_optical  = X_te_opt
        self._y_train = y_tr
        self._y_test  = y_te

        explained = self.pca.explained_variance_ratio_.sum() * 100
        log.info('PCA variance explained: %.1f%%  Train: %d  Test: %d',
                 explained, len(y_tr), len(y_te))

        return X_tr_opt, X_te_opt, y_tr, y_te, self.label_encoder

    def transform(self, X_raw: np.ndarray) -> np.ndarray:
        """Transform new raw features through the fitted pipeline."""
        X_scaled = self.scaler.transform(np.atleast_2d(X_raw).astype(np.float64))
        X_pca    = self.pca.transform(X_scaled)
        X_norm   = self._pca_minmax(X_pca, fit=False)
        return self._mzm_encode(X_norm)

    def get_split_optical(self):
        """Return cached (X_train_opt, X_test_opt, y_train, y_test)."""
        if self._X_train_optical is None:
            raise RuntimeError('Call fit_transform() first.')
        return self._X_train_optical, self._X_test_optical, self._y_train, self._y_test

    def _pca_minmax(self, X: np.ndarray, fit: bool) -> np.ndarray:
        if fit:
            self._pca_min = X.min(axis=0)
            self._pca_max = X.max(axis=0)
        denom = np.where(self._pca_max - self._pca_min == 0, 1.0,
                         self._pca_max - self._pca_min)
        return np.clip((X - self._pca_min) / denom, 0.0, 1.0)

    @staticmethod
    def _mzm_encode(X: np.ndarray) -> np.ndarray:
        """MZM: u(t) = sin(pi/2 * x(t)),  x in [0,1] -> u in [0,1]."""
        return np.sin(np.pi / 2.0 * X).astype(np.float32)
