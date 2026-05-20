"""
reservoir/readout.py
====================
Ridge Regression readout layer for the Photonic RC-IDS.

Only the readout is trained (the reservoir is fixed).
Uses sklearn RidgeClassifier with closed-form L2 solution.
"""
from __future__ import annotations
import logging
from typing import Optional, Tuple
import numpy as np
from scipy.special import softmax as scipy_softmax
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

log = logging.getLogger(__name__)


class RidgeReadout:
    """
    Ridge Classifier readout with Softmax probability output.

    Parameters
    ----------
    alpha : float   Ridge L2 regularisation (default 1e-3).
    temperature : float  Softmax temperature.
    """

    def __init__(self, alpha: float = 1e-3, temperature: float = 1.0) -> None:
        self.alpha = alpha
        self.tau   = temperature
        self.clf   = RidgeClassifier(alpha=alpha, fit_intercept=True)
        self.classes_: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RidgeReadout':
        """Train Ridge on reservoir state vectors."""
        self.clf.fit(X, y)
        self.classes_  = self.clf.classes_
        self.is_fitted = True
        log.info('RidgeReadout fitted: X=%s, classes=%s', X.shape, self.classes_)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return Softmax class probabilities [N, n_classes]."""
        if not self.is_fitted:
            raise RuntimeError('Call fit() first.')
        dec = self.clf.decision_function(X)
        if dec.ndim == 1:
            dec = np.column_stack([-dec, dec])
        return scipy_softmax(dec / max(self.tau, 1e-6), axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard class labels."""
        if not self.is_fitted:
            raise RuntimeError('Call fit() first.')
        return self.clf.predict(X)

    def evaluate(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[float, float, Optional[float]]:
        """Return (accuracy, f1_weighted, roc_auc_macro)."""
        y_pred   = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        f1       = f1_score(y, y_pred, average='weighted', zero_division=0)
        roc_auc  = None
        try:
            proba  = self.predict_proba(X)
            y_bin  = label_binarize(y, classes=self.classes_)
            n_cls  = len(self.classes_)
            if n_cls == 2:
                roc_auc = roc_auc_score(y_bin, proba[:, 1])
            elif n_cls > 2:
                roc_auc = roc_auc_score(y_bin, proba, multi_class='ovr', average='macro')
        except Exception as e:
            log.warning('ROC-AUC failed: %s', e)
        return accuracy, f1, roc_auc

    def get_weights(self) -> dict:
        """Export model weights for hot-swap."""
        return {'coef': self.clf.coef_.copy(),
                'intercept': self.clf.intercept_.copy(),
                'classes': self.classes_.copy()}

    def set_weights(self, w: dict) -> None:
        """Hot-swap weights from a retrained model."""
        self.clf.coef_      = w['coef']
        self.clf.intercept_ = w['intercept']
        self.classes_       = w['classes']
        self.clf.classes_   = w['classes']
        self.is_fitted      = True
        log.info('RidgeReadout weights hot-swapped. New shape: %s', w['coef'].shape)
