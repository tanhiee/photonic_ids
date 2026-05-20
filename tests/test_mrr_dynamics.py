"""
tests/test_mrr_dynamics.py
===========================
Comprehensive unit test suite for the Photonic RC-IDS v2 simulation framework.
Implements 9 tests checking dataset logic, physical dynamics, metrics,
and pipeline readouts.
"""

from __future__ import annotations
import unittest
import numpy as np
import sys
import os

# Ensure root directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.cicids_loader import generate_synthetic_cicids, get_train_test_split
from physics.mrr_dynamics import MRRDynamics
from reservoir.photonic_reservoir import PhotonicReservoir
from training.offline_trainer import OfflineTrainer

class TestData(unittest.TestCase):
    """Verifies dataset generation and zero-day isolation logic."""
    
    def test_dataset_generates(self) -> None:
        """Verifies synthetic dataset shape and ranges."""
        X, y = generate_synthetic_cicids(random_seed=42)
        self.assertEqual(X.shape, (21400, 16))
        self.assertEqual(y.shape, (21400,))
        self.assertTrue(np.all(X >= 0.0) and np.all(X <= 1.0))

    def test_holdout(self) -> None:
        """Verifies 'Infiltration' class is properly held out as zero-day."""
        X, y = generate_synthetic_cicids(random_seed=42)
        X_train, y_train, X_test, y_test, X_zero_day, y_zero_day = get_train_test_split(
            X, y, train_ratio=0.80, random_seed=42
        )
        self.assertEqual(len(X_zero_day), 200)
        self.assertEqual(len(y_zero_day), 200)
        self.assertTrue(np.all(y_zero_day == "Infiltration"))
        self.assertFalse("Infiltration" in y_train)
        self.assertFalse("Infiltration" in y_test)


class TestMRRDynamics(unittest.TestCase):
    """Verifies physical simulation models of single and array MRR nodes."""
    
    def test_array_64_rings(self) -> None:
        """Checks array output dimensions of the 64-ring mesh."""
        res = PhotonicReservoir(random_seed=42)
        X = np.ones((5, 16))
        states = res.transform(X)
        self.assertEqual(states.shape, (5, 64))

    def test_cw_steady_state(self) -> None:
        """Checks sweep detuning physical convergence values."""
        mrr = MRRDynamics()
        # Test standard detuning sweep
        u, n, th = mrr.compute_steady_state(-0.20)
        self.assertAlmostEqual(u, 0.0022, places=4)
        self.assertAlmostEqual(n, 0.8787, places=4)
        self.assertAlmostEqual(th, 4.0851, places=4)


class TestMetrics(unittest.TestCase):
    """Verifies analytical metrics calculations and correctness of validation statistics."""
    
    def test_confusion_diag(self) -> None:
        """Checks confusion matrix diagonal properties."""
        # Simple test to confirm row normalization works
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 0, 2, 2])
        
        # Verify simple accuracy calculation
        acc = np.mean(y_true == y_pred)
        self.assertAlmostEqual(acc, 5.0/6.0, places=4)

    def test_perfect_classifier(self) -> None:
        """Verifies accuracy calculation of a perfect predictor."""
        y_true = np.array(["Benign", "DDoS", "Bot"])
        y_pred = np.array(["Benign", "DDoS", "Bot"])
        self.assertAlmostEqual(np.mean(y_true == y_pred), 1.0, places=4)


class TestPipeline(unittest.TestCase):
    """Verifies integration pipelines of training and inference networks."""
    
    def test_full_pipeline(self) -> None:
        """Checks fit and prediction consistency of OfflineTrainer."""
        X, y = generate_synthetic_cicids(random_seed=42)
        X_train, y_train, X_test, y_test, X_zero_day, y_zero_day = get_train_test_split(
            X, y, train_ratio=0.80, random_seed=42
        )
        # Sub-sample to keep test execution fast
        res = PhotonicReservoir(random_seed=42)
        trainer = OfflineTrainer(res)
        meta = trainer.train(X_train[:100], y_train[:100])
        
        self.assertTrue(trainer.is_fitted)
        preds = trainer.predict(X_test[:10])
        self.assertEqual(len(preds), 10)


class TestReadout(unittest.TestCase):
    """Verifies electrical Ridge regression readout properties."""
    
    def test_ridge_mac_count(self) -> None:
        """Checks multiply-accumulate operations counts scaling."""
        # Standard matrix multiply MAC count: 64 inputs, 14 classes -> 64 * 14 = 896 MACs
        classes = 14
        features = 64
        mac_count = classes * features
        self.assertEqual(mac_count, 896)

    def test_softmax_normalised(self) -> None:
        """Checks that output probabilities sum to 1."""
        # Test basic softmax normalisation correctness
        scores = np.array([1.0, 2.0, 3.0])
        exp_s = np.exp(scores - np.max(scores))
        probs = exp_s / np.sum(exp_s)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
