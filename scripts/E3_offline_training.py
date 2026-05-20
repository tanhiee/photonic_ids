"""
E3_offline_training.py — Milestone 3: Full Offline Training Pipeline
=====================================================================
Runs the complete offline training phase:
  1. Load CICIDS2017 (or mock data)
  2. Preprocess: MinMaxScaler -> PCA(79->16) -> MZM
  3. Photonic Reservoir: 8x8 MRR mesh forward pass
  4. Ridge Readout: fit and evaluate
  5. Print metrics: Accuracy, F1, ROC-AUC

Usage:
    python scripts/E3_offline_training.py
    python scripts/E3_offline_training.py --data path/to/CICIDS2017.csv
"""
import sys, os, argparse, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%H:%M:%S')

from config import SystemConfig
from training_eval.offline_trainer import OfflineTrainer


def main():
    parser = argparse.ArgumentParser(description='E3: Offline Training')
    parser.add_argument('--data', type=str, default=None, help='CICIDS2017 CSV path')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    cfg = SystemConfig()
    cfg.data.random_seed = args.seed

    print('\n=== E3: All-Optical Photonic RC-IDS — Offline Training ===')
    trainer = OfflineTrainer(cfg)
    metrics, _ = trainer.run(dataset_path=args.data)

    print('\n╔══════════════════════════════════════╗')
    print('║  PHOTONIC RC-IDS — OFFLINE RESULTS   ║')
    print('╠══════════════════════════════════════╣')
    print(str(metrics))
    print('╚══════════════════════════════════════╝')


if __name__ == '__main__':
    main()
