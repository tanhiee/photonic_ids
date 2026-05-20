#!/usr/bin/env bash
# ==============================================================================
#   Photonic IDS Simulation — Master Runner Script
#   Target: Unix Bash / Git Bash on Windows
# ==============================================================================

# Ensure logs directory exists
mkdir -p logs

LOG_FILE="logs/run_$(date +%Y%m%d_%H%M%S).log"
MODE=${1:-"full"}

echo "══════════════════════════════════════════════════════════════════════════════"
echo "  Photonic IDS Simulation  —  Master Runner"
echo "  Python: python3.12 (3.12)"
echo "  Log   : $LOG_FILE"
echo "  Chế độ: $MODE"
echo "══════════════════════════════════════════════════════════════════════════════"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "══════════════════════════════════════════════════════════════════════════════"
echo "▶  Unit tests"
echo "══════════════════════════════════════════════════════════════════════════════"
python -m unittest tests/test_mrr_dynamics.py -v
echo "✅  Unit tests  — OK"
echo

# Helper function to run scripts and format output
run_step() {
    local step_num=$1
    local name=$2
    local script_path=$3
    
    echo "══════════════════════════════════════════════════════════════════════════════"
    echo "▶  $step_num — $name"
    echo "══════════════════════════════════════════════════════════════════════════════"
    python "$script_path"
    echo "✅  $step_num — $name  — OK"
    echo
}

run_step "01" "Single MRR" "scripts/01_simulate_single_mrr.py"
run_step "02" "Reservoir array" "scripts/02_simulate_reservoir.py"
run_step "03" "Offline train" "scripts/03_train_offline.py"
run_step "04" "Evaluate" "scripts/04_evaluate.py"
run_step "05" "Ablation" "scripts/05_ablation_study.py"
run_step "06" "Zero-Day" "scripts/06_zero_day_test.py"
run_step "07" "Capacity" "scripts/07_computational_capacity.py"
run_step "08" "Deployment" "scripts/08_deployment_economics.py"
run_step "09" "Adversarial" "scripts/09_adversarial_robustness.py"

echo "══════════════════════════════════════════════════════════════════════════════"
echo "  🎉  Hoàn thành — log đầy đủ: $LOG_FILE"
echo "══════════════════════════════════════════════════════════════════════════════"
