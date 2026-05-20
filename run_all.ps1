# ==============================================================================
#   Photonic IDS Simulation — Windows PowerShell Master Runner
# ==============================================================================

# Force UTF-8 encoding for Python outputs to prevent Cp1252 charmap encoding errors
$env:PYTHONIOENCODING = "utf-8"

# Ensure logs directory exists
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = "logs/run_$timestamp.log"

Write-Output "======================================================================"
Write-Output "  Photonic IDS Simulation  -  Master Runner (PowerShell)"
Write-Output "  Python: python (3.12)"
Write-Output "  Log   : $logFile"
Write-Output "  Mode  : full"
Write-Output "======================================================================"

# Start transcript to capture all outputs to log
Start-Transcript -Path $logFile -Append -Force | Out-Null

Write-Output "======================================================================"
Write-Output "  Unit tests"
Write-Output "======================================================================"
python tests/test_mrr_dynamics.py -v
Write-Output "  Unit tests  - OK`n"

function Run-Step {
    param (
        [string]$StepNum,
        [string]$Name,
        [string]$ScriptPath
    )
    Write-Output "======================================================================"
    Write-Output "  $StepNum - $Name"
    Write-Output "======================================================================"
    python $ScriptPath
    Write-Output "  $StepNum - $Name  - OK`n"
}

Run-Step -StepNum "01" -Name "Single MRR" -ScriptPath "scripts/01_simulate_single_mrr.py"
Run-Step -StepNum "02" -Name "Reservoir array" -ScriptPath "scripts/02_simulate_reservoir.py"
Run-Step -StepNum "03" -Name "Offline train" -ScriptPath "scripts/03_train_offline.py"
Run-Step -StepNum "04" -Name "Evaluate" -ScriptPath "scripts/04_evaluate.py"
Run-Step -StepNum "05" -Name "Ablation" -ScriptPath "scripts/05_ablation_study.py"
Run-Step -StepNum "06" -Name "Zero-Day" -ScriptPath "scripts/06_zero_day_test.py"
Run-Step -StepNum "07" -Name "Capacity" -ScriptPath "scripts/07_computational_capacity.py"
Run-Step -StepNum "08" -Name "Deployment" -ScriptPath "scripts/08_deployment_economics.py"
Run-Step -StepNum "09" -Name "Adversarial" -ScriptPath "scripts/09_adversarial_robustness.py"

Stop-Transcript | Out-Null

Write-Output "======================================================================"
Write-Output "  Complete - log saved: $logFile"
Write-Output "======================================================================"
