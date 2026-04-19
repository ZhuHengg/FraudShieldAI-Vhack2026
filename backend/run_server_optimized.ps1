# run_server_optimized.ps1 — High Performance API Launcher
# Launches Uvicorn with multiple workers for better concurrency.

Write-Host "--- FraudShield AI: Starting Optimized Backend Server ---" -ForegroundColor Cyan

# Ensure we are in the correct directory
$ScriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptPath

# 1. Activate Virtual Environment
if (Test-Path "venv\Scripts\Activate.ps1") {
    . venv\Scripts\Activate.ps1
}

# 2. Run Uvicorn in production-like mode
# --workers 4: handles multiple users simultaneously
# No --reload: maximum performance
Write-Host ">> Launching with 4 workers..." -ForegroundColor Gray
Write-Host ">> Access API Docs: http://localhost:8000/docs" -ForegroundColor Gray
Write-Host ">> Stop with Ctrl+C" -ForegroundColor Gray
Write-Host ""

uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 4
