# run_locust_ui.ps1 — FraudShield AI Load Test Launcher
# Launches the Locust web interface at http://localhost:8089

Write-Host "--- FraudShield AI: Starting Load Test Interface ---" -ForegroundColor Cyan

# Ensure we are in the correct directory
$ScriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptPath

# 1. Activate Virtual Environment if it exists
if (Test-Path "../venv/Scripts/Activate.ps1") {
    Write-Host "[1/2] Activating virtual environment..." -ForegroundColor Gray
    . ../venv/Scripts/Activate.ps1
} else {
    Write-Host "[!] Virtual environment not found. Ensure 'pip install locust' is run." -ForegroundColor Yellow
}

# 2. Run Locust pointing to the local backend
Write-Host "[2/2] Launching Locust UI..." -ForegroundColor Green
Write-Host ">> Target Host: http://localhost:8000" -ForegroundColor Gray
Write-Host ">> Access Dashboard: http://localhost:8089" -ForegroundColor Gray
Write-Host ">> Stop with Ctrl+C" -ForegroundColor Gray
Write-Host ""

locust -f locustfile.py --host http://localhost:8000
