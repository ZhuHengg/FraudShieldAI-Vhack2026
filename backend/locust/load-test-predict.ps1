# ------------------------------------------------------------------
# load-test-predict.ps1 — ML Engine Only (predict endpoint)
# ------------------------------------------------------------------
# Tests ONLY the /predict endpoint — pure ML scoring latency.
# No database endpoints, no SHAP, no health checks.
# ------------------------------------------------------------------

Write-Host ""
Write-Host "--- FraudShield AI: Predict-Only Load Test ---" -ForegroundColor Cyan
Write-Host ">> Testing ONLY the /predict ML scoring endpoint." -ForegroundColor Yellow
Write-Host ">> Target Host: http://localhost:8000" -ForegroundColor Yellow
Write-Host ">> Access Dashboard: http://localhost:8089" -ForegroundColor Yellow
Write-Host ""

# Activate venv
if (Test-Path "../venv/Scripts/Activate.ps1") {
    . ../venv/Scripts/Activate.ps1
}

locust -f locustfile.py --tags predict --host http://localhost:8000
