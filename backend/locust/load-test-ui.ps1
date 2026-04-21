# load-test-ui.ps1 — FraudShield AI Load Test Shortcut
# Directly launches the Locust UI script from within the locust directory.

if (Test-Path "$PSScriptRoot\run_locust_ui.ps1") {
    & "$PSScriptRoot\run_locust_ui.ps1"
} else {
    Write-Error "Could not find run_locust_ui.ps1 in the current directory."
}
