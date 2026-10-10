# AisleSense one command (Windows PowerShell 5.1+).
#   powershell -File run.ps1                                  # dashboard on http://127.0.0.1:8000/ (synth_aisle)
#   powershell -File run.ps1 -Site atc -Mode replay -Speed 600
#   powershell -File run.ps1 -Site synth_aisle -Headless -Seconds 20
# Builds frontend/dist first if it is missing (npm ci + build), then starts run.py with the project venv.
param(
    [string]$Site = "synth_aisle",
    [string]$Mode = "",
    [double]$Speed = 1.0,
    [switch]$Headless,
    [double]$Seconds = 60,
    [int]$Port = 8000
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$py = $env:AISLE_PY
if (-not $py) {
    $venv = Join-Path $root "..\venv\Scripts\python.exe"
    if (Test-Path $venv) { $py = (Resolve-Path $venv).Path } else { $py = "python" }
}

if (-not (Test-Path (Join-Path $root "frontend\dist\index.html"))) {
    Write-Host "frontend/dist missing: building it (npm ci + npm run build)"
    & npm --prefix frontend ci
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & npm --prefix frontend run build
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$args_ = @("run.py", "--site", $Site, "--speed", "$Speed", "--port", "$Port")
if ($Mode) { $args_ += @("--mode", $Mode) }
if ($Headless) { $args_ += @("--headless", "--seconds", "$Seconds") }
& $py @args_
exit $LASTEXITCODE
