param(
    [switch]$SkipInstall,
    [switch]$NoClean
)

$ErrorActionPreference = "Stop"

function Resolve-RepoRoot {
    $scriptDir = Split-Path -Parent $PSCommandPath
    return (Resolve-Path (Join-Path $scriptDir "..")).Path
}

function Require-Path {
    param(
        [string]$Path,
        [string]$Description
    )

    if (-not (Test-Path $Path)) {
        throw "$Description not found: $Path"
    }
}

$repoRoot = Resolve-RepoRoot
Set-Location $repoRoot

$python = Get-Command python -ErrorAction Stop
Write-Host "Using Python: $($python.Source)"
python --version

Require-Path (Join-Path $repoRoot "ui_main.py") "UI entrypoint"
Require-Path (Join-Path $repoRoot "main.py") "CLI entrypoint"
Require-Path (Join-Path $repoRoot "src\TeamControl\utils\ipconfig.yaml") "Network config"
Require-Path (Join-Path $repoRoot "calibration.json") "Calibration config"
Require-Path (Join-Path $repoRoot "tuning.json") "Tuning config"

if (-not $SkipInstall) {
    Write-Host "Installing build dependencies..."
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -e .
    python -m pip install pyinstaller
}

$cleanArgs = @()
if (-not $NoClean) {
    $cleanArgs += "--clean"
}

$commonArgs = @(
    "--noconfirm",
    "--onedir",
    "--paths", "src",
    "--add-data", "src\TeamControl\utils\ipconfig.yaml;TeamControl\utils"
) + $cleanArgs

Write-Host "Building TeamControl UI..."
python -m PyInstaller @commonArgs --windowed --name TeamControl ui_main.py

Write-Host "Building TeamControl CLI..."
python -m PyInstaller @commonArgs --console --name TeamControlCLI main.py

$releaseDirs = @(
    (Join-Path $repoRoot "dist\TeamControl"),
    (Join-Path $repoRoot "dist\TeamControlCLI")
)

foreach ($dir in $releaseDirs) {
    Require-Path $dir "Build output"

    Copy-Item -Force (Join-Path $repoRoot "calibration.json") (Join-Path $dir "calibration.json")
    Copy-Item -Force (Join-Path $repoRoot "tuning.json") (Join-Path $dir "tuning.json")
}

Write-Host ""
Write-Host "Build complete."
Write-Host "UI app:  dist\TeamControl\TeamControl.exe"
Write-Host "CLI app: dist\TeamControlCLI\TeamControlCLI.exe"
Write-Host ""
Write-Host "Release the whole dist\TeamControl folder for UI users."
