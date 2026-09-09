<#
.SYNOPSIS
  Start the Green-Weight backend and dashboard together (local dev).

.DESCRIPTION
  Launches uvicorn (api.py) and the Vite dev server in two child windows.

  The backend MUST run with cwd = backend/src/green_weight, not
  backend/src: every module uses bare same-directory imports
  (`from config import get_config`), so `-m green_weight.api` from
  backend/src fails. See CLAUDE.md, "Running the Project".

  The dashboard reaches the backend through Vite's /api proxy
  (vite.config.js), so both must be up for the UI to show "online".

.PARAMETER LoadModels
  Set GREEN_WEIGHT_LOAD_MODELS=1, so /infer attempts real GPU inference
  instead of the labelled mock responses. Needs CUDA + HF_TOKEN; without
  them api.py logs a warning and stays in mock mode.

.EXAMPLE
  .\run_dev.ps1
  .\run_dev.ps1 -LoadModels
#>
param(
    [switch]$LoadModels,
    [int]$BackendPort = 8000
)

$ErrorActionPreference = 'Stop'
$root     = $PSScriptRoot
$backend  = Join-Path $root 'backend\src\green_weight'
$frontend = Join-Path $root 'frontend'

if (-not (Test-Path (Join-Path $frontend 'node_modules'))) {
    Write-Host 'Installing frontend dependencies (first run)...' -ForegroundColor Yellow
    Push-Location $frontend; npm install; Pop-Location
}

$envPrefix = if ($LoadModels) { '$env:GREEN_WEIGHT_LOAD_MODELS=1; ' } else { '' }

Write-Host "Backend  -> http://localhost:$BackendPort" -ForegroundColor Green
Start-Process powershell -ArgumentList @(
    '-NoExit', '-Command',
    "cd '$backend'; $envPrefix python -m uvicorn api:app --reload --port $BackendPort"
)

Write-Host 'Dashboard -> http://localhost:5173' -ForegroundColor Green
Start-Process powershell -ArgumentList @(
    '-NoExit', '-Command',
    "cd '$frontend'; npm run dev"
)

Write-Host ''
Write-Host 'Both started in separate windows. Close those windows to stop.' -ForegroundColor Cyan
