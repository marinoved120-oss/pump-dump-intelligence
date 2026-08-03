param(
    [string]$Source = "..\pump-dump-research-v0.1.3"
)

$ErrorActionPreference = "Stop"
$sourceRaw = Join-Path $Source "data\raw"
$destinationRaw = Join-Path $PSScriptRoot "data\raw"

if (-not (Test-Path $sourceRaw)) {
    Write-Error "Raw data folder not found: $sourceRaw"
    exit 2
}

New-Item -ItemType Directory -Force -Path $destinationRaw | Out-Null
Copy-Item -Path (Join-Path $sourceRaw "*") -Destination $destinationRaw -Recurse -Force
Write-Host "Raw datasets copied to $destinationRaw" -ForegroundColor Green
Write-Host "Next: docker compose run --rm research rebuild" -ForegroundColor Cyan
