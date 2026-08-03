param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$SourceProject
)

$ErrorActionPreference = "Stop"
$source = (Resolve-Path $SourceProject).Path
$sourceRaw = Join-Path $source "data\raw"
$targetRaw = Join-Path $PSScriptRoot "data\raw"

if (-not (Test-Path $sourceRaw)) {
    throw "Raw data folder not found: $sourceRaw"
}

New-Item -ItemType Directory -Force -Path $targetRaw | Out-Null
Copy-Item (Join-Path $sourceRaw "*") $targetRaw -Recurse -Force
Write-Host "Raw Binance datasets copied to $targetRaw" -ForegroundColor Green
Write-Host "Next: docker compose run --rm research rebuild" -ForegroundColor Cyan
