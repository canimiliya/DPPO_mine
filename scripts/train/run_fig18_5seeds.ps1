param(
    [ValidateSet('hopper-v2','walker2d-v2','halfcheetah-v2')]
    [string]$EnvName = 'hopper-v2',
    [ValidateSet('DPPO','Gaussian-MLP')]
    [string]$Method = 'DPPO'
)

$ErrorActionPreference = 'Stop'
foreach ($seed in 42,43,44,45,46) {
    $script = Join-Path $PSScriptRoot 'run_fig18_seed42.ps1'
    & $script -EnvName $EnvName -Method $Method -Seed $seed
    if ($LASTEXITCODE -ne 0) { throw "Figure 18 $Method $EnvName seed $seed failed" }
}
