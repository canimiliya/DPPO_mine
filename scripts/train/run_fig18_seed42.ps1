param(
    [ValidateSet('hopper-v2','walker2d-v2','halfcheetah-v2')]
    [string]$EnvName = 'hopper-v2',
    [ValidateSet('DPPO','Gaussian-MLP')]
    [string]$Method = 'DPPO',
    [int]$Seed = 42
)

$ErrorActionPreference = 'Stop'
$Project = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$SourceWsl = ($Project -replace '\\','/' -replace '^([A-Za-z]):', '/$1' ).ToLower()
$DataWsl = "$SourceWsl/data"
$LogWsl = "$SourceWsl/logs"
$MujocoWsl = "$SourceWsl/.runtime/mujoco210"
$VenvWsl = "$SourceWsl/.runtime/venv"
$Config = if ($Method -eq 'DPPO') { 'ppo_diffusion_mlp' } else { 'ppo_gaussian_mlp' }
$ConfigDir = "$SourceWsl/source/dppo_v0.6/cfg/gym/scratch/$EnvName"

$bash = "cd '$SourceWsl/source/dppo_v0.6'; export DPPO_DATA_DIR='$DataWsl'; export DPPO_LOG_DIR='$LogWsl'; export MUJOCO_PY_MUJOCO_PATH='$MujocoWsl'; export MUJOCO_GL=osmesa; export PATH='$VenvWsl/bin`:$PATH'; export LD_LIBRARY_PATH='$MujocoWsl/bin`:/usr/lib/x86_64-linux-gnu`:/usr/lib'; exec '$VenvWsl/bin/python' script/run.py --config-name=$Config --config-dir='$ConfigDir' seed=$Seed"
wsl.exe -- bash -lc $bash
if ($LASTEXITCODE -ne 0) { throw "Figure 18 $Method $EnvName failed with exit code $LASTEXITCODE" }
