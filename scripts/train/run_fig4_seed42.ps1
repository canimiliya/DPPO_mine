param(
    [ValidateSet('hopper-v2','walker2d-v2','halfcheetah-v2')]
    [string]$EnvName = 'hopper-v2',
    [ValidateSet('DRWR','DAWR','DIPO','IDQL','DQL','QSM','DPPO')]
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
$Checkpoint = "$SourceWsl/checkpoints/official/${EnvName}_medium_pre_diffusion_mlp_ta4_td20_state_3000.pt"
$Config = @{ DRWR='ft_rwr_diffusion_mlp'; DAWR='ft_awr_diffusion_mlp'; DIPO='ft_dipo_diffusion_mlp'; IDQL='ft_idql_diffusion_mlp'; DQL='ft_dql_diffusion_mlp'; QSM='ft_qsm_diffusion_mlp'; DPPO='ft_ppo_diffusion_mlp' }[$Method]
$ConfigDir = "$SourceWsl/source/dppo_v0.6/cfg/gym/finetune/$EnvName"

$bash = "cd '$SourceWsl/source/dppo_v0.6'; export DPPO_DATA_DIR='$DataWsl'; export DPPO_LOG_DIR='$LogWsl'; export MUJOCO_PY_MUJOCO_PATH='$MujocoWsl'; export MUJOCO_GL=osmesa; export PATH='$VenvWsl/bin`:$PATH'; export LD_LIBRARY_PATH='$MujocoWsl/bin`:/usr/lib/x86_64-linux-gnu`:/usr/lib'; exec '$VenvWsl/bin/python' script/run.py --config-name=$Config --config-dir='$ConfigDir' seed=$Seed base_policy_path='$Checkpoint'"
wsl.exe -- bash -lc $bash
if ($LASTEXITCODE -ne 0) { throw "Figure 4 $Method $EnvName failed with exit code $LASTEXITCODE" }
