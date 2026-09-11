#!/usr/bin/env bash
set -u

PROJECT=/mnt/d/Desktop/my_project/paper_reproduction/DPPO
SOURCE="$PROJECT/source/dppo_v0.6"
PYTHON="$PROJECT/.runtime/venv/bin/python"

export DPPO_DATA_DIR="$PROJECT/data"
export DPPO_LOG_DIR="$PROJECT/logs"
export MUJOCO_PY_MUJOCO_PATH="$PROJECT/.runtime/mujoco210"
export MUJOCO_GL=osmesa
export PATH="$PROJECT/.runtime/venv/bin:$PATH"
export LD_LIBRARY_PATH="$PROJECT/.runtime/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/usr/lib"
export DPPO_WANDB_ENTITY=local
export WANDB_MODE=disabled

cd "$SOURCE"
"$PYTHON" script/run.py \
  --config-name=ft_ppo_diffusion_mlp \
  --config-dir="$SOURCE/cfg/gym/finetune/hopper-v2" \
  seed=42 \
  train.n_train_itr=2 \
  base_policy_path="$PROJECT/checkpoints/official/hopper_medium_pre_diffusion_mlp_ta4_td20_state_3000.pt" \
  > /tmp/dppo_telemetry_train.log 2>&1 &
train_pid=$!

"$PYTHON" "$PROJECT/scripts/diagnostics/monitor_telemetry.py" \
  --pid "$train_pid" \
  --output "$PROJECT/reports/telemetry_patched_short.csv" \
  --interval 1 \
  > /tmp/dppo_telemetry_monitor.log 2>&1 &
monitor_pid=$!

wait "$train_pid"
train_status=$?
wait "$monitor_pid"
cat /tmp/dppo_telemetry_train.log
cat /tmp/dppo_telemetry_monitor.log
exit "$train_status"
