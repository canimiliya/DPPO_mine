#!/usr/bin/env bash
set -u

PROJECT=/mnt/d/Desktop/my_project/paper_reproduction/DPPO
SOURCE="$PROJECT/source/dppo_v0.6"
PYTHON="$PROJECT/.runtime/venv/bin/python"
OUT=/mnt/d/AgentData/DPPO-P4-PREP/benchmark_ram
TRAIN_LOG="$OUT/train.stdout.log"
TELEMETRY="$OUT/telemetry.csv"
RUN_LOG=/mnt/d/AgentData/DPPO-P4-PREP/logs/can_official_size_ram/run.log

mkdir -p "$OUT" "$(dirname "$RUN_LOG")"
export DPPO_DATA_DIR="$PROJECT/data"
export DPPO_LOG_DIR=/mnt/d/AgentData/DPPO-P4-PREP/logs
export DPPO_WANDB_ENTITY=local
export WANDB_MODE=disabled
export MUJOCO_PY_MUJOCO_PATH="$PROJECT/.runtime/mujoco210"
export MUJOCO_GL=osmesa
export PATH="$PROJECT/.runtime/venv/bin:$PATH"
export LD_LIBRARY_PATH="$PROJECT/.runtime/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/usr/lib"

cd "$SOURCE"
"$PYTHON" script/run.py \
  --config-name=ft_ppo_diffusion_mlp \
  --config-dir=cfg/robomimic/finetune/can \
  base_policy_path="$PROJECT/logs/robomimic-pretrain/can/can_pre_diffusion_mlp_ta4_td20/2024-06-28_13-29-54/checkpoint/state_5000.pt" \
  normalization_path="$PROJECT/data/robomimic/can/normalization.npz" \
  logdir=/mnt/d/AgentData/DPPO-P4-PREP/logs/can_official_size_ram \
  env.n_envs=50 \
  train.n_steps=300 \
  train.n_train_itr=6 \
  > "$TRAIN_LOG" 2>&1 &
train_pid=$!

"$PYTHON" "$PROJECT/scripts/diagnostics/monitor_telemetry.py" \
  --pid "$train_pid" \
  --output "$TELEMETRY" \
  --interval 1 \
  > "$OUT/telemetry.stdout.log" 2>&1 &
monitor_pid=$!

wait "$train_pid"
train_status=$?
wait "$monitor_pid" || true
cp "$RUN_LOG" "$OUT/run.log" 2>/dev/null || true
echo "TRAIN_PID=$train_pid"
echo "TRAIN_STATUS=$train_status"
echo "TRAIN_LOG=$TRAIN_LOG"
echo "RUN_LOG=$RUN_LOG"
echo "TELEMETRY=$TELEMETRY"
exit "$train_status"
