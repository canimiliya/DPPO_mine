#!/usr/bin/env bash
set -u

METHOD="${1:?usage: run_hopper_method_benchmark.sh DPPO|IDQL|DIPO telemetry.csv}"
TELEMETRY="${2:?usage: run_hopper_method_benchmark.sh METHOD telemetry.csv}"
PROJECT=/mnt/d/Desktop/my_project/paper_reproduction/DPPO
SOURCE="$PROJECT/source/dppo_v0.6"
PYTHON="$PROJECT/.runtime/venv/bin/python"
DATA="$PROJECT/data"
LOGS="$PROJECT/logs"
MUJOCO="$PROJECT/.runtime/mujoco210"
CHECKPOINT="$PROJECT/checkpoints/official/hopper_medium_pre_diffusion_mlp_ta4_td20_state_3000.pt"

case "$METHOD" in
  DPPO) CONFIG=ft_ppo_diffusion_mlp ;;
  IDQL) CONFIG=ft_idql_diffusion_mlp ;;
  DIPO) CONFIG=ft_dipo_diffusion_mlp ;;
  *) echo "Unsupported method: $METHOD" >&2; exit 2 ;;
esac

export DPPO_DATA_DIR="$DATA"
export DPPO_LOG_DIR="$LOGS"
export MUJOCO_PY_MUJOCO_PATH="$MUJOCO"
export MUJOCO_GL=osmesa
export PATH="$PROJECT/.runtime/venv/bin:$PATH"
export LD_LIBRARY_PATH="$MUJOCO/bin:/usr/lib/x86_64-linux-gnu:/usr/lib"
export DPPO_WANDB_ENTITY=local
export WANDB_MODE=disabled

RAW="$PROJECT/reports/${METHOD,,}_hopper_benchmark_stdout.log"
START_UNIX="$(date +%s)"
START_ISO="$(date --iso-8601=seconds)"
{
  echo "benchmark_method=$METHOD"
  echo "benchmark_start_unix=$START_UNIX"
  echo "benchmark_start_iso=$START_ISO"
  echo "config=$CONFIG"
  echo "seed=42"
  echo "override=train.n_train_itr=5"
  echo "official_config_overrides=none_other_than_n_train_itr"
} > "$RAW"

cd "$SOURCE"
"$PYTHON" script/run.py \
  --config-name="$CONFIG" \
  --config-dir="$SOURCE/cfg/gym/finetune/hopper-v2" \
  seed=42 \
  train.n_train_itr=5 \
  base_policy_path="$CHECKPOINT" \
  >> "$RAW" 2>&1 &
train_pid=$!

"$PYTHON" "$PROJECT/scripts/diagnostics/monitor_telemetry.py" \
  --pid "$train_pid" \
  --output "$TELEMETRY" \
  --interval 1 \
  > "$PROJECT/reports/${METHOD,,}_hopper_benchmark_monitor.log" 2>&1 &
monitor_pid=$!

wait "$train_pid"
train_status=$?
wait "$monitor_pid"
monitor_status=$?
END_UNIX="$(date +%s)"
END_ISO="$(date --iso-8601=seconds)"
{
  echo "benchmark_end_unix=$END_UNIX"
  echo "benchmark_end_iso=$END_ISO"
  echo "benchmark_exit_code=$train_status"
  echo "monitor_exit_code=$monitor_status"
} >> "$RAW"
cat "$RAW"
exit "$train_status"
