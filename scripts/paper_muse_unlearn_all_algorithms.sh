#!/bin/bash

#SBATCH --output=/home/joaoabitante/Sout/%j__%x.out
#SBATCH --error=/home/joaoabitante/Sout/%j__%x.out

#SBATCH --nodes=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=52G
#SBATCH --time=2-00:00:00
#SBATCH --gpus=4

set -euo pipefail
export HYDRA_FULL_ERROR=1

export MASTER_PORT=$(
python - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
)
echo "Master Port: ${MASTER_PORT}"

MODEL="Llama-2-7b-hf"

DATA_SPLITS=(
  "News"
  "Books"
)

PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=1e-05

# Keep this exact order:
# 1) GA, 2) GA+GDR, 3) GA+KLR, 4) NPO, 5) NPO+GDR, 6) NPO+KLR
METHODS=(
  "GA"
  "GA+GDR"
  "GA+KLR"
  "NPO"
  "NPO+GDR"
  "NPO+KLR"
)

# Internal trainer mapping (same order as METHODS).
TRAINERS=(
  "GradAscent"
  "GradDiff"
  "GradDiff"
  "NPO"
  "NPO"
  "NPO"
)

# Retain loss override (same order as METHODS).
# Empty = do not override retain loss type.
RETAIN_LOSS_TYPES=(
  ""
  "NLL"
  "KL"
  ""
  "NLL"
  "KL"
)

# Fill these vectors in the same METHODS order above.
# Note: GA ignores alpha (kept for index alignment).
ALPHAS=(
  "0"
  "100"
  "100"
  "0"
  "0.1"
  "0.1"
)

EPOCHS=(
  "10"
  "10"
  "-"
  "-"
  "10"
  "5"
)

if [[ "${#METHODS[@]}" -ne 6 ]]; then
  echo "Expected exactly 6 methods." >&2
  exit 1
fi

if [[ "${#ALPHAS[@]}" -ne "${#METHODS[@]}" ]]; then
  echo "ALPHAS length (${#ALPHAS[@]}) must match METHODS length (${#METHODS[@]})." >&2
  exit 1
fi

if [[ "${#EPOCHS[@]}" -ne "${#METHODS[@]}" ]]; then
  echo "EPOCHS length (${#EPOCHS[@]}) must match METHODS length (${#METHODS[@]})." >&2
  exit 1
fi

for data_split in "${DATA_SPLITS[@]}"; do
  retain_logs_path="saves/eval/muse_${MODEL}_${data_split}_retrain/MUSE_EVAL.json"
  echo "--- Data split: ${data_split} ---"
  echo "Retain logs: ${retain_logs_path}"

  for idx in "${!METHODS[@]}"; do
    method="${METHODS[$idx]}"
    trainer="${TRAINERS[$idx]}"
    retain_loss_type="${RETAIN_LOSS_TYPES[$idx]}"
    alpha="${ALPHAS[$idx]}"
    epochs="${EPOCHS[$idx]}"

    # Pure NPO must use alpha=0 and default retain loss (NLL in trainer config).
    if [[ "${method}" == "NPO" ]]; then
      alpha="0"
      retain_loss_type=""
    fi

    method_slug="${method//+/_}"
    task_name="muse_${MODEL}_${data_split}_${method_slug}_alpha-${alpha}_lr-${LEARNING_RATE}_ep-${epochs}"
    train_output_dir="saves/unlearn/norm_calculation/${method_slug}/${data_split}"

    method_args=()
    if [[ "${trainer}" != "GradAscent" ]]; then
      method_args+=("trainer.method_args.alpha=${alpha}")
    fi
    if [[ -n "${retain_loss_type}" ]]; then
      method_args+=("trainer.method_args.retain_loss_type=${retain_loss_type}")
    fi

    echo
    echo "=== Method: ${method} | Trainer: ${trainer} | Alpha: ${alpha} | Epochs: ${epochs} ==="
    echo "Train output: ${train_output_dir}"

    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
      --config_file configs/accelerate/default_config.yaml \
      --num_processes=4 \
      --main_process_port "${MASTER_PORT}" \
      src/train.py --config-name=unlearn.yaml \
      experiment=unlearn/muse/default.yaml \
      model="${MODEL}" \
      data_split="${data_split}" \
      trainer="${trainer}" \
      task_name="${task_name}" \
      paths.output_dir="${train_output_dir}" \
      retain_logs_path="${retain_logs_path}" \
      trainer.args.per_device_train_batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE}" \
      trainer.args.gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
      trainer.args.learning_rate="${LEARNING_RATE}" \
      trainer.args.num_train_epochs="${epochs}" \
      trainer.args.ddp_find_unused_parameters=true \
      trainer.args.gradient_checkpointing=true \
      trainer.args.eval_strategy=no \
      trainer.args.eval_on_start=False \
      trainer.args.do_eval=False \
      "${method_args[@]}"
  done
done
