#!/bin/bash

#SBATCH --output=/home/joaoabitante/Sout/%j__%x.out
#SBATCH --error=/home/joaoabitante/Sout/%j__%x.out

#SBATCH --nodes=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=52G
#SBATCH --time=2-00:00:00
#SBATCH --gpus=3

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

RED='\e[31m'
YELLOW='\e[33m'
NC='\e[0m'

echo -e "${YELLOW}Master Port: ${MASTER_PORT}${NC}"

MODEL="Llama-2-7b-hf"
DATA_SPLITS=(
  "News"
  "Books"
)

PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=1e-05
LORA_DROPOUT=0.05
TARGET_MODULE_VALUES='["q_proj","v_proj","k_proj","o_proj","gate_proj","down_proj","up_proj"]'

# Keep this exact order:
# 1) GA+GDR, 2) GA+KLR, 3) NPO+GDR, 4) NPO+KLR
METHODS=(
  "GA+GDR"
  "GA+KLR"
  "NPO+GDR"
  "NPO+KLR"
)

TRAINERS=(
  "GradDiff"
  "GradDiff"
  "NPO"
  "NPO"
)

RETAIN_LOSS_TYPES=(
  "NLL"
  "KL"
  "NLL"
  "KL"
)

ALPHAS=(
  "100"
  "100"
  "0.1"
  "0.1"
)

EPOCHS=(
  "5"
  "5"
  "5"
  "5"
)

if [[ "${#METHODS[@]}" -ne 4 ]]; then
  echo -e "${RED}Expected exactly 4 methods.${NC}" >&2
  exit 1
fi

for arr_name in TRAINERS RETAIN_LOSS_TYPES ALPHAS EPOCHS; do
  eval "arr_len=\${#${arr_name}[@]}"
  if [[ "${arr_len}" -ne "${#METHODS[@]}" ]]; then
    echo -e "${RED}${arr_name} length (${arr_len}) must match METHODS length (${#METHODS[@]}).${NC}" >&2
    exit 1
  fi
done

get_lora_params() {
  local split="$1"
  local method="$2"

  case "${split}|${method}" in
    "Books|GA+GDR") echo "16 16" ;;
    "News|GA+GDR") echo "64 64" ;;
    "Books|GA+KLR") echo "16 16" ;;
    "News|GA+KLR") echo "64 64" ;;
    "Books|NPO+GDR") echo "64 128" ;;
    "News|NPO+GDR") echo "16 16" ;;
    "Books|NPO+KLR") echo "16 32" ;;
    "News|NPO+KLR") echo "64 128" ;;
    *)
      return 1
      ;;
  esac
}

for data_split in "${DATA_SPLITS[@]}"; do
  retain_logs_path="saves/eval/muse_${MODEL}_${data_split}_retrain/MUSE_EVAL.json"
  echo -e "${RED}--- Data split: ${data_split} ---${NC}"
  echo -e "${RED}Retain logs: ${retain_logs_path}${NC}"

  for idx in "${!METHODS[@]}"; do
    method="${METHODS[$idx]}"
    trainer="${TRAINERS[$idx]}"
    retain_loss_type="${RETAIN_LOSS_TYPES[$idx]}"
    alpha="${ALPHAS[$idx]}"
    epochs="${EPOCHS[$idx]}"

    if ! read -r rank lora_alpha < <(get_lora_params "${data_split}" "${method}"); then
      echo -e "${RED}Missing LoRA config for ${method} on ${data_split}.${NC}" >&2
      exit 1
    fi

    if ! [[ "${rank}" =~ ^[0-9]+$ && "${lora_alpha}" =~ ^[0-9]+$ ]]; then
      echo -e "${RED}LoRA rank/alpha for ${method} on ${data_split} must be non-negative integers.${NC}" >&2
      exit 1
    fi

    if (( rank == 0 || lora_alpha == 0 )); then
      echo -e "${YELLOW}Skipping ${method} on ${data_split}: rank=${rank}, lora_alpha=${lora_alpha}.${NC}"
      echo -e "${YELLOW}Set GA_KLR_NEWS_RANK and GA_KLR_NEWS_LORA_ALPHA to enable this run.${NC}"
      continue
    fi

    method_slug="${method//+/_}"
    task_name="muse_${MODEL}_${data_split}_${method_slug}_alpha-${alpha}_lr-${LEARNING_RATE}_ep-${epochs}_r-${rank}_la-${lora_alpha}"
    train_output_dir="saves/unlearn/norm_calculation/${method_slug}/${data_split}"

    method_args=(
      "trainer.method_args.alpha=${alpha}"
      "trainer.method_args.retain_loss_type=${retain_loss_type}"
    )

    echo
    echo -e "${RED}=== Method: ${method} | Trainer: ${trainer} | Alpha: ${alpha} | Epochs: ${epochs} ===${NC}"
    echo -e "${RED}LoRA rank: ${rank} | LoRA alpha: ${lora_alpha}${NC}"
    echo -e "${RED}Train output: ${train_output_dir}${NC}"

    CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch \
      --config_file configs/accelerate/default_config.yaml \
      --num_processes=3 \
      --main_process_port "${MASTER_PORT}" \
      src/train.py --config-name=unlearn.yaml \
      experiment=unlearn/muse/default.yaml \
      adapter=lora \
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
      model.lora_config.target_modules="${TARGET_MODULE_VALUES}" \
      model.lora_config.r="${rank}" \
      model.lora_config.lora_alpha="${lora_alpha}" \
      model.lora_config.lora_dropout="${LORA_DROPOUT}" \
      "${method_args[@]}"
  done
done
