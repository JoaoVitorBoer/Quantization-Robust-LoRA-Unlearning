#!/bin/bash

#SBATCH --job-name=grid_tofu
#SBATCH --output=/home/joaoabitante/Sout/%j__%x.out
#SBATCH --error=/home/joaoabitante/Sout/%j__%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=10GB
#SBATCH --time=2-00:00:00
#SBATCH --gpus=2

set -euo pipefail

export HYDRA_FULL_ERROR=1

RED='\e[31m'
NC='\e[0m'

NUM_GPUS=2
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
EVAL_CUDA_DEVICE="${EVAL_CUDA_DEVICE:-0}"

RESULTS_ROOT="${RESULTS_ROOT:-saves/unlearn/adam_results}"
MODELS_ROOT="${MODELS_ROOT:-saves/unlearn/adam_models}"

MODELS=(
  "Llama-3.2-1B-Instruct"
  "Llama-3.2-3B-Instruct"
  #"Llama-3.1-8B-Instruct"
)

# Format: "method_tag trainer experiment_cfg"
# Comment out entries here to run a subset of methods.
METHODS=(
  #"GA GradAscent unlearn/tofu/default.yaml"
  #"GradDiff GradDiff unlearn/tofu/default.yaml"
  #"NPO NPO unlearn/tofu/default.yaml"
  "SimNPO SimNPO unlearn/tofu/default.yaml"
)

LRS=(1e-5 2e-5 5e-5)
EPOCHS=(5 10)

SPLITS=(
  "forget01 holdout01 retain99"
  "forget05 holdout05 retain95"
  "forget10 holdout10 retain90"
)

PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
EFFECTIVE_BATCH_SIZE=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS))

mkdir -p "${RESULTS_ROOT}" "${MODELS_ROOT}"

echo -e "${RED}Using per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE}, gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}, effective_batch_size=${EFFECTIVE_BATCH_SIZE}.${NC}"

export MASTER_PORT=$(
python - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
)

echo -e "${RED}Master Port: ${MASTER_PORT}${NC}"
echo -e "${RED}Results root: ${RESULTS_ROOT}${NC}"
echo -e "${RED}Models root: ${MODELS_ROOT}${NC}"

for split in "${SPLITS[@]}"; do
  read -r forget_split holdout_split retain_split <<< "${split}"
  echo -e "${RED}--- Split: forget=${forget_split} | holdout=${holdout_split} | retain=${retain_split} ---${NC}"

  for model in "${MODELS[@]}"; do
    model_path="open-unlearning/tofu_${model}_full"
    retain_logs_path="saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json"

    for method_cfg in "${METHODS[@]}"; do
      read -r method_tag trainer experiment_cfg <<< "${method_cfg}"

      for lr in "${LRS[@]}"; do
        for epochs in "${EPOCHS[@]}"; do
          task_name="tofu_${model}_${forget_split}_${method_tag}_Adam_lr-${lr}_ep-${epochs}"
          train_output_dir="${MODELS_ROOT}/${method_tag}/${forget_split}/${model}/lr-${lr}_ep-${epochs}"
          eval_output_dir="${RESULTS_ROOT}/${method_tag}/${forget_split}/${model}/lr-${lr}_ep-${epochs}"

          mkdir -p "${train_output_dir}" "${eval_output_dir}"

          echo
          echo -e "${RED}=== Method: ${method_tag} | Trainer: ${trainer} | LR: ${lr} | Epochs: ${epochs} | Model: ${model} | Split: ${forget_split} ===${NC}"
          echo -e "${RED}Train output: ${train_output_dir}${NC}"

          CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" accelerate launch \
            --config_file configs/accelerate/default_config.yaml \
            --num_processes="${NUM_GPUS}" \
            --main_process_port "${MASTER_PORT}" \
            src/train.py --config-name=unlearn.yaml \
            experiment="${experiment_cfg}" \
            trainer="${trainer}" \
            task_name="${task_name}" \
            paths.output_dir="${train_output_dir}" \
            model="${model}" \
            forget_split="${forget_split}" \
            holdout_split="${holdout_split}" \
            retain_split="${retain_split}" \
            retain_logs_path="${retain_logs_path}" \
            model.model_args.pretrained_model_name_or_path="${model_path}" \
            trainer.args.per_device_train_batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE}" \
            trainer.args.gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
            trainer.args.learning_rate="${lr}" \
            trainer.args.num_train_epochs="${epochs}" \
            trainer.args.ddp_find_unused_parameters=true \
            trainer.args.gradient_checkpointing=true \
            trainer.args.eval_strategy=no \
            trainer.args.eval_on_start=False \
            trainer.args.do_eval=False \
            trainer.args.optim=paged_adamw_32bit

          if [[ ! -f "${train_output_dir}/config.json" ]]; then
            echo -e "${RED}Skipping eval for ${task_name}: ${train_output_dir}/config.json not found.${NC}"
            continue
          fi

          echo -e "${RED}=== Eval: ${task_name} | Output: ${eval_output_dir} ===${NC}"

          CUDA_VISIBLE_DEVICES="${EVAL_CUDA_DEVICE}" python src/eval.py \
            experiment=eval/tofu/default.yaml \
            forget_split="${forget_split}" \
            holdout_split="${holdout_split}" \
            model="${model}" \
            task_name="${task_name}_eval" \
            paths.output_dir="${eval_output_dir}" \
            retain_logs_path="${retain_logs_path}" \
            model.model_args.pretrained_model_name_or_path="${train_output_dir}"
        done
      done
    done
  done
done
