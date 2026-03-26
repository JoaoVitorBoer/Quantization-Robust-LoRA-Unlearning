#!/bin/bash

#SBATCH --job-name=grid_muon_tofu
#SBATCH --output=/home/joaoabitante/Sout/%j__%x.out
#SBATCH --error=/home/joaoabitante/Sout/%j__%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --gpus=3

set -euo pipefail

export HYDRA_FULL_ERROR=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RED='\e[31m'
NC='\e[0m'

NUM_GPUS=3
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2}"
EVAL_CUDA_DEVICE="${EVAL_CUDA_DEVICE:-0}"

RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/saves/unlearn/muon_results}"
MODELS_ROOT="${MODELS_ROOT:-${REPO_ROOT}/saves/unlearn/muon_models}"

MODELS=(
  "Llama-3.2-1B-Instruct"
  "Llama-3.2-3B-Instruct"
  "Llama-3.1-8B-Instruct"
)

# Format: "method_tag trainer experiment_cfg"
# Comment out entries here to run a subset of methods.
METHODS=(
  "GA GradAscent unlearn/tofu/default.yaml"
  "GradDiff GradDiff unlearn/tofu/default.yaml"
  "NPO NPO unlearn/tofu/default.yaml"
  "SimNPO SimNPO unlearn/tofu/default.yaml"
)

LRS=(5e-6 1e-5 2e-5)

SPLITS=(
  "forget01 holdout01 retain99"
  "forget05 holdout05 retain95"
  "forget10 holdout10 retain90"
)

# Muon baseline.
MUON_MOMENTUM="${MUON_MOMENTUM:-0.95}"
MUON_NS_STEPS="${MUON_NS_STEPS:-5}"
MUON_ADJUST_LR_FN="${MUON_ADJUST_LR_FN:-match_rms_adamw}"
ADAMW_BETAS="${ADAMW_BETAS:-[0.9,0.95]}"
ADAMW_EPS="${ADAMW_EPS:-1e-8}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"

PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=3
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
    retain_logs_path="${REPO_ROOT}/saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json"

    for method_cfg in "${METHODS[@]}"; do
      read -r method_tag trainer experiment_cfg <<< "${method_cfg}"

      for lr in "${LRS[@]}"; do
        task_name="tofu_${model}_${forget_split}_${method_tag}_Muon_lr-${lr}"
        train_output_dir="${MODELS_ROOT}/${method_tag}/${forget_split}/${model}/lr-${lr}"
        eval_output_dir="${RESULTS_ROOT}/${method_tag}/${forget_split}/${model}/lr-${lr}"

        mkdir -p "${train_output_dir}" "${eval_output_dir}"

        echo
        echo -e "${RED}=== Method: ${method_tag} | Trainer: ${trainer} | LR: ${lr} | Model: ${model} | Split: ${forget_split} ===${NC}"
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
          trainer.args.weight_decay="${WEIGHT_DECAY}" \
          trainer.args.ddp_find_unused_parameters=true \
          trainer.args.gradient_checkpointing=true \
          trainer.args.eval_strategy=no \
          trainer.args.eval_on_start=False \
          trainer.args.do_eval=False \
          trainer.method_args.optimizer_name=muon \
          trainer.method_args.optimizer_kwargs.muon_momentum="${MUON_MOMENTUM}" \
          trainer.method_args.optimizer_kwargs.muon_ns_steps="${MUON_NS_STEPS}" \
          trainer.method_args.optimizer_kwargs.muon_adjust_lr_fn="${MUON_ADJUST_LR_FN}" \
          trainer.method_args.optimizer_kwargs.muon_weight_decay="${WEIGHT_DECAY}" \
          trainer.method_args.optimizer_kwargs.adamw_betas="${ADAMW_BETAS}" \
          trainer.method_args.optimizer_kwargs.adamw_eps="${ADAMW_EPS}" \
          trainer.method_args.optimizer_kwargs.adamw_weight_decay="${WEIGHT_DECAY}"

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
