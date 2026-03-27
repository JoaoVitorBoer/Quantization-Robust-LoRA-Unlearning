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
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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

RED='\e[31m'
NC='\e[0m'   # no color / reset

MODEL="Llama-2-7b-hf"

DATA_SPLITS=(
"News"
#"Books"
)

TRAINERS=("NPO")

# Keep these the same
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=5
EPOCHS=(5)
LRS=(7e-4 3e-4 1e-4)

ALPHAS=(10)

TARGET_MODULE_VALUES='["q_proj","v_proj","k_proj","o_proj","gate_proj","down_proj","up_proj"]'
LORA_DROPOUT=0.05
RANK=64
LORA_ALPHA=128

# Keep evaluation in these precisions
PRECISIONS=(fp 8bit 4bit)

# -------------------------------
# Optional retain loss type:
# - Default: do not pass anything
# - Enable: export RETAIN_LOSS_TYPE=KL
# -------------------------------
RETAIN_LOSS_TYPE="KL"

retain_loss_args=()
retain_loss_tag=""

if [[ -n "${RETAIN_LOSS_TYPE}" ]]; then
  retain_loss_args=("trainer.method_args.retain_loss_type=${RETAIN_LOSS_TYPE}")
  retain_loss_tag="_retain-${RETAIN_LOSS_TYPE}"
fi

for data_split in "${DATA_SPLITS[@]}"; do
  retain_logs_path="saves/eval/muse_${MODEL}_${data_split}_retrain/MUSE_EVAL.json"
  echo -e "${RED}--- Data split: ${data_split} | retain_logs_path: ${retain_logs_path} ---${NC}"

  for trainer in "${TRAINERS[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
      for lr in "${LRS[@]}"; do
        for epochs in "${EPOCHS[@]}"; do

          task_name="muse_${MODEL}_${data_split}_${trainer}_alpha-${alpha}_lr-${lr}_ep-${epochs}${retain_loss_tag}"
          train_output_dir="saves/unlearn/paper_models/${trainer}/muse/${data_split}/${MODEL}/alpha-${alpha}/lr-${lr}_ep-${epochs}${retain_loss_tag}"

          echo
          echo -e "${RED}=== Trainer: ${trainer} | Alpha: ${alpha} | LR: ${lr} | Epochs: ${epochs} ===${NC}"
          echo -e "${RED}Train output: ${train_output_dir}${NC}"

          CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch \
            --config_file configs/accelerate/default_config.yaml \
            --num_processes=3  \
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
              trainer.args.learning_rate="${lr}" \
              trainer.args.num_train_epochs="${epochs}" \
              trainer.args.ddp_find_unused_parameters=true \
              trainer.args.gradient_checkpointing=true \
              trainer.args.eval_strategy=no \
              trainer.args.eval_on_start=False \
              trainer.method_args.alpha="${alpha}" \
              adapter=lora \
              model.lora_config.target_modules="${TARGET_MODULE_VALUES}" \
              model.lora_config.r="${RANK}" \
              model.lora_config.lora_alpha="${LORA_ALPHA}" \
              model.lora_config.lora_dropout="${LORA_DROPOUT}" \
              "${retain_loss_args[@]}"

          # Evaluation: fp / 8bit / 4bit (unchanged behavior)
          for precision in "${PRECISIONS[@]}"; do
            quant_override=()
            precision_tag="${precision}"

            case "${precision}" in
              fp)
                precision_tag="fp"
                ;;
              8bit)
                quant_override=("model.quantization_config=8bit")
                ;;
              4bit)
                quant_override=("model.quantization_config=4bit")
                ;;
              *)
                echo -e "${RED}Unknown precision ${precision}${NC}" >&2
                exit 1
                ;;
            esac

            eval_task_name="${task_name}_eval_${precision_tag}"
            eval_output_dir="saves/unlearn/paper_results/${trainer}_${data_split}_alpha-${alpha}_lr-${lr}_ep-${epochs}${retain_loss_tag}/${precision_tag}"

            echo -e "${RED}---- Eval precision: ${precision_tag} | Output: ${eval_output_dir}${NC}"

            CUDA_VISIBLE_DEVICES=0 python src/eval.py \
              experiment=eval/muse/default.yaml \
              model="${MODEL}" \
              data_split="${data_split}" \
              task_name="${eval_task_name}" \
              paths.output_dir="${eval_output_dir}" \
              retain_logs_path="${retain_logs_path}" \
              model.model_handler=LoRAModelForEvaluation \
              model.model_args.pretrained_model_name_or_path="${train_output_dir}" \
              "${quant_override[@]}"
          done

        done
      done
    done
  done
done