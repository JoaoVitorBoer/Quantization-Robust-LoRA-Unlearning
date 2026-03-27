#!/bin/bash

#SBATCH --output=/home/joaoabitante/Sout/%j__%x.out
#SBATCH --error=/home/joaoabitante/Sout/%j__%x.out

#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=30G
#SBATCH --time=2-00:00:00
#SBATCH --gpus=4

set -euo pipefail

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
NC='\e[0m'   # no color / reset

echo "Master Port: ${MASTER_PORT}"

PER_DEVICE_TRAIN_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=2

# 5 EPOCHS FOLLOWING THE SATIMP PAPER
# BS 16

MODELS=(
  "Llama-2-7b-chat-hf"
)

TRAINER_EXPERIMENTS=(
  "SatImp unlearn/tofu/default.yaml"
)

ADAPTERS=(
  "base"
  "lora"
  "qlora"
)

PRECISIONS=(
  "fp"
  "8bit"
  "4bit"
)

LRS=(1e-5)
BETA1S=(5.0 6.0)
BETA2=1.0
ALPHAS=(1.0 0.1 0.01)

SPLITS=(
  "forget01 holdout01 retain99"
  "forget05 holdout05 retain95"
  "forget10 holdout10 retain90"
)

for split in "${SPLITS[@]}"; do
  read -r forget_split holdout_split retain_split <<<"${split}"
  echo -e "${RED}--- Split: forget=${forget_split} | holdout=${holdout_split} | retain=${retain_split} ---${NC}"

  for model in "${MODELS[@]}"; do
    model_path="open-unlearning/tofu_${model}_full"
    retain_logs_path="saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json"
    echo -e "${RED}Model: ${model} | Path: ${model_path}${NC}"

    for trainer_experiment in "${TRAINER_EXPERIMENTS[@]}"; do
      IFS=' ' read -r trainer experiment_cfg <<<"${trainer_experiment}"

      for adapter in "${ADAPTERS[@]}"; do
        adapter_overrides=()
        adapter_tag="${adapter}"

        case "${adapter}" in
          base)
            adapter_tag="base"
            adapter_overrides=("trainer.args.num_train_epochs=5")
            ;;
          lora)
            adapter_tag="lora"
            adapter_overrides=("adapter=lora")
            ;;
          qlora)
            adapter_tag="qlora"
            adapter_overrides=("adapter=lora" "model.quantization_config=qlora")
            ;;
          *)
            echo "Unknown adapter ${adapter}" >&2
            exit 1
            ;;
        esac

        for lr in "${LRS[@]}"; do
          for beta1 in "${BETA1S[@]}"; do
            for alpha in "${ALPHAS[@]}"; do
              task_name="tofu_${model}_${forget_split}_${trainer}_${adapter_tag}_lr${lr}_beta1${beta1}_beta2${BETA2}_alpha${alpha}"
              train_output_dir="saves/unlearn/${trainer}/tofu/${forget_split}/${model}/${adapter_tag}/lr-${lr}_beta1-${beta1}_beta2-${BETA2}_alpha-${alpha}"

              echo
              echo -e "${RED}=== Trainer: ${trainer} | Adapter: ${adapter_tag} | Task: ${task_name} ===${NC}"
              echo -e "${RED}Train output: ${train_output_dir}${NC}"

              CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/default_config.yaml --num_processes=4 --main_process_port "${MASTER_PORT}" \
                src/train.py --config-name=unlearn.yaml \
                experiment="${experiment_cfg}" \
                trainer="${trainer}" \
                task_name="${task_name}" \
                model="${model}" \
                forget_split="${forget_split}" \
                retain_split="${retain_split}" \
                paths.output_dir="${train_output_dir}" \
                model.model_args.pretrained_model_name_or_path="${model_path}" \
                retain_logs_path="${retain_logs_path}" \
                trainer.args.per_device_train_batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE}" \
                trainer.args.gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
                trainer.args.ddp_find_unused_parameters=true \
                trainer.args.gradient_checkpointing=true \
                trainer.args.eval_strategy=no \
                trainer.args.eval_on_start=False \
                trainer.args.learning_rate="${lr}" \
                trainer.method_args.beta1="${beta1}" \
                trainer.method_args.beta2="${BETA2}" \
                trainer.method_args.alpha="${alpha}" \
                "${adapter_overrides[@]}"

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
                    echo "Unknown precision ${precision}" >&2
                    exit 1
                    ;;
                esac

                eval_task_name="${task_name}_eval_${precision_tag}"
                eval_output_dir="saves/unlearn/${trainer}/grid_results/tofu/${forget_split}/${model}/${adapter_tag}/${precision_tag}/lr-${lr}_beta1-${beta1}_beta2-${BETA2}_alpha-${alpha}"

                eval_overrides=("model.model_args.pretrained_model_name_or_path=${train_output_dir}")
                if [[ "${adapter_tag}" != "base" ]]; then
                  eval_overrides+=("model.model_handler=LoRAModelForEvaluation")
                fi
                eval_overrides+=("${quant_override[@]}")

                echo -e "${RED}---- Eval precision: ${precision_tag} | Output: ${eval_output_dir}${NC}"

                CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" python src/eval.py \
                  experiment=eval/tofu/default.yaml \
                  forget_split="${forget_split}" \
                  holdout_split="${holdout_split}" \
                  model="${model}" \
                  task_name="${eval_task_name}" \
                  paths.output_dir="${eval_output_dir}" \
                  retain_logs_path="${retain_logs_path}" \
                  "${eval_overrides[@]}"
                done
              done
            done
          done
        done
      done
    done
done
