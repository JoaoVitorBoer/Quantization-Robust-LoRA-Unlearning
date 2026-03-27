#!/bin/bash

#SBATCH --output=/home/joaoabitante/Sout/%j__%x.out
#SBATCH --error=/home/joaoabitante/Sout/%j__%x.out

#SBATCH --nodes=1                        # Number of nodes to use
#SBATCH --cpus-per-task=16               # Number of CPU cores per task
#SBATCH --mem=30G                        # Memory per node (e.g., 30 gigabytes)
#SBATCH --time=2-00:00:00                # Wall-clock time limit (HH:MM:SS)
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
echo "Master Port: ${MASTER_PORT}"

PER_DEVICE_TRAIN_BATCH_SIZE=2  # On two GPUs this yields an effective batch size of 32
GRADIENT_ACCUMULATION_STEPS=2

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

SPLITS=(
  "forget01 holdout01 retain99"
  "forget05 holdout05 retain95"
  "forget10 holdout10 retain90"
)

for split in "${SPLITS[@]}"; do
  read -r forget_split holdout_split retain_split <<<"${split}"
  echo "--- Split: forget=${forget_split} | holdout=${holdout_split} | retain=${retain_split} ---"

  for model in "${MODELS[@]}"; do
    model_path="open-unlearning/tofu_${model}_full"
    retain_logs_path="saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json"
    echo "Model: ${model} | Path: ${model_path}"

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

        task_name="tofu_${model}_${forget_split}_${trainer}_${adapter_tag}"
        train_output_dir="saves/unlearn/${model}/${forget_split}/${trainer}/${adapter_tag}"

        echo
        echo "=== Trainer: ${trainer} | Adapter: ${adapter_tag} | Task: ${task_name} ==="
        echo "Train output: ${train_output_dir}"

        CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/default_config.yaml --num_processes=4 --main_process_port "${MASTER_PORT}" --tee "0:3"\
          src/train.py --config-name=unlearn.yaml \
          experiment="${experiment_cfg}" \
          trainer="${trainer}" \
          task_name="${task_name}" \
          model="${model}" \
          forget_split="${forget_split}" \
          paths.output_dir="${train_output_dir}" \
          retain_split="${retain_split}" \
          model.model_args.pretrained_model_name_or_path="${model_path}" \
          retain_logs_path="${retain_logs_path}" \
          trainer.args.per_device_train_batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE}" \
          trainer.args.gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
          trainer.args.ddp_find_unused_parameters=true \
          trainer.args.gradient_checkpointing=true \
          trainer.args.eval_strategy=no \
          trainer.args.eval_on_start=False \
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
          eval_output_dir="saves/unlearn/${trainer}/tofu/${forget_split}/${model}/${adapter_tag}/evals/${precision_tag}"

          eval_overrides=("model.model_args.pretrained_model_name_or_path=${train_output_dir}")
          if [[ "${adapter_tag}" != "base" ]]; then
            eval_overrides+=("model.model_handler=LoRAModelForEvaluation")
          fi
          eval_overrides+=("${quant_override[@]}")

          echo "---- Eval precision: ${precision_tag} | Output: ${eval_output_dir}"

          CUDA_VISIBLE_DEVICES=0 python src/eval.py \
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
