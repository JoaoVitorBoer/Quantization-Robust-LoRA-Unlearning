#!/bin/bash

#SBATCH --output=/home/joaoabitante/Sout/%j__%x.out
#SBATCH --error=/home/joaoabitante/Sout/%j__%x.out

#SBATCH --nodes=1                        # Number of nodes to use
#SBATCH --cpus-per-task=16               # Number of CPU cores per task
#SBATCH --mem=30G                        # Memory per node (e.g., 16 gigabytes)
#SBATCH --time=2-00:00:00                # Wall-clock time limit (HH:MM:SS)
#SBATCH --gpus=3

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

BASE_MODEL="Llama-2-7b-hf"

PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=8

DATA_SPLITS=(
  "News"
  "Books"
)

TRAINERS=(
  "GradAscent"
  "NPO"
  "SimNPO"
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

for data_split in "${DATA_SPLITS[@]}"; do
  retain_logs_path="saves/eval/muse_${BASE_MODEL}_${data_split}_retrain/MUSE_EVAL.json"
  echo "--- Data split: ${data_split} | retain_logs_path: ${retain_logs_path} ---"

  for trainer in "${TRAINERS[@]}"; do
    for adapter in "${ADAPTERS[@]}"; do
      adapter_overrides=()
      adapter_tag="${adapter}"
      experiment_cfg="unlearn/muse/default.yaml"
      model_cfg="${BASE_MODEL}"

      case "${adapter}" in
        base)
          adapter_tag="base"
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

      task_name="muse_${BASE_MODEL}_${data_split}_${trainer}_${adapter_tag}"
      train_output_dir="saves/unlearn/${BASE_MODEL}/${data_split}/${trainer}/${adapter_tag}"

      echo
      echo "=== Trainer: ${trainer} | Adapter: ${adapter_tag} | Task: ${task_name} ==="
      echo "Train output: ${train_output_dir}"

      CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --config_file configs/accelerate/default_config.yaml --num_processes=3 --main_process_port "${MASTER_PORT}" \
        src/train.py --config-name=unlearn.yaml \
        experiment="${experiment_cfg}" \
        model="${model_cfg}" \
        data_split="${data_split}" \
        trainer="${trainer}" \
        task_name="${task_name}" \
        paths.output_dir="${train_output_dir}" \
        retain_logs_path="${retain_logs_path}" \
        trainer.args.per_device_train_batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE}" \
        trainer.args.gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=true \
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
        eval_output_dir="saves/unlearn/${trainer}/muse/${data_split}/${BASE_MODEL}/${adapter_tag}/evals/${precision_tag}"

        eval_overrides=("model.model_args.pretrained_model_name_or_path=${train_output_dir}")
        if [[ "${adapter_tag}" != "base" ]]; then
          eval_overrides+=("model.model_handler=LoRAModelForEvaluation")
        fi
        eval_overrides+=("${quant_override[@]}")

        echo "---- Eval precision: ${precision_tag} | Output: ${eval_output_dir}"

        CUDA_VISIBLE_DEVICES=0 python src/eval.py \
          experiment=eval/muse/default.yaml \
          model="${model_cfg}" \
          data_split="${data_split}" \
          task_name="${eval_task_name}" \
          paths.output_dir="${eval_output_dir}" \
          retain_logs_path="${retain_logs_path}" \
          "${eval_overrides[@]}"
      done
    done
  done
done
