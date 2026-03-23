#!/bin/bash

#SBATCH --output=/home/joaoabitante/Sout/%j__%x.out
#SBATCH --error=/home/joaoabitante/Sout/%j__%x.out

#SBATCH --nodes=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --gpus=2

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
  "Books"
  "News"
)

# Edit this list if you only want to run a subset of methods.
TRAINERS=(
  #"GradAscent"
  "GradDiff"
  # "NPO"
  # "SimNPO"
)

PER_DEVICE_TRAIN_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=8

MUON_MOMENTUM=0.95
MUON_NS_STEPS=5
MUON_ADJUST_LR_FN=match_rms_adamw
ADAMW_BETAS="[0.9,0.95]"
ADAMW_EPS=1e-8

for data_split in "${DATA_SPLITS[@]}"; do
  retain_logs_path="saves/eval/muse_${MODEL}_${data_split}_retrain/MUSE_EVAL.json"
  echo "--- Data split: ${data_split} | retain_logs_path: ${retain_logs_path} ---"

  for trainer in "${TRAINERS[@]}"; do
    task_name="muse_${MODEL}_${data_split}_${trainer}_Muon"
    train_output_dir="saves/unlearn/muon/${MODEL}/${data_split}/${trainer}"

    echo
    echo "=== Trainer: ${trainer} | Task: ${task_name} ==="
    echo "Train output: ${train_output_dir}"

    CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
      --config_file configs/accelerate/default_config.yaml \
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
      trainer.args.ddp_find_unused_parameters=true \
      trainer.args.gradient_checkpointing=true \
      trainer.args.eval_strategy=no \
      trainer.args.eval_on_start=False \
      trainer.args.do_eval=False \
      trainer.method_args.optimizer_name=muon \
      trainer.method_args.optimizer_kwargs.muon_momentum="${MUON_MOMENTUM}" \
      trainer.method_args.optimizer_kwargs.muon_ns_steps="${MUON_NS_STEPS}" \
      trainer.method_args.optimizer_kwargs.muon_adjust_lr_fn="${MUON_ADJUST_LR_FN}" \
      trainer.method_args.optimizer_kwargs.adamw_betas="${ADAMW_BETAS}" \
      trainer.method_args.optimizer_kwargs.adamw_eps="${ADAMW_EPS}"

    CUDA_VISIBLE_DEVICES=0 python src/eval.py \
      experiment=eval/muse/default.yaml \
      model="${MODEL}" \
      data_split="${data_split}" \
      task_name="${task_name}" \
      model.model_args.pretrained_model_name_or_path="${train_output_dir}" \
      paths.output_dir="${train_output_dir}/evals" \
      retain_logs_path="${retain_logs_path}"
  done
done
