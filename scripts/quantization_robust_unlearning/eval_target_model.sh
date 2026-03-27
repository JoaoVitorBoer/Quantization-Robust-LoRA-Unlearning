#!/bin/bash

#SBATCH --output=/home/joaoabitante/Sout/%j__%x.out
#SBATCH --error=/home/joaoabitante/Sout/%j__%x.out

#SBATCH --nodes=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=30GB
#SBATCH --time=2-00:00:00
#SBATCH --gpus=2

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
    "Books"
    "News"
)

# Keep evaluation in these precisions
PRECISIONS=(fp 8bit 4bit)

for data_split in "${DATA_SPLITS[@]}"; do
  retain_logs_path="saves/eval/muse_${MODEL}_${data_split}_retrain/MUSE_EVAL.json"
  base_model_path="muse-bench/MUSE-${data_split}_target"
  task_name="muse_${MODEL}_${data_split}_target"
  echo -e "${RED}--- Data split: ${data_split} | target: ${base_model_path} ---${NC}"

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
    eval_output_dir="saves/unlearn/paper_results/target_${data_split}_${MODEL}/${precision_tag}"

    echo -e "${RED}---- Eval precision: ${precision_tag} | Output: ${eval_output_dir}${NC}"

    CUDA_VISIBLE_DEVICES=0,1 python src/eval.py \
      experiment=eval/muse/default.yaml \
      model="${MODEL}" \
      data_split="${data_split}" \
      task_name="${eval_task_name}" \
      paths.output_dir="${eval_output_dir}" \
      retain_logs_path="${retain_logs_path}" \
      model.model_args.pretrained_model_name_or_path="${base_model_path}" \
      "${quant_override[@]}"
  done
done
