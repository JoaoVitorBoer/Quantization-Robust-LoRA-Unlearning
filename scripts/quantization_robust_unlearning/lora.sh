#!/bin/bash

#SBATCH --output=/home/joaoabitante/Sout/%j__%x.out
#SBATCH --error=/home/joaoabitante/Sout/%j__%x.out

#SBATCH --nodes=1                        # Number of nodes to use
#SBATCH --cpus-per-task=16               # Number of CPU cores per task
#SBATCH --mem=30G                        # Memory per node (e.g., 16 gigabytes)
#SBATCH --time=2-00:00:00                  # Wall-clock time limit (HH:MM:SS)
#SBATCH --gpus=3

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"


per_device_train_batch_size=2
gradient_accumulation_steps=8


model=Llama-2-7b-hf

data_splits=(
    "Books"
)

trainers=(
    "SimNPO"
)

# #########################################################
# #################### MUSE Unlearning ####################
# #########################################################


for data_split in "${data_splits[@]}"; do
    for trainer in "${trainers[@]}"; do

        task_name=lora_muse_${model}_${data_split}_${trainer}

        echo ===================================================================
        echo ================== TRAINING =====================================
        echo ===================================================================

        CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --config_file configs/accelerate/default_config.yaml --num_processes=3 --main_process_port $MASTER_PORT \
        src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/muse/default.yaml \
        adapter=lora \
        model=${model} \
        data_split=${data_split} \
        trainer=${trainer} \
        task_name=${task_name} \
        retain_logs_path=saves/eval/muse_Llama-2-7b-hf_${data_split}_retrain/MUSE_EVAL.json \
        trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
        trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=true\
        model.quantization_config=qlora \

        echo ===================================================================
        echo ================== EVALUATION =====================================
        echo ===================================================================

        CUDA_VISIBLE_DEVICES=0 python src/eval.py \
        experiment=eval/muse/default.yaml \
        data_split=${data_split} \
        task_name=${task_name} \
        model=${model} \
        model.model_handler=LoRAModelForEvaluation \
        model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
        paths.output_dir=saves/unlearn/${trainer}/evals \
        retain_logs_path=saves/eval/muse_Llama-2-7b-hf_${data_split}_retrain/MUSE_EVAL.json
    done
done

