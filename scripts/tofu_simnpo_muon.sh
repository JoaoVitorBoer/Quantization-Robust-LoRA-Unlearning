#!/bin/bash

set -euo pipefail
export HYDRA_FULL_ERROR=1

#SBATCH --output=/home/joaoabitante/Sout/%j__%x.out
#SBATCH --error=/home/joaoabitante/Sout/%j__%x.out

#SBATCH --nodes=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --gpus=2

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

models=(
    # "Llama-3.2-1B-Instruct"
    "Llama-3.2-3B-Instruct"
    # "Llama-3.1-8B-Instruct"
)
splits=(
    "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)

per_device_train_batch_size=4 # on two gpus would make effective batch size 32
gradient_accumulation_steps=4

simnpo_delta=0.0
simnpo_beta=0.7
simnpo_alpha=0.1
simnpo_gamma=1.0

muon_momentum=0.95
muon_ns_steps=5
muon_adjust_lr_fn=match_rms_adamw
adamw_betas="[0.9,0.95]"
adamw_eps=1e-8


########################################################################################################################
####################################### Unlearn TOFU models with SimNPO + Muon ########################################
########################################################################################################################


for split in "${splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)

    for model in "${models[@]}"; do
        task_name=tofu_${model}_${forget_split}_SimNPO_Muon
        model_path=open-unlearning/tofu_${model}_full
        train_output_dir=saves/unlearn/${task_name}

        echo ${task_name}: Unlearning ${model_path} using SimNPO with Muon

        # Unlearn
        CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
        src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default.yaml \
        trainer=SimNPO \
        task_name=${task_name} \
        paths.output_dir=${train_output_dir} \
        model=${model} \
        forget_split=${forget_split} \
        retain_split=${retain_split} \
        model.model_args.pretrained_model_name_or_path=${model_path} \
        retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
        trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
        trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=true \
        trainer.args.eval_strategy=no \
        trainer.args.eval_on_start=False \
        trainer.args.do_eval=False \
        trainer.method_args.delta=$simnpo_delta \
        trainer.method_args.beta=$simnpo_beta \
        trainer.method_args.alpha=$simnpo_alpha \
        trainer.method_args.gamma=$simnpo_gamma \
        trainer.method_args.optimizer_name=muon \
        trainer.method_args.optimizer_kwargs.muon_momentum=$muon_momentum \
        trainer.method_args.optimizer_kwargs.muon_ns_steps=$muon_ns_steps \
        trainer.method_args.optimizer_kwargs.muon_adjust_lr_fn=$muon_adjust_lr_fn \
        trainer.method_args.optimizer_kwargs.adamw_betas=${adamw_betas} \
        trainer.method_args.optimizer_kwargs.adamw_eps=$adamw_eps

        # if [[ ! -f "${train_output_dir}/config.json" ]]; then
        #     echo "Skipping eval for ${task_name}: ${train_output_dir}/config.json not found."
        #     continue
        # fi

        # Eval
        CUDA_VISIBLE_DEVICES=0 python src/eval.py \
        experiment=eval/tofu/default.yaml \
        forget_split=${forget_split} \
        holdout_split=${holdout_split} \
        model=${model} \
        task_name=${task_name} \
        model.model_args.pretrained_model_name_or_path=${train_output_dir} \
        paths.output_dir=${train_output_dir}/evals \
        retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json
    done
done
