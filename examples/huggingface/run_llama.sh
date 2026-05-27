#!/bin/bash
#SBATCH --argos=no
#SBATCH --account=project_2001659
#SBATCH --partition=gpumedium
#SBATCH --time=15
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:gh200:4
#SBATCH --nodes=1
#SBATCH --mem=240G
#SBATCH --cpus-per-task=288

module purge
module load csc-tools
module load python-pytorch

source /scratch/project_2001659/amoisala/liger_kernel_fork/Liger-Kernel/venv/bin/activate

# We are putting the cache in the ramdisk, stored in
# memory. Alternatively store it to the project's scratch.

export HF_HOME=/scratch/$SLURM_JOB_ACCOUNT/$USER/hf-cache/
mkdir -p $HF_HOME

python -m torch.distributed.run --nnodes=1 --nproc-per-node=4 training.py \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --eval_strategy "no" \
    --save_strategy "no" \
    --learning_rate 6e-6 \
    --weight_decay 0.05 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --include_num_input_tokens_seen \
    --report_to none \
    --fsdp "full_shard auto_wrap" \
    --fsdp_config config/fsdp_config.json \
    --seed 42 \
    --use_liger True \
    --output_dir alpaca_finetuning \
    --gradient_checkpointing False \
    --max_seq_length 512
