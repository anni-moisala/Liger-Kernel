#!/bin/bash
#SBATCH --argos=no
#SBATCH --account=project_2001659
#SBATCH --partition=gpumedium
#SBATCH --time=60
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:gh200:4
#SBATCH --nodes=1
#SBATCH --mem=240G
#SBATCH --cpus-per-task=288

module purge
module load csc-tools
module load python-pytorch

source /scratch/project_2001659/amoisala/Liger-Kernel/venv/bin/activate

# We are putting the cache in the ramdisk, stored in
# memory. Alternatively store it to the project's scratch.

#export HF_HOME=/scratch/$SLURM_JOB_ACCOUNT/$USER/hf-cache/
export HF_HOME=/dev/shm/$USER/hf-cache
export TORCHINDUCTOR_CACHE_DIR=/dev/shm/$USER/
mkdir -p $HF_HOME

## Benchmarking Script
## Runs the training script with different configurations and logs the results

MODEL_TYPE="mistral"
MODEL_PATH="mistralai/Mistral-7B-v0.1"
USE_LIGER_VALUES=("True" "False")
BATCH_SIZE_VALUES=(64 128 192)
NUM_REP=5
MAX_STEPS=20
DATASET_PATH="tatsu-lab/alpaca"

RESULTS_DIR=./results
mkdir -p $RESULTS_DIR

for USE_LIGER in "${USE_LIGER_VALUES[@]}"; do
    for BATCH_SIZE in "${BATCH_SIZE_VALUES[@]}"; do
        echo "Running with use_liger=$USE_LIGER and batch_size=$BATCH_SIZE"

        for ((i=1; i<=NUM_REP; i++)); do

            LOG_FILE="${RESULTS_DIR}/${MODEL_TYPE}_use_liger_${USE_LIGER}_batch_size_${BATCH_SIZE}_rep_${i}.log"

            if python -m torch.distributed.run --nnodes=1 --nproc-per-node=4 training.py \
                --bf16 True \
                --num_train_epochs 1 \
                --max_steps $MAX_STEPS \
                --model_name $MODEL_PATH \
                --dataset $DATASET_PATH \
                --per_device_train_batch_size $BATCH_SIZE \
                --per_device_eval_batch_size 16 \
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
                --use_liger $USE_LIGER \
                --output_dir model_output_dir \
		        --gradient_checkpointing False \
                --max_seq_length 512 \
                > "$LOG_FILE" 2>&1
            then
                echo "Run succeeded"
            else
                EXIT_CODE=$?
                echo "Run failed with exit code $EXIT_CODE" | tee -a "$LOG_FILE"

                if grep -qi "out of memory" "$LOG_FILE"; then
                    echo "OOM detected, continuing benchmark..." | tee -a "$LOG_FILE"
                else
                    echo "Non-OOM failure, continuing anyway..." | tee -a "$LOG_FILE"
                fi
            fi
            sleep 5
        done
    done
done
