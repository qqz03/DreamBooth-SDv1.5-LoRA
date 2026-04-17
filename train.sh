#!/bin/bash
set -e

# -- Usage --------------------------------------------------------------------
# bash train.sh A    → Config A: rank=128
# bash train.sh B    → Config B: rank=16
# ----------------------------------------------------------------------------

CONFIG=${1:-A}

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export DIFFUSERS_OFFLINE=1

MODEL_NAME="/home/qianqz/Model/stable-diffsion-v1.5-fp32"
INSTANCE_DIR="/home/qianqz/DreamBooth/resize-data"
CLASS_DIR="/home/qianqz/DreamBooth/class-data"

# -- Experimental Configuration: rank is the only controlled variable ---------
if [ "$CONFIG" = "A" ]; then
    LORA_RANK=128
    DESC="rank=128 high-capacity"
elif [ "$CONFIG" = "B" ]; then
    LORA_RANK=16
    DESC="rank=16 regularized"
else
    echo "Usage: bash train.sh A or bash train.sh B"; exit 1
fi

# -- Fixed Hyperparameters: identical across both configurations --------------
MAX_STEPS=600
BATCH_SIZE=1
LR=1e-4
LR_SCHEDULER="constant"
LR_WARMUP=0
SEED=42
CHECKPOINT_STEPS=200
NUM_CLASS_IMAGES=200

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/home/qianqz/DreamBooth/output/config${CONFIG}_${TIMESTAMP}"
LOG_FILE="${OUTPUT_DIR}/train.log"

mkdir -p $CLASS_DIR $OUTPUT_DIR

echo "============================================================"
echo " Config ${CONFIG}: ${DESC}"
echo " Timestamp        : $TIMESTAMP"
echo " Output directory : $OUTPUT_DIR"
echo " Fixed parameters : lr=${LR} scheduler=${LR_SCHEDULER} steps=${MAX_STEPS}"
echo " Controlled factor: rank=${LORA_RANK}"
echo " Start time       : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

TRAIN_START=$(date +%s)

accelerate launch \
~/diffusers/examples/dreambooth/train_dreambooth_lora.py \
--pretrained_model_name_or_path=$MODEL_NAME \
--instance_data_dir=$INSTANCE_DIR \
--class_data_dir=$CLASS_DIR \
--output_dir=$OUTPUT_DIR \
--instance_prompt="a photo of sks dragon toy" \
--class_prompt="a photo of dragon toy" \
--with_prior_preservation \
--prior_loss_weight=1.0 \
--num_class_images=$NUM_CLASS_IMAGES \
--resolution=512 \
--train_batch_size=$BATCH_SIZE \
--gradient_accumulation_steps=1 \
--learning_rate=$LR \
--lr_scheduler=$LR_SCHEDULER \
--lr_warmup_steps=$LR_WARMUP \
--max_train_steps=$MAX_STEPS \
--rank=$LORA_RANK \
--mixed_precision="no" \
--seed=$SEED \
--checkpointing_steps=$CHECKPOINT_STEPS \
2>&1 | tee "$LOG_FILE"

ELAPSED=$(( $(date +%s) - TRAIN_START ))
echo ""
echo "============================================================"
echo " Training completed. Elapsed time: $(( ELAPSED/3600 ))h $(( (ELAPSED%3600)/60 ))m $(( ELAPSED%60 ))s"
echo " Output directory : $OUTPUT_DIR"
echo "============================================================"
