#!/bin/bash
set -e

# VLA Full Training Pipeline
# This script runs the full training pipeline in the specified order:
# 1. Optional: Pre-train Sensor & Robot State Encoders.
# 2. Main VLA Training using a 2-Stage strategy (Cache -> Live).

# =====================================
# General Training Configuration
# =====================================
# Total epochs are split between Stage 1 (cache) and Stage 2 (live)
TOTAL_MAIN_EPOCHS=30
STAGE1_RATIO=0.5 # 90% of training with cache

# Pre-training epochs (if enabled)
SENSOR_PRETRAIN_EPOCHS=50
ROBOT_PRETRAIN_EPOCHS=300

# Batch sizes
PRETRAIN_BATCH_SIZE=32
MAIN_BATCH_SIZE=4
GRAD_ACCUM=8

# Validation split ratio
VAL_SPLIT=0.05

# Calculate epochs for each stage
STAGE1_EPOCHS=$(printf "%.0f" $(echo "$TOTAL_MAIN_EPOCHS * $STAGE1_RATIO" | bc))
STAGE2_EPOCHS=$(($TOTAL_MAIN_EPOCHS - $STAGE1_EPOCHS))

# Checkpoint paths for resuming
FM_CHECKPOINT="./checkpoints/flow_matching_latest.pt"
REG_CHECKPOINT="./checkpoints/regression_latest.pt"
SENSOR_CLIP_CHECKPOINT="./checkpoints/sensor_clip_best.pth"
ROBOT_STATE_MAE_CHECKPOINT="./checkpoints/robot_state_mae_best.pth"

# Number of GPUs
NUM_GPUS=4

# Fixed training parameters
LR=1e-4
WEIGHT_DECAY=0.01
SENSOR_ENABLED="--sensor_enabled"
FUSION="concat"
DATASET_PATH="/home/najo/NAS/VLA/dataset/New_dataset"
IMG_HEIGHT=360
IMG_WIDTH=640
ANNOTATION_PATH="vlm_annotations.json"
IMPORTANT_WEIGHT=10.0

# VL model is always frozen, as per user's instruction.
FINETUNE_ARGS="--finetune_vl none"

# =================================================================
# 1. PRE-TRAINING (OPTIONAL - UNCOMMENT TO RUN)
# =================================================================

# # --- 1.1 Sensor Encoder Pre-training (CLIP-style) ---
# echo ""
# echo "=============== 1.1 SENSOR ENCODER PRE-TRAINING (CLIP) ==============="
# echo "Epochs: $SENSOR_PRETRAIN_EPOCHS, Batch Size: $PRETRAIN_BATCH_SIZE"
# torchrun --nproc_per_node=$NUM_GPUS TRAIN_SensorImage_CLIP.py \
#     --epochs $SENSOR_PRETRAIN_EPOCHS \
#     --batch_size $PRETRAIN_BATCH_SIZE \
#     --learning_rate $LR \
#     --new_dataset_path $DATASET_PATH \
#     --val_split $VAL_SPLIT \
#     --checkpoint_dir ./checkpoints \
#     --annotation_path $ANNOTATION_PATH \
#     --important_weight $IMPORTANT_WEIGHT \
#     --resume_from $SENSOR_CLIP_CHECKPOINT
# echo "=============== SENSOR ENCODER PRE-TRAINING COMPLETE ==============="
# echo ""


# # --- 1.2 Robot State Encoder Pre-training (MAE) ---
# echo ""
# echo "=============== 1.2 ROBOT STATE ENCODER PRE-TRAINING (MAE) ==============="
# echo "Epochs: $ROBOT_PRETRAIN_EPOCHS, Batch Size: 128"
# torchrun --nproc_per_node=$NUM_GPUS TRAIN_RobotState_MAE.py \
#     --epochs $ROBOT_PRETRAIN_EPOCHS \
#     --batch_size 128 \
#     --learning_rate 3e-4 \
#     --dataset_path $DATASET_PATH \
#     --val_split $VAL_SPLIT \
#     --checkpoint_dir ./checkpoints \
#     --resume_from $ROBOT_STATE_MAE_CHECKPOINT
# echo "=============== ROBOT STATE ENCODER PRE-TRAINING COMPLETE ==============="
# echo ""


# =================================================================
# 2. MAIN VLA TRAINING (REGRESSION)
# =================================================================

# # --- 2.1 Regression Training: Stage 1 (Cache Mode) ---
# echo ""
# echo "=============== 2.1 REGRESSION TRAINING (STAGE 1: CACHE) ==============="
# echo "Epochs: $STAGE1_EPOCHS, Batch Size: $MAIN_BATCH_SIZE, Grad Accum: $GRAD_ACCUM"
# torchrun --nproc_per_node=$NUM_GPUS TRAIN_Regression.py \
#     --epochs $STAGE1_EPOCHS \
#     --batch_size $MAIN_BATCH_SIZE \
#     --grad_accum $GRAD_ACCUM \
#     --lr $LR \
#     --image_resize_height $IMG_HEIGHT \
#     --image_resize_width $IMG_WIDTH \
#     --fusion_strategy $FUSION \
#     $SENSOR_ENABLED \
#     $FINETUNE_ARGS \
#     --val_split $VAL_SPLIT \
#     --load_sensor_encoder_checkpoint $SENSOR_CLIP_CHECKPOINT \
#     --load_robot_state_encoder_checkpoint $ROBOT_STATE_MAE_CHECKPOINT \
#     --use_cache
# echo "=============== REGRESSION STAGE 1 COMPLETE ==============="
# echo ""

# # --- 2.2 Regression Training: Stage 2 (Live Mode) ---
# echo ""
# echo "=============== 2.2 REGRESSION TRAINING (STAGE 2: LIVE) ==============="
# echo "Epochs: $STAGE2_EPOCHS, Batch Size: $MAIN_BATCH_SIZE, Grad Accum: $GRAD_ACCUM"
# torchrun --nproc_per_node=$NUM_GPUS TRAIN_Regression.py \
#     --epochs $STAGE2_EPOCHS \
#     --batch_size $MAIN_BATCH_SIZE \
#     --grad_accum $GRAD_ACCUM \
#     --lr $LR \
#     --image_resize_height $IMG_HEIGHT \
#     --image_resize_width $IMG_WIDTH \
#     --fusion_strategy $FUSION \
#     $SENSOR_ENABLED \
#     $FINETUNE_ARGS \
#     --val_split $VAL_SPLIT \
#     --resume $REG_CHECKPOINT
# echo "=============== REGRESSION STAGE 2 COMPLETE ==============="
# echo ""


# =================================================================
# 3. MAIN VLA TRAINING (FLOW MATCHING)
# =================================================================

# --- 3.1 Flow Matching Training: Stage 1 (Cache Mode) ---
# echo ""
# echo "=============== 3.1 FLOW MATCHING TRAINING (STAGE 1: CACHE) ==============="
# echo "Epochs: $STAGE1_EPOCHS, Batch Size: $MAIN_BATCH_SIZE, Grad Accum: $GRAD_ACCUM"
# torchrun --nproc_per_node=$NUM_GPUS TRAIN_FlowMatching.py \
#     --epochs $STAGE1_EPOCHS \
#     --batch_size $MAIN_BATCH_SIZE \
#     --grad_accum $GRAD_ACCUM \
#     --lr $LR \
#     --image_resize_height $IMG_HEIGHT \
#     --image_resize_width $IMG_WIDTH \
#     --fusion_strategy $FUSION \
#     $SENSOR_ENABLED \
#     $FINETUNE_ARGS \
#     --val_split $VAL_SPLIT \
#     --load_sensor_encoder_checkpoint $SENSOR_CLIP_CHECKPOINT \
#     --load_robot_state_encoder_checkpoint $ROBOT_STATE_MAE_CHECKPOINT \
#     --use_cache \
#     --resume $FM_CHECKPOINT
# echo "=============== FLOW MATCHING STAGE 1 COMPLETE ==============="
# echo ""

# --- 3.2 Flow Matching Training: Stage 2 (Live Mode) ---
echo ""
echo "=============== 3.2 FLOW MATCHING TRAINING (STAGE 2: LIVE) ==============="
echo "Epochs: $STAGE2_EPOCHS, Batch Size: $MAIN_BATCH_SIZE, Grad Accum: $GRAD_ACCUM"
torchrun --nproc_per_node=$NUM_GPUS TRAIN_FlowMatching.py \
    --epochs $STAGE2_EPOCHS \
    --batch_size $MAIN_BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --lr $LR \
    --image_resize_height $IMG_HEIGHT \
    --image_resize_width $IMG_WIDTH \
    --fusion_strategy $FUSION \
    $SENSOR_ENABLED \
    $FINETUNE_ARGS \
    --val_split $VAL_SPLIT \
    --resume $FM_CHECKPOINT
echo "=============== FLOW MATCHING STAGE 2 COMPLETE ==============="
echo ""

echo "✅✅✅ VLA FULL TRAINING PIPELINE FINISHED ✅✅✅"
