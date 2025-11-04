#!/bin/bash

# Flow Matching Training Script for VLA
# Using Optimal Transport Conditional Flow Matching (OT-CFM)
# Based on Pi0 paper: https://arxiv.org/pdf/2410.24164v1

# Number of GPUs
NUM_GPUS=4

# Training configuration
MODEL_TYPE="flow_matching"  # Use flow matching
EPOCHS=100
BATCH_SIZE=32  # Per GPU
GRAD_ACCUM=4  # Effective batch size = 4 GPUs × 1 × 4 = 16
LR=1e-4
WEIGHT_DECAY=0.01

# Sensor configuration
SENSOR_ENABLED=true
SENSOR_LOSS_WEIGHT=2.0

# Fusion strategy
FUSION="concat"  # Options: concat, cross_attention, gated, none

# Data paths
OLD_DATA="/home/najo/NAS/VLA/dataset/recv_all_*"
NEW_DATA="/home/najo/NAS/VLA/dataset/New_dataset"

# Sampling weights (new:old ratio)
OLD_WEIGHT=1.0
NEW_WEIGHT=3.0

# Image resize for faster training
IMG_HEIGHT=360
IMG_WIDTH=640

echo "🚀 Starting Flow Matching Training"
echo "   Model: ${MODEL_TYPE}"
echo "   GPUs: ${NUM_GPUS}"
echo "   Batch Size (per GPU): ${BATCH_SIZE}"
echo "   Gradient Accumulation: ${GRAD_ACCUM}"
echo "   Effective Batch Size: $((NUM_GPUS * BATCH_SIZE * GRAD_ACCUM))"
echo "   Learning Rate: ${LR}"
echo "   Sensor Enabled: ${SENSOR_ENABLED}"
echo "   Fusion Strategy: ${FUSION}"
echo ""

# Run training with torchrun
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29500 \
    TRAIN_Unified.py \
    --model-type ${MODEL_TYPE} \
    --mode train \
    --dataset_dir "${OLD_DATA}" \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --grad_accum ${GRAD_ACCUM} \
    --lr ${LR} \
    --sensor_enabled \
    --sensor_loss_weight ${SENSOR_LOSS_WEIGHT} \
    --fusion_strategy ${FUSION} \
    --image_resize_height ${IMG_HEIGHT} \
    --image_resize_width ${IMG_WIDTH} \
    --val_split 0.1 \
    --num_workers 4 \
    --sched_on step \
    --resume /home/najo/NAS/VLA/Insertion_VLAv2/checkpoints/flow_matching_latest.pt


echo ""
echo "✅ Flow Matching Training Complete!"
echo "   Checkpoints saved in: ./checkpoints/"
echo "   Model type: flow_matching"
