#!/bin/bash
set -e

# VLA Full Training Pipeline
# This script runs the full training pipeline in the specified order:
# 0. Build VL Cache (required for cache mode training)
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
ROBOT_PRETRAIN_EPOCHS=200

# Batch sizes
PRETRAIN_BATCH_SIZE=32
MAIN_BATCH_SIZE=8
GRAD_ACCUM=8

# Validation split ratio
VAL_SPLIT=0.05

# Calculate epochs for each stage
STAGE1_EPOCHS=$(printf "%.0f" $(echo "$TOTAL_MAIN_EPOCHS * $STAGE1_RATIO" | bc))
STAGE2_EPOCHS=$(($TOTAL_MAIN_EPOCHS - $STAGE1_EPOCHS))

# Checkpoint paths for resuming
FM_CHECKPOINT="./checkpoints/flow_matching_best.pt"
REG_CHECKPOINT="./checkpoints/regression_best.pt"
SENSOR_CLIP_CHECKPOINT="./checkpoints/sensor_clip_best.pth"
ROBOT_STATE_MAE_CHECKPOINT="./checkpoints/robot_state_mae_best.pth"

# Number of GPUs
NUM_GPUS=4

# Fixed training parameters
LR=1e-4
WEIGHT_DECAY=0.01
SENSOR_ENABLED="--sensor_enabled"
FUSION="concat"
CACHE_ROOT="/home/najo/NAS/VLA/dataset/cache"
QWEN_CACHE_ROOT="$CACHE_ROOT/qwen_vl_features"
DATASET_PATHS=("/home/najo/NAS/VLA/dataset/New_dataset" "/home/najo/NAS/VLA/dataset/New_dataset2")
IMG_HEIGHT=360
IMG_WIDTH=640
ANNOTATION_PATH="vlm_annotations.json"
IMPORTANT_WEIGHT=10.0

# VL model is always frozen, as per user's instruction.
FINETUNE_ARGS="--finetune_vl none"

# =================================================================
# 1. PRE-TRAINING (OPTIONAL - UNCOMMENT TO RUN)
# =================================================================

# --- 1.0 Build CLIP VLM Cache (for Sensor Encoder Pre-training) ---
# echo ""
# echo "=============== 1.0 SENSOR CLIP VLM CACHE BUILDING ==============="
# echo "Building VLM feature cache for CLIP pre-training (in parallel across $NUM_GPUS GPUs)..."
# torchrun --nproc_per_node=$NUM_GPUS cache_clip_vlm_features.py \
#     --new_dataset_paths "${DATASET_PATHS[@]}" \
#     --annotation_path $ANNOTATION_PATH \
#     --cache_root $CACHE_ROOT \
#     --vlm_model "Qwen/Qwen2.5-VL-7B-Instruct"
# echo "=============== SENSOR CLIP VLM CACHE BUILDING COMPLETE ==============="
# echo ""

# # --- 1.2 Robot State Encoder Pre-training (MAE) ---
# echo ""
# echo "=============== 1.2 ROBOT STATE ENCODER PRE-TRAINING (MAE) ==============="
# echo "Epochs: $ROBOT_PRETRAIN_EPOCHS, Batch Size: 128"
# torchrun --nproc_per_node=$NUM_GPUS TRAIN_RobotState_MAE.py \
#     --epochs $ROBOT_PRETRAIN_EPOCHS \
#     --batch_size 128 \
#     --learning_rate 3e-4 \
#     --dataset_paths "${DATASET_PATHS[@]}" \
#     --val_split $VAL_SPLIT \
#     --checkpoint_dir ./checkpoints \
#     --resume_from $ROBOT_STATE_MAE_CHECKPOINT
# echo "=============== ROBOT STATE ENCODER PRE-TRAINING COMPLETE ==============="
# echo ""

# --- 1.1 Sensor Encoder Pre-training (CLIP-style) ---
# echo ""
# echo "=============== 1.1 SENSOR ENCODER PRE-TRAINING (CLIP) ==============="
# echo "Epochs: $SENSOR_PRETRAIN_EPOCHS, Batch Size: $PRETRAIN_BATCH_SIZE"
# torchrun --nproc_per_node=$NUM_GPUS TRAIN_SensorImage_CLIP.py \
#     --epochs $SENSOR_PRETRAIN_EPOCHS \
#     --batch_size $PRETRAIN_BATCH_SIZE \
#     --learning_rate $LR \
#     --new_dataset_paths "${DATASET_PATHS[@]}" \
#     --val_split $VAL_SPLIT \
#     --checkpoint_dir ./checkpoints \
#     --annotation_path $ANNOTATION_PATH \
#     --important_weight $IMPORTANT_WEIGHT \
#     --cache_root $CACHE_ROOT \
#     --vlm_model "Qwen/Qwen2.5-VL-7B-Instruct" \
#     --resume_from $SENSOR_CLIP_CHECKPOINT
# echo "=============== SENSOR ENCODER PRE-TRAINING COMPLETE ==============="
# echo ""

# =================================================================
# 0. VL CACHE BUILDING (REQUIRED FOR CACHE MODE)
# =================================================================
# echo ""
# echo "=============== 0. VL CACHE BUILDING ==============="
# echo "Building VL feature cache for faster training..."
# torchrun --nproc_per_node=$NUM_GPUS TRAIN_FlowMatching.py \
#     --mode cache \
#     --batch_size $MAIN_BATCH_SIZE \
#     --num_workers 8 \
#     --image_resize_height $IMG_HEIGHT \
#     --image_resize_width $IMG_WIDTH \
#     --cache_loader_only \
#     --cache_root $QWEN_CACHE_ROOT
# echo "=============== VL CACHE BUILDING COMPLETE ==============="
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
#     --use_cache \
#     --cache_root $QWEN_CACHE_ROOT \
#     --resume $REG_CHECKPOINT
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
#     --cache_root $QWEN_CACHE_ROOT \
#     --resume $REG_CHECKPOINT
# echo "=============== REGRESSION STAGE 2 COMPLETE ==============="
# echo ""


# =================================================================
# 3. MAIN VLA TRAINING (FLOW MATCHING)
# =================================================================

# --- 3.1 Flow Matching Training: Stage 1 (Cache Mode) ---
echo ""
echo "=============== 3.1 FLOW MATCHING TRAINING (STAGE 1: CACHE) ==============="
echo "Epochs: $STAGE1_EPOCHS, Batch Size: $MAIN_BATCH_SIZE, Grad Accum: $GRAD_ACCUM"
torchrun --nproc_per_node=$NUM_GPUS TRAIN_FlowMatching.py \
    --epochs $STAGE1_EPOCHS \
    --batch_size $MAIN_BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --lr $LR \
    --image_resize_height $IMG_HEIGHT \
    --image_resize_width $IMG_WIDTH \
    --fusion_strategy $FUSION \
    $SENSOR_ENABLED \
    $FINETUNE_ARGS \
    --val_split $VAL_SPLIT \
    --load_sensor_encoder_checkpoint $SENSOR_CLIP_CHECKPOINT \
    --load_robot_state_encoder_checkpoint $ROBOT_STATE_MAE_CHECKPOINT \
    --use_cache \
    --cache_root $QWEN_CACHE_ROOT \
    # --resume $FM_CHECKPOINT
echo "=============== FLOW MATCHING STAGE 1 COMPLETE ==============="
echo ""

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
    --cache_root $QWEN_CACHE_ROOT \
    # --resume $FM_CHECKPOINT
echo "=============== FLOW MATCHING STAGE 2 COMPLETE ==============="
echo ""

echo "✅✅✅ VLA FULL TRAINING PIPELINE FINISHED ✅✅✅"

# =================================================================
# 4. MODEL TEST 
# =================================================================
#
# python benchmark_realtime_inference.py \
#     --checkpoint-regression checkpoints/regression_best.pt \
#     --checkpoint-flow checkpoints/flow_matching_best.pt \
#     --dataset-dir /home/najo/NAS/VLA/dataset/New_dataset/Blue_point/episode_20251030_025856 \
#     --cache-root ./cache/qwen_vl_features \
#     --device cuda:0 \
#     --num-iterations 20 \
#     --compare-views \
#     --parallel-view-encoding

# # 센서 인코더 UAMP 등 성능 분석 시각화

# python analyze_sensor_embeddings.py \
#     --sensor-checkpoint checkpoints/sensor_clip_best.pth \
#     --dataset-paths \
#     /home/najo/NAS/VLA/dataset/New_dataset2/Green_point/data_collection_20251108_054442 \
#     --output-dir analysis/sensor_tsne \
#     --max-samples-per-episode 200 \
#     --method both \
#     --device cuda:0

# # 로봇 엔코더 성능 분석

# python analyze_robot_reproducibility.py \
#     --task-dirs /home/najo/NAS/VLA/dataset/New_dataset2/Yellow_point \
#     --exclude-episodes /home/najo/NAS/VLA/dataset/New_dataset2/Yellow_point/data_collection_20251108_043306 \
#     --output-dir analysis/reproducibility \
#     --target-length 200 \
#     --use-median

# python reconstruct_robot_states.py \
#        --episode-roots /home/najo/NAS/VLA/dataset/New_dataset2/Yellow_point \
#        --exclude-episodes /home/najo/NAS/VLA/dataset/New_dataset2/Yellow_point/data_collection_20251108_043306 \
#        --checkpoint checkpoints/robot_state_mae_best.pth \
#        --window-size 100 --stride 20 --mask-ratio 0.0 \
#        --output-root analysis/reconstructions \
#        --output-name robot_states_recon.npz \
#        --device cuda:0 --dtype bfloat16 --verbose

# python analyze_robot_reproducibility.py \
#        --task-dirs /home/najo/NAS/VLA/dataset/New_dataset2/Yellow_point \
#        --exclude-episodes /home/najo/NAS/VLA/dataset/New_dataset2/Yellow_point/data_collection_20251108_043306 \
#        --output-dir analysis/reproducibility \
#        --target-length 200 --use-median \
#        --recon-file-name robot_states_recon.npz \
#        --recon-key poses \
#        --recon-root /home/najo/NAS/VLA/Insertion_VLAv2/analysis/reconstructions

# =================================================================
# 4. ABLATION STUDIES (FOR REGRESSION MODEL)
# =================================================================
#
# NOTE: This section is for running ablation studies on the TRAIN_Regression.py script.
# The commands below use Stage 1 (cache mode) for faster experimentation.
# It is recommended to run each experiment separately and record the results from wandb.
#
# -----------------------------------------------------------------
# --- Experiment Group 1: Modality Ablation (All Views)
# -----------------------------------------------------------------

# --- Exp 1.1: Vision Only (All Views) ---
# echo "--- Running Ablation: Vision Only (All Views) ---"
# torchrun --nproc_per_node=$NUM_GPUS TRAIN_Regression.py \
#     --epochs $STAGE1_EPOCHS \
#     --batch_size $MAIN_BATCH_SIZE \
#     --grad_accum $GRAD_ACCUM \
#     --lr $LR \
#     --fusion_strategy "concat" \
#     --disable-sensor \
#     --disable-robot-state \
#     --use_cache

# --- Exp 1.2: Vision + Sensor Only ---
# echo "--- Running Ablation: Vision + Sensor Only ---"
# torchrun --nproc_per_node=$NUM_GPUS TRAIN_Regression.py \
#     --epochs $STAGE1_EPOCHS \
#     --batch_size $MAIN_BATCH_SIZE \
#     --grad_accum $GRAD_ACCUM \
#     --lr $LR \
#     --fusion_strategy "concat" \
#     --disable-robot-state \
#     --use_cache

# --- Exp 1.3: Vision + Robot State Only ---
# echo "--- Running Ablation: Vision + Robot State Only ---"
# torchrun --nproc_per_node=$NUM_GPUS TRAIN_Regression.py \
#     --epochs $STAGE1_EPOCHS \
#     --batch_size $MAIN_BATCH_SIZE \
#     --grad_accum $GRAD_ACCUM \
#     --lr $LR \
#     --fusion_strategy "concat" \
#     --disable-sensor \
#     --use_cache

# --- Exp 1.4: Full Model (All Views, All Modalities) ---
# This is the standard full run for comparison.
# echo "--- Running Ablation: Full Model (All Views) ---"
# torchrun --nproc_per_node=$NUM_GPUS TRAIN_Regression.py \
#     --epochs $STAGE1_EPOCHS \
#     --batch_size $MAIN_BATCH_SIZE \
#     --grad_accum $GRAD_ACCUM \
#     --lr $LR \
#     --fusion_strategy "concat" \
#     --use_cache

# -----------------------------------------------------------------
# --- Experiment Group 2: Per-View Importance (with Full Modalities)
# -----------------------------------------------------------------
# This loop runs a separate training for each individual view.

# for view_num in {1..5}; do
#     echo "--- Running Ablation: View $view_num Only (Full Modalities) ---"
#     torchrun --nproc_per_node=$NUM_GPUS TRAIN_Regression.py \
#         --epochs $STAGE1_EPOCHS \
#         --batch_size $MAIN_BATCH_SIZE \
#         --grad_accum $GRAD_ACCUM \
#         --lr $LR \
#         --fusion_strategy "concat" \
#         --views $view_num \
#         --use_cache
# done

echo "✅ Ablation study section added. Uncomment the desired experiments to run."
