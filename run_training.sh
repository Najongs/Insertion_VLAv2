export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4

torchrun --nproc_per_node=4 training/A5st_VLA_TRAIN_Unified.py \
    --model-type regression \
    --mode cache

# torchrun --nproc_per_node=4 training/A5st_VLA_TRAIN_Unified.py \
#     --model-type diffusion \
#     --mode cache

# torchrun --nproc_per_node=4 \
#         training/A5st_VLA_TRAIN_Unified.py \
#         --model-type diffusion \
#         --dataset_dir /home/najo/NAS/VLA/dataset \
#         --batch_size 4 \
#         --grad_accum 16 \
#         --lr 1e-4 \
#         --epochs 100 \
#         --diffusion_timesteps 50 \
#         --sensor_enabled \
#         --sensor_loss_weight 2.0 \
#         --fusion_strategy concat \
#         --val_split 0.1 \
#         --num_workers 8 \
#         --resume /home/najo/NAS/VLA/Insertion_VLA/checkpoints/diffusion_latest.pt


torchrun --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    training/A5st_VLA_TRAIN_Unified.py \
    --model-type regression \
    --mode train \
    --dataset_dir /home/najo/NAS/VLA/dataset \
    --batch_size 32 \
    --grad_accum 16 \
    --lr 5e-5 \
    --sensor_lr 5e-4 \
    --min_lr 1e-6 \
    --epochs 100 \
    --sensor_enabled \
    --sensor_loss_weight 2.0 \
    --fusion_strategy concat \
    --image_resize_height 360 \
    --image_resize_width 640 \
    --val_split 0.05 \
    --num_workers 8 \
    --resume /home/najo/NAS/VLA/Insertion_VLA/checkpoints/regression_best.pt



# Test script to evaluate trained VLA model on a demonstration trajectory

# This script:
# 1. Loads a trained checkpoint (regression or diffusion)
# 2. Loads a demonstration (old or new dataset format)
# 3. Generates predicted actions for the entire trajectory
# 4. Compares with ground truth actions
# 5. Visualizes results (trajectory plot, error metrics, sample images)

# Usage:
# # Test Regression model (New Dataset)

# # Test Diffusion model (Old Dataset)
# python scripts/test_dataset_v2.py \
#     --model-type diffusion \
#     --checkpoint ./checkpoints/diffusion_best.pt \
#     --demo-dir /home/najo/NAS/VLA/Insertion_VLA/Make_dataset/New_dataset/Eye_trocar/episode_20251030_041238 \
#     --dataset-type new \
#     --output-dir ./test_results_diff_new_eye4 \
#     --vlm-reuse-count 3 \
#     --device cuda:0

# python scripts/test_dataset_v2.py \
#     --model-type diffusion \
#     --checkpoint ./checkpoints/diffusion_best.pt \
#     --demo-dir /home/najo/NAS/VLA/Insertion_VLA/Make_dataset/New_dataset/Eye_trocar/episode_20251030_040048 \
#     --dataset-type new \
#     --output-dir ./test_results_diff_new_eye7 \
#     --vlm-reuse-count 3 \
#     --device cuda:1

# python scripts/test_dataset_v2.py \
#     --model-type diffusion \
#     --checkpoint ./checkpoints/diffusion_best.pt \
#     --demo-dir /home/najo/NAS/VLA/Insertion_VLA/Make_dataset/New_dataset/Eye_trocar/episode_20251030_035954 \
#     --dataset-type new \
#     --output-dir ./test_results_diff_new_eye6 \
#     --vlm-reuse-count 3 \
#     --device cuda:2

# python scripts/test_dataset_v2.py \
#      --model-type diffusion \
#      --checkpoint ./checkpoints/diffusion_best.pt \
#      --demo-dir /home/najo/NAS/VLA/Insertion_VLA/Make_dataset/New_dataset/Eye_trocar/episode_20251030_041049 \
#      --dataset-type new \
#      --output-dir ./test_results_diff_new_eye5 \
#      --vlm-reuse-count 3 \
#      --device cuda:3

# /home/najo/NAS/VLA/Insertion_VLA/Make_dataset/New_dataset/Eye_trocar/episode_20251030_040800
# /home/najo/NAS/VLA/Insertion_VLA/Make_dataset/New_dataset/Eye_trocar/episode_20251030_040943
# /home/najo/NAS/VLA/Insertion_VLA/Make_dataset/New_dataset/Eye_trocar/episode_20251030_040048

# /home/najo/NAS/VLA/Insertion_VLA/Make_dataset/New_dataset/White_point/episode_20251030_034812
# /home/najo/NAS/VLA/Insertion_VLA/Make_dataset/New_dataset/White_point/episode_20251030_035351
# /home/najo/NAS/VLA/Insertion_VLA/Make_dataset/New_dataset/White_point/episode_20251030_035154

# /home/najo/NAS/VLA/Insertion_VLA/Make_dataset/New_dataset/Green_point/episode_20251030_030630
# /home/najo/NAS/VLA/Insertion_VLA/Make_dataset/New_dataset/Green_point/episode_20251030_031315

