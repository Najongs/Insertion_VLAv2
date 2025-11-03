"""
Test script to evaluate trained VLA model on a demonstration trajectory

This script:
1. Loads a trained checkpoint (regression or diffusion)
2. Loads a demonstration (old or new dataset format)
3. Generates predicted actions for the entire trajectory
4. Compares with ground truth actions
5. Visualizes results (trajectory plot, error metrics, sample images)

Usage:
    # Test Regression model (New Dataset)
    python examples/test_trained_model_trajectory.py \
        --model-type regression \
        --checkpoint ./checkpoints/regression_best.pt \
        --demo-dir /home/najo/NAS/VLA/Insertion_VLA/Make_dataset/New_dataset/Blue_point/episode_20251030_025119 \
        --dataset-type new \
        --output-dir ./test_results_reg_new
        # --vlm-reuse-count is now optional, defaults to 3 for regression

    # Test Diffusion model (Old Dataset)
    python examples/test_trained_model_trajectory.py \
        --model-type diffusion \
        --checkpoint ./checkpoints/diffusion_best.pt \
        --demo-dir /home/najo/NAS/VLA/dataset/White_silicone_white_circle/recv_all_20251027_170308 \
        --dataset-type old \
        --output-dir ./test_results_diff_old
        # --vlm-reuse-count is now optional, defaults to 1 for diffusion
"""
import time
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import all required models and datasets
from models.model_with_sensor import Not_freeze_QwenVLAWithSensor
from models.model_with_sensor_diffusion import QwenVLAWithSensorDiffusion
from vla_datasets.IntegratedDataset import insertionMeca500DatasetWithSensor
from vla_datasets.NewAsyncDataset import NewAsyncInsertionDataset


def load_model(checkpoint_path: str, model_type: str, device: str = "cuda"):
    """
    Load trained model from checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        model_type: 'regression' or 'diffusion'
        device: Device to load model on

    Returns:
        model: Loaded model in eval mode
    """
    print(f"🔄 Loading checkpoint from: {checkpoint_path}")
    print(f"   Model Type: {model_type.upper()}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # ⬇️ [수정] 
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 
    
    # 
    # 
    device_map = device if "cuda" in device else "cpu"
    print(f"   Mapping model to device: {device_map}")
    # ⬆️ [수정] 

    if model_type == 'regression':
        model = Not_freeze_QwenVLAWithSensor(
            vl_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
            action_dim=7,
            horizon=8,
            hidden_dim=1024,
            finetune_vl="none",  # 
            sensor_enabled=True,
            sensor_input_channels=1026,
            sensor_temporal_length=65,
            sensor_output_dim=3072,
            fusion_strategy="concat",
            image_resize_height=360,
            image_resize_width=640,
            device_map=device_map, # 
        ).to(device)
    
    elif model_type == 'diffusion':
        model = QwenVLAWithSensorDiffusion(
            vl_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
            action_dim=7,
            horizon=8,
            hidden_dim=1024,
            sensor_enabled=True,
            fusion_strategy="concat",
            diffusion_timesteps=50, # 
        ).to(device)
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Load state dict
    if "model_state_dict" in checkpoint:
        # Handle weights saved from DDP
        state_dict = checkpoint["model_state_dict"]
        # Remove 'module.' prefix if present
        if all(k.startswith('module.') for k in state_dict.keys()):
            print("   (Removing 'module.' prefix from state dict)")
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded model weights from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'val_loss' in checkpoint:
            print(f"   Validation loss: {checkpoint['val_loss']:.6f}")
    else:
        # Try loading directly
        model.load_state_dict(checkpoint, strict=False)
        print("✅ Loaded model weights (direct format)")

    model.eval()
    return model

def load_demo_trajectory(
    demo_dir: str,
    dataset_type: str,
    horizon: int = 8,
    sensor_window_size: int = 65,
    vlm_reuse_count: int = 1,
    action_expert_hz: int = 10,
):
    """
    Load a single demonstration trajectory

    Args:
        demo_dir: Path to demonstration directory (or episode dir for 'new')
        dataset_type: 'old' or 'new'
        horizon: Action horizon
        sensor_window_size: Sensor window size
        vlm_reuse_count: VLM reuse count (1 for step-by-step test)
        action_expert_hz: Action expert HZ (for 'new' dataset)

    Returns:
        dataset: Dataset object for the demonstration
    """
    print(f"📂 Loading demonstration from: {demo_dir}")
    print(f"   Dataset Type: {dataset_type.upper()}")

    if dataset_type == 'old':
        dataset = insertionMeca500DatasetWithSensor(
            trajectory_dir=demo_dir,
            horizon=horizon,
            instruction="Perform insertion task",
            sensor_window_size=sensor_window_size,
            view_selection=['left', 'oak'],
            cache_sensor_windows=True
        )
    elif dataset_type == 'new':
        dataset = NewAsyncInsertionDataset(
            episode_dir=demo_dir,
            horizon=horizon,
            vlm_reuse_count=vlm_reuse_count, # 
            action_expert_hz=action_expert_hz,
            instruction="Perform insertion task"
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    print(f"✅ Loaded {len(dataset)} timesteps")
    
    if hasattr(dataset, 'has_sensor'):
        # 'old' dataset has this attribute
        print(f"   Has sensor data: {dataset.has_sensor}")
    else:
        # 'new' dataset checks sensor data per-sample
        print(f"   (Sensor data status is checked per-sample for 'new' dataset)")

    return dataset
def predict_trajectory(model, dataset, model_type: str, device: str = "cuda"):
    """
    Generate predictions for entire trajectory

    Args:
        model: Trained model
        dataset: Dataset containing the demonstration
        model_type: 'regression' or 'diffusion'
        device: Device to run inference on

    Returns:
        predictions: List of predicted action sequences
        ground_truths: List of ground truth action sequences
        metadata: Additional info (images, sensor data, etc.)
    """
    print(f"\n🤖 Generating predictions for {len(dataset)} timesteps...")

    predictions = []
    ground_truths = []
    metadata = {
        'images': [],
        'sensor_data': [],
        'cache_keys': [],
        'inference_times': [] # ⬅️ 
    }

    model.eval()
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Predicting"):
            sample = dataset[idx]

            # Prepare inputs
            images = sample['images']
            instruction = [sample['instruction']]
            gt_actions = sample['actions'].unsqueeze(0).to(device, dtype=torch.bfloat16)
            
            has_sensor = sample.get('has_sensor', False)
            sensor_data = sample['sensor_data'].unsqueeze(0).to(device, dtype=torch.bfloat16) if has_sensor else None
            
            cache_key = [sample['cache_key']]

            # ⬇️ [수정] 
            # 
            torch.cuda.synchronize() # 
            start_time = time.time()
            # ⬆️ [수정]

            # Run inference
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                
                if model_type == 'regression':
                    pred_actions, _ = model(
                        text_inputs=instruction,
                        image_inputs=[images],
                        z_chunk=gt_actions,  # Passed for validation/metrics
                        cache_keys=cache_key,
                        sensor_data=sensor_data,
                    )
                
                elif model_type == 'diffusion':
                    if not hasattr(model, 'predict_action'):
                        raise NotImplementedError(
                            "Diffusion model must have a 'predict_action' method for inference."
                        )
                    
                    pred_actions = model.predict_action(
                        text_inputs=instruction,
                        image_inputs=[images],
                        sensor_data=sensor_data,
                        cache_keys=cache_key
                    )
                    # Ensure shape is [1, horizon, 7]
                    if pred_actions.dim() == 2: # If it returns [H, 7]
                        pred_actions = pred_actions.unsqueeze(0) # Add batch dim

            # ⬇️ [수정] 
            # 
            torch.cuda.synchronize() # 
            end_time = time.time()
            inference_time_ms = (end_time - start_time) * 1000.0
            metadata['inference_times'].append(inference_time_ms)
            # ⬆️ [수정]

            # Store results
            predictions.append(pred_actions.cpu().float().numpy()[0])  # [horizon, 7]
            ground_truths.append(gt_actions.cpu().float().numpy()[0]) # [horizon, 7]

            # Store metadata (only first few for visualization)
            if idx < 10:
                metadata['images'].append(images)
                metadata['cache_keys'].append(cache_key[0])
                if sensor_data is not None:
                    metadata['sensor_data'].append(sensor_data.cpu().float().numpy()[0])

    predictions = np.array(predictions)  # [T, horizon, 7]
    ground_truths = np.array(ground_truths)  # [T, horizon, 7]

    print(f"✅ Generated predictions: {predictions.shape}")

    return predictions, ground_truths, metadata


def compute_metrics(predictions, ground_truths):
    """
    Compute evaluation metrics

    Args:
        predictions: Predicted actions [T, horizon, 7]
        ground_truths: Ground truth actions [T, horizon, 7]

    Returns:
        metrics: Dictionary of metrics
    """
    print("\n📊 Computing metrics...")

    # MSE per dimension (averaged over timesteps and horizon)
    mse_per_dim = np.mean((predictions - ground_truths) ** 2, axis=(0, 1))

    # Overall MSE
    overall_mse = np.mean((predictions - ground_truths) ** 2)

    # MAE per dimension
    mae_per_dim = np.mean(np.abs(predictions - ground_truths), axis=(0, 1))

    # Overall MAE
    overall_mae = np.mean(np.abs(predictions - ground_truths))

    # Per-timestep error (for trajectory visualization)
    per_timestep_mse = np.mean((predictions - ground_truths) ** 2, axis=(1, 2))

    # Per-horizon error (first action vs full horizon)
    first_action_mse = np.mean((predictions[:, 0, :] - ground_truths[:, 0, :]) ** 2)
    full_horizon_mse = overall_mse

    # Correlation analysis (trend similarity)
    correlations = []
    for dim in range(predictions.shape[2]):  # For each action dimension
        # Use first action of each timestep for correlation
        pred_traj = predictions[:, 0, dim]
        gt_traj = ground_truths[:, 0, dim]
        if np.std(pred_traj) > 1e-6 and np.std(gt_traj) > 1e-6:
            corr = np.corrcoef(pred_traj, gt_traj)[0, 1]
        else:
            corr = 0.0
        correlations.append(corr)

    correlations = np.array(correlations)

    metrics = {
        'overall_mse': overall_mse,
        'overall_mae': overall_mae,
        'mse_per_dim': mse_per_dim,
        'mae_per_dim': mae_per_dim,
        'per_timestep_mse': per_timestep_mse,
        'first_action_mse': first_action_mse,
        'full_horizon_mse': full_horizon_mse,
        'correlations': correlations,
    }

    print(f"✅ Overall MSE: {overall_mse:.6f}")
    print(f"   Overall MAE: {overall_mae:.6f}")
    print(f"   First action MSE: {first_action_mse:.6f}")
    print(f"   Full horizon MSE: {full_horizon_mse:.6f}")
    print(f"   Per-dimension MSE: {mse_per_dim}")
    print(f"   Per-dimension MAE: {mae_per_dim}")
    print(f"   Correlations (trend): {correlations}")

    return metrics
def visualize_results(predictions, ground_truths, metrics, metadata, output_dir: Path):
    print(f"\n📈 Creating visualizations in: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    action_names = ['dx', 'dy', 'dz', 'da', 'db', 'dr', 'gripper']

    # ✅ 1. Combined Position + Orientation Trajectory (3x2 layout)
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    pos_dims = [0, 1, 2]
    ori_dims = [3, 4, 5]
    pos_names = ['dx (X-axis)', 'dy (Y-axis)', 'dz (Z-axis)']
    ori_names = ['da (Alpha)', 'db (Beta)', 'dr (Gamma)']

    # ⬇️ [수정] 
    # Calculate shared Y-axis limits for position (dx, dy, dz)
    all_pos_gt = ground_truths[:, 0, pos_dims]
    all_pos_pred = predictions[:, 0, pos_dims]
    pos_min = min(np.min(all_pos_gt), np.min(all_pos_pred))
    pos_max = max(np.max(all_pos_gt), np.max(all_pos_pred))
    pos_range = pos_max - pos_min if pos_max > pos_min else 1.0
    pos_margin = pos_range * 0.1  # Add 10% margin
    pos_ylim = (pos_min - pos_margin, pos_max + pos_margin)

    # Calculate shared Y-axis limits for orientation (da, db, dr)
    all_ori_gt = ground_truths[:, 0, ori_dims]
    all_ori_pred = predictions[:, 0, ori_dims]
    ori_min = min(np.min(all_ori_gt), np.min(all_ori_pred))
    ori_max = max(np.max(all_ori_gt), np.max(all_ori_pred))
    ori_range = ori_max - ori_min if ori_max > ori_min else 1.0
    ori_margin = ori_range * 0.1  # Add 10% margin
    ori_ylim = (ori_min - ori_margin, ori_max + ori_margin)
    # ⬆️ [수정] 여기까지

    for i in range(3):
        # Position subplot (left column)
        ax_pos = axes[i, 0]
        gt_pos = ground_truths[:, 0, pos_dims[i]]
        pred_pos = predictions[:, 0, pos_dims[i]]
        timesteps = np.arange(len(gt_pos))
        ax_pos.plot(timesteps, gt_pos, 'b-', label='GT', linewidth=2.8, marker='o', markersize=4, markevery=5)
        ax_pos.plot(timesteps, pred_pos, 'r--', label='Pred', linewidth=2.8, marker='s', markersize=4, markevery=5, alpha=0.8)
        ax_pos.set_title(pos_names[i], fontsize=17, fontweight='bold')
        ax_pos.set_ylabel('Δ Position', fontsize=16)
        ax_pos.grid(True, alpha=0.3)
        ax_pos.set_ylim(pos_ylim) # ⬅️ 
        if i == 2:
            ax_pos.set_xlabel('Timestep', fontsize=15)
        if i == 0:
            ax_pos.legend(fontsize=14, loc='best')

        # Orientation subplot (right column)
        ax_ori = axes[i, 1]
        gt_ori = ground_truths[:, 0, ori_dims[i]]
        pred_ori = predictions[:, 0, ori_dims[i]]
        timesteps = np.arange(len(gt_ori))
        ax_ori.plot(timesteps, gt_ori, 'b-', label='GT', linewidth=2.8, marker='o', markersize=4, markevery=5)
        ax_ori.plot(timesteps, pred_ori, 'r--', label='Pred', linewidth=2.8, marker='s', markersize=4, markevery=5, alpha=0.8)
        ax_ori.set_title(ori_names[i], fontsize=17, fontweight='bold')
        ax_ori.set_ylabel('Δ Orientation', fontsize=16)
        ax_ori.grid(True, alpha=0.3)
        ax_ori.set_ylim(ori_ylim) # ⬅️ 
        if i == 2:
            ax_ori.set_xlabel('Timestep', fontsize=15)
        if i == 0:
            ax_ori.legend(fontsize=14, loc='best')

    plt.suptitle('Pose Trajectory (Position + Orientation)', fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'trajectory_pose_combined.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: trajectory_pose_combined.png")
    plt.close()

    # 2. Plot per-timestep MSE
    fig, ax = plt.subplots(figsize=(12, 6))
    timesteps = np.arange(len(metrics['per_timestep_mse']))
    ax.plot(timesteps, metrics['per_timestep_mse'], 'g-', linewidth=2)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('MSE')
    ax.set_title('Per-Timestep Mean Squared Error')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'per_timestep_error.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: per_timestep_error.png")
    plt.close()

    # 3. Bar plot of per-dimension errors and correlations
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    x = np.arange(7)
    ax1.bar(x, metrics['mse_per_dim'], color='steelblue', alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(action_names, rotation=45)
    ax1.set_ylabel('MSE')
    ax1.set_title('Mean Squared Error per Action Dimension')
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.bar(x, metrics['mae_per_dim'], color='coral', alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(action_names, rotation=45)
    ax2.set_ylabel('MAE')
    ax2.set_title('Mean Absolute Error per Action Dimension')
    ax2.grid(True, alpha=0.3, axis='y')

    # Correlation plot (trend similarity)
    colors = ['green' if c > 0.8 else 'orange' if c > 0.5 else 'red' for c in metrics['correlations']]
    ax3.bar(x, metrics['correlations'], color=colors, alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(action_names, rotation=45)
    ax3.set_ylabel('Correlation')
    ax3.set_ylim(-1, 1)
    ax3.axhline(y=0.8, color='g', linestyle='--', alpha=0.5, label='Good (>0.8)')
    ax3.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Fair (>0.5)')
    ax3.set_title('Correlation (Trend Similarity)')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'error_per_dimension.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: error_per_dimension.png")
    plt.close()

    # 4. Full horizon comparison (show all 8 actions for selected timesteps)
    # Select 4 representative timesteps
    T = predictions.shape[0]
    selected_timesteps = [0, T//3, 2*T//3, T-1]

    fig, axes = plt.subplots(4, 7, figsize=(21, 12))

    for row_idx, t in enumerate(selected_timesteps):
        for dim in range(7):
            ax = axes[row_idx, dim]

            # Plot horizon for this timestep and dimension
            horizon_steps = np.arange(predictions.shape[1])
            gt_horizon = ground_truths[t, :, dim]
            pred_horizon = predictions[t, :, dim]

            ax.plot(horizon_steps, gt_horizon, 'b-o', label='GT', linewidth=2, markersize=4)
            ax.plot(horizon_steps, pred_horizon, 'r--s', label='Pred', linewidth=2, markersize=4, alpha=0.7)

            if row_idx == 0:
                ax.set_title(f'{action_names[dim]}', fontsize=10)
            if dim == 0:
                ax.set_ylabel(f't={t}', fontsize=10)
            if row_idx == len(selected_timesteps) - 1:
                ax.set_xlabel('Horizon', fontsize=8)

            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=6, loc='best')

    plt.suptitle('Full Horizon Comparison (Selected Timesteps)', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'horizon_comparison.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: horizon_comparison.png")
    plt.close()

    # 5. Sample images (if available)
    if metadata['images']:
        import cv2

        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        for i, ax in enumerate(axes.flatten()):
            if i < len(metadata['images']):
                # Load first image from the view
                if not metadata['images'][i]: # Handle empty list
                    ax.text(0.5, 0.5, 'Image\nList Empty', ha='center', va='center')
                    ax.axis('off')
                    continue

                img_uri = metadata['images'][i][0]  # First view
                img_path = img_uri.replace('file://', '')

                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        ax.imshow(img)
                        ax.set_title(f'Timestep {i}', fontsize=10)
                        ax.axis('off')
                    else:
                        ax.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center')
                        ax.axis('off')
                except Exception as e:
                    ax.text(0.5, 0.5, f'Error:\n{str(e)[:20]}', ha='center', va='center', fontsize=8)
                    ax.axis('off')
            else:
                ax.axis('off')

        plt.suptitle('Sample Images from Trajectory', fontsize=14, y=0.98)
        plt.tight_layout()
        plt.savefig(output_dir / 'sample_images.png', dpi=150, bbox_inches='tight')
        print(f"   ✅ Saved: sample_images.png")
        plt.close()

    inference_stats = {}
    if metadata['inference_times']:
        times_ms = np.array(metadata['inference_times'])
        inference_stats = {
            'mean_ms': float(np.mean(times_ms)),
            'std_ms': float(np.std(times_ms)),
            'median_ms': float(np.median(times_ms)),
            'p95_ms': float(np.percentile(times_ms, 95)),
            'p99_ms': float(np.percentile(times_ms, 99)),
            'min_ms': float(np.min(times_ms)),
            'max_ms': float(np.max(times_ms)),
            'fps_avg': float(1000.0 / np.mean(times_ms))
        }
        print("\n⏱️ Inference Time Stats:")
        print(f"   Mean: {inference_stats['mean_ms']:.2f} ms")
        print(f"   Median: {inference_stats['median_ms']:.2f} ms")
        print(f"   P95: {inference_stats['p95_ms']:.2f} ms")
        print(f"   Avg FPS: {inference_stats['fps_avg']:.2f} Hz")
    
    metrics_json = {
        'inference_stats_ms': inference_stats, # 
        'overall_mse': float(metrics['overall_mse']),
        'overall_mae': float(metrics['overall_mae']),
        'first_action_mse': float(metrics['first_action_mse']),
        'full_horizon_mse': float(metrics['full_horizon_mse']),
        'mse_per_dim': metrics['mse_per_dim'].tolist(),
        'mae_per_dim': metrics['mae_per_dim'].tolist(),
        'correlations': metrics['correlations'].tolist(),
        'action_names': action_names,
        'interpretation': {
            'mse': 'Lower is better - measures absolute error magnitude',
            'mae': 'Lower is better - measures average absolute error',
            'correlation': 'Higher is better (0~1) - measures trend similarity. >0.8 is good, >0.5 is fair',
            'note': 'High correlation with high MSE means the model captures the trend but has scale/offset issues'
        }
    }

    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_json, f, indent=2)
    print(f"   ✅ Saved: metrics.json")


def main():
    parser = argparse.ArgumentParser(description="Test trained VLA model on demonstration trajectory")
    
    # --- New/Modified Arguments ---
    parser.add_argument(
        "--model-type",
        type=str,
        choices=['regression', 'diffusion'],
        required=True,
        help="Type of model to test"
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=['old', 'new'],
        default='new',
        help="Type of dataset format ('old' for insertionMeca500, 'new' for NewAsyncInsertion)"
    )
    parser.add_argument(
        "--demo-dir",
        type=str,
        default="/home/najo/NAS/VLA/Insertion_VLA/Make_dataset/New_dataset/Blue_point/episode_20251030_025119",
        help="Path to demonstration directory (for 'old') or episode directory (for 'new')"
    )
    
    # ⬇️ [FIX] Set default=None to dynamically assign it based on model type
    parser.add_argument(
        "--vlm-reuse-count",
        type=int,
        default=None, 
        help="VLM reuse count. Default: 3 for regression, 1 for diffusion."
    )
    # ⬆️ [FIX]
    
    parser.add_argument(
        "--action-expert-hz",
        type=int,
        default=10,
        help="Action expert HZ (for 'new' dataset)"
    )
    # --- End New/Modified Arguments ---
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/regression_best.pt",
        help="Path to checkpoint file (e.g., regression_best.pt or diffusion_best.pt)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./test_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=8,
        help="Action horizon"
    )
    parser.add_argument(
        "--sensor-window-size",
        type=int,
        default=65,
        help="Sensor window size (65 = 100ms @ 650Hz)"
    )

    args = parser.parse_args()
    
    # ⬇️ [FIX] Automatically set default vlm_reuse_count based on model type
    if args.vlm_reuse_count is None:
        if args.model_type == 'regression':
            args.vlm_reuse_count = 3
        else: # diffusion
            args.vlm_reuse_count = 1
        print(f"ℹ️ vlm_reuse_count not set. Defaulting to {args.vlm_reuse_count} for {args.model_type} model.")
    # ⬆️ [FIX]

    print("="*80)
    print("🧪 Testing Trained VLA Model on Demonstration Trajectory")
    print("="*80)
    print(f"Model Type: {args.model_type.upper()}")
    print(f"Dataset Type: {args.dataset_type.upper()}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Demo directory: {args.demo_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"VLM Reuse Count: {args.vlm_reuse_count}") # 
    print("="*80)

    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print(f"   Available checkpoints:")
        ckpt_dir = Path("./checkpoints")
        if ckpt_dir.exists():
            for ckpt in ckpt_dir.glob("*.pt"):
                print(f"   - {ckpt.name}")
        return

    # Check if demo directory exists
    if not Path(args.demo_dir).exists():
        print(f"❌ Demo directory not found: {args.demo_dir}")
        return

    # Load model
    model = load_model(
        args.checkpoint, 
        model_type=args.model_type, 
        device=args.device
    )

    # Load demonstration
    dataset = load_demo_trajectory(
        args.demo_dir,
        dataset_type=args.dataset_type,
        horizon=args.horizon,
        sensor_window_size=args.sensor_window_size,
        vlm_reuse_count=args.vlm_reuse_count,
        action_expert_hz=args.action_expert_hz
    )

    # Generate predictions
    predictions, ground_truths, metadata = predict_trajectory(
        model, 
        dataset, 
        model_type=args.model_type, 
        device=args.device
    )

    # Compute metrics
    metrics = compute_metrics(predictions, ground_truths)

    # Visualize results
    output_dir = Path(args.output_dir)
    visualize_results(predictions, ground_truths, metrics, metadata, output_dir)

    print("\n" + "="*80)
    print("✅ Testing complete!")
    print(f"📁 Results saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()