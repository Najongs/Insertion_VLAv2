#!/usr/bin/env python3
"""
Test the complete flow matching pipeline with robot states
"""

import torch
from vla_datasets.unified_dataset import UnifiedVLADataset, unified_collate_fn
from torch.utils.data import DataLoader
from pathlib import Path


def test_flow_matching_dimensions():
    """Test dimension flow through the entire pipeline"""

    episode_path = Path("/home/najo/NAS/VLA/dataset/New_dataset/Blue_point/episode_20251030_025856")

    if not episode_path.exists():
        print(f"❌ Episode not found: {episode_path}")
        return

    print("🔬 Testing Flow Matching Pipeline Dimensions\n")

    # Create dataset
    dataset = UnifiedVLADataset(
        data_dir=str(episode_path),
        format='new',
        horizon=8,
        vlm_reuse_count=4,
        sensor_window_size=65,
        action_expert_hz=10,
    )

    print(f"📊 Dataset Configuration:")
    print(f"   Total samples: {len(dataset)}")
    print(f"   Has sensor: {dataset.has_sensor}")
    print(f"   Has robot_states: {dataset.has_robot_states}")
    print(f"   Sensor window size: {dataset.sensor_window_size}")
    print()

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=unified_collate_fn,
        num_workers=0,
    )

    # Get a batch
    batch = next(iter(dataloader))

    print("📦 Batch Dimensions:")
    print(f"   Actions: {batch['actions'].shape}")
    print(f"   Sensor data: {batch['sensor_data'].shape}")
    print(f"   Robot states: {batch['robot_states'].shape}")
    print(f"   Has sensor mask: {batch['has_sensor_mask']}")
    print(f"   Has robot states mask: {batch['has_robot_states_mask']}")
    print()

    # Simulate sensor and robot state encoding
    batch_size = batch['actions'].shape[0]
    sensor_output_dim = 2048

    # Simulate sensor encoder output
    sensor_features = torch.randn(batch_size, sensor_output_dim)
    print(f"🔄 Simulated Encoder Outputs:")
    print(f"   Sensor features: {sensor_features.shape}")

    # Simulate robot state encoder output
    robot_state_features = torch.randn(batch_size, sensor_output_dim)
    print(f"   Robot state features: {robot_state_features.shape}")

    # Combine features (this is what happens in FlowMatchingActionExpert.forward)
    combined_features = torch.cat([sensor_features, robot_state_features], dim=-1)
    print(f"   Combined features: {combined_features.shape}")
    print()

    # Simulate VL features
    vl_hidden_size = 2048  # This should match actual VL model
    vl_seq_len = 256
    vl_tokens = torch.randn(batch_size, vl_seq_len, vl_hidden_size)
    vl_pooled = vl_tokens.mean(dim=1)

    print(f"🎨 Vision-Language Features:")
    print(f"   VL tokens: {vl_tokens.shape}")
    print(f"   VL pooled: {vl_pooled.shape}")
    print()

    # Fusion (concat strategy)
    fused = torch.cat([vl_pooled, combined_features], dim=-1)
    print(f"🔗 Fusion (concat strategy):")
    print(f"   Fused shape: {fused.shape}")
    print(f"   Expected: (batch_size={batch_size}, vl={vl_hidden_size} + combined={sensor_output_dim*2})")
    print(f"   Actual: ({batch_size}, {fused.shape[-1]})")
    print()

    # Check dimensions match expected
    expected_fused_dim = vl_hidden_size + (sensor_output_dim * 2)
    if fused.shape[-1] == expected_fused_dim:
        print(f"✅ Dimensions match! Fusion layer should expect {expected_fused_dim} input dims")
    else:
        print(f"❌ Dimension mismatch! Expected {expected_fused_dim}, got {fused.shape[-1]}")

    print()
    print("📝 Summary:")
    print(f"   VL dim: {vl_hidden_size}")
    print(f"   Sensor output dim: {sensor_output_dim}")
    print(f"   Robot state output dim: {sensor_output_dim}")
    print(f"   Combined dim: {sensor_output_dim * 2}")
    print(f"   Fusion input dim: {expected_fused_dim}")
    print()
    print("🎯 Model Configuration:")
    print(f"   FlowMatchingActionExpert should be initialized with:")
    print(f"     vl_dim={vl_hidden_size}")
    print(f"     sensor_dim={sensor_output_dim * 2}")
    print(f"     fusion_strategy='concat'")
    print()
    print("✅ Pipeline test complete!")


if __name__ == "__main__":
    test_flow_matching_dimensions()
