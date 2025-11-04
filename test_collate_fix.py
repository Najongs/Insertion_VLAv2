#!/usr/bin/env python3
"""
Test that robot_states is included in batched data after collate_fn fix
"""

import torch
from vla_datasets.unified_dataset import UnifiedVLADataset, unified_collate_fn
from torch.utils.data import DataLoader
from pathlib import Path


def test_collate_with_robot_states():
    """Test that collate_fn includes robot_states"""

    # Use new format dataset
    episode_path = Path("/home/najo/NAS/VLA/dataset/New_dataset/Blue_point/episode_20251030_025856")

    if not episode_path.exists():
        print(f"❌ Episode not found: {episode_path}")
        return

    print("🔬 Testing collate function with robot_states...\n")

    # Create dataset
    dataset = UnifiedVLADataset(
        data_dir=str(episode_path),
        format='new',
        horizon=8,
        vlm_reuse_count=4,
        sensor_window_size=65,
        action_expert_hz=10,
    )

    print(f"📊 Dataset info:")
    print(f"   Total samples: {len(dataset)}")
    print(f"   Has sensor: {dataset.has_sensor}")
    print(f"   Has robot_states: {dataset.has_robot_states}")
    print()

    # Test single item
    print("🔍 Testing single __getitem__:")
    sample = dataset[0]
    print(f"   Keys: {sample.keys()}")
    print(f"   robot_states shape: {sample['robot_states'].shape}")
    print(f"   robot_states dtype: {sample['robot_states'].dtype}")
    print()

    # Create dataloader with collate_fn
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=unified_collate_fn,
        num_workers=0,
    )

    # Test batching
    print("🔍 Testing batch from DataLoader:")
    batch = next(iter(dataloader))
    print(f"   Batch keys: {batch.keys()}")

    if "robot_states" in batch:
        print(f"   ✅ robot_states in batch!")
        print(f"   robot_states shape: {batch['robot_states'].shape}")
        print(f"   robot_states dtype: {batch['robot_states'].dtype}")
        print(f"   has_robot_states_mask: {batch['has_robot_states_mask']}")
    else:
        print(f"   ❌ robot_states NOT in batch!")

    print()
    print("✅ Test complete!")


if __name__ == "__main__":
    test_collate_with_robot_states()
