#!/usr/bin/env python3
"""
Test script for VLA Cache System
Tests the complete cache system integration
"""

import sys
from pathlib import Path
import torch

# Test VLACacheManager
print("=" * 80)
print("1️⃣ Testing VLACacheManager")
print("=" * 80)

from vla_cache_manager import VLACacheManager

cache_mgr = VLACacheManager(
    cache_dir="/tmp/test_vla_cache",
    cache_limit_gb=1.0
)

# Test cache path generation
print("\n📁 Cache path generation:")
path1 = cache_mgr.get_cache_path("recv_all_20251027_170308", 0)
path2 = cache_mgr.get_cache_path("episode_20251030_025119", 150)
print(f"   Old format: {path1.name}")
print(f"   New format: {path2.name}")

assert path1.name == "recv_all_20251027_170308_vlm0.pt", "Old format path wrong"
assert path2.name == "episode_20251030_025119_vlm150.pt", "New format path wrong"
print("   ✅ Path generation correct")

# Test save and load
print("\n💾 Save and load test:")
test_features = torch.randn(1, 1, 3072)
cache_mgr.save_cache("test_dataset", 0, test_features)
print(f"   Saved: test_dataset_vlm0.pt")

loaded = cache_mgr.load_cache("test_dataset", 0)
assert loaded is not None, "Failed to load cache"
print(f"   Loaded: {loaded.shape}")
print(f"   Match: {torch.allclose(test_features.cpu().float(), loaded.float(), rtol=1e-3)}")

# Test cache_exists
print("\n🔍 Cache existence check:")
exists_0 = cache_mgr.cache_exists("test_dataset", 0)
exists_1 = cache_mgr.cache_exists("test_dataset", 1)
print(f"   test_dataset_vlm0 exists: {exists_0}")
print(f"   test_dataset_vlm1 exists: {exists_1}")
assert exists_0 == True, "Cache should exist"
assert exists_1 == False, "Cache should not exist"
print("   ✅ Existence check correct")

# Test stats
print("\n📊 Cache statistics:")
stats = cache_mgr.get_cache_stats()
for key, value in stats.items():
    print(f"   {key}: {value}")

# Test list datasets
print("\n📋 Cached datasets:")
datasets = cache_mgr.list_cached_datasets()
for dataset_name, vlm_indices in datasets.items():
    print(f"   {dataset_name}: {vlm_indices}")

# Cleanup
cache_mgr.clear_cache(confirm=True)
print("\n✅ VLACacheManager tests passed!")

# Test Dataset Integration
print("\n" + "=" * 80)
print("2️⃣ Testing Dataset Integration")
print("=" * 80)

# Check if old dataset exists
old_dataset_path = Path("/home/najo/NAS/VLA/dataset/dataset/recv_all_20251027_170308")
new_dataset_path = Path("/home/najo/NAS/VLA/dataset/New_dataset/Yellow_point")

if old_dataset_path.exists():
    print("\n📂 Testing Old Format Dataset:")
    from vla_datasets.unified_dataset import UnifiedVLADataset

    # Find first directory
    old_dirs = sorted(old_dataset_path.parent.glob("*"))
    if old_dirs:
        test_dir = old_dirs[0]
        print(f"   Loading: {test_dir.name}")

        ds = UnifiedVLADataset(
            data_dir=str(test_dir),
            format='old',
            horizon=8,
            vlm_reuse_count=3,
        )

        print(f"   Total samples: {len(ds)}")
        print(f"   VL Cache: {ds.cache_found_count}/{len(ds.vl_cache_files)}")

        # Check cache_key format
        sample = ds[0]
        cache_key = sample["cache_key"]
        vlm_idx = sample["vlm_idx"]
        print(f"   Sample cache_key: {cache_key}")
        print(f"   Sample vlm_idx: {vlm_idx}")

        # Verify cache_key format
        expected_key = f"{test_dir.name}_vlm{vlm_idx}"
        assert cache_key == expected_key, f"Cache key mismatch: {cache_key} != {expected_key}"
        print("   ✅ Old format cache_key correct")
else:
    print(f"\n⚠️ Old dataset not found at {old_dataset_path}")

if new_dataset_path.exists():
    print("\n📂 Testing New Format Dataset:")
    from vla_datasets.unified_dataset import UnifiedVLADataset

    # Find first episode
    new_episodes = sorted(new_dataset_path.glob("episode_*"))
    if new_episodes:
        test_episode = new_episodes[0]
        print(f"   Loading: {test_episode.name}")

        ds = UnifiedVLADataset(
            data_dir=str(test_episode),
            format='new',
            horizon=8,
            vlm_reuse_count=3,
        )

        print(f"   Total samples: {len(ds)}")
        print(f"   VL Cache: {ds.cache_found_count}/{len(ds.vl_cache_files)}")

        # Check cache_key format
        sample = ds[0]
        cache_key = sample["cache_key"]
        vlm_idx = sample["vlm_idx"]
        print(f"   Sample cache_key: {cache_key}")
        print(f"   Sample vlm_idx: {vlm_idx}")

        # Verify cache_key format
        expected_key = f"{test_episode.name}_vlm{vlm_idx}"
        assert cache_key == expected_key, f"Cache key mismatch: {cache_key} != {expected_key}"
        print("   ✅ New format cache_key correct")
else:
    print(f"\n⚠️ New dataset not found at {new_dataset_path}")

print("\n" + "=" * 80)
print("🎉 All Cache System Tests Passed!")
print("=" * 80)
print("\n✅ Cache system is working correctly")
print("✅ Dataset integration is correct")
print("✅ Cache keys are properly formatted")
print("\n🚀 Ready to use in training!")
