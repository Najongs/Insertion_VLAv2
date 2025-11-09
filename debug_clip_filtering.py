"""
Debug script to check how many samples are being filtered in CLIP training
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
from pathlib import Path
from functools import partial
import sys

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from vla_datasets.unified_dataset import UnifiedVLADataset, create_unified_dataloader
from TRAIN_SensorImage_CLIP import SensorImageCLIPDataset, clip_collate_fn

# Load VLM annotations
annotation_path = Path("vlm_annotations.json")
annotations = {}
if annotation_path.exists():
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)
    print(f"✓ Loaded {len(annotations)} VLM annotations")

# Create dataset
print("\nCreating dataset...")
unified_dataset = create_unified_dataloader(
    new_dataset_paths=["/home/najo/NAS/VLA/dataset/New_dataset", "/home/najo/NAS/VLA/dataset/New_dataset2"],
    old_dataset_patterns=[],
    new_weight=3.0,
    batch_size=32,
    num_workers=0,  # Single thread for debugging
    shuffle=False,
    return_dataset=True
)

clip_dataset = SensorImageCLIPDataset(unified_dataset, vlm_annotations=annotations, use_augmentation=False)
print(f"✓ Dataset created with {len(clip_dataset)} samples")

# Create dataloader with small batch
collate_fn_with_window = partial(clip_collate_fn, window_size=60)
from torch.utils.data import DataLoader
dataloader = DataLoader(
    clip_dataset,
    batch_size=32,
    num_workers=0,
    collate_fn=collate_fn_with_window,
    shuffle=False
)

# Analyze first few batches
print("\n" + "="*80)
print("Analyzing sample filtering...")
print("="*80)

total_samples = 0
total_filtered = 0
total_valid = 0
batch_count = 0
max_batches = 10  # Check first 10 batches

for batch_idx, batch in enumerate(dataloader):
    if batch_idx >= max_batches:
        break

    episode_ids = batch["episode_ids"]
    timestamps = batch["timestamps"]
    batch_size = len(episode_ids)

    valid_in_batch = 0
    filtered_in_batch = 0

    print(f"\n--- Batch {batch_idx + 1}/{max_batches} (size: {batch_size}) ---")

    for i, (episode_id, timestamp) in enumerate(zip(episode_ids, timestamps)):
        total_samples += 1

        if episode_id is None or timestamp is None:
            filtered_in_batch += 1
            total_filtered += 1
            if i < 3:  # Show first 3 samples
                print(f"  Sample {i}: episode_id={episode_id}, timestamp={timestamp} → FILTERED (None)")
            continue

        # Get the target_found_timestamp for this episode
        episode_data = annotations.get(episode_id, {})
        target_found_timestamp = episode_data.get("target_found_timestamp")

        if target_found_timestamp is not None:
            try:
                current_ts = float(timestamp)
                target_ts = float(target_found_timestamp)

                # Check if this sample would be included in training
                if current_ts >= target_ts:
                    valid_in_batch += 1
                    total_valid += 1
                    if i < 3:
                        print(f"  Sample {i}: {episode_id}, ts={current_ts:.3f}, target={target_ts:.3f} → VALID (Δ={current_ts-target_ts:.3f}s)")
                else:
                    filtered_in_batch += 1
                    total_filtered += 1
                    if i < 3:
                        print(f"  Sample {i}: {episode_id}, ts={current_ts:.3f}, target={target_ts:.3f} → FILTERED (Δ={current_ts-target_ts:.3f}s)")
            except (ValueError, TypeError) as e:
                filtered_in_batch += 1
                total_filtered += 1
                if i < 3:
                    print(f"  Sample {i}: {episode_id} → FILTERED (conversion error: {e})")
        else:
            filtered_in_batch += 1
            total_filtered += 1
            if i < 3:
                print(f"  Sample {i}: {episode_id} → FILTERED (no target_found_timestamp)")

    print(f"  Batch summary: {valid_in_batch} VALID, {filtered_in_batch} FILTERED ({valid_in_batch/batch_size*100:.1f}% valid)")
    batch_count += 1

# Final statistics
print("\n" + "="*80)
print("FINAL STATISTICS")
print("="*80)
print(f"Total samples analyzed: {total_samples}")
print(f"Valid samples (weight=1.0): {total_valid} ({total_valid/total_samples*100:.1f}%)")
print(f"Filtered samples (weight=0.0): {total_filtered} ({total_filtered/total_samples*100:.1f}%)")
print(f"\nExpected loss behavior:")
if total_valid == 0:
    print("  ⚠️  ALL samples filtered → Loss will be 0!")
elif total_valid < total_samples * 0.1:
    print(f"  ⚠️  Only {total_valid/total_samples*100:.1f}% samples used → Loss will be very noisy and training inefficient")
else:
    print(f"  ✓ {total_valid/total_samples*100:.1f}% samples used → Training should work")
