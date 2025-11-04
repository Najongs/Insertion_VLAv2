#!/usr/bin/env python3
"""
Test loading speed comparison: CSV vs NPZ for robot states
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path


def test_csv_loading(csv_path: Path, iterations=10):
    """Test CSV loading speed"""
    times = []
    joint_cols = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
    pose_cols = ['pose_x', 'pose_y', 'pose_z', 'pose_a', 'pose_b', 'pose_r']
    use_cols = joint_cols + pose_cols

    for _ in range(iterations):
        start = time.perf_counter()
        df = pd.read_csv(csv_path, usecols=use_cols)
        joints = df[joint_cols].to_numpy(dtype=np.float32)
        poses = df[pose_cols].to_numpy(dtype=np.float32)
        robot_states = np.concatenate([joints, poses], axis=1)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.mean(times), np.std(times), robot_states.shape


def test_npz_loading(npz_path: Path, iterations=10):
    """Test NPZ loading speed"""
    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        with np.load(npz_path) as data:
            robot_states = data['robot_states'].astype(np.float32)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.mean(times), np.std(times), robot_states.shape


def main():
    episode_path = Path("/home/najo/NAS/VLA/dataset/New_dataset/Blue_point/episode_20251030_025856")
    csv_path = episode_path / "robot_states.csv"
    npz_path = episode_path / "robot_states.npz"

    if not csv_path.exists() or not npz_path.exists():
        print(f"❌ Files not found!")
        print(f"   CSV: {csv_path.exists()}")
        print(f"   NPZ: {npz_path.exists()}")
        return

    print("🔬 Testing robot states loading speed...\n")
    print(f"📁 Episode: {episode_path.name}")
    print(f"   CSV size: {csv_path.stat().st_size / 1024:.1f} KB")
    print(f"   NPZ size: {npz_path.stat().st_size / 1024:.1f} KB")
    print()

    # Test CSV
    print("⏱️  Testing CSV loading (10 iterations)...")
    csv_mean, csv_std, csv_shape = test_csv_loading(csv_path, iterations=10)
    print(f"   Mean: {csv_mean*1000:.2f} ms ± {csv_std*1000:.2f} ms")
    print(f"   Shape: {csv_shape}")
    print()

    # Test NPZ
    print("⏱️  Testing NPZ loading (10 iterations)...")
    npz_mean, npz_std, npz_shape = test_npz_loading(npz_path, iterations=10)
    print(f"   Mean: {npz_mean*1000:.2f} ms ± {npz_std*1000:.2f} ms")
    print(f"   Shape: {npz_shape}")
    print()

    # Comparison
    speedup = csv_mean / npz_mean
    print(f"📊 Results:")
    print(f"   Speedup: {speedup:.2f}x faster")
    print(f"   Time saved: {(csv_mean - npz_mean)*1000:.2f} ms per load")
    print()

    # Estimate for full dataset
    num_episodes = 13
    loads_per_epoch = num_episodes * 10  # Assume ~10 samples per episode
    time_saved_per_epoch = (csv_mean - npz_mean) * loads_per_epoch
    print(f"💡 Estimated time saved per epoch ({loads_per_epoch} loads):")
    print(f"   {time_saved_per_epoch:.2f} seconds")


if __name__ == "__main__":
    main()
