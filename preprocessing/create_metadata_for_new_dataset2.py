#!/usr/bin/env python3
"""
Create metadata.json for New_dataset2 episodes (data_collection_* format)

This script generates metadata.json files for episodes in New_dataset2
to make them compatible with UnifiedVLADataset.

Usage:
    python preprocessing/create_metadata_for_new_dataset2.py \
        --dataset_path /home/najo/NAS/VLA/dataset/New_dataset2
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def create_metadata(episode_dir):
    """
    Create metadata.json for a single episode

    Args:
        episode_dir: Path to episode directory (e.g., data_collection_20251108_055533)
    """
    episode_path = Path(episode_dir)
    episode_name = episode_path.name

    # Find files
    sensor_npz = list(episode_path.glob("sensor_data_*.npz"))
    robot_csv = list(episode_path.glob("robot_state_*.csv"))

    if not sensor_npz or not robot_csv:
        print(f"⚠️  Skipping {episode_name}: Missing sensor_data.npz or robot_state.csv")
        return False

    sensor_npz = sensor_npz[0]
    robot_csv = robot_csv[0]

    # Load raw sensor data and create windows
    try:
        sensor_raw = np.load(sensor_npz, allow_pickle=True)

        # New_dataset2 format: timestamps, forces, alines
        timestamps = sensor_raw['timestamps']  # (N,)
        forces = sensor_raw['forces']  # (N,)
        alines = sensor_raw['alines']  # (N, 1025)

        # Calculate sensor Hz from timestamps
        if len(timestamps) > 1:
            intervals = np.diff(timestamps)
            mean_interval = np.mean(intervals)
            sensor_hz = int(round(1.0 / mean_interval)) if mean_interval > 0 else 650
        else:
            sensor_hz = 650

        # Create 650-sample windows (sliding window approach)
        window_size = 650
        sensor_windows = []

        if len(timestamps) >= window_size:
            # Create windows every 200ms (~5Hz, matching camera rate)
            step = int(sensor_hz * 0.2)  # 650Hz * 0.2s = 130 samples
            if step == 0:
                step = 1

            for start_idx in range(0, len(timestamps) - window_size + 1, step):
                end_idx = start_idx + window_size
                window_forces = forces[start_idx:end_idx]  # (650,)
                window_alines = alines[start_idx:end_idx]  # (650, 1025)
                window_timestamp = timestamps[end_idx - 1]  # Last sample timestamp

                # Combine force (650, 1) and aline (650, 1025) → (650, 1026)
                window_data = np.concatenate([
                    window_forces.reshape(-1, 1),  # (650, 1)
                    window_alines                   # (650, 1025)
                ], axis=1)

                sensor_windows.append({
                    'timestamp': window_timestamp,
                    'data': window_data
                })

            sensor_windows_count = len(sensor_windows)
        else:
            sensor_windows_count = 0

    except Exception as e:
        print(f"⚠️  Error loading sensor data from {sensor_npz}: {e}")
        return False

    # Load robot states to get counts and frequency
    try:
        robot_df = pd.read_csv(robot_csv)
        robot_states_count = len(robot_df)

        # Calculate robot frequency from timestamps
        if 'origin_timestamp' in robot_df.columns:
            timestamps = robot_df['origin_timestamp'].values
        elif 'recv_timestamp' in robot_df.columns:
            timestamps = robot_df['recv_timestamp'].values
        else:
            print(f"⚠️  No timestamp column found in {robot_csv}")
            return False

        if len(timestamps) > 1:
            intervals = np.diff(timestamps)
            mean_interval = np.mean(intervals)
            robot_hz = int(round(1.0 / mean_interval)) if mean_interval > 0 else 100
        else:
            robot_hz = 100  # Default

        start_time = float(timestamps[0])
        end_time = float(timestamps[-1])

    except Exception as e:
        print(f"⚠️  Error loading robot states from {robot_csv}: {e}")
        return False

    # Find camera views
    images_dir = episode_path / "images" if (episode_path / "images").exists() else episode_path
    camera_views = sorted([d.name for d in images_dir.iterdir() if d.is_dir() and d.name.startswith("View")])

    # Count camera frames (from first view)
    camera_frames_count = 0
    if camera_views:
        first_view_path = images_dir / camera_views[0]
        if first_view_path.exists():
            camera_frames_count = len(list(first_view_path.glob("*.jpg")))

    # Create metadata
    metadata = {
        "episode_name": episode_name,
        "start_time": start_time,
        "end_time": end_time,
        "camera_views": camera_views,
        "sensor_hz": sensor_hz,
        "robot_hz": robot_hz,
        "sensor_window_size": 650,  # 650Hz × 1s = 650 samples
        "action_horizon": 8,
        "action_dim": 7,
        "camera_frames_count": camera_frames_count,
        "sensor_windows_count": sensor_windows_count,
        "robot_states_count": robot_states_count
    }

    # Save windowed sensor data
    if sensor_windows:
        windowed_sensor_path = episode_path / "sensor_data.npz"
        window_timestamps = np.array([w['timestamp'] for w in sensor_windows])
        window_data = np.stack([w['data'] for w in sensor_windows])  # (N, 650, 1026)

        np.savez(windowed_sensor_path,
                timestamps=window_timestamps,
                data=window_data)
        print(f"✓ Created windowed sensor_data.npz: shape {window_data.shape}")

    # Save metadata.json
    metadata_path = episode_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Created metadata.json for {episode_name}")
    print(f"   Robot Hz: {robot_hz}, Sensor windows: {sensor_windows_count}, Robot states: {robot_states_count}")

    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Create metadata.json for New_dataset2")
    parser.add_argument('--dataset_path', type=str,
                       default="/home/najo/NAS/VLA/dataset/New_dataset2",
                       help='Path to New_dataset2 root directory')
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)

    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        return

    # Find all data_collection_* episodes
    episodes = []
    for color_dir in dataset_path.iterdir():
        if color_dir.is_dir():
            for episode_dir in color_dir.glob("data_collection_*"):
                if episode_dir.is_dir():
                    episodes.append(episode_dir)

    print(f"📁 Found {len(episodes)} episodes in {dataset_path}")
    print(f"🔧 Creating metadata.json files...\n")

    success_count = 0
    for episode_dir in tqdm(episodes, desc="Processing episodes"):
        if create_metadata(episode_dir):
            success_count += 1

    print(f"\n✅ Successfully created metadata for {success_count}/{len(episodes)} episodes")


if __name__ == "__main__":
    main()
