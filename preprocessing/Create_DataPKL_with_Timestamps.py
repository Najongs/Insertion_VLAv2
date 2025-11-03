#!/usr/bin/env python3
"""
Create data.pkl with Timestamp-based Action Delta Calculation

이 스크립트는:
1. CSV에서 타임스탬프를 읽어서 주기 확인
2. 주기에 맞춰 action delta 계산 (예: 10Hz = 100ms 간격)
3. data.pkl 파일 생성 (빠른 로딩)

주기 계산:
- Robot Hz: CSV에서 타임스탬프 간격 계산
- Action Hz: 10Hz (100ms)
- Delta = pose[t+100ms] - pose[t]

Usage:
    python preprocessing/Create_DataPKL_with_Timestamps.py \
        --dataset_dirs /path/to/dataset1 /path/to/dataset2
"""

import os
import sys
import glob
import csv
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def analyze_csv_frequency(csv_path):
    """
    CSV 타임스탬프를 분석해서 실제 주파수 계산

    Returns:
        robot_hz: 평균 주파수
        timestamps: 전체 타임스탬프 리스트
    """
    timestamps = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = float(row.get("origin_timestamp", row.get("recv_timestamp")))
            timestamps.append(ts)

    if len(timestamps) < 2:
        return None, timestamps

    # 연속된 타임스탬프 간격 계산
    intervals = np.diff(timestamps)
    mean_interval = np.mean(intervals)
    robot_hz = 1.0 / mean_interval if mean_interval > 0 else 100.0

    return robot_hz, timestamps


def calculate_actions_with_frequency(poses, timestamps, action_hz=10.0, robot_hz=None):
    """
    타임스탬프 기반으로 주기에 맞춰 action delta 계산

    Args:
        poses: (T, 6) numpy array of poses
        timestamps: (T,) numpy array of timestamps
        action_hz: Target action frequency (default: 10Hz)
        robot_hz: Actual robot frequency (auto-detected if None)

    Returns:
        actions: (N, 7) numpy array of action deltas
        action_timestamps: (N,) corresponding timestamps
    """
    if robot_hz is None:
        # Auto-detect from timestamps
        intervals = np.diff(timestamps)
        mean_interval = np.mean(intervals)
        robot_hz = 1.0 / mean_interval if mean_interval > 0 else 100.0

    # Calculate target interval in seconds
    target_interval = 1.0 / action_hz  # e.g., 0.1s for 10Hz

    print(f"   Detected robot Hz: {robot_hz:.1f}")
    print(f"   Target action Hz: {action_hz:.1f}")
    print(f"   Target interval: {target_interval*1000:.1f}ms")

    actions = []
    action_timestamps_list = []

    # Sample poses at action_hz intervals
    i = 0
    while i < len(timestamps):
        current_time = timestamps[i]
        target_time = current_time + target_interval

        # Find the closest timestamp to target_time
        # Search within reasonable range (target_time ± 20ms)
        search_start = i + 1
        search_end = min(len(timestamps), i + int(robot_hz * target_interval * 1.5))

        if search_start >= len(timestamps):
            break

        # Find closest match
        best_idx = None
        best_diff = float('inf')

        for j in range(search_start, search_end):
            diff = abs(timestamps[j] - target_time)
            if diff < best_diff:
                best_diff = diff
                best_idx = j

        # Only accept if within tolerance (e.g., 20ms)
        if best_idx is not None and best_diff < 0.020:  # 20ms tolerance
            # Calculate delta
            delta_pose = poses[best_idx] - poses[i]

            # Add gripper dimension
            delta_action = np.concatenate([delta_pose, [1.0]], axis=0)

            actions.append(delta_action)
            action_timestamps_list.append(current_time)

            # Move to next sample
            i = best_idx
        else:
            # Skip if no good match found
            i += 1

    if len(actions) == 0:
        return np.zeros((0, 7), dtype=np.float32), np.array([])

    return np.array(actions, dtype=np.float32), np.array(action_timestamps_list)


def create_data_pkl_for_episode(episode_dir, action_hz=10.0):
    """
    단일 에피소드에 대해 data.pkl 생성

    Args:
        episode_dir: Path to episode (e.g., recv_all_20251027_170308)
        action_hz: Action frequency in Hz (default: 10)
    """
    episode_dir = Path(episode_dir)

    print(f"\n{'='*80}")
    print(f"Processing: {episode_dir.name}")
    print(f"{'='*80}")

    # 1. Load robot CSV
    csv_files = list(episode_dir.glob("robot_state_*.csv"))
    if not csv_files:
        print(f"❌ No robot_state CSV found")
        return False

    csv_path = csv_files[0]
    print(f"📄 Loading CSV: {csv_path.name}")

    # Analyze frequency
    robot_hz, timestamps = analyze_csv_frequency(csv_path)

    if robot_hz is None:
        print(f"❌ Cannot detect robot frequency")
        return False

    # Load CSV with pandas
    df = pd.read_csv(csv_path)

    # Extract poses
    pose_cols = ['pose_x', 'pose_y', 'pose_z', 'pose_a', 'pose_b', 'pose_r']
    poses = df[pose_cols].values.astype(np.float32)
    timestamps = np.array(timestamps)

    print(f"   Robot states: {len(poses)}")
    print(f"   Duration: {timestamps[-1] - timestamps[0]:.2f}s")

    # 2. Calculate actions with timestamp-based frequency
    actions, action_timestamps = calculate_actions_with_frequency(
        poses, timestamps, action_hz=action_hz, robot_hz=robot_hz
    )

    print(f"   Generated actions: {len(actions)}")
    print(f"   Action timestep: {1000.0/action_hz:.1f}ms")

    if len(actions) == 0:
        print(f"❌ No actions generated")
        return False

    # 3. Load images
    print(f"\n📸 Loading image paths...")
    image_data = {}

    # Scan for all views
    for view_num in range(1, 6):
        view_dir = episode_dir / f"View{view_num}"
        if not view_dir.exists():
            continue

        # Check for left/right subdirs
        for subdir in ['left', 'right']:
            subdir_path = view_dir / subdir
            if subdir_path.exists():
                images = sorted(subdir_path.glob("*.jpg"))
                if images:
                    view_key = f"View{view_num}_{subdir}"
                    image_data[view_key] = [str(img.absolute()) for img in images]
                    print(f"   {view_key}: {len(images)} images")

        # Also check direct view folder (for View5/OAK)
        images = sorted(view_dir.glob("*.jpg"))
        if images:
            view_key = f"View{view_num}"
            image_data[view_key] = [str(img.absolute()) for img in images]
            print(f"   {view_key}: {len(images)} images")

    if not image_data:
        print(f"❌ No images found")
        return False

    # 4. Load sensor data (if exists)
    sensor_data_path = episode_dir / f"sensor_data_{episode_dir.name.replace('recv_all_', '')}.npz"
    sensor_data = None

    if sensor_data_path.exists():
        print(f"\n💾 Loading sensor data: {sensor_data_path.name}")
        sensor_npz = np.load(sensor_data_path)

        # Extract force and FPI
        if 'force' in sensor_npz and 'aline' in sensor_npz:
            force = sensor_npz['force']
            aline = sensor_npz['aline']

            # Combine: (T, 1) + (T, 1025) = (T, 1026)
            sensor_combined = np.concatenate([
                force[:, None],
                aline
            ], axis=-1).astype(np.float32)

            # Align sensor data length with actions (approximate)
            # Take samples at action intervals
            sensor_indices = np.linspace(0, len(sensor_combined)-1, len(actions), dtype=int)
            sensor_data = sensor_combined[sensor_indices]

            print(f"   Sensor data: {sensor_data.shape}")
            sensor_data = {
                'force': sensor_data[:, 0:1],
                'fpi': sensor_data[:, 1:]
            }
        else:
            print(f"   ⚠️ Sensor data format not recognized")
    else:
        print(f"   ℹ️ No sensor data found")

    # 5. Infer instruction from parent folder
    parent_name = episode_dir.parent.name.lower()
    if "white" in parent_name and "circle" in parent_name:
        instruction = "Insert into the white square silicone with a white circle sticker attached"
    elif "needle" in parent_name and "trocar" in parent_name:
        instruction = "Insert the needle through the trocar into the eye phantom model"
    else:
        instruction = "Perform the insertion task"

    # 6. Create data.pkl
    data = {
        'action': actions,  # (N, 7)
        'image': image_data,  # dict of view_key -> list of paths
        'sensor_data': sensor_data,  # dict of 'force', 'fpi' or None
        'instruction': instruction,
        'metadata': {
            'episode_name': episode_dir.name,
            'robot_hz': robot_hz,
            'action_hz': action_hz,
            'num_actions': len(actions),
            'num_robot_states': len(poses),
            'duration_seconds': timestamps[-1] - timestamps[0],
            'has_sensor': sensor_data is not None,
        }
    }

    # Save
    output_path = episode_dir / "data.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"\n✅ Saved: {output_path}")
    print(f"   Actions: {len(actions)}")
    print(f"   Views: {len(image_data)}")
    print(f"   Sensor: {'Yes' if sensor_data else 'No'}")
    print(f"   Instruction: {instruction}")

    return True


def process_all_datasets(dataset_dirs, action_hz=10.0):
    """
    여러 데이터셋 디렉토리를 처리

    Args:
        dataset_dirs: List of dataset directory paths
        action_hz: Action frequency in Hz
    """
    all_episodes = []

    for dataset_dir in dataset_dirs:
        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            print(f"⚠️ Directory not found: {dataset_dir}")
            continue

        # Find all recv_all_* or episode_* directories
        episodes = sorted([
            d for d in dataset_path.iterdir()
            if d.is_dir() and (d.name.startswith("recv_all_") or d.name.startswith("episode_"))
        ])

        all_episodes.extend(episodes)

    if not all_episodes:
        print(f"⚠️ No episodes found in provided directories")
        return

    print(f"{'='*80}")
    print(f"Found {len(all_episodes)} episodes to process")
    print(f"{'='*80}")

    success_count = 0
    fail_count = 0

    for episode_dir in all_episodes:
        try:
            success = create_data_pkl_for_episode(episode_dir, action_hz=action_hz)
            if success:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"❌ Error processing {episode_dir.name}: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1

    print(f"\n{'='*80}")
    print(f"✅ Processing Complete!")
    print(f"{'='*80}")
    print(f"   Success: {success_count}/{len(all_episodes)}")
    print(f"   Failed: {fail_count}/{len(all_episodes)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create data.pkl with timestamp-based action calculation")
    parser.add_argument(
        "--dataset_dirs",
        type=str,
        nargs="+",
        default=[
            "/home/najo/NAS/VLA/dataset/White_silicone_white_circle",
            "/home/najo/NAS/VLA/dataset/Needle_insertion_eye_trocar",
            "/home/najo/NAS/VLA/dataset/OCT_insertion",
            "/home/najo/NAS/VLA/dataset/part1",
        ],
        help="Paths to dataset directories"
    )
    parser.add_argument(
        "--action_hz",
        type=float,
        default=10.0,
        help="Action frequency in Hz (default: 10)"
    )

    args = parser.parse_args()

    process_all_datasets(args.dataset_dirs, action_hz=args.action_hz)
