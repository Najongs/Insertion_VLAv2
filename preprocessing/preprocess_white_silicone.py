"""
Preprocessing script for White_silicone_white_circle dataset

This script processes raw CSV-based data and generates JSON files for training.

Steps:
1. Load robot state from CSV (robot_state_*.csv)
2. For each camera view, find closest robot state for each image (within 10ms)
3. Generate JSON files with matched data
4. Include sensor data reference for later loading

Based on Data_preprocessing.ipynb
"""

import os
import glob
import csv
import json
import bisect
import numpy as np
from pathlib import Path
from tqdm import tqdm


def load_robot_csv_white_silicone(csv_path):
    """
    Load robot CSV for White_silicone_white_circle dataset

    CSV columns: recv_timestamp, origin_timestamp, send_timestamp, force_placeholder,
                 joint_1~6, pose_x, pose_y, pose_z, pose_a, pose_b, pose_r
    """
    timestamps, data = [], []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Use origin_timestamp (column 1) as main timestamp
            ts = float(row["origin_timestamp"])
            timestamps.append(ts)
            data.append({
                "timestamp": ts,
                "joint_angles": [
                    float(row[f"joint_{i}"]) for i in range(1, 7)
                ],
                "ee_pose": [
                    float(row["pose_x"]), float(row["pose_y"]), float(row["pose_z"]),
                    float(row["pose_a"]), float(row["pose_b"]), float(row["pose_r"])
                ],
            })
    return timestamps, data


def extract_timestamp_white_silicone(img_path: str):
    """
    Extract timestamp from White_silicone image filename

    Examples:
    - zed_41182735_left_1761551500.174.jpg → 1761551500.174
    - oak_1944301011169A4800_1761551500.174.jpg → 1761551500.174
    """
    try:
        filename = Path(img_path).name
        # Remove .jpg extension
        stem = filename.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
        # Split by underscore
        parts = stem.split('_')
        # The last part should be the timestamp with decimal (e.g., '1761551500.174')
        timestamp_str = parts[-1]
        return float(timestamp_str)
    except (ValueError, IndexError):
        return None


def get_closest_item(ts_list, target_ts):
    """Binary search for closest timestamp index"""
    idx = bisect.bisect_left(ts_list, target_ts)
    if idx == 0:
        return 0
    if idx >= len(ts_list):
        return len(ts_list) - 1
    before, after = ts_list[idx - 1], ts_list[idx]
    return idx - 1 if abs(before - target_ts) < abs(after - target_ts) else idx


def get_closest_robot_state(ts_list, robot_data, ts_img):
    """Get robot state closest to image timestamp"""
    idx = get_closest_item(ts_list, ts_img)
    return robot_data[idx]


def build_singleview_dataset_white_silicone(episode_dir, max_diff_s=0.010, save_json=True):
    """
    Generate JSON for each camera view in White_silicone_white_circle dataset

    Args:
        episode_dir: Path to recv_all_YYYYMMDD_HHMMSS directory
        max_diff_s: Maximum time difference between image and robot state (default: 10ms)
        save_json: Whether to save JSON files

    Returns:
        Number of valid samples generated
    """
    episode_dir = Path(episode_dir)
    print(f"\n{'='*80}")
    print(f"Processing: {episode_dir.name}")
    print(f"{'='*80}")

    # Find robot state CSV
    csv_files = list(episode_dir.glob("robot_state_*.csv"))
    if not csv_files:
        print(f"⚠️ No robot state CSV found in {episode_dir}")
        return 0

    csv_path = csv_files[0]
    print(f"📄 Loading robot states from: {csv_path.name}")
    ts_list, robot_data = load_robot_csv_white_silicone(csv_path)
    print(f"   Loaded {len(robot_data)} robot states")

    # Find camera views
    # View1-4 have left/right subdirs, View5 (OAK) has images directly
    view_configs = []

    for view_num in range(1, 5):
        view_dir = episode_dir / f"View{view_num}"
        if view_dir.exists():
            for side in ["left", "right"]:
                side_dir = view_dir / side
                if side_dir.exists():
                    view_configs.append((f"View{view_num}_{side}", side_dir))

    # View5 (OAK)
    view5_dir = episode_dir / "View5"
    if view5_dir.exists():
        view_configs.append(("View5_oak", view5_dir))

    print(f"🎥 Found {len(view_configs)} camera views")

    total_samples = 0

    for view_key, img_dir in view_configs:
        print(f"\n📸 Processing {view_key}...")

        # Find all images
        img_paths = sorted(glob.glob(str(img_dir / "*.jpg")))

        if len(img_paths) == 0:
            print(f"   ⚠️ No images found")
            continue

        # Extract timestamps
        img_data = []
        for img_path in img_paths:
            ts = extract_timestamp_white_silicone(img_path)
            if ts is not None:
                img_data.append((img_path, ts))

        if len(img_data) == 0:
            print(f"   ⚠️ No valid timestamps found")
            continue

        print(f"   Found {len(img_data)} images with valid timestamps")

        # Match with robot states
        dataset = []
        filtered_time = 0

        for img_path, ts_img in tqdm(img_data, desc=f"   Matching {view_key}"):
            robot_state = get_closest_robot_state(ts_list, robot_data, ts_img)
            diff_robot = abs(robot_state["timestamp"] - ts_img)

            if diff_robot > max_diff_s:
                filtered_time += 1
                continue

            dataset.append({
                "timestamp": ts_img,
                "image": str(Path(img_path).resolve()),
                "robot_state": robot_state,
                "time_diff_robot": diff_robot,
            })

        print(f"\n   [📊 Results for {view_key}]")
        print(f"    - Total frames: {len(img_data)}")
        print(f"    - Kept: {len(dataset)}")
        print(f"    - Filtered (Δt > {max_diff_s*1000:.1f}ms): {filtered_time}")

        # Save JSON
        if save_json and len(dataset) > 0:
            out_path = episode_dir / f"{episode_dir.name}_{view_key}_single.json"
            with open(out_path, "w") as f:
                json.dump(dataset, f, indent=2)
            print(f"    💾 Saved: {out_path.name} ({len(dataset)} samples)")
            total_samples += len(dataset)

    print(f"\n✅ Completed: {episode_dir.name}")
    print(f"   Total valid samples: {total_samples}")
    print(f"{'='*80}\n")

    return total_samples


def process_all_white_silicone_sessions(base_dir, max_diff_s=0.010):
    """
    Process all recv_all_* sessions in White_silicone_white_circle directory

    Args:
        base_dir: Path to White_silicone_white_circle directory
        max_diff_s: Maximum time difference for matching (default: 10ms)
    """
    base_dir = Path(base_dir)

    if not base_dir.exists():
        print(f"❌ Directory not found: {base_dir}")
        return

    # Find all recv_all_* directories
    sessions = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("recv_all_")])

    if not sessions:
        print(f"⚠️ No recv_all_* directories found in {base_dir}")
        return

    print(f"🔍 Found {len(sessions)} sessions to process:")
    for s in sessions:
        print(f"   - {s.name}")

    print(f"\n{'#'*80}")
    print(f"Starting preprocessing...")
    print(f"{'#'*80}\n")

    total_samples_all = 0

    for session_dir in sessions:
        try:
            samples = build_singleview_dataset_white_silicone(
                session_dir,
                max_diff_s=max_diff_s,
                save_json=True
            )
            total_samples_all += samples
        except Exception as e:
            print(f"❌ Error processing {session_dir.name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'#'*80}")
    print(f"✅ All preprocessing completed!")
    print(f"   Processed {len(sessions)} sessions")
    print(f"   Total valid samples: {total_samples_all}")
    print(f"{'#'*80}\n")


# =====================================
# Main Execution
# =====================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess White_silicone_white_circle dataset")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/home/najo/NAS/VLA/Insertion_VLA/dataset/White_silicone_white_circle",
        help="Path to White_silicone_white_circle directory"
    )
    parser.add_argument(
        "--max_diff",
        type=float,
        default=0.010,
        help="Maximum time difference in seconds (default: 0.010 = 10ms)"
    )
    parser.add_argument(
        "--single_session",
        type=str,
        default=None,
        help="Process only a single session (e.g., recv_all_20251027_165107)"
    )

    args = parser.parse_args()

    if args.single_session:
        # Process single session
        session_path = Path(args.base_dir) / args.single_session
        if not session_path.exists():
            print(f"❌ Session directory not found: {session_path}")
        else:
            build_singleview_dataset_white_silicone(
                session_path,
                max_diff_s=args.max_diff,
                save_json=True
            )
    else:
        # Process all sessions
        process_all_white_silicone_sessions(
            args.base_dir,
            max_diff_s=args.max_diff
        )
