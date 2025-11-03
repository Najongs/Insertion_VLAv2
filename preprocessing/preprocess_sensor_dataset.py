"""
Unified Preprocessing Script for Sensor-based VLA Datasets

Supports:
- White_silicone_white_circle
- Needle_insertion_eye_trocar

Key Features:
- Matches images to robot states (within 10ms)
- Extracts sensor data for the INTERVAL between images (not single point)
- Auto-generates instructions from folder names
- Handles missing right camera images
"""

import os
import glob
import csv
import json
import bisect
import numpy as np
from pathlib import Path
from tqdm import tqdm


# =====================================
# Instruction Templates
# =====================================
INSTRUCTION_TEMPLATES = {
    "white_silicone_white_circle": "Insert into the white square silicone with a white circle sticker attached",
    "needle_insertion_eye_trocar": "Insert the needle through the trocar into the eye phantom model",
    "default": "Perform the insertion task"
}


def get_instruction_from_folder(folder_name: str) -> str:
    """
    Generate instruction from folder name

    Args:
        folder_name: Name of the dataset folder

    Returns:
        Instruction string
    """
    folder_lower = folder_name.lower()

    if "white_silicone" in folder_lower or "white_circle" in folder_lower:
        return INSTRUCTION_TEMPLATES["white_silicone_white_circle"]
    elif "needle" in folder_lower and "trocar" in folder_lower:
        return INSTRUCTION_TEMPLATES["needle_insertion_eye_trocar"]
    else:
        return INSTRUCTION_TEMPLATES["default"]


# =====================================
# CSV Loading
# =====================================
def load_robot_csv(csv_path):
    """
    Load robot CSV for sensor datasets (recv_all_* format)

    CSV columns: recv_timestamp, origin_timestamp, send_timestamp, force_placeholder,
                 joint_1~6, pose_x, pose_y, pose_z, pose_a, pose_b, pose_r
    """
    timestamps, data = [], []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Use origin_timestamp as main timestamp
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


# =====================================
# Timestamp Extraction
# =====================================
def extract_timestamp(img_path: str):
    """
    Extract timestamp from image filename

    Supports formats:
    - zed_41182735_left_1761551500.174.jpg → 1761551500.174
    - oak_1944301011169A4800_1761551500.174.jpg → 1761551500.174
    """
    try:
        filename = Path(img_path).name
        stem = filename.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
        parts = stem.split('_')
        # Last part should be the timestamp with decimal
        timestamp_str = parts[-1]
        return float(timestamp_str)
    except (ValueError, IndexError):
        return None


# =====================================
# Matching Functions
# =====================================
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


def calculate_image_intervals(timestamps):
    """
    Calculate statistics about image intervals

    Args:
        timestamps: List of image timestamps

    Returns:
        dict with mean, median, min, max intervals
    """
    if len(timestamps) < 2:
        return {"mean": 1.0, "median": 1.0, "min": 1.0, "max": 1.0}

    intervals = np.diff(timestamps)
    return {
        "mean": float(np.mean(intervals)),
        "median": float(np.median(intervals)),
        "min": float(np.min(intervals)),
        "max": float(np.max(intervals)),
        "std": float(np.std(intervals))
    }


# =====================================
# Main Processing Function
# =====================================
def build_dataset_with_interval_sensor(
    episode_dir,
    max_diff_s=0.010,
    sensor_rate=650,
    save_json=True
):
    """
    Generate JSON with sensor data based on IMAGE INTERVALS

    Key Change: Instead of extracting sensor data at a single timestamp,
    we extract sensor data for the INTERVAL between consecutive images.

    Args:
        episode_dir: Path to recv_all_YYYYMMDD_HHMMSS directory
        max_diff_s: Maximum time difference between image and robot state (default: 10ms)
        sensor_rate: Sensor sampling rate in Hz (default: 650)
        save_json: Whether to save JSON files

    Returns:
        Number of valid samples generated
    """
    episode_dir = Path(episode_dir)
    parent_folder = episode_dir.parent.name

    print(f"\n{'='*80}")
    print(f"Processing: {episode_dir.name}")
    print(f"Parent folder: {parent_folder}")
    print(f"{'='*80}")

    # Generate instruction from parent folder name
    instruction = get_instruction_from_folder(parent_folder)
    print(f"📝 Generated instruction: {instruction}")

    # Find robot state CSV
    csv_files = list(episode_dir.glob("robot_state_*.csv"))
    if not csv_files:
        print(f"⚠️ No robot state CSV found")
        return 0

    csv_path = csv_files[0]
    print(f"📄 Loading robot states from: {csv_path.name}")
    ts_list, robot_data = load_robot_csv(csv_path)
    print(f"   Loaded {len(robot_data)} robot states")

    # Find camera views (check if left/right subdirs exist)
    view_configs = []

    for view_num in range(1, 5):
        view_dir = episode_dir / f"View{view_num}"
        if view_dir.exists():
            # Check if left/right subdirectories exist
            left_dir = view_dir / "left"
            right_dir = view_dir / "right"

            if left_dir.exists():
                view_configs.append((f"View{view_num}_left", left_dir))
            if right_dir.exists():
                view_configs.append((f"View{view_num}_right", right_dir))

            # If no subdirs, check if images are directly in view_dir
            if not left_dir.exists() and not right_dir.exists():
                imgs = list(view_dir.glob("*.jpg"))
                if imgs:
                    view_configs.append((f"View{view_num}", view_dir))

    # View5 (OAK)
    view5_dir = episode_dir / "View5"
    if view5_dir.exists():
        view_configs.append(("View5_oak", view5_dir))

    print(f"🎥 Found {len(view_configs)} camera views:")
    for name, _ in view_configs:
        print(f"   - {name}")

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
            ts = extract_timestamp(img_path)
            if ts is not None:
                img_data.append((img_path, ts))

        if len(img_data) < 2:
            print(f"   ⚠️ Not enough images with valid timestamps (need at least 2)")
            continue

        # Calculate image intervals
        timestamps = [ts for _, ts in img_data]
        interval_stats = calculate_image_intervals(timestamps)

        print(f"   Found {len(img_data)} images")
        print(f"   Image intervals: mean={interval_stats['mean']:.3f}s, "
              f"median={interval_stats['median']:.3f}s")

        # Calculate expected sensor samples per interval
        expected_samples = int(interval_stats['mean'] * sensor_rate)
        print(f"   Expected sensor samples per interval: ~{expected_samples}")

        # Match with robot states and create dataset
        dataset = []
        filtered_time = 0

        for i, (img_path, ts_img) in enumerate(tqdm(img_data, desc=f"   Matching {view_key}")):
            # Get robot state for this image
            robot_state = get_closest_robot_state(ts_list, robot_data, ts_img)
            diff_robot = abs(robot_state["timestamp"] - ts_img)

            if diff_robot > max_diff_s:
                filtered_time += 1
                continue

            # Calculate interval for sensor data
            # If this is the first image, use mean interval going backwards
            # If this is not the first image, use actual interval from previous
            if i == 0:
                interval_duration = interval_stats['mean']
                ts_start = ts_img - interval_duration
                ts_end = ts_img
            else:
                ts_prev = img_data[i-1][1]
                ts_start = ts_prev
                ts_end = ts_img
                interval_duration = ts_end - ts_start

            dataset.append({
                "timestamp": ts_img,
                "image": str(Path(img_path).resolve()),
                "robot_state": robot_state,
                "time_diff_robot": diff_robot,
                "sensor_interval": {
                    "start": ts_start,
                    "end": ts_end,
                    "duration": interval_duration
                }
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


def process_all_sessions(base_dirs, max_diff_s=0.010, sensor_rate=650):
    """
    Process all recv_all_* sessions in multiple base directories

    Args:
        base_dirs: List of base directory paths
        max_diff_s: Maximum time difference for matching (default: 10ms)
        sensor_rate: Sensor sampling rate in Hz (default: 650)
    """
    all_sessions = []

    for base_dir in base_dirs:
        base_path = Path(base_dir)
        if not base_path.exists():
            print(f"⚠️ Directory not found: {base_dir}")
            continue

        # Find all recv_all_* directories
        sessions = sorted([d for d in base_path.iterdir()
                          if d.is_dir() and d.name.startswith("recv_all_")])
        all_sessions.extend(sessions)

    if not all_sessions:
        print(f"⚠️ No recv_all_* directories found")
        return

    print(f"🔍 Found {len(all_sessions)} sessions to process:")
    for s in all_sessions:
        print(f"   - {s.parent.name}/{s.name}")

    print(f"\n{'#'*80}")
    print(f"Starting preprocessing...")
    print(f"{'#'*80}\n")

    total_samples_all = 0

    for session_dir in all_sessions:
        try:
            samples = build_dataset_with_interval_sensor(
                session_dir,
                max_diff_s=max_diff_s,
                sensor_rate=sensor_rate,
                save_json=True
            )
            total_samples_all += samples
        except Exception as e:
            print(f"❌ Error processing {session_dir.name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'#'*80}")
    print(f"✅ All preprocessing completed!")
    print(f"   Processed {len(all_sessions)} sessions")
    print(f"   Total valid samples: {total_samples_all}")
    print(f"{'#'*80}\n")


# =====================================
# Main Execution
# =====================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess sensor-based VLA datasets")
    parser.add_argument(
        "--base_dirs",
        type=str,
        nargs="+",
        default=[
            "/home/najo/NAS/VLA/Insertion_VLA/dataset/White_silicone_white_circle",
            "/home/najo/NAS/VLA/Insertion_VLA/dataset/Needle_insertion_eye_trocar"
        ],
        help="Paths to base directories containing recv_all_* sessions"
    )
    parser.add_argument(
        "--max_diff",
        type=float,
        default=0.010,
        help="Maximum time difference in seconds (default: 0.010 = 10ms)"
    )
    parser.add_argument(
        "--sensor_rate",
        type=int,
        default=650,
        help="Sensor sampling rate in Hz (default: 650)"
    )
    parser.add_argument(
        "--single_session",
        type=str,
        default=None,
        help="Process only a single session (provide full path)"
    )

    args = parser.parse_args()

    if args.single_session:
        # Process single session
        session_path = Path(args.single_session)
        if not session_path.exists():
            print(f"❌ Session directory not found: {session_path}")
        else:
            build_dataset_with_interval_sensor(
                session_path,
                max_diff_s=args.max_diff,
                sensor_rate=args.sensor_rate,
                save_json=True
            )
    else:
        # Process all sessions
        process_all_sessions(
            args.base_dirs,
            max_diff_s=args.max_diff,
            sensor_rate=args.sensor_rate
        )
