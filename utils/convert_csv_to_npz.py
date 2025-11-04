#!/usr/bin/env python3
"""
CSV to NPZ Converter for Robot States

Converts robot_states.csv files to robot_states.npz for faster loading.
This provides 10-100x speedup during dataset initialization.

Usage:
    # Convert single directory
    python utils/convert_csv_to_npz.py --dir /path/to/episode_dir

    # Convert all episodes in a dataset
    python utils/convert_csv_to_npz.py --dataset /path/to/New_dataset

    # Dry run (show what would be converted)
    python utils/convert_csv_to_npz.py --dataset /path/to/New_dataset --dry-run
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def convert_csv_to_npz(csv_path: Path, npz_path: Path, verbose: bool = True):
    """Convert a single CSV file to NPZ format"""

    if not csv_path.exists():
        if verbose:
            print(f"⚠️ CSV file not found: {csv_path}")
        return False

    if npz_path.exists():
        if verbose:
            print(f"⏭️  NPZ already exists: {npz_path.name}")
        return False

    try:
        # Define column names
        joint_cols = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        pose_cols = ['pose_x', 'pose_y', 'pose_z', 'pose_a', 'pose_b', 'pose_r']
        use_cols = joint_cols + pose_cols

        # Read CSV
        df = pd.read_csv(csv_path, usecols=use_cols)

        # Convert to numpy arrays
        joints = df[joint_cols].to_numpy(dtype=np.float32)  # (N, 6)
        poses = df[pose_cols].to_numpy(dtype=np.float32)    # (N, 6)
        robot_states = np.concatenate([joints, poses], axis=1)  # (N, 12)

        # Save as NPZ (compressed)
        np.savez_compressed(
            npz_path,
            robot_states=robot_states,
            joints=joints,
            poses=poses
        )

        if verbose:
            csv_size = csv_path.stat().st_size / 1024 / 1024  # MB
            npz_size = npz_path.stat().st_size / 1024 / 1024  # MB
            print(f"✅ Converted: {csv_path.name} ({csv_size:.2f}MB) → {npz_path.name} ({npz_size:.2f}MB, {len(robot_states)} samples)")

        return True

    except Exception as e:
        if verbose:
            print(f"❌ Failed to convert {csv_path}: {e}")
        return False


def convert_directory(dir_path: Path, dry_run: bool = False, verbose: bool = True):
    """Convert robot_states.csv in a single directory"""
    csv_path = dir_path / "robot_states.csv"
    npz_path = dir_path / "robot_states.npz"

    if dry_run:
        if csv_path.exists() and not npz_path.exists():
            print(f"[DRY RUN] Would convert: {csv_path}")
        return False

    return convert_csv_to_npz(csv_path, npz_path, verbose=verbose)


def convert_dataset(dataset_path: Path, dry_run: bool = False):
    """Convert all CSV files in a dataset (all tasks/episodes)"""

    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        return

    # Collect all episode directories
    episode_dirs = []
    for task_dir in dataset_path.iterdir():
        if not task_dir.is_dir():
            continue
        for episode_dir in task_dir.iterdir():
            if episode_dir.is_dir() and episode_dir.name.startswith('episode_'):
                csv_path = episode_dir / "robot_states.csv"
                npz_path = episode_dir / "robot_states.npz"
                if csv_path.exists() and not npz_path.exists():
                    episode_dirs.append(episode_dir)

    if not episode_dirs:
        print("✅ No CSV files to convert (all already NPZ or no CSV files found)")
        return

    print(f"📦 Found {len(episode_dirs)} CSV files to convert")

    if dry_run:
        print("\n[DRY RUN] The following files would be converted:")
        for ep_dir in episode_dirs:
            print(f"  - {ep_dir}")
        return

    # Convert with progress bar
    converted_count = 0
    failed_count = 0

    for ep_dir in tqdm(episode_dirs, desc="Converting CSV → NPZ"):
        success = convert_directory(ep_dir, dry_run=False, verbose=False)
        if success:
            converted_count += 1
        else:
            failed_count += 1

    print(f"\n✅ Conversion complete!")
    print(f"   Converted: {converted_count}")
    print(f"   Failed: {failed_count}")
    print(f"   Total: {len(episode_dirs)}")


def main():
    parser = argparse.ArgumentParser(description='Convert robot_states.csv to NPZ format')
    parser.add_argument('--dir', type=str, help='Single directory to convert')
    parser.add_argument('--dataset', type=str, help='Dataset root path (converts all episodes)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be converted without actually converting')

    args = parser.parse_args()

    if args.dir:
        dir_path = Path(args.dir)
        convert_directory(dir_path, dry_run=args.dry_run)
    elif args.dataset:
        dataset_path = Path(args.dataset)
        convert_dataset(dataset_path, dry_run=args.dry_run)
    else:
        parser.print_help()
        print("\n❌ Error: Must specify either --dir or --dataset")


if __name__ == "__main__":
    main()
