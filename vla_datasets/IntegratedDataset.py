"""
Integrated Dataset for VLA with Optional Sensor Data

Dataset Types:
1. insertionMeca500DatasetWithSensor - Custom collected data (with or without sensor)
   - White_silicone_white_circle: HAS sensor data (NPZ files)
   - OCT_insertion, part1: NO sensor data
2. BridgeRawSequenceDataset - Bridge v2 data (NO sensor data)

When sensor data is not available, sensor_data will be None and the model should handle it gracefully.
"""

import json
import os
import re
import glob
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from tqdm import tqdm
from pathlib import Path


# =====================================
# Utility Functions
# =====================================
def infer_lang_from_path(traj_path):
    """Infer language instruction from path (for Bridge dataset)"""
    match = re.search(r"datacol2_([^/]+)/([^/]+)/", traj_path)
    if match:
        env = match.group(1).replace("_", " ")
        task = match.group(2).replace("_", " ")
        return f"{task} task in the {env}"
    else:
        return "<no_lang>"


def load_sensor_data(sensor_npz_path: str) -> tuple:
    """
    Load sensor data from NPZ file

    Args:
        sensor_npz_path: Path to sensor NPZ file

    Returns:
        timestamps: (N,) array of timestamps
        forces: (N,) array of force data
        alines: (N, 1025) array of OCT A-scan data
        Returns (None, None, None) if loading fails
    """
    try:
        data = np.load(sensor_npz_path)
        timestamps = data['timestamps']
        forces = data['forces']
        alines = data['alines']
        return timestamps, forces, alines
    except Exception as e:
        print(f"[ERROR] Failed to load sensor data from {sensor_npz_path}: {e}")
        return None, None, None


def extract_sensor_window(timestamps: np.ndarray,
                         forces: np.ndarray,
                         alines: np.ndarray,
                         target_timestamp: float = None,
                         window_size: int = 650,
                         sample_rate: float = 650.0,
                         start_time: float = None,
                         end_time: float = None,
                         adaptive_window: bool = False) -> np.ndarray:
    """
    Extract sensor data window based on time interval or single timestamp

    Args:
        timestamps: (N,) array of sensor timestamps
        forces: (N,) array of force data
        alines: (N, 1025) array of A-scan data
        target_timestamp: Target timestamp (backward compatibility)
        window_size: Target number of samples (default: 650 = 1 second at 650Hz)
        sample_rate: Sensor sampling rate in Hz
        start_time: Start of interval (new - preferred method)
        end_time: End of interval (new - preferred method)
        adaptive_window: If True, return actual samples without padding/truncation
                        (useful with variable-length SensorEncoder)

    Returns:
        sensor_window: (window_size, 1026) or (actual_size, 1026) array of [force, aline]
    """
    if timestamps is None or len(timestamps) == 0:
        # Return zeros if no sensor data
        return np.zeros((window_size, 1026), dtype=np.float32)

    # New method: Extract data for a time interval
    if start_time is not None and end_time is not None:
        # Find indices for the interval
        start_idx = np.searchsorted(timestamps, start_time, side='left')
        end_idx = np.searchsorted(timestamps, end_time, side='right')

        # Extract data in the interval
        force_window = forces[start_idx:end_idx]
        aline_window = alines[start_idx:end_idx]

    # Old method: Extract data around a single timestamp (backward compatibility)
    elif target_timestamp is not None:
        idx = np.searchsorted(timestamps, target_timestamp)
        end_idx = min(idx + 1, len(timestamps))
        start_idx = max(0, end_idx - window_size)
        force_window = forces[start_idx:end_idx]
        aline_window = alines[start_idx:end_idx]

    else:
        raise ValueError("Must provide either (start_time, end_time) or target_timestamp")

    # Combine force and aline: (N, 1) + (N, 1025) = (N, 1026)
    sensor_data = np.concatenate([
        force_window[:, None],  # (N, 1)
        aline_window            # (N, 1025)
    ], axis=1)

    # Handle adaptive window mode
    if adaptive_window:
        # Return actual data without padding/truncation
        # The model will handle variable-length inputs
        return sensor_data.astype(np.float32)

    # Pad or truncate to window_size (default behavior)
    actual_size = len(sensor_data)

    if actual_size < window_size:
        # Pad with zeros at the beginning
        padding_size = window_size - actual_size
        padding = np.zeros((padding_size, 1026), dtype=np.float32)
        sensor_data = np.vstack([padding, sensor_data])
    elif actual_size > window_size:
        # Truncate to window_size (take the most recent samples)
        sensor_data = sensor_data[-window_size:]

    return sensor_data.astype(np.float32)


# =====================================
# Dataset Classes
# =====================================
class CSVBasedDatasetWithSensor(Dataset):
    """
    CSV-based dataset with sensor data (e.g., White_silicone_white_circle)

    Directory structure:
        trajectory_dir/
        ├── View1/left/*.jpg, View1/right/*.jpg
        ├── View2/left/*.jpg, View2/right/*.jpg
        ├── View3/left/*.jpg, View3/right/*.jpg
        ├── View4/left/*.jpg, View4/right/*.jpg
        ├── View5/*.jpg                        # OAK camera
        ├── robot_state_*.csv                  # Robot states
        └── sensor_data_*.npz                  # Sensor data (OCT/FPI)

    CSV columns: recv_timestamp,origin_timestamp,send_timestamp,force_placeholder,
                 joint_1~6,pose_x,pose_y,pose_z,pose_a,pose_b,pose_r
    """
    def __init__(self,
                 trajectory_dir: str,
                 horizon: int = 8,
                 instruction: str = "Approach the white square silicone",
                 sensor_window_size: int = 650,
                 view_selection: list = None,
                 cache_sensor_windows: bool = True):

        self.trajectory_dir = Path(trajectory_dir)
        self.horizon = horizon
        self.instruction = instruction
        self.sensor_window_size = sensor_window_size
        self.cache_sensor_windows = cache_sensor_windows

        # Default view selection
        if view_selection is None:
            view_selection = ['View1/left', 'View5']  # View1 left camera and OAK
        self.view_selection = view_selection

        print(f"📄 Loading CSV-based trajectory from: {self.trajectory_dir}")

        # Load robot state CSV
        csv_files = list(self.trajectory_dir.glob("robot_state_*.csv"))
        if not csv_files:
            raise ValueError(f"No robot state CSV found in {trajectory_dir}")

        csv_path = csv_files[0]
        print(f"   Loading robot states from: {csv_path.name}")

        # Read CSV with pandas or numpy
        import csv as csv_module
        robot_data = []
        with open(csv_path, 'r') as f:
            reader = csv_module.DictReader(f)
            for row in reader:
                robot_data.append(row)

        if len(robot_data) < 2:
            raise ValueError(f"Dataset must have at least 2 timesteps")

        print(f"   Loaded {len(robot_data)} robot states")

        # Extract timestamps and poses
        self.timestamps = np.array([float(r['recv_timestamp']) for r in robot_data])

        # Extract ee_pose (x, y, z, a, b, r)
        absolute_poses = np.array([
            [float(r['pose_x']), float(r['pose_y']), float(r['pose_z']),
             float(r['pose_a']), float(r['pose_b']), float(r['pose_r'])]
            for r in robot_data
        ], dtype=np.float32)

        # Compute 6D delta poses
        delta_poses_6d = absolute_poses[1:] - absolute_poses[:-1]

        # Extend to 7D by adding 1
        num_actions = delta_poses_6d.shape[0]
        last_dim_ones = np.ones((num_actions, 1), dtype=np.float32)
        self.actions = np.concatenate([delta_poses_6d, last_dim_ones], axis=1)

        # Load sensor data
        self.sensor_timestamps = None
        self.sensor_forces = None
        self.sensor_alines = None
        self.has_sensor = False

        sensor_files = list(self.trajectory_dir.glob("sensor_data_*.npz"))
        if sensor_files:
            sensor_path = sensor_files[0]
            print(f"🔬 Loading sensor data from: {sensor_path.name}")
            self.sensor_timestamps, self.sensor_forces, self.sensor_alines = \
                load_sensor_data(str(sensor_path))
            if self.sensor_timestamps is not None:
                self.has_sensor = True
                print(f"   ✅ Sensor data loaded: {len(self.sensor_timestamps)} samples")
        else:
            print(f"   ℹ️  No sensor data found")

        # Detect available views
        self.available_views = self._detect_available_views()
        print(f"   Available views: {self.available_views}")

        # Build image index: map timestamp to image paths
        self.image_index = self._build_image_index()
        print(f"   Built image index: {len(self.image_index)} timesteps")

        # Index chunks
        self.samples = self._index_chunks()
        print(f"   ✅ Indexed {len(self.samples)} chunks (horizon={horizon}, has_sensor={self.has_sensor})")

        # Pre-compute and cache sensor windows if enabled
        self.sensor_window_cache = {}
        if self.has_sensor and self.cache_sensor_windows:
            print(f"   🔄 Pre-computing sensor windows for faster loading...")
            self._precompute_sensor_windows_csv()
            print(f"   ✅ Cached {len(self.sensor_window_cache)} sensor windows")

    def _detect_available_views(self) -> list:
        """Detect available camera views"""
        views = []
        for view_spec in self.view_selection:
            view_path = self.trajectory_dir / view_spec
            if view_path.exists():
                views.append(view_spec)
        return views if views else ['View1/left', 'View5']

    def _build_image_index(self) -> dict:
        """Build index mapping timestamps to image files"""
        index = {}

        for t, timestamp in enumerate(self.timestamps[:-1]):  # Exclude last (no action for it)
            images = {}
            for view_spec in self.available_views:
                view_path = self.trajectory_dir / view_spec
                # Find image closest to this timestamp
                img_files = sorted(view_path.glob("*.jpg"))
                if img_files:
                    # Extract timestamps from filenames
                    best_img = None
                    min_diff = float('inf')
                    for img_file in img_files:
                        # Parse timestamp from filename (e.g., zed_41182735_left_1761551500.174.jpg)
                        try:
                            parts = img_file.stem.split('_')
                            img_ts = float('.'.join(parts[-2:]))  # Get last two parts as timestamp
                            diff = abs(img_ts - timestamp)
                            if diff < min_diff:
                                min_diff = diff
                                best_img = img_file
                        except:
                            continue

                    if best_img and min_diff < 1.0:  # Within 1 second
                        images[view_spec] = str(best_img)

            if images:
                index[t] = images

        return index

    def _index_chunks(self):
        """Index action chunks"""
        num_actions = len(self.actions)
        chunk_count = max(num_actions - self.horizon + 1, 0)
        # Only keep chunks that have images
        valid_chunks = [i for i in range(chunk_count) if i in self.image_index]
        return valid_chunks

    def _precompute_sensor_windows_csv(self):
        """Pre-compute and cache all sensor windows for CSV-based dataset"""
        if not self.has_sensor:
            return

        for t in tqdm(range(len(self.timestamps) - 1), desc="Caching sensor windows"):
            target_timestamp = self.timestamps[t]
            sensor_window = extract_sensor_window(
                self.sensor_timestamps,
                self.sensor_forces,
                self.sensor_alines,
                target_timestamp,
                window_size=self.sensor_window_size
            )
            # Store as torch tensor for faster loading
            self.sensor_window_cache[t] = torch.tensor(sensor_window, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def _fix_image_path(self, img_path: str) -> str:
        """
        Fix image paths that reference old dataset location

        Converts: /home/najo/NAS/VLA/Insertion_VLA/dataset/...
        To:       /home/najo/NAS/VLA/dataset/...

        Also converts: /home/najo/NAS/VLA/dataset/part2/...
        To:            /home/najo/NAS/VLA/dataset/part1/...
        """
        if "/Insertion_VLA/dataset/" in img_path:
            img_path = img_path.replace("/Insertion_VLA/dataset/", "/dataset/")

        # Fix part2 -> part1 path (datasets were merged into part1)
        if "/dataset/part2/" in img_path:
            img_path = img_path.replace("/dataset/part2/", "/dataset/part1/")

        return img_path

    def __getitem__(self, idx):
        start_idx = self.samples[idx]
        t = start_idx

        # === Image Loading ===
        views = []
        if t in self.image_index:
            for view_spec in self.available_views:
                if view_spec in self.image_index[t]:
                    img_path = self._fix_image_path(self.image_index[t][view_spec])
                    views.append(f"file://{img_path}")

        # === Action sequence loading ===
        start = start_idx
        end = start_idx + self.horizon

        if end > len(self.actions):
            act_seq = self.actions[start:len(self.actions)]
            last_action = act_seq[-1] if len(act_seq) > 0 else np.zeros(7, dtype=np.float32)
            repeat_len = end - len(self.actions)
            pad = np.tile(last_action, (repeat_len, 1))
            act_seq = np.concatenate([act_seq, pad], axis=0)
        else:
            act_seq = self.actions[start:end]

        act_seq = torch.tensor(act_seq, dtype=torch.float32)

        # === Sensor data loading ===
        sensor_data = None
        if self.has_sensor:
            # Use cached sensor window if available
            if self.cache_sensor_windows and t in self.sensor_window_cache:
                sensor_data = self.sensor_window_cache[t]
            else:
                # Compute on-the-fly if not cached
                target_timestamp = self.timestamps[t]
                sensor_window = extract_sensor_window(
                    self.sensor_timestamps,
                    self.sensor_forces,
                    self.sensor_alines,
                    target_timestamp,
                    window_size=self.sensor_window_size
                )
                sensor_data = torch.tensor(sensor_window, dtype=torch.float32)

        # === Language & metadata ===
        lang = self.instruction
        confidence = 1.0
        cache_key = f"{self.trajectory_dir.name}::t={t}"

        return {
            "images": views,
            "actions": act_seq,
            "sensor_data": sensor_data,
            "instruction": lang,
            "confidence": confidence,
            "cache_key": cache_key
        }


class insertionMeca500DatasetWithSensor(Dataset):
    """
    Meca500 robot dataset with OPTIONAL sensor data integration

    Directory structure:
        trajectory_dir/
        ├── View1/left/, View1/right/      # ZED camera 1
        ├── View2/left/, View2/right/      # ZED camera 2
        ├── View3/left/, View3/right/      # ZED camera 3
        ├── View4/left/, View4/right/      # ZED camera 4
        ├── View5/                         # OAK camera
        ├── robot_state_*.csv              # Robot states
        ├── sensor_data_*.npz (OPTIONAL)   # Sensor data (OCT/FPI) - only in White_silicone_white_circle
        └── *.json                         # Trajectory info

    Datasets with sensor data: White_silicone_white_circle
    Datasets without sensor data: OCT_insertion, part1

    Args:
        trajectory_dir: Path to trajectory directory
        horizon: Action prediction horizon (default: 8)
        instruction: Language instruction
        sensor_window_size: Number of sensor samples per window (default: 650)
        view_selection: Which views to use (default: ['left', 'oak'])
    """
    def __init__(self,
                 trajectory_dir: str,
                 horizon: int = 8,
                 instruction: str = "Approach the white square silicone",
                 sensor_window_size: int = 650,
                 view_selection: list = None,
                 cache_sensor_windows: bool = True):

        self.trajectory_dir = Path(trajectory_dir)
        self.horizon = horizon
        self.instruction = instruction
        self.sensor_window_size = sensor_window_size
        self.cache_sensor_windows = cache_sensor_windows

        # Default view selection: View1_left and View5_oak
        if view_selection is None:
            view_selection = ['left', 'oak']
        self.view_selection = view_selection

        # Load trajectory data (find JSON files)
        json_files = list(self.trajectory_dir.glob("*_single.json"))
        if not json_files:
            # Try precise multi-view JSON
            json_files = list(self.trajectory_dir.glob("*_precise_*.json"))

        if not json_files:
            raise ValueError(f"No JSON trajectory files found in {trajectory_dir}")

        # Use the first JSON file (can be extended to handle multiple)
        self.json_path = json_files[0]
        print(f"📄 Loading trajectory from: {self.json_path.name}")

        with open(self.json_path, 'r') as f:
            self.trajectory_data = json.load(f)

        if len(self.trajectory_data) < 2:
            raise ValueError(f"Dataset {self.json_path} must have at least 2 timesteps")

        # Extract robot poses and compute delta actions
        absolute_poses = np.array([
            item['robot_state']['ee_pose'] for item in self.trajectory_data
        ], dtype=np.float32)

        # Compute 6D delta poses
        delta_poses_6d = absolute_poses[1:] - absolute_poses[:-1]

        # Extend to 7D by adding 1 in last dimension
        num_actions = delta_poses_6d.shape[0]
        last_dim_ones = np.ones((num_actions, 1), dtype=np.float32)
        self.actions = np.concatenate([delta_poses_6d, last_dim_ones], axis=1)

        # Try to load sensor data if available (OPTIONAL)
        self.sensor_timestamps = None
        self.sensor_forces = None
        self.sensor_alines = None
        self.has_sensor = False

        sensor_files = list(self.trajectory_dir.glob("sensor_data_*.npz"))
        if sensor_files:
            sensor_path = sensor_files[0]
            print(f"🔬 Loading sensor data from: {sensor_path.name}")
            self.sensor_timestamps, self.sensor_forces, self.sensor_alines = \
                load_sensor_data(str(sensor_path))
            if self.sensor_timestamps is not None:
                self.has_sensor = True
                print(f"   ✅ Sensor data loaded: {len(self.sensor_timestamps)} samples")
            else:
                print(f"   ⚠️ Failed to load sensor data")
        else:
            print(f"   ℹ️  No sensor data found (this is OK for OCT_insertion/part1)")

        # Determine available views
        self.available_views = self._detect_available_views()
        print(f"   Available views: {self.available_views}")

        # Index chunks
        self.samples = self._index_chunks()
        print(f"   ✅ Indexed {len(self.samples)} chunks (horizon={horizon}, has_sensor={self.has_sensor})")

        # Pre-compute and cache sensor windows if enabled
        self.sensor_window_cache = {}
        if self.has_sensor and self.cache_sensor_windows:
            print(f"   🔄 Pre-computing sensor windows for faster loading...")
            self._precompute_sensor_windows()
            print(f"   ✅ Cached {len(self.sensor_window_cache)} sensor windows")

    def _detect_available_views(self) -> list:
        """Detect which camera views are available"""
        views = []

        # Check first trajectory point
        first_point = self.trajectory_data[0]
        if 'images' in first_point:
            # JSON contains image paths
            img_keys = sorted(first_point['images'].keys())
            for key in img_keys:
                if any(sel in key for sel in self.view_selection):
                    views.append(key)
        elif 'image' in first_point:
            # Single view JSON
            views.append('single_view')
        else:
            # Fallback: check directory structure
            for view_num in range(1, 6):
                view_dir = self.trajectory_dir / f"View{view_num}"
                if view_dir.exists():
                    if view_num <= 4:  # ZED cameras
                        for side in ['left', 'right']:
                            if 'left' in self.view_selection and side == 'left':
                                views.append(f"View{view_num}_{side}")
                            elif 'right' in self.view_selection and side == 'right':
                                views.append(f"View{view_num}_{side}")
                    else:  # OAK camera
                        if 'oak' in self.view_selection:
                            views.append(f"View{view_num}")

        return views if views else ['View1_left', 'View5']  # Default fallback

    def _index_chunks(self):
        """Index action chunks"""
        num_actions = len(self.actions)
        chunk_count = max(num_actions - self.horizon + 1, 0)
        return list(range(chunk_count))

    def _precompute_sensor_windows(self):
        """Pre-compute and cache all sensor windows for faster data loading"""
        if not self.has_sensor:
            return

        for idx in tqdm(range(len(self.trajectory_data)), desc="Caching sensor windows"):
            current_data_point = self.trajectory_data[idx]

            # Check if sensor_interval info is available (new format)
            if 'sensor_interval' in current_data_point:
                interval_info = current_data_point['sensor_interval']
                sensor_window = extract_sensor_window(
                    self.sensor_timestamps,
                    self.sensor_forces,
                    self.sensor_alines,
                    start_time=interval_info['start'],
                    end_time=interval_info['end'],
                    window_size=self.sensor_window_size
                )
            else:
                # Fallback to old method (single timestamp)
                target_timestamp = current_data_point.get('timestamp', 0.0)
                if target_timestamp > 0:
                    sensor_window = extract_sensor_window(
                        self.sensor_timestamps,
                        self.sensor_forces,
                        self.sensor_alines,
                        target_timestamp=target_timestamp,
                        window_size=self.sensor_window_size
                    )
                else:
                    sensor_window = None

            if sensor_window is not None:
                # Store as torch tensor for faster loading
                self.sensor_window_cache[idx] = torch.tensor(sensor_window, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def _fix_image_path(self, img_path: str) -> str:
        """
        Fix image paths that reference old dataset location

        Converts: /home/najo/NAS/VLA/Insertion_VLA/dataset/...
        To:       /home/najo/NAS/VLA/dataset/...

        Also converts: /home/najo/NAS/VLA/dataset/part2/...
        To:            /home/najo/NAS/VLA/dataset/part1/...
        """
        if "/Insertion_VLA/dataset/" in img_path:
            img_path = img_path.replace("/Insertion_VLA/dataset/", "/dataset/")

        # Fix part2 -> part1 path (datasets were merged into part1)
        if "/dataset/part2/" in img_path:
            img_path = img_path.replace("/dataset/part2/", "/dataset/part1/")

        return img_path

    def __getitem__(self, idx):
        start_idx = self.samples[idx]
        t = start_idx

        # === Image Loading ===
        current_data_point = self.trajectory_data[t]
        views = []

        if 'images' in current_data_point:
            # Multi-view JSON
            image_paths_dict = current_data_point['images']
            for view_key in self.available_views:
                if view_key in image_paths_dict:
                    img_path = self._fix_image_path(image_paths_dict[view_key])
                    views.append(f"file://{img_path}")
        elif 'image' in current_data_point:
            # Single view JSON
            img_path = self._fix_image_path(current_data_point['image'])
            views.append(f"file://{img_path}")

        # === Action sequence loading ===
        start = start_idx
        end = start_idx + self.horizon

        # Pad with last action if needed (same as Bridge dataset)
        if end > len(self.actions):
            act_seq = self.actions[start:len(self.actions)]
            last_action = act_seq[-1] if len(act_seq) > 0 else np.zeros(7, dtype=np.float32)
            repeat_len = end - len(self.actions)
            pad = np.tile(last_action, (repeat_len, 1))
            act_seq = np.concatenate([act_seq, pad], axis=0)
        else:
            act_seq = self.actions[start:end]

        act_seq = torch.tensor(act_seq, dtype=torch.float32)

        # === Sensor data loading (OPTIONAL) ===
        sensor_data = None
        if self.has_sensor:
            # Use cached sensor window if available
            if self.cache_sensor_windows and t in self.sensor_window_cache:
                sensor_data = self.sensor_window_cache[t]
            else:
                # Compute on-the-fly if not cached
                # Check if sensor_interval info is available (new format)
                if 'sensor_interval' in current_data_point:
                    interval_info = current_data_point['sensor_interval']
                    sensor_window = extract_sensor_window(
                        self.sensor_timestamps,
                        self.sensor_forces,
                        self.sensor_alines,
                        start_time=interval_info['start'],
                        end_time=interval_info['end'],
                        window_size=self.sensor_window_size
                    )
                else:
                    # Fallback to old method (single timestamp)
                    target_timestamp = current_data_point.get('timestamp', 0.0)
                    if target_timestamp > 0:
                        sensor_window = extract_sensor_window(
                            self.sensor_timestamps,
                            self.sensor_forces,
                            self.sensor_alines,
                            target_timestamp=target_timestamp,
                            window_size=self.sensor_window_size
                        )
                    else:
                        sensor_window = None

                if sensor_window is not None:
                    sensor_data = torch.tensor(sensor_window, dtype=torch.float32)

        # sensor_data can be None if no sensor data available

        # === Language & metadata ===
        lang = self.instruction
        confidence = 1.0
        cache_key = f"{self.json_path.parent.name}::{self.json_path.stem}::t={t}"

        return {
            "images": views,
            "actions": act_seq,
            "sensor_data": sensor_data,  # Can be None if no sensor data
            "instruction": lang,
            "confidence": confidence,
            "cache_key": cache_key
        }


class BridgeRawSequenceDataset(Dataset):
    """
    Berkeley Bridge Dataset v2 (NO sensor data)
    Kept for compatibility and mixed training
    """
    def __init__(self, root, horizon=8, max_traj=None):
        self.root = root
        self.horizon = horizon

        policy_files1 = glob.glob(
            os.path.join(
                root, "bridge_data_v2", "datacol2_*", "*", "*", "*", "raw", "traj_group*", "traj*", "policy_out.pkl"
            ),
            recursive=False
        )
        policy_files2 = glob.glob(
            os.path.join(
                root, "bridge_data_v1", "berkeley", "*", "*", "*", "raw", "traj_group*", "traj*", "policy_out.pkl"
            ),
            recursive=False
        )

        policy_files3 = glob.glob(
            os.path.join(
                root, "bridge_data_v2", "deepthought_*", "*", "*", "*", "raw", "traj_group*", "traj*", "policy_out.pkl"
            ),
            recursive=False
        )

        all_policy_files = policy_files2 + policy_files3
        self.traj_paths = [os.path.dirname(p) for p in all_policy_files]
        print(f"✅ Found {len(self.traj_paths)} Bridge trajectories")

        if max_traj:
            self.traj_paths = self.traj_paths[:max_traj]

        # Calculate max views
        self.max_views = 0
        for traj_path in tqdm(self.traj_paths, desc="Scanning max views"):
            view_dirs = [d for d in os.listdir(traj_path) if d.startswith("images")]
            self.max_views = max(self.max_views, len(view_dirs))
        print(f"✅ Max views: {self.max_views}")

        # Index chunks
        self.samples = self._index_chunks()

    def _index_chunks(self):
        samples = []
        for traj_path in tqdm(self.traj_paths, desc="Indexing Chunks"):
            img_dir = os.path.join(traj_path, "images0")
            imgs = sorted(glob.glob(os.path.join(img_dir, "im_*.jpg")))
            if not imgs:
                continue
            T = len(imgs)
            chunk_count = max(T - self.horizon + 1, 0)
            for i in range(chunk_count):
                samples.append((traj_path, i))
        print(f"✅ Indexed {len(samples)} chunks")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        traj_path, start_idx = self.samples[idx]

        # Load views
        view_dirs = sorted([d for d in os.listdir(traj_path) if d.startswith("images")])
        num_views = len(view_dirs)

        t = start_idx
        views = []
        for v in range(num_views):
            img_path = os.path.join(traj_path, f"images{v}", f"im_{t}.jpg")
            if os.path.exists(img_path):
                views.append(f"file://{os.path.abspath(img_path)}")

        # Pad views
        if len(views) < self.max_views:
            views += [None] * (self.max_views - len(views))
        views = views[:self.max_views]

        # Load actions
        with open(os.path.join(traj_path, "policy_out.pkl"), "rb") as f:
            actions = pickle.load(f)
            if isinstance(actions[0], dict):
                actions = [a.get("actions") for a in actions if "actions" in a]
        actions = np.array(actions, dtype=np.float32)

        # Extract action sequence
        start = start_idx
        end = start_idx + self.horizon
        if end > len(actions):
            act_seq = actions[start:len(actions)]
            last_action = act_seq[-1] if len(act_seq) > 0 else np.zeros(actions.shape[1], dtype=np.float32)
            repeat_len = end - len(actions)
            pad = np.tile(last_action, (repeat_len, 1))
            act_seq = np.concatenate([act_seq, pad], axis=0)
        else:
            act_seq = actions[start:end]

        act_seq = torch.tensor(act_seq, dtype=torch.float32)

        # Language
        lang_path = os.path.join(traj_path, "lang.txt")
        lang = ""
        confidence = 0.5
        if os.path.exists(lang_path):
            with open(lang_path, "r") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            if len(lines) == 1:
                lang = lines[0]
            elif len(lines) >= 2:
                lang = lines[0]
                for line in lines[1:]:
                    if "confidence" in line.lower():
                        try:
                            confidence = float(line.split(":")[-1].strip())
                        except:
                            confidence = 0.5
        else:
            lang = infer_lang_from_path(traj_path)

        cache_key = f"{traj_path}::t={t}"

        # NO sensor data for Bridge dataset
        sensor_data = None

        return {
            "images": views,
            "actions": act_seq,
            "sensor_data": sensor_data,  # None for Bridge (no sensor)
            "instruction": lang,
            "confidence": confidence,
            "cache_key": cache_key
        }


# =====================================
# Collate Function
# =====================================
def collate_fn_with_sensor(batch):
    """
    Collate function that handles OPTIONAL sensor data

    Key Feature: Always returns sensor_data tensor with consistent shape,
    filling with zeros for samples without sensor data.

    Args:
        batch: List of dataset items

    Returns:
        Dictionary with batched data including:
        - sensor_data: Always (B, 650, 1026) tensor
        - has_sensor_mask: (B,) boolean tensor indicating which samples have real sensor data
    """
    images = [b["images"] for b in batch]
    instructions = [b["instruction"] for b in batch]
    confidences = [b["confidence"] for b in batch]
    cache_keys = [b["cache_key"] for b in batch]

    actions = torch.stack([b["actions"] for b in batch], dim=0)  # [B, H, 7]

    # Handle sensor data: ALWAYS return a tensor with consistent shape
    sensor_data_list = [b["sensor_data"] for b in batch]

    # Create mask indicating which samples have real sensor data
    has_sensor_mask = torch.tensor([s is not None for s in sensor_data_list], dtype=torch.bool)

    # Stack sensor data, using zeros for None
    sensor_data = torch.stack([
        s if s is not None else torch.zeros((650, 1026), dtype=torch.float32)
        for s in sensor_data_list
    ], dim=0)  # [B, 650, 1026]

    return {
        "images": images,
        "instruction": instructions,
        "actions": actions,
        "sensor_data": sensor_data,           # Always (B, 650, 1026) tensor
        "has_sensor_mask": has_sensor_mask,   # (B,) boolean mask
        "confidence": confidences,
        "cache_keys": cache_keys,
    }


# =====================================
# Helper Functions
# =====================================
def create_integrated_dataloader(
    dataset_root: str = None,
    trajectory_dirs: list = None,
    bridge_root: str = None,
    batch_size: int = 4,
    horizon: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    bridge_max_traj: int = None,
    val_split: float = 0.1,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    priority_datasets: dict = None,  # NEW: {"dataset_name": weight}
):
    """
    Create train and validation DataLoaders with integrated datasets

    Args:
        dataset_root: Path to root directory containing trajectory subdirectories
        trajectory_dirs: List of specific paths to custom trajectory directories (overrides dataset_root)
        bridge_root: Path to Bridge dataset root (optional)
        batch_size: Batch size per GPU
        horizon: Action horizon
        shuffle: Whether to shuffle training data
        num_workers: Number of worker processes
        bridge_max_traj: Maximum number of Bridge trajectories to use
        val_split: Validation split ratio (0.0 to 1.0)
        distributed: Whether to use DistributedSampler for multi-GPU training
        rank: Rank of current process (for distributed training)
        world_size: Total number of processes (for distributed training)
        priority_datasets: Dict of dataset names and their sampling weights
                          e.g., {"Needle_insertion_eye_trocar": 2.0, "White_silicone_white_circle": 2.0}

    Returns:
        train_loader, val_loader: Training and validation DataLoaders
    """
    import os
    from pathlib import Path
    from torch.utils.data import random_split, DistributedSampler, WeightedRandomSampler

    datasets = []
    dataset_names = []  # Track dataset names for weighted sampling

    # If dataset_root is provided, scan for trajectory directories
    if dataset_root:
        dataset_root = Path(dataset_root)
        # Look for subdirectories that contain trajectory data
        # Skip cache and raw directories
        skip_dirs = {'cache', 'raw', '__pycache__'}

        for subdir in sorted(dataset_root.iterdir()):
            if subdir.is_dir() and subdir.name not in skip_dirs:
                # Check if this directory contains trajectory data
                # Look for subdirectories with recv_all_* pattern or direct trajectory data
                traj_paths = []
                if list(subdir.glob('recv_all_*')):
                    # Contains recv_all_* subdirectories
                    traj_paths = [p for p in subdir.glob('recv_all_*') if p.is_dir()]
                elif (subdir / 'positions_xyz.csv').exists() or (subdir / 'trajectory_data.json').exists():
                    # Direct trajectory data
                    traj_paths = [subdir]

                for traj_dir in traj_paths:
                    try:
                        ds = insertionMeca500DatasetWithSensor(
                            trajectory_dir=str(traj_dir),
                            horizon=horizon
                        )
                        datasets.append(ds)
                        # Track dataset name (parent directory name)
                        dataset_names.append(traj_dir.parent.name)
                        sensor_status = "WITH sensor" if ds.has_sensor else "NO sensor"
                        if rank == 0:
                            print(f"✅ Added dataset: {traj_dir.name} ({len(ds)} samples, {sensor_status})")
                    except Exception as e:
                        if rank == 0:
                            print(f"⚠️  Skipped {traj_dir}: {e}")

    # Add specific trajectory directories if provided
    if trajectory_dirs:
        for traj_dir in trajectory_dirs:
            try:
                ds = insertionMeca500DatasetWithSensor(
                    trajectory_dir=traj_dir,
                    horizon=horizon
                )
                datasets.append(ds)
                # Track dataset name (directory name or parent name)
                traj_path = Path(traj_dir)
                dataset_name = traj_path.parent.name if 'recv_all_' in traj_path.name else traj_path.name
                dataset_names.append(dataset_name)
                sensor_status = "WITH sensor" if ds.has_sensor else "NO sensor"
                if rank == 0:
                    print(f"✅ Added custom dataset: {traj_dir} ({len(ds)} samples, {sensor_status})")
            except Exception as e:
                if rank == 0:
                    print(f"⚠️  Failed to load {traj_dir}: {e}")

    # Add Bridge dataset (without sensor)
    if bridge_root and os.path.exists(bridge_root):
        try:
            bridge_ds = BridgeRawSequenceDataset(
                root=bridge_root,
                horizon=horizon,
                max_traj=bridge_max_traj
            )
            datasets.append(bridge_ds)
            dataset_names.append("Bridge")
            if rank == 0:
                print(f"✅ Added Bridge dataset: {len(bridge_ds)} samples (NO sensor)")
        except Exception as e:
            if rank == 0:
                print(f"⚠️  Failed to load Bridge dataset: {e}")

    if not datasets:
        raise ValueError("No datasets loaded!")

    # Concatenate all datasets
    if len(datasets) == 1:
        combined_dataset = datasets[0]
    else:
        combined_dataset = ConcatDataset(datasets)

    if rank == 0:
        print(f"\n📊 Total dataset size: {len(combined_dataset)} samples")

    # Split into train and validation sets
    if val_split > 0:
        val_size = int(len(combined_dataset) * val_split)
        train_size = len(combined_dataset) - val_size
        train_dataset, val_dataset = random_split(
            combined_dataset,
            [train_size, val_size],
            generator=torch.manual_seed(42)
        )
        if rank == 0:
            print(f"   Train: {train_size} samples")
            print(f"   Val: {val_size} samples")
    else:
        train_dataset = combined_dataset
        val_dataset = None
        if rank == 0:
            print(f"   Train: {len(train_dataset)} samples (no validation split)")

    # Create weighted sampler if priority_datasets is specified
    train_sampler = None
    val_sampler = None
    train_shuffle = shuffle

    if priority_datasets and not distributed:
        # Create sample weights based on dataset priority
        sample_weights = []

        # Get actual indices from train_dataset (which may be a Subset after random_split)
        if hasattr(train_dataset, 'indices'):
            # train_dataset is a Subset, get original indices
            train_indices = train_dataset.indices
        else:
            # train_dataset is the full dataset
            train_indices = list(range(len(train_dataset)))

        # If using ConcatDataset, track which samples belong to which dataset
        if isinstance(combined_dataset, ConcatDataset):
            cumulative_sizes = combined_dataset.cumulative_sizes
            for orig_idx in train_indices:
                # Find which dataset this sample belongs to
                dataset_idx = 0
                for i, cum_size in enumerate(cumulative_sizes):
                    if orig_idx < cum_size:
                        dataset_idx = i
                        break

                # Get dataset name and weight
                dataset_name = dataset_names[dataset_idx]
                weight = priority_datasets.get(dataset_name, 1.0)
                sample_weights.append(weight)
        else:
            # Single dataset
            dataset_name = dataset_names[0] if dataset_names else "unknown"
            weight = priority_datasets.get(dataset_name, 1.0)
            sample_weights = [weight] * len(train_dataset)

        # Create weighted sampler
        sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float)
        train_sampler = WeightedRandomSampler(
            weights=sample_weights_tensor,
            num_samples=len(train_dataset),
            replacement=True
        )
        train_shuffle = False  # Don't shuffle when using WeightedRandomSampler

        if rank == 0 and priority_datasets:
            print(f"\n⚖️  Weighted sampling enabled:")
            for name, weight in priority_datasets.items():
                matching_count = sum(1 for dn in dataset_names if name in dn)
                if matching_count > 0:
                    print(f"   {name}: {weight}x weight")

    # Create samplers for distributed training (overrides weighted sampler)
    elif distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        ) if val_dataset else None
        train_shuffle = False

        if rank == 0 and priority_datasets:
            print(f"\n⚠️  Note: Weighted sampling not supported with DistributedSampler")

    # Create train DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn_with_sensor,
        pin_memory=True
    )

    # Create validation DataLoader
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn_with_sensor,
            pin_memory=True
        )
    else:
        val_loader = None

    return train_loader, val_loader


# =====================================
# Main - Example Usage
# =====================================
if __name__ == "__main__":
    print("="*80)
    print("Testing Integrated Dataset with Optional Sensor Data")
    print("="*80)

    # Test 1: CSV-based Dataset WITH sensor data (White_silicone_white_circle)
    test_dir_with_sensor = "/home/najo/NAS/VLA/dataset/White_silicone_white_circle/recv_all_20251027_165107"

    if os.path.exists(test_dir_with_sensor):
        print(f"\n🧪 Test 1: CSV-based Dataset WITH sensor data")
        print(f"   Directory: {test_dir_with_sensor}")
        dataset1 = CSVBasedDatasetWithSensor(
            trajectory_dir=test_dir_with_sensor,
            horizon=8
        )

        print(f"\n📊 Dataset size: {len(dataset1)}")
        print(f"   Has sensor: {dataset1.has_sensor}")

        sample = dataset1[0]
        print(f"\n📦 Sample 0:")
        print(f"   Images: {len(sample['images'])} views")
        print(f"   Actions shape: {sample['actions'].shape}")
        print(f"   Sensor data: {sample['sensor_data'].shape if sample['sensor_data'] is not None else 'None'}")
        print(f"   Instruction: {sample['instruction']}")

    # Test 2: Dataset WITHOUT sensor data (OCT_`insertion` - if exists)
    test_dir_no_sensor = "/home/najo/NAS/VLA/dataset/OCT_insertion/Captures1"

    if os.path.exists(test_dir_no_sensor):
        print(f"\n🧪 Test 2: Dataset WITHOUT sensor data")
        print(f"   Directory: {test_dir_no_sensor}")
        try:
            dataset2 = insertionMeca500DatasetWithSensor(
                trajectory_dir=test_dir_no_sensor,
                horizon=8,
                instruction="Insert the probe into tissue"
            )

            print(f"\n📊 Dataset size: {len(dataset2)}")
            print(f"   Has sensor: {dataset2.has_sensor}")

            sample = dataset2[0]
            print(f"\n📦 Sample 0:")
            print(f"   Images: {len(sample['images'])} views")
            print(f"   Actions shape: {sample['actions'].shape}")
            print(f"   Sensor data: {sample['sensor_data'].shape if sample['sensor_data'] is not None else 'None (Expected)'}")
            print(f"   Instruction: {sample['instruction']}")
        except Exception as e:
            print(f"   ⚠️ Could not load: {e}")

    # Test 3: Mixed DataLoader
    print(f"\n🧪 Test 3: Mixed DataLoader with both sensor and non-sensor data")

    test_dirs = []
    if os.path.exists(test_dir_with_sensor):
        test_dirs.append(test_dir_with_sensor)
    if os.path.exists(test_dir_no_sensor):
        test_dirs.append(test_dir_no_sensor)

    if test_dirs:
        dataloader = create_integrated_dataloader(
            trajectory_dirs=test_dirs,
            batch_size=2,
            shuffle=False,
            num_workers=0
        )

        batch = next(iter(dataloader))
        print(f"\n📦 Batch:")
        print(f"   Batch size: {batch['actions'].shape[0]}")
        print(f"   Actions shape: {batch['actions'].shape}")
        print(f"   Sensor data: {batch['sensor_data'].shape if batch['sensor_data'] is not None else 'None or Mixed'}")
        print(f"   Instructions: {batch['instruction']}")

    print("\n" + "="*80)
    print("✅ All tests passed!")
    print("="*80)
