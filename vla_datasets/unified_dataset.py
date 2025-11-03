"""
Unified VLA Dataset (통합 데이터셋)

두 가지 데이터 포맷을 지원하는 통합 데이터셋:
1. Old Format (AsyncInsertionMeca500DatasetWithSensor): data.pkl 기반
2. New Format (NewAsyncInsertionDataset): metadata.json + sensor_data.npz 기반

Key Features:
- VL feature caching support
- Memory-optimized with mmap
- Weighted random sampling
- Async VLM update pattern (reuse_count)

Usage:
    # Old format dataset
    ds = UnifiedVLADataset(
        data_dir="/path/to/recv_all_xxx",
        format='old',
        horizon=8,
        vlm_reuse_count=3
    )

    # New format dataset
    ds = UnifiedVLADataset(
        data_dir="/path/to/episode_xxx",
        format='new',
        horizon=8,
        vlm_reuse_count=3
    )

    # Create weighted dataloader
    loader = create_unified_dataloader(
        old_dataset_patterns=["/path/to/old/*"],
        new_dataset_path="/path/to/new",
        old_weight=1.0,
        new_weight=3.0,
        batch_size=4
    )
"""

import os
import gc
import glob
import json
import pickle
from pathlib import Path
from typing import Literal, Optional, List, Dict, Any
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler


# =====================================
# Utility Functions
# =====================================

def _safe_joblib_load(pkl_path):
    """Try joblib (mmap), fallback to pickle"""
    try:
        import joblib
        return joblib.load(pkl_path, mmap_mode='r')
    except Exception:
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)


def _load_sensor_compound(traj_dir: Path, T_actions: int):
    """Load sensor data from .npy or data.pkl"""
    npy_path = traj_dir / "sensor_data.npy"
    if npy_path.exists():
        arr = np.load(npy_path, mmap_mode='r')
        if arr.ndim == 2 and arr.shape == (T_actions, 1026):
            return arr

    data = _safe_joblib_load(traj_dir / "data.pkl")
    sensor_raw = data.get("sensor_data")
    if sensor_raw is None:
        return np.zeros((T_actions, 1026), dtype=np.float32)

    if isinstance(sensor_raw, dict):
        fpi_data = sensor_raw.get("fpi", np.zeros((T_actions, 1025), dtype=np.float32))
        force_data = sensor_raw.get("force", np.zeros((T_actions, 1), dtype=np.float32))
        return np.column_stack((force_data, fpi_data)).astype(np.float32, copy=False)
    else:
        return np.asarray(sensor_raw, dtype=np.float32)


# =====================================
# Unified VLA Dataset
# =====================================

class UnifiedVLADataset(Dataset):
    """
    통합 VLA 데이터셋 - Old/New format 자동 감지

    Args:
        data_dir: 데이터 디렉토리 경로
        format: 'auto', 'old', 'new'
        horizon: Action prediction horizon (default: 8)
        vlm_reuse_count: VL feature reuse count (default: 3)
        sensor_window_size: Sensor window size (default: 65 for async, 650 for full)
        action_expert_hz: Action expert frequency (default: 10 Hz)
        cache_root: VL cache root directory
    """
    def __init__(
        self,
        data_dir: str,
        format: Literal['auto', 'old', 'new'] = 'auto',
        horizon: int = 8,
        vlm_reuse_count: int = 3,
        sensor_window_size: int = 65,
        action_expert_hz: int = 10,
        instruction: Optional[str] = None,
        cache_root: str = "/home/najo/NAS/VLA/dataset/cache/qwen_vl_features",
    ):
        self.data_dir = Path(data_dir)
        self.cache_root = Path(cache_root)
        self.horizon = int(horizon)
        self.vlm_reuse_count = int(vlm_reuse_count)
        self.sensor_window_size = int(sensor_window_size)
        self.action_expert_hz = int(action_expert_hz)
        self.sensor_hz = 650

        # Auto-detect format
        if format == 'auto':
            format = self._detect_format()

        self.format = format

        # Load data based on format
        if format == 'old':
            self._load_old_format(instruction)
        elif format == 'new':
            self._load_new_format(instruction)
        else:
            raise ValueError(f"Unknown format: {format}")

        # Pre-scan VL cache files (optimization)
        self._scan_vl_cache()

        print(f"📦 Loaded {self.data_dir.name} ({self.format} format)")
        print(f"   Samples: {self._total_samples}, Sensor: {self.has_sensor}, VL Cache: {self.cache_found_count}/{len(self.vl_cache_files)}")

    def _detect_format(self) -> str:
        """Auto-detect dataset format"""
        if (self.data_dir / "metadata.json").exists():
            return 'new'
        elif (self.data_dir / "data.pkl").exists():
            return 'old'
        else:
            raise ValueError(f"Cannot detect format for {self.data_dir}")

    def _load_old_format(self, instruction: Optional[str]):
        """Load old format (data.pkl based)"""
        data_file = self.data_dir / "data.pkl"
        if not data_file.exists():
            raise FileNotFoundError(f"data.pkl not found in {self.data_dir}")

        data = _safe_joblib_load(data_file)

        # Actions
        actions = data.get("action")
        if actions is None:
            raise ValueError(f"'action' missing in {data_file}")
        self.actions = np.asarray(actions, dtype=np.float32)
        T = len(self.actions)

        # Images
        self.images = data.get("image", {}) or {}

        # Instruction
        self.instruction = instruction or data.get("instruction", "Perform needle insertion task.")

        # Sensor data
        self.has_sensor = ("sensor_data" in data) and (data["sensor_data"] is not None)
        if self.has_sensor:
            npy_path = self.data_dir / "sensor_data.npy"
            if npy_path.exists():
                self.sensor_data = np.load(npy_path, mmap_mode='r')
                if not (self.sensor_data.ndim == 2 and self.sensor_data.shape == (T, 1026)):
                    self.sensor_data = _load_sensor_compound(self.data_dir, T)
            else:
                self.sensor_data = _load_sensor_compound(self.data_dir, T)
        else:
            self.sensor_data = np.zeros((T, 1026), dtype=np.float32)

        del data
        gc.collect()

        # Compute sample count
        self.action_step_size = self.horizon
        self.max_action_steps = (T - self.horizon) // self.horizon
        if self.max_action_steps < 0:
            self.max_action_steps = 0
        self._total_samples = self.max_action_steps * self.vlm_reuse_count

        # For old format, robot_hz is implicit from action spacing
        self.robot_hz = 100  # Default assumption
        self.action_interval = self.action_step_size

    def _load_new_format(self, instruction: Optional[str]):
        """Load new format (metadata.json + npz based)"""
        meta_path = self.data_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {self.data_dir}")

        with open(meta_path, 'r') as f:
            self.meta = json.load(f)

        self.robot_hz = self.meta.get("robot_hz", 100)
        self.sensor_hz = self.meta.get("sensor_hz", 650)
        self.action_interval = int(self.robot_hz / self.action_expert_hz)
        self.vlm_interval = self.action_interval * self.vlm_reuse_count

        # Instruction
        task_name = self.data_dir.parent.name.replace('_', ' ')
        self.instruction = instruction or f"Perform {task_name} insertion task."

        # Sensor data (mmap)
        self.sensor_path = self.data_dir / "sensor_data.npz"
        self.sensor_npz = None
        self._load_sensor_metadata()

        # Robot states (pose only)
        csv_path = self.data_dir / "robot_states.csv"
        use_cols = ['pose_x', 'pose_y', 'pose_z', 'pose_a', 'pose_b', 'pose_r']
        try:
            df = pd.read_csv(csv_path, usecols=use_cols)
        except Exception as e:
            print(f"⚠️ Fallback full read for {csv_path}: {e}")
            df = pd.read_csv(csv_path)
        self.poses = df[use_cols].to_numpy(dtype=np.float32)
        self.num_poses = len(self.poses)

        # Compute actions from poses (will be done in __getitem__)
        self.actions = None  # Not pre-computed for new format

        # Image paths
        img_dir = self.data_dir / "images"
        self.images = {}
        for view_name in self.meta.get("camera_views", []):
            view_dir = img_dir / view_name
            if view_dir.exists():
                files = sorted(view_dir.glob("*.jpg"), key=lambda x: float(x.stem))
                self.images[view_name] = [str(f) for f in files]

        # Sample count
        self.num_actions = (self.num_poses - self.action_interval) // self.action_interval
        self._total_samples = self.num_actions

        # For new format
        self.action_step_size = self.action_interval
        self.max_action_steps = self.num_actions

    def _load_sensor_metadata(self):
        """Load sensor metadata without keeping file handle open (new format)"""
        try:
            with np.load(self.sensor_path) as npz:
                self.sensor_timestamps = npz['timestamps'][:]
                self.sensor_windows_shape = npz['data'].shape
                self.sensor_length = self.sensor_windows_shape[0]
                self.has_sensor = True
        except FileNotFoundError:
            print(f"⚠️ Sensor file not found: {self.sensor_path}")
            self.sensor_timestamps = np.array([])
            self.sensor_windows_shape = (0, 0, 0)
            self.sensor_length = 0
            self.has_sensor = False
        except Exception as e:
            print(f"⚠️ Error loading sensor metadata {self.sensor_path}: {e}")
            self.sensor_length = 0
            self.has_sensor = False

    def _get_sensor_npz(self):
        """Lazy load sensor npz with mmap_mode (new format)"""
        if self.sensor_npz is None or not hasattr(self.sensor_npz, 'f'):
            try:
                self.sensor_npz = np.load(self.sensor_path, mmap_mode='r')
            except FileNotFoundError:
                return None
        return self.sensor_npz

    def _scan_vl_cache(self):
        """Pre-scan VL cache files (optimization)"""
        self.vl_cache_files = {}

        if self.format == 'old':
            # Old format: vlm_idx based on action steps
            for action_step in range(self.max_action_steps):
                vlm_idx = min(action_step * self.action_step_size, len(self.actions) - 1)
                if vlm_idx not in self.vl_cache_files:
                    cache_path = self.cache_root / f"{self.data_dir.name}_vlm{vlm_idx}.pt"
                    self.vl_cache_files[vlm_idx] = cache_path if cache_path.exists() else None

        else:  # new format
            # New format: vlm_idx based on vlm_interval
            num_vlm_steps = (self._total_samples + self.vlm_reuse_count - 1) // self.vlm_reuse_count
            for i in range(num_vlm_steps):
                vlm_idx = i * self.vlm_interval
                cache_path = self.cache_root / f"{self.data_dir.name}_vlm{vlm_idx}.pt"
                self.vl_cache_files[vlm_idx] = cache_path if cache_path.exists() else None

        self.cache_found_count = sum(1 for p in self.vl_cache_files.values() if p is not None)

    def __len__(self):
        return self._total_samples

    def __getstate__(self):
        """Prepare for pickling - close file handles"""
        state = self.__dict__.copy()
        if self.format == 'new':
            state['sensor_npz'] = None
        return state

    def __setstate__(self, state):
        """Restore after unpickling"""
        self.__dict__.update(state)
        if self.format == 'new':
            self.sensor_npz = None

    def __getitem__(self, idx):
        if idx >= self._total_samples:
            raise IndexError

        if self.format == 'old':
            return self._getitem_old(idx)
        else:
            return self._getitem_new(idx)

    def _getitem_old(self, idx):
        """Get item for old format"""
        action_step = idx // self.vlm_reuse_count
        reuse_step = idx % self.vlm_reuse_count
        vlm_idx = min(action_step * self.action_step_size, len(self.actions) - 1)

        # VL cache or image paths
        vl_cache, image_paths = self._load_vl_or_images(vlm_idx)

        # Sensor window
        sensor_start = action_step * self.action_step_size
        sensor_end = sensor_start + self.sensor_window_size
        sensor_window = self._get_sensor_window_old(sensor_start, sensor_end)

        # Actions
        action_start = action_step * self.action_step_size
        action_end = action_start + self.horizon
        actions = self._get_actions_old(action_start, action_end)

        cache_key = f"{self.data_dir.name}_vlm{vlm_idx}"

        return {
            "instruction": self.instruction,
            "images": image_paths,
            "vl_cache": vl_cache,
            "sensor_data": torch.from_numpy(sensor_window),
            "actions": torch.from_numpy(actions),
            "has_sensor": bool(self.has_sensor),
            "cache_key": cache_key,
            "vlm_idx": int(vlm_idx),
            "reuse_step": int(reuse_step),
            "confidence": 1.0 if self.has_sensor else 0.5,
        }

    def _getitem_new(self, idx):
        """Get item for new format"""
        reuse_step = idx % self.vlm_reuse_count
        action_step = idx
        vlm_idx = (idx // self.vlm_reuse_count) * self.vlm_interval

        # VL cache or image paths
        vl_cache, image_paths = self._load_vl_or_images(vlm_idx)

        # Sensor window
        sensor_window = self._get_sensor_window_new(idx)

        # Actions (computed from poses)
        actions = self._get_actions_new(action_step)

        cache_key = f"{self.data_dir.name}_vlm{vlm_idx}"

        return {
            "instruction": self.instruction,
            "images": image_paths,
            "vl_cache": vl_cache,
            "sensor_data": torch.from_numpy(sensor_window),
            "actions": torch.from_numpy(actions),
            "has_sensor": bool(self.has_sensor),
            "cache_key": cache_key,
            "vlm_idx": int(vlm_idx),
            "reuse_step": int(reuse_step),
            "confidence": 1.0,
        }

    def _load_vl_or_images(self, vlm_idx):
        """Load VL cache or return image paths"""
        vl_cache = None
        image_paths = []

        cache_path = self.vl_cache_files.get(vlm_idx)

        if cache_path:
            try:
                vl_cache = torch.load(cache_path, map_location="cpu")
                return vl_cache, None
            except Exception as e:
                print(f"⚠️ Failed to load VL cache {cache_path.name}: {e}")

        # Fallback to image paths
        if isinstance(self.images, dict):
            for view_name in sorted(self.images.keys()):
                view_images = self.images[view_name]
                if len(view_images) > 0:
                    img_idx = min(vlm_idx, len(view_images) - 1)
                    if self.format == 'old':
                        img_path = view_images[img_idx]
                        image_paths.append(img_path if img_path else "")
                    else:
                        image_paths.append(view_images[img_idx])

        return vl_cache, image_paths

    def _get_sensor_window_old(self, start, end):
        """Get sensor window for old format"""
        T_sensor = len(self.sensor_data) if hasattr(self.sensor_data, '__len__') else 0

        if T_sensor == 0 or start >= T_sensor:
            return np.zeros((self.sensor_window_size, 1026), dtype=np.float32)

        sw = self.sensor_data[start:min(end, T_sensor)]
        if sw.shape[0] < self.sensor_window_size:
            pad = np.zeros((self.sensor_window_size - sw.shape[0], 1026), dtype=np.float32)
            return np.concatenate([sw, pad], axis=0)
        return sw

    def _get_sensor_window_new(self, idx):
        """Get sensor window for new format"""
        if not self.has_sensor:
            return np.zeros((self.sensor_window_size, 1026), dtype=np.float32)

        s_npz = self._get_sensor_npz()
        if s_npz is None or self.sensor_length == 0:
            return np.zeros((self.sensor_window_size, 1026), dtype=np.float32)

        sensor_idx = min(idx, self.sensor_length - 1)
        sensor_window = np.array(s_npz['data'][sensor_idx], dtype=np.float32)

        # Ensure correct shape
        if sensor_window.shape[0] < self.sensor_window_size:
            pad = np.zeros((self.sensor_window_size - sensor_window.shape[0],
                          sensor_window.shape[1]), dtype=np.float32)
            sensor_window = np.concatenate([sensor_window, pad], axis=0)

        return sensor_window

    def _get_actions_old(self, start, end):
        """Get actions for old format"""
        T_action = len(self.actions)

        if start >= T_action:
            return np.zeros((self.horizon, 7), dtype=np.float32)

        act = self.actions[start:min(end, T_action)]
        if act.shape[0] < self.horizon:
            last = act[-1] if act.shape[0] > 0 else np.zeros((7,), dtype=np.float32)
            pad = np.tile(last, (self.horizon - act.shape[0], 1))
            return np.concatenate([act, pad], axis=0)
        return act

    def _get_actions_new(self, action_step):
        """Get actions for new format (computed from pose deltas)"""
        actions = []

        for i in range(self.horizon):
            start_pose_idx = (action_step + i) * self.action_interval
            end_pose_idx = start_pose_idx + self.action_interval

            if end_pose_idx >= self.num_poses:
                break

            delta_pose = self.poses[end_pose_idx] - self.poses[start_pose_idx]
            delta_action = np.concatenate([delta_pose, [1.0]], axis=0)
            actions.append(delta_action)

        # Pad to fixed horizon
        if not actions:
            return np.zeros((self.horizon, 7), dtype=np.float32)
        elif len(actions) < self.horizon:
            pad = np.tile(actions[-1], (self.horizon - len(actions), 1))
            actions = np.concatenate([actions, pad], axis=0)

        return np.array(actions, dtype=np.float32)


# =====================================
# Collate Function
# =====================================

def unified_collate_fn(batch):
    """
    통합 collate function for batching

    Returns:
        Dictionary with batched data
    """
    instructions = [b["instruction"] for b in batch]
    image_lists = [b["images"] for b in batch]
    vl_features = [b["vl_cache"] for b in batch]

    # Pad sensor data to max length
    sensor_tensors = [b["sensor_data"] for b in batch]
    max_sensor_len = max(t.shape[0] for t in sensor_tensors)
    padded_sensors = []
    for sensor in sensor_tensors:
        if sensor.shape[0] < max_sensor_len:
            pad = torch.zeros((max_sensor_len - sensor.shape[0], sensor.shape[1]),
                            dtype=sensor.dtype)
            padded_sensors.append(torch.cat([sensor, pad], dim=0))
        else:
            padded_sensors.append(sensor)
    sensor_data = torch.stack(padded_sensors, dim=0)

    actions = torch.stack([b["actions"] for b in batch], dim=0)
    has_sensor_mask = torch.tensor([b["has_sensor"] for b in batch], dtype=torch.bool)
    cache_keys = [b["cache_key"] for b in batch]
    vlm_indices = [b["vlm_idx"] for b in batch]
    reuse_steps = [b["reuse_step"] for b in batch]
    confidence = [b["confidence"] for b in batch]

    return {
        "instruction": instructions,
        "images": image_lists,
        "vl_cache": vl_features,
        "sensor_data": sensor_data,
        "actions": actions,
        "has_sensor_mask": has_sensor_mask,
        "cache_keys": cache_keys,
        "vlm_indices": vlm_indices,
        "reuse_steps": reuse_steps,
        "confidence": confidence,
    }


# =====================================
# Unified Dataloader Builder
# =====================================

def create_unified_dataloader(
    old_dataset_patterns: List[str] = None,
    new_dataset_path: Optional[str] = None,
    old_weight: float = 1.0,
    new_weight: float = 3.0,
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = True,
    horizon: int = 8,
    vlm_reuse_count: int = 3,
    sensor_window_size: int = 65,
    action_expert_hz: int = 10,
    cache_root: str = "/home/najo/NAS/VLA/dataset/cache/qwen_vl_features",
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
):
    """
    통합 데이터로더 생성

    Args:
        old_dataset_patterns: Old format dataset patterns (glob)
        new_dataset_path: New format dataset root path
        old_weight: Weight for old datasets
        new_weight: Weight for new datasets
        batch_size: Batch size
        num_workers: Number of workers
        shuffle: Whether to shuffle
        horizon: Action prediction horizon
        vlm_reuse_count: VL feature reuse count
        sensor_window_size: Sensor window size
        action_expert_hz: Action expert frequency
        cache_root: VL cache root
        distributed: Whether to use distributed sampling
        rank: Process rank (for distributed)
        world_size: Total processes (for distributed)

    Returns:
        DataLoader
    """
    datasets = []
    dataset_weights = []

    # Load old format datasets
    if old_dataset_patterns:
        print("📦 Loading old format datasets...")
        for pattern in old_dataset_patterns:
            expanded_paths = glob.glob(pattern)
            for traj_dir in expanded_paths:
                try:
                    ds = UnifiedVLADataset(
                        data_dir=traj_dir,
                        format='old',
                        horizon=horizon,
                        vlm_reuse_count=vlm_reuse_count,
                        sensor_window_size=sensor_window_size,
                        action_expert_hz=action_expert_hz,
                        cache_root=cache_root,
                    )
                    datasets.append(ds)
                    dataset_weights.extend([old_weight] * len(ds))
                except Exception as e:
                    print(f"⚠️ Failed to load old dataset {traj_dir}: {e}")

    # Load new format datasets
    if new_dataset_path:
        print("\n📦 Loading new format datasets...")
        new_path = Path(new_dataset_path)
        if new_path.exists():
            for task_dir in new_path.iterdir():
                if not task_dir.is_dir():
                    continue
                task_name = task_dir.name.replace('_', ' ')
                instruction = f"Perform {task_name} insertion task"

                for episode_dir in task_dir.iterdir():
                    if not episode_dir.is_dir() or not episode_dir.name.startswith('episode_'):
                        continue
                    try:
                        ds = UnifiedVLADataset(
                            data_dir=str(episode_dir),
                            format='new',
                            horizon=horizon,
                            vlm_reuse_count=vlm_reuse_count,
                            sensor_window_size=sensor_window_size,
                            action_expert_hz=action_expert_hz,
                            instruction=instruction,
                            cache_root=cache_root,
                        )
                        datasets.append(ds)
                        dataset_weights.extend([new_weight] * len(ds))
                    except Exception as e:
                        print(f"⚠️ Failed to load new dataset {episode_dir}: {e}")
        else:
            print(f"⚠️ New dataset path not found: {new_dataset_path}")

    if not datasets:
        raise ValueError("No datasets loaded!")

    full_dataset = ConcatDataset(datasets)

    print(f"\n📊 Total dataset statistics:")
    print(f"   Total samples: {len(full_dataset)}")
    print(f"   Old dataset samples: {sum(1 for w in dataset_weights if w == old_weight)}")
    print(f"   New dataset samples: {sum(1 for w in dataset_weights if w == new_weight)}")
    print(f"   Sampling ratio (new:old): {new_weight}:{old_weight}")

    # Create sampler
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            full_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle
        )
        shuffle = False
    elif shuffle and old_weight != new_weight:
        sampler = WeightedRandomSampler(
            weights=dataset_weights,
            num_samples=len(dataset_weights),
            replacement=True,
        )
        shuffle = False

    # Create dataloader
    dataloader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=unified_collate_fn,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
        pin_memory=True,
    )

    return dataloader


# Backward compatibility aliases
AsyncInsertionMeca500DatasetWithSensor = lambda **kwargs: UnifiedVLADataset(format='old', **kwargs)
NewAsyncInsertionDataset = lambda **kwargs: UnifiedVLADataset(format='new', **kwargs)
async_collate_fn_with_sensor = unified_collate_fn
create_weighted_async_dataloader = create_unified_dataloader


# =====================================
# Test Code
# =====================================

if __name__ == "__main__":
    print("🧪 Testing Unified VLA Dataset...")

    # Test old format
    print("\n=== Testing Old Format ===")
    old_test_dir = "/home/najo/NAS/VLA/dataset/White_silicone_white_circle/recv_all_20251027_170308"
    if Path(old_test_dir).exists():
        try:
            ds_old = UnifiedVLADataset(
                data_dir=old_test_dir,
                format='old',
                horizon=8,
                vlm_reuse_count=3,
                sensor_window_size=65,
            )
            print(f"✅ Old dataset loaded: {len(ds_old)} samples")

            if len(ds_old) > 0:
                sample = ds_old[0]
                print(f"   Instruction: {sample['instruction']}")
                print(f"   Sensor shape: {sample['sensor_data'].shape}")
                print(f"   Actions shape: {sample['actions'].shape}")
                print(f"   Has sensor: {sample['has_sensor']}")
        except Exception as e:
            print(f"⚠️ Old format test failed: {e}")
    else:
        print(f"⚠️ Old test directory not found: {old_test_dir}")

    # Test new format
    print("\n=== Testing New Format ===")
    new_test_dir = "/home/najo/NAS/VLA/Insertion_VLA/Make_dataset/New_dataset/Blue_point/episode_20251030_025119"
    if Path(new_test_dir).exists():
        try:
            ds_new = UnifiedVLADataset(
                data_dir=new_test_dir,
                format='new',
                horizon=8,
                vlm_reuse_count=3,
                sensor_window_size=650,
            )
            print(f"✅ New dataset loaded: {len(ds_new)} samples")

            if len(ds_new) > 0:
                sample = ds_new[0]
                print(f"   Instruction: {sample['instruction']}")
                print(f"   Sensor shape: {sample['sensor_data'].shape}")
                print(f"   Actions shape: {sample['actions'].shape}")
                print(f"   Has sensor: {sample['has_sensor']}")
        except Exception as e:
            print(f"⚠️ New format test failed: {e}")
    else:
        print(f"⚠️ New test directory not found: {new_test_dir}")

    print("\n✅ All tests completed!")
