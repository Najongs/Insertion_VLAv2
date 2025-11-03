import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import gc

# ============================================================
# NewAsyncInsertionDataset with VL Cache (Optimized)
# ============================================================

class NewAsyncInsertionDataset(Dataset):
    """
    RAM-optimized dataset for new async data format.
    ✅ [Optimized] Pre-scans VL cache in __init__ to reduce I/O in __getitem__.
    """

    def __init__(
        self,
        episode_dir,
        horizon=8,
        vlm_reuse_count=3,
        action_expert_hz=10,
        instruction=None,
        cache_root="/home/najo/NAS/VLA/dataset/cache/qwen_vl_features",
    ):
        self.episode_dir = Path(episode_dir)
        self.cache_root = Path(cache_root)
        self.horizon = int(horizon)
        self.vlm_reuse_count = int(vlm_reuse_count)
        self.action_expert_hz = int(action_expert_hz)

        # --- Load metadata (lightweight JSON)
        meta_path = self.episode_dir / "metadata.json"
        with open(meta_path, 'r') as f:
            self.meta = json.load(f)

        self.robot_hz = self.meta.get("robot_hz", 100)
        self.sensor_hz = self.meta.get("sensor_hz", 650)
        self.sensor_window_size = self.meta.get("sensor_window_size", 650)

        self.action_interval = int(self.robot_hz / self.action_expert_hz)
        self.vlm_interval = self.action_interval * self.vlm_reuse_count

        # --- Instruction
        task_name = self.episode_dir.parent.name.replace('_', ' ')
        self.instruction = instruction or f"Perform {task_name} insertion task."

        # --- Sensor data (mmap)
        self.sensor_path = self.episode_dir / "sensor_data.npz"
        self.sensor_npz = None
        self._load_sensor_metadata()

        # --- Robot states (pose only)
        csv_path = self.episode_dir / "robot_states.csv"
        use_cols = ['pose_x', 'pose_y', 'pose_z', 'pose_a', 'pose_b', 'pose_r']
        try:
            df = pd.read_csv(csv_path, usecols=use_cols)
        except Exception as e:
            print(f"⚠️ Fallback full read for {csv_path}: {e}")
            df = pd.read_csv(csv_path)
        self.poses = df[use_cols].to_numpy(dtype=np.float32)
        self.num_poses = len(self.poses)

        # --- Image paths (lazy list of str)
        img_dir = self.episode_dir / "images"
        self.image_views = {}
        for view_name in self.meta.get("camera_views", []):
            view_dir = img_dir / view_name
            if view_dir.exists():
                files = sorted(view_dir.glob("*.jpg"), key=lambda x: float(x.stem))
                self.image_views[view_name] = [str(f) for f in files]

        # --- Sample count
        self.num_actions = (self.num_poses - self.action_interval) // self.action_interval
        self.total_samples = self.num_actions

        # ⬇️ [OPTIMIZATION] Pre-scan all VL cache files once
        self.vl_cache_files = {} # Map vlm_idx -> cache_path or None
        num_vlm_steps = (self.total_samples + self.vlm_reuse_count - 1) // self.vlm_reuse_count
        
        for i in range(num_vlm_steps):
            vlm_idx = i * self.vlm_interval
            cache_path = self.cache_root / f"{self.episode_dir.name}_vlm{vlm_idx}.pt"
            if cache_path.exists():
                self.vl_cache_files[vlm_idx] = cache_path
            else:
                self.vl_cache_files[vlm_idx] = None
        
        cache_found_count = sum(1 for p in self.vl_cache_files.values() if p is not None)
        print(f"📦 Loaded {self.episode_dir.name}: poses={self.num_poses}, sensor={self.sensor_length}, samples={self.total_samples} (Found {cache_found_count}/{num_vlm_steps} VL caches)")
        # ⬆️ [OPTIMIZATION] End

    def _load_sensor_metadata(self):
        """Load sensor metadata without keeping file handle open"""
        try:
            with np.load(self.sensor_path) as npz:
                self.sensor_timestamps = npz['timestamps'][:]
                self.sensor_windows_shape = npz['data'].shape
                self.sensor_length = self.sensor_windows_shape[0]
        except FileNotFoundError:
            print(f"⚠️ Sensor file not found: {self.sensor_path}, setting length to 0")
            self.sensor_timestamps = np.array([])
            self.sensor_windows_shape = (0, 0, 0) # (len, window, features)
            self.sensor_length = 0
        except Exception as e:
            print(f"⚠️ Error loading sensor metadata {self.sensor_path}: {e}")
            self.sensor_length = 0


    def _get_sensor_npz(self):
        """Lazy load sensor npz with mmap_mode"""
        if self.sensor_npz is None or not hasattr(self.sensor_npz, 'f'):
            # Re-open mmap handle if it was closed (e.g., after pickling)
            try:
                self.sensor_npz = np.load(self.sensor_path, mmap_mode='r')
            except FileNotFoundError:
                # Handle case where file doesn't exist (e.g., _load_sensor_metadata warned)
                return None
        return self.sensor_npz

    def __getstate__(self):
        """Prepare object for pickling - close file handles"""
        state = self.__dict__.copy()
        state['sensor_npz'] = None  # Close mmap handle
        return state

    def __setstate__(self, state):
        """Restore object after unpickling"""
        self.__dict__.update(state)
        self.sensor_npz = None # Will be re-opened by _get_sensor_npz on first access

    # ----------------------
    # [REMOVED] _load_vl_cache_feature (merged into __getitem__)
    # ----------------------

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if idx >= self.total_samples:
            raise IndexError

        # --- Index mapping ---
        reuse_step = idx % self.vlm_reuse_count
        action_step = idx
        vlm_idx = (idx // self.vlm_reuse_count) * self.vlm_interval

        # ⬇️ [OPTIMIZATION] Load VL cache from pre-scanned dictionary
        vl_cache = None
        image_paths = []
        cache_path = self.vl_cache_files.get(vlm_idx) # Fast dict lookup

        if cache_path:
            try:
                vl_cache = torch.load(cache_path, map_location="cpu")
                image_paths = None  # cache loaded → no image paths needed
            except Exception as e:
                print(f"⚠️ Failed to load VL cache {cache_path.name}: {e}")
                cache_path = None # Force fallback to images
        
        if not cache_path: # Fallback if cache_path is None or loading failed
            vl_cache = None
            for view_name in sorted(self.image_views.keys()):
                imgs = self.image_views[view_name]
                if len(imgs) > 0:
                    img_idx = min(vlm_idx, len(imgs) - 1)
                    image_paths.append(imgs[img_idx])
        # ⬆️ [OPTIMIZATION] End


        # --- Sensor window ---
        has_sensor = False
        sensor_window = np.zeros((self.sensor_window_size, self.sensor_windows_shape[-1] if self.sensor_length > 0 else 1026), dtype=np.float32) # Default empty
        
        s_npz = self._get_sensor_npz()
        if s_npz is not None and self.sensor_length > 0:
            sensor_idx = min(idx, self.sensor_length - 1)
            sensor_window = np.array(s_npz['data'][sensor_idx], dtype=np.float32)
            has_sensor = True
        
        # --- Action chunk (computed from pose deltas) ---
        actions = []
        for i in range(self.horizon):
            start_pose_idx = (action_step + i) * self.action_interval
            end_pose_idx = start_pose_idx + self.action_interval
            if end_pose_idx >= self.num_poses:
                break
            delta_pose = self.poses[end_pose_idx] - self.poses[start_pose_idx]
            delta_action = np.concatenate([delta_pose, [1.0]], axis=0) # [dx, dy, dz, da, db, dr, 1.0]
            actions.append(delta_action)

        # Pad to fixed horizon
        if not actions:
            # If no actions could be computed (e.g., end of trajectory)
            actions = np.zeros((self.horizon, 7), dtype=np.float32)
        elif len(actions) < self.horizon:
            pad = np.tile(actions[-1], (self.horizon - len(actions), 1))
            actions = np.concatenate([actions, pad], axis=0)

        cache_key_str = f"{self.episode_dir.name}_vlm{vlm_idx}"
        
        return {
            "instruction": self.instruction,
            "images": image_paths,
            "vl_cache": vl_cache,
            "sensor_data": torch.from_numpy(sensor_window),
            "actions": torch.from_numpy(np.array(actions, dtype=np.float32)),
            "has_sensor": has_sensor,
            # "cache_key": f"{self.episode_dir.name}_idx{idx}", # 
            "cache_key": cache_key_str, # 
            "vlm_idx": vlm_idx,
            "reuse_step": reuse_step,
            "confidence": 1.0,
        }
        
from torch.utils.data import WeightedRandomSampler, ConcatDataset
from vla_datasets.AsyncIntegratedDataset import async_collate_fn_with_sensor

def create_weighted_async_dataloader(
    old_dataset_patterns,
    new_dataset_path,
    old_dataset_weight=1.0,
    new_dataset_weight=3.0,
    batch_size=4,
    num_workers=4,
    shuffle=True,
    horizon=8,
    vlm_reuse_count=3,
    action_expert_hz=10,
    cache_root="/home/najo/NAS/VLA/dataset/cache/qwen_vl_features",
):
    import glob
    from vla_datasets.AsyncIntegratedDataset import AsyncInsertionMeca500DatasetWithSensor

    datasets = []
    dataset_weights = []

    # --- Load old datasets
    print("📦 Loading old datasets...")
    for pattern in old_dataset_patterns:
        expanded_paths = glob.glob(pattern)
        for traj_dir in expanded_paths:
            try:
                ds = AsyncInsertionMeca500DatasetWithSensor(
                    trajectory_dir=traj_dir,
                    horizon=horizon,
                    vlm_reuse_count=vlm_reuse_count,
                    sensor_window_size=65,
                )
                datasets.append(ds)
                dataset_weights.extend([old_dataset_weight] * len(ds))
            except Exception as e:
                print(f"⚠️ Failed to load old dataset {traj_dir}: {e}")

    # --- Load new datasets
    print("\n📦 Loading new datasets...")
    new_dataset_path = Path(new_dataset_path)
    if new_dataset_path.exists():
        for task_dir in new_dataset_path.iterdir():
            if not task_dir.is_dir():
                continue
            task_name = task_dir.name.replace('_', ' ')
            instruction = f"Perform {task_name} insertion task"
            for episode_dir in task_dir.iterdir():
                if not episode_dir.is_dir() or not episode_dir.name.startswith('episode_'):
                    continue
                try:
                    ds = NewAsyncInsertionDataset(
                        episode_dir=episode_dir,
                        horizon=horizon,
                        vlm_reuse_count=vlm_reuse_count,
                        action_expert_hz=action_expert_hz,
                        instruction=instruction,
                        cache_root=cache_root,
                    )
                    datasets.append(ds)
                    dataset_weights.extend([new_dataset_weight] * len(ds))
                except Exception as e:
                    print(f"⚠️ Failed to load new dataset {episode_dir}: {e}")
    else:
        print(f"⚠️ New dataset path not found: {new_dataset_path}")

    if not datasets:
        raise ValueError("No datasets loaded!")

    full_dataset = ConcatDataset(datasets)

    print(f"\n📊 Total dataset statistics:")
    print(f"   Total samples: {len(full_dataset)}")
    print(f"   Old dataset samples: {sum(1 for w in dataset_weights if w == old_dataset_weight)}")
    print(f"   New dataset samples: {sum(1 for w in dataset_weights if w == new_dataset_weight)}")
    print(f"   Sampling ratio (new:old): {new_dataset_weight}:{old_dataset_weight}")

    sampler = None
    if shuffle and new_dataset_weight != old_dataset_weight:
        sampler = WeightedRandomSampler(
            weights=dataset_weights,
            num_samples=len(dataset_weights),
            replacement=True,
        )
        shuffle = False

    dataloader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=async_collate_fn_with_sensor,
        prefetch_factor=4 if num_workers > 0 else None,
        # ⬇️ [OPTIMIZATION]
        persistent_workers=(num_workers > 0), # Keep workers alive
        pin_memory=True,                      # Speed up CPU->GPU transfer
        # ⬆️ [OPTIMIZATION]
    )

    return dataloader


if __name__ == "__main__":
    # Test new async dataset
    print("🧪 Testing NewAsyncInsertionDataset...")

    test_dir = "Make_dataset/New_dataset/Blue_point/episode_20251030_025119"

    if Path(test_dir).exists():
        dataset = NewAsyncInsertionDataset(
            episode_dir=test_dir,
            horizon=8,
            vlm_reuse_count=3,
            action_expert_hz=10,
        )

        print(f"\n✅ Dataset loaded successfully!")
        print(f"   Total samples: {len(dataset)}")

        # Test first few samples
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            print(f"\n📦 Sample {i}:")
            print(f"   Instruction: {sample['instruction']}")
            print(f"   Images: {len(sample['images'])} views")
            print(f"   Sensor shape: {sample['sensor_data'].shape}")
            print(f"   Actions shape: {sample['actions'].shape}")
            print(f"   Has sensor: {sample['has_sensor']}")
            print(f"   VLM idx: {sample['vlm_idx']}")
            print(f"   Reuse step: {sample['reuse_step']}")

            # Check data values
            print(f"   Action stats: mean={sample['actions'].mean():.4f}, std={sample['actions'].std():.4f}")
    else:
        print(f"⚠️ Test directory not found: {test_dir}")
