"""
Async Integrated Dataset (Low-Memory Optimized + VL Cache Integration)
- Supports loading precomputed Qwen-VL features from cache
- Falls back to raw image paths if cache not found
"""

import os
import glob
import gc
import pickle
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Utilities
# -----------------------------

def _safe_joblib_load(pkl_path):
    """Try joblib (mmap), fallback to pickle."""
    try:
        import joblib
        return joblib.load(pkl_path, mmap_mode='r')
    except Exception:
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)

def _read_meta_len_actions_and_has_sensor(traj_dir: Path):
    data_file = traj_dir / "data.pkl"
    data = _safe_joblib_load(data_file)
    actions = data.get("action")
    if actions is None:
        raise ValueError(f"No 'action' in {data_file}")
    T = len(actions)
    has_sensor = ("sensor_data" in data) and (data["sensor_data"] is not None)
    return T, has_sensor

def _compute_total_samples(T, horizon, vlm_reuse_count):
    if T <= 0 or horizon <= 0:
        return 0
    max_action_steps = (T - horizon) // horizon
    if max_action_steps < 0:
        max_action_steps = 0
    return max_action_steps * vlm_reuse_count

def _load_sensor_compound(traj_dir: Path, T_actions: int):
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

# -----------------------------
# Core Dataset
# -----------------------------
# -----------------------------
# Core Dataset (Optimized)
# -----------------------------

class AsyncInsertionMeca500DatasetWithSensor(Dataset):
    """
    Memory-optimized async dataset with optional Qwen-VL cache loading.
    ✅ [Optimized] Pre-scans VL cache in __init__ to reduce I/O in __getitem__.
    """

    def __init__(
        self,
        trajectory_dir,
        horizon=8,
        vlm_reuse_count=3,
        sensor_window_size=65,
        action_expert_hz=10,
        prefer_npy_sensor=True,
        cache_root="/home/najo/NAS/VLA/dataset/cache/qwen_vl_features",
    ):
        self.trajectory_dir = Path(trajectory_dir)
        self.cache_root = Path(cache_root)
        self.horizon = int(horizon)
        self.vlm_reuse_count = int(vlm_reuse_count)
        self.sensor_window_size = int(sensor_window_size)
        self.action_expert_hz = int(action_expert_hz)
        self.sensor_hz = 650

        data_file = self.trajectory_dir / "data.pkl"
        if not data_file.exists():
            raise FileNotFoundError(f"data.pkl not found in {self.trajectory_dir}")

        data = _safe_joblib_load(data_file)
        actions = data.get("action")
        if actions is None:
            raise ValueError(f"'action' missing in {data_file}")
        self.actions = np.asarray(actions, dtype=np.float32)
        T = len(self.actions)

        self.images = data.get("image", {}) or {}
        self.instruction = data.get("instruction", "Perform needle insertion task.")
        self.has_sensor = ("sensor_data" in data) and (data["sensor_data"] is not None)

        if self.has_sensor:
            if prefer_npy_sensor and (self.trajectory_dir / "sensor_data.npy").exists():
                self.sensor_data = np.load(self.trajectory_dir / "sensor_data.npy", mmap_mode='r')
                if not (self.sensor_data.ndim == 2 and self.sensor_data.shape == (T, 1026)):
                    self.sensor_data = _load_sensor_compound(self.trajectory_dir, T)
            else:
                self.sensor_data = _load_sensor_compound(self.trajectory_dir, T)
        else:
            self.sensor_data = np.zeros((T, 1026), dtype=np.float32)

        del data
        gc.collect()

        self.action_step_size = self.horizon
        self.max_action_steps = (T - self.horizon) // self.horizon
        if self.max_action_steps < 0:
            self.max_action_steps = 0
        self._total_samples = self.max_action_steps * self.vlm_reuse_count
        
        # ⬇️ [OPTIMIZATION] Pre-scan all VL cache files once
        self.vl_cache_files = {} # Map vlm_idx -> cache_path or None
        
        for action_step in range(self.max_action_steps):
            vlm_idx = min(action_step * self.action_step_size, T - 1)
            
            # Avoid re-checking if index already scanned
            if vlm_idx not in self.vl_cache_files:
                cache_path = self.cache_root / f"{self.trajectory_dir.name}_vlm{vlm_idx}.pt"
                if cache_path.exists():
                    self.vl_cache_files[vlm_idx] = cache_path
                else:
                    self.vl_cache_files[vlm_idx] = None
        
        cache_found_count = sum(1 for p in self.vl_cache_files.values() if p is not None)
        # ⬆️ [OPTIMIZATION] End

        print(f"📦 AsyncDataset(optimized+cache): {self.trajectory_dir.name}")
        print(f"   T(actions): {T}, total samples: {self._total_samples}")
        print(f"   has_sensor: {self.has_sensor}, (Found {cache_found_count}/{len(self.vl_cache_files)} VL caches)")
        # (cache_root M-logs removed for brevity)

    def __len__(self):
        return self._total_samples

    def _resolve_indices(self, idx):
        if not (0 <= idx < self._total_samples):
            raise IndexError(idx)
        action_step = idx // self.vlm_reuse_count
        reuse_step = idx % self.vlm_reuse_count
        vlm_idx = min(action_step * self.action_step_size, len(self.actions) - 1)
        action_start = action_step * self.action_step_size
        action_end = action_start + self.horizon
        sensor_start = action_start
        sensor_end = sensor_start + self.sensor_window_size
        return action_step, reuse_step, vlm_idx, sensor_start, sensor_end, action_start, action_end

    # ----------------------
    # [REMOVED] _load_vl_cache_feature (merged into __getitem__)
    # ----------------------

    def __getitem__(self, idx):
        _, reuse_step, vlm_idx, s0, s1, a0, a1 = self._resolve_indices(idx)

        # ⬇️ [OPTIMIZATION] Load VL cache from pre-scanned dictionary
        vl_cache = None
        image_paths = []
        cache_path = self.vl_cache_files.get(vlm_idx) # Fast dict lookup

        if cache_path: # If cache_path is not None
            try:
                vl_cache = torch.load(cache_path, map_location="cpu")
                image_paths = None  # cache loaded → no image paths needed
            except Exception as e:
                print(f"⚠️ Failed to load VL cache {cache_path.name}: {e}")
                cache_path = None # Force fallback to images
        
        if not cache_path: # Fallback if cache_path is None or loading failed
            vl_cache = None
            if isinstance(self.images, dict):
                for view_name in sorted(self.images.keys()):
                    view_images = self.images[view_name]
                    if vlm_idx < len(view_images):
                        img_path = view_images[vlm_idx]
                        image_paths.append(img_path if img_path else "")
                    else:
                        image_paths.append(view_images[-1] if len(view_images) > 0 else "")
            else:
                image_paths = []
        # ⬆️ [OPTIMIZATION] End


        # --- Sensor window ---
        T_sensor = getattr(self.sensor_data, "shape", [0])[0] if hasattr(self.sensor_data, "shape") else 0
        if T_sensor == 0 or s0 >= T_sensor:
            sensor_window = np.zeros((self.sensor_window_size, 1026), dtype=np.float32)
        else:
            sw = self.sensor_data[s0:min(s1, T_sensor)]
            if sw.shape[0] < self.sensor_window_size:
                pad = np.zeros((self.sensor_window_size - sw.shape[0], 1026), dtype=np.float32)
                sensor_window = np.concatenate([sw, pad], axis=0)
            else:
                sensor_window = sw

        # --- Actions ---
        T_action = len(self.actions)
        if a0 >= T_action:
            actions = np.zeros((self.horizon, 7), dtype=np.float32)
        else:
            act = self.actions[a0:min(a1, T_action)]
            if act.shape[0] < self.horizon:
                last = act[-1] if act.shape[0] > 0 else np.zeros((7,), dtype=np.float32)
                pad = np.tile(last, (self.horizon - act.shape[0], 1))
                actions = np.concatenate([act, pad], axis=0)
            else:
                actions = act

        cache_key_str = f"{self.trajectory_dir.name}_vlm{vlm_idx}"

        return {
            "instruction": self.instruction,
            "images": image_paths,
            "vl_cache": vl_cache,
            "sensor_data": torch.from_numpy(sensor_window),
            "actions": torch.from_numpy(actions),
            "has_sensor": bool(self.has_sensor),
            # "cache_key": f"{self.trajectory_dir.name}_vlm{vlm_idx}_idx{idx}", # 
            "cache_key": cache_key_str, # 
            "vlm_idx": int(vlm_idx),
            "reuse_step": int(reuse_step),
            "confidence": 1.0 if self.has_sensor else 0.5,
        }
# -----------------------------
# Collate
# -----------------------------

def async_collate_fn_with_sensor(batch):
    instructions = [b["instruction"] for b in batch]
    image_lists = [b["images"] for b in batch]
    vl_features = [b["vl_cache"] for b in batch]

    sensor_tensors = [b["sensor_data"] for b in batch]
    max_sensor_len = max(t.shape[0] for t in sensor_tensors)
    padded_sensors = []
    for sensor in sensor_tensors:
        if sensor.shape[0] < max_sensor_len:
            pad = torch.zeros((max_sensor_len - sensor.shape[0], sensor.shape[1]), dtype=sensor.dtype)
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
        "vl_cache": vl_features,  # now passed to model forward()
        "sensor_data": sensor_data,
        "actions": actions,
        "has_sensor_mask": has_sensor_mask,
        "cache_keys": cache_keys,
        "vlm_indices": vlm_indices,
        "reuse_steps": reuse_steps,
        "confidence": confidence,
    }

# -----------------------------
# LazyConcatDataset + loader (unchanged)
# -----------------------------

class LazyConcatDataset(Dataset):
    def __init__(
        self,
        traj_dirs,
        horizon=8,
        vlm_reuse_count=3,
        sensor_window_size=65,
        action_expert_hz=10,
        prefer_npy_sensor=True,
        max_cache=2,
        cache_root="/home/najo/NAS/VLA/dataset/cache/qwen_vl_features",
    ):
        self.traj_dirs = [Path(p) for p in traj_dirs]
        self.horizon = int(horizon)
        self.vlm_reuse_count = int(vlm_reuse_count)
        self.sensor_window_size = int(sensor_window_size)
        self.action_expert_hz = int(action_expert_hz)
        self.prefer_npy_sensor = bool(prefer_npy_sensor)
        self.cache_root = Path(cache_root)

        self._lengths = []
        for d in self.traj_dirs:
            T, _ = _read_meta_len_actions_and_has_sensor(d)
            self._lengths.append(_compute_total_samples(T, self.horizon, self.vlm_reuse_count))

        self._cumsums = np.cumsum([0] + self._lengths)
        self._total_len = int(self._cumsums[-1])
        self._cache = OrderedDict()
        self._max_cache = max(1, int(max_cache))

    def __len__(self):
        return self._total_len

    def _find_ds(self, idx):
        ds_idx = int(np.searchsorted(self._cumsums, idx, side='right') - 1)
        local_idx = idx - int(self._cumsums[ds_idx])
        return ds_idx, local_idx

    def _get_dataset_from_cache(self, ds_idx):
        key = ds_idx
        if key in self._cache:
            ds = self._cache.pop(key)
            self._cache[key] = ds
            return ds
        traj_dir = self.traj_dirs[ds_idx]
        ds = AsyncInsertionMeca500DatasetWithSensor(
            trajectory_dir=traj_dir,
            horizon=self.horizon,
            vlm_reuse_count=self.vlm_reuse_count,
            sensor_window_size=self.sensor_window_size,
            action_expert_hz=self.action_expert_hz,
            prefer_npy_sensor=self.prefer_npy_sensor,
            cache_root=self.cache_root,
        )
        self._cache[key] = ds
        if len(self._cache) > self._max_cache:
            old_key, old_ds = self._cache.popitem(last=False)
            del old_ds
            gc.collect()
        return ds

    def __getitem__(self, idx):
        ds_idx, local_idx = self._find_ds(idx)
        ds = self._get_dataset_from_cache(ds_idx)
        return ds[local_idx]
    
def create_async_integrated_dataloader(
    dataset_patterns,
    batch_size=1,
    num_workers=0,
    shuffle=True,
    horizon=8,
    vlm_reuse_count=3,
    sensor_window_size=65,
    action_expert_hz=10,
    prefer_npy_sensor=True,
    max_cache=2,
    cache_root="/home/najo/NAS/VLA/dataset/cache/qwen_vl_features",
    pin_memory=False, # 
):
    if isinstance(dataset_patterns, (str, Path)):
        dataset_patterns = [str(dataset_patterns)]
    traj_dirs = []
    for pat in dataset_patterns:
        traj_dirs.extend(sorted(glob.glob(pat)))
    if not traj_dirs:
        raise ValueError("No trajectories matched the given patterns.")
    
    dataset = LazyConcatDataset(
        traj_dirs=traj_dirs,
        horizon=horizon,
        vlm_reuse_count=vlm_reuse_count,
        sensor_window_size=sensor_window_size,
        action_expert_hz=action_expert_hz,
        prefer_npy_sensor=prefer_npy_sensor,
        max_cache=max_cache,
        cache_root=cache_root,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=async_collate_fn_with_sensor,
        # ⬇️ [OPTIMIZATION]
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=(num_workers > 0), # Keep workers alive
        pin_memory=True,                      # Speed up CPU->GPU transfer
        # ⬆️ [OPTIMIZATION]
    )
    return loader


# -----------------------------
# Quick test
# -----------------------------

if __name__ == "__main__":
    print("🧪 Testing Low-Memory AsyncIntegratedDataset...")

    test_dir = "/home/najo/NAS/VLA/dataset/White_silicone_white_circle/recv_all_20251027_170308"
    if Path(test_dir).exists():
        # Single-trajectory test
        ds = AsyncInsertionMeca500DatasetWithSensor(
            trajectory_dir=test_dir,
            horizon=8,
            vlm_reuse_count=3,
            sensor_window_size=65,
            action_expert_hz=10,
            prefer_npy_sensor=True,
        )
        print(f"✅ Single dataset len: {len(ds)}")
        if len(ds) > 0:
            s = ds[0]
            print("📦 First sample:")
            print("   instr:", s["instruction"])
            print("   #views:", len(s["images"]))
            print("   sensor:", tuple(s["sensor_data"].shape))
            print("   actions:", tuple(s["actions"].shape))
            print("   has_sensor:", s["has_sensor"])
            print("   vlm_idx:", s["vlm_idx"], "reuse_step:", s["reuse_step"])

        # Dataloader test
        dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=async_collate_fn_with_sensor)
        batch = next(iter(dl))
        print("📦 First batch:")
        print("   sensor:", tuple(batch["sensor_data"].shape))
        print("   actions:", tuple(batch["actions"].shape))
        print("   vlm_indices:", batch["vlm_indices"])
        print("   reuse_steps:", batch["reuse_steps"])

        # LazyConcat test (same dir twice for demo)
        loader = create_async_integrated_dataloader(
            [test_dir, test_dir],
            batch_size=2,
            num_workers=0,
            shuffle=True,
            horizon=8,
            vlm_reuse_count=3,
            sensor_window_size=65,
            action_expert_hz=10,
            prefer_npy_sensor=True,
            max_cache=2,
            pin_memory=False,
        )
        b = next(iter(loader))
        print("📦 LazyConcat batch:")
        print("   sensor:", tuple(b["sensor_data"].shape))
        print("   actions:", tuple(b["actions"].shape))
    else:
        print(f"⚠️ Test directory not found: {test_dir}")
