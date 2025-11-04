# Real-time Inference Fix Summary

## Problem

The benchmark was not loading actual images for VL encoding, resulting in:
- VL encoding taking only **36ms** (text-only) instead of **1400ms+** (with 5 images)
- Dataset returning `images=None` because VL cache was available

## Root Cause

1. **Dataset behavior**: `UnifiedVLADataset._load_vl_or_images()` returns `(vl_cache, None)` when cache exists
2. **Image storage structure**: Images are stored in `data_dir/images/{camera_view}/*.jpg`, not as direct `frame_*.png` files
3. **Benchmark reconstruction**: The benchmark was trying to reconstruct paths as `frame_{idx:06d}.png`, which doesn't exist

## Solution

Modified `benchmark_realtime_inference.py` to load actual image paths from the dataset's `images` dict:

### Before (Incorrect)
```python
# Tried to construct paths as frame_{idx:06d}.png
img_path = data_dir / f"frame_{frame_idx:06d}.png"
```

### After (Correct)
```python
# Load from dataset.images dict (same as _load_vl_or_images)
if hasattr(dataset, 'images') and isinstance(dataset.images, dict):
    for view_name in sorted(dataset.images.keys()):
        view_images = dataset.images[view_name]
        if len(view_images) > 0:
            img_idx = min(vlm_idx, len(view_images) - 1)
            image_paths.append(view_images[img_idx])
```

## Results

### After All Fixes

**Regression Model:**
- ✅ VL encoding: **1452.82 ms** (99.3% of total time)
- ✅ Sensor encoding: **6.63 ms** (0.5% of total time)
- ✅ Action prediction: **4.34 ms** (0.3% of total time)
- ✅ **Total: 1463.78 ms** (0.68 FPS)

**Flow Matching Model:**
- ✅ VL encoding: **1459.15 ms** (97.3% of total time)
- ✅ Sensor encoding: **8.26 ms** (0.6% of total time)
- ✅ Action prediction: **32.29 ms** (2.2% of total time)
- ✅ **Total: 1499.70 ms** (0.67 FPS)

**Key Insights:**
- Flow Matching action prediction is **7.4x slower** than Regression (32.29ms vs 4.34ms)
- VL encoding dominates inference time (**97-99%** of total)
- Both sensor and action prediction are fast enough for real-time control
- For 10Hz control (100ms/cycle), async VL encoding with VLM reuse is essential

## Key Changes in `benchmark_realtime_inference.py`

### 1. Image Loading Fix (Lines 108-133)

Modified `prepare_sample()` to:
1. Extract `vlm_idx` from sample
2. Load images from `dataset.images` dict using the same logic as `unified_dataset.py`
3. Select appropriate frame for each camera view

### 2. Model Initialization Fix (Lines 596, 600, 647, 651)

Fixed model initialization to respect `--disable-sensor` and `--disable-robot-states` flags:

```python
# Before (hardcoded to True)
sensor_enabled=True,
robot_state_enabled=True,

# After (respects CLI flags)
sensor_enabled=not args.disable_sensor,
robot_state_enabled=not args.disable_robot_states,
```

This prevents dimension mismatch errors when comparing models with/without sensors.

## Verification

Run the quick benchmark:
```bash
bash benchmark_quick_test.sh
```

Expected VL encoding time:
- **Text-only**: ~30-50ms (incorrect)
- **5 images**: ~1400-1500ms (correct) ✅

## Notes

- VL encoding dominates inference time (**99.1%** of total time)
- For real-time 10Hz control (100ms per cycle), async VL encoding with VLM reuse is essential
- Sensor encoding is very fast (**0.5%** of total time)
- Action prediction is very fast (**0.4%** of total time)

---

**Date**: 2025-01-04
**Status**: ✅ Fixed and verified
