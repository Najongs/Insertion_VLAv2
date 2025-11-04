# Robot States Loading Fix

## Problem

Training failed with dimension mismatch error because `robot_states` was `None` during training, even though the dataset correctly loaded robot states data.

**Error**:
```
RuntimeError: Dimension mismatch in fusion_proj: Expected 6144, got 4096
vl_pooled: 2048, combined_features: 2048
```

**Expected**: `combined_features = sensor(2048) + robot_state(2048) = 4096`
**Actual**: `combined_features = 2048` (only sensor, no robot_state)

## Root Cause

The `unified_collate_fn` in `vla_datasets/unified_dataset.py` was **NOT including `robot_states`** in the batched output dictionary.

### Dataset __getitem__ ✅ (Working)
```python
def _getitem_new(self, idx):
    # ...
    return {
        "instruction": self.instruction,
        "images": image_paths,
        "vl_cache": vl_cache,
        "sensor_data": torch.from_numpy(sensor_window),
        "robot_states": torch.from_numpy(robot_state_window),  # ✅ Included
        "actions": torch.from_numpy(actions),
        "has_sensor": bool(self.has_sensor),
        "has_robot_states": bool(self.has_robot_states),  # ✅ Included
        # ...
    }
```

### Collate Function ❌ (Broken)
```python
def unified_collate_fn(batch):
    # ...
    return {
        "instruction": instructions,
        "images": image_lists,
        "vl_cache": vl_features,
        "sensor_data": sensor_data,  # ✅ Included
        # ❌ robot_states MISSING!
        "actions": actions,
        "has_sensor_mask": has_sensor_mask,  # ✅ Included
        # ❌ has_robot_states_mask MISSING!
        # ...
    }
```

## Solution

Updated `unified_collate_fn` in `vla_datasets/unified_dataset.py:643-701` to:

1. **Pad robot_states** to max length (same as sensor data)
2. **Stack robot_states** into batch tensor
3. **Add robot_states** to output dictionary
4. **Add has_robot_states_mask** to output dictionary

### Fixed Collate Function ✅
```python
def unified_collate_fn(batch):
    # ... existing code for sensor_data ...

    # Pad robot states to max length (same as sensor data)
    robot_state_tensors = [b["robot_states"] for b in batch]
    max_robot_len = max(t.shape[0] for t in robot_state_tensors)
    padded_robot_states = []
    for robot_state in robot_state_tensors:
        if robot_state.shape[0] < max_robot_len:
            pad = torch.zeros((max_robot_len - robot_state.shape[0], robot_state.shape[1]),
                            dtype=robot_state.dtype)
            padded_robot_states.append(torch.cat([robot_state, pad], dim=0))
        else:
            padded_robot_states.append(robot_state)
    robot_states = torch.stack(padded_robot_states, dim=0)

    # ... existing code ...

    has_robot_states_mask = torch.tensor([b["has_robot_states"] for b in batch], dtype=torch.bool)

    return {
        "instruction": instructions,
        "images": image_lists,
        "vl_cache": vl_features,
        "sensor_data": sensor_data,
        "robot_states": robot_states,  # ✅ Now included!
        "actions": actions,
        "has_sensor_mask": has_sensor_mask,
        "has_robot_states_mask": has_robot_states_mask,  # ✅ Now included!
        "cache_keys": cache_keys,
        "vlm_indices": vlm_indices,
        "reuse_steps": reuse_steps,
        "confidence": confidence,
    }
```

## Verification

Created `test_collate_fix.py` to verify the fix:

```bash
$ python test_collate_fix.py
```

**Results**:
```
✅ robot_states in batch!
   robot_states shape: torch.Size([4, 65, 12])
   robot_states dtype: torch.float32
   has_robot_states_mask: tensor([True, True, True, True])
```

## Expected Dimensions

With the fix, the model should now receive:

- **VL features**: `(B, seq_len, 2048)` → pooled to `(B, 2048)`
- **Sensor features**: `(B, 65, 1026)` → encoded to `(B, 2048)`
- **Robot state features**: `(B, 65, 12)` → encoded to `(B, 2048)`
- **Combined features**: `(B, 4096)` = sensor(2048) + robot_state(2048)
- **Fusion input**: `(B, 6144)` = vl(2048) + combined(4096)

## Additional Fixes

Also added debug logging in `TRAIN_Unified.py:637-671` to track:
1. Batch keys at first step
2. Robot_states shape when loaded
3. Warning when robot_states not loaded

This will help detect similar issues in the future.

## Files Changed

1. **vla_datasets/unified_dataset.py**:
   - Line 643-701: Updated `unified_collate_fn` to include robot_states

2. **TRAIN_Unified.py**:
   - Line 637-671: Added debug logging for robot_states

3. **test_collate_fix.py** (new):
   - Verification test for collate function

## Summary

The issue was a **missing field in the collate function** - a classic data pipeline bug where individual samples had the data, but batching dropped it. The fix ensures robot_states properly flows from dataset → collate → dataloader → training loop → model.
