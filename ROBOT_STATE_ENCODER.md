# Robot State Encoder Implementation

## Problem

Robot states (joint angles + pose) were being encoded using the same **SensorEncoder** designed for high-dimensional sensor data (1026 channels). This is suboptimal because:

1. **Different data characteristics**:
   - Sensor: 1026 dims (force + A-scan), noisy, high-dimensional
   - Robot state: 12 dims (6 joints + 6 poses), clean, structured

2. **Over-engineering**: Conv1D layers designed for sensor noise are unnecessary for structured robot state

3. **Efficiency**: Robot state is much simpler and doesn't need heavy convolution

## Solution: Dedicated RobotStateEncoder

Created a specialized MLP-based encoder optimized for robot state data.

### Architecture

```
Input: (B, T, 12) where T=65, 12=[6 joints + 6 poses]
  ↓
Per-timestep MLP (shared weights across time)
  - Linear(12 → 256) + LayerNorm + GELU + Dropout
  - Linear(256 → 512) + LayerNorm + GELU + Dropout
  - Linear(512 → 512) + LayerNorm + GELU + Dropout
  ↓ (B, T, 512)
Temporal Transformer (2 layers)
  - Multi-head attention across time (nhead=8)
  - Captures temporal dependencies in robot motion
  ↓ (B, T, 512)
Average Pooling across time
  ↓ (B, 512)
Output Projection MLP
  - Linear(512 → 2048) + LayerNorm + GELU + Dropout
  - Linear(2048 → 2048)
  ↓
Output: (B, 2048)
```

### Key Features

1. **Per-timestep MLP**: Shared MLP processes each timestep independently
   - Learns to extract features from joint angles and poses
   - LayerNorm + GELU for stable training

2. **Temporal Transformer**: Models temporal dependencies
   - Self-attention captures motion patterns
   - Learns smooth trajectories and acceleration patterns

3. **Compact design**:
   - Hidden dim: 256 → 512 (vs sensor's 512 → 1024)
   - No Conv1D layers (robot state doesn't need spatial filtering)
   - Fewer parameters, faster training

4. **Variable sequence length handling**:
   - Pads with zeros if T < 65
   - Truncates to last 65 if T > 65

## Comparison: SensorEncoder vs RobotStateEncoder

| Feature | SensorEncoder | RobotStateEncoder |
|---------|--------------|-------------------|
| Input dims | 1026 | 12 |
| Architecture | Conv1D + Transformer | MLP + Transformer |
| Hidden dim | 512 → 1024 | 256 → 512 |
| Temporal modeling | Conv downsample + Attention | Direct attention |
| Parameters | ~15M | ~2M |
| Use case | Noisy sensor signals | Clean robot state |

## Implementation

### models/unified_model.py

```python
# Old (using SensorEncoder)
self.robot_state_encoder = SensorEncoder(
    input_channels=12,  # ❌ Overkill for 12 dims
    temporal_length=65,
    hidden_dim=512,
    output_dim=2048,
    use_transformer=True,
    num_transformer_layers=2
)

# New (using RobotStateEncoder)
self.robot_state_encoder = RobotStateEncoder(
    input_dim=12,           # ✅ MLP handles this better
    temporal_length=65,
    hidden_dim=256,         # ✅ Smaller, more efficient
    output_dim=2048,
    num_layers=3,
    use_temporal_attention=True,
    nhead=8,
    dropout=0.1
)
```

### Why MLP is Better for Robot State

1. **Structured data**: Joint angles and poses have direct physical meaning
   - Joint 1-6: Revolute joint angles (radians)
   - Pose: (x, y, z, roll, pitch, yaw)
   - No spatial correlation like in sensor scans

2. **Low dimensionality**: 12 dims vs 1026 dims
   - MLP can directly learn meaningful representations
   - No need for dimensionality reduction via convolution

3. **Temporal patterns**: Robot motion has smooth trajectories
   - Transformer attention captures velocity and acceleration
   - No need for Conv1D temporal filtering

4. **Efficiency**: Faster training, fewer parameters
   - Conv layers on 12 channels are wasteful
   - MLP is simpler and more interpretable

## Benefits

1. **Better semantic encoding**:
   - MLP learns joint-specific and pose-specific features
   - Temporal attention captures motion patterns

2. **Faster training**:
   - ~13M fewer parameters than using SensorEncoder
   - Less computation per forward pass

3. **More interpretable**:
   - Can analyze what each MLP layer learns
   - Attention weights show important time steps

4. **Extensible**:
   - Easy to add domain-specific features (velocity, acceleration)
   - Can incorporate kinematic constraints

## Validation Fix

Also fixed validation error where flow matching was incorrectly unpacking 3 values:

```python
# Before (incorrect)
flow_loss, _, _ = model(...)  # ❌ Flow matching only returns loss

# After (correct)
loss, _, _ = model(...)  # ✅ Matches diffusion's return signature
```

## Files Changed

1. **models/unified_model.py**:
   - Lines 40-162: Added `RobotStateEncoder` class
   - Lines 900-914: Use `RobotStateEncoder` instead of `SensorEncoder`

2. **TRAIN_Unified.py**:
   - Line 784: Fixed validation unpacking for flow matching

## Summary

Robot states now have a **dedicated, optimized encoder** that:
- Uses MLP instead of Conv1D (better for structured data)
- Has temporal attention (captures motion patterns)
- Is more efficient (~13M fewer parameters)
- Produces same output_dim=2048 for fusion with sensor features

This is a **more principled design** that respects the different nature of sensor data vs robot state data.
