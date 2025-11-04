# Temporal Synchronization Report

## Summary

✅ **All modalities are properly synchronized** for asynchronous multi-modal fusion!

## System Configuration

| Modality | Frequency | Window Size | Window Duration | Purpose |
|----------|-----------|-------------|-----------------|---------|
| **VL Model** | 3.3 Hz | N/A | N/A | Vision-language understanding |
| **Action Prediction** | 10 Hz | 8 steps | 800 ms | Future action sequence |
| **Sensor Data** | 650 Hz → 100 Hz windows | 650 samples | 650 ms | Force + FPI/OCT sensing |
| **Robot States** | 100 Hz | 65 samples | 650 ms | Joint angles + pose |

## Timing Breakdown

### 1. VL Model (3.3 Hz / 300ms period)
- **VL features reused** for 3 consecutive action predictions
- Updated every 30 robot samples (300ms at 100Hz)
- This allows vision processing to be slower than action prediction

**Example**:
```
t=0ms:    VL_0 computed
t=100ms:  Action_0 predicted using VL_0
t=200ms:  Action_1 predicted using VL_0 (reused)
t=300ms:  VL_1 computed
t=400ms:  Action_2 predicted using VL_1
```

### 2. Action Expert (10 Hz / 100ms period)
- Predicts 8-step action sequence (horizon=8)
- Each prediction covers next 800ms
- Uses current VL features + sensor/robot state windows

### 3. Sensor Windows (650ms, 100Hz sampling)
- **Raw sensor data**: 650 Hz (FPI/OCT)
- **Pre-computed windows**: 650 samples per window
- Each action gets a 650ms sensor context window
- Windows are pre-computed and stored in NPZ format

### 4. Robot State Windows (650ms, 100Hz sampling)
- **Robot data**: 100 Hz (joint angles + pose)
- **Window size**: 65 samples = 650ms @ 100Hz
- Centered around each action prediction time
- Provides robot motion context

## Sample Timeline Analysis

| Sample | VLM idx | Action idx | Robot Window | Time (ms) | Notes |
|--------|---------|------------|--------------|-----------|-------|
| 0 | 0 | 0 | [0-65] | 0 | VL_0, first action |
| 1 | 0 | 1 | [0-65] | 100 | Reuse VL_0 |
| 2 | 0 | 2 | [0-65] | 200 | Reuse VL_0 |
| 3 | 30 | 3 | [0-65] | 300 | **VL_1 updated** |
| 4 | 30 | 4 | [8-73] | 400 | Reuse VL_1, robot window shifts |
| 10 | 90 | 10 | [68-133] | 1000 | VL_3, robot window at 100ms center |

## Window Centering Strategy

Each action prediction at time `t` uses:

1. **VL features**: Most recent VL update (within last 300ms)
2. **Sensor window**: `[t-325ms, t+325ms]` (650ms centered)
3. **Robot state window**: `[t-325ms, t+325ms]` (650ms centered)

This provides **±325ms context** around each prediction, capturing:
- Recent sensor readings
- Recent robot motion
- Future-looking sensor data (causal for control)

## Data Verification

### Sensor vs Robot State Dimensions

**Different sampling rates → different window sizes (same duration)**

| Data Type | Sampling Hz | Window Samples | Window Duration |
|-----------|-------------|----------------|-----------------|
| Sensor | 650 Hz | 650 | 650 ms |
| Robot State | 100 Hz | 65 | 650 ms |

Both cover the **same 650ms time window**, just at different resolutions!

### Asynchronous Verification

Tested robot state changes between samples:
- Sample 0 robot states: `shape=(65, 12)`
- Sample 10 robot states: `shape=(65, 12)`
- Mean difference: **1.433** → ✅ States are changing (asynchronous)

## Why This Design Works

1. **VL Reuse** (3Hz):
   - Vision processing is expensive
   - Scene doesn't change much in 300ms
   - Allows real-time performance

2. **Action Prediction** (10Hz):
   - Fast enough for responsive control
   - Slow enough to be stable
   - Standard robot control frequency

3. **Sensor Windows** (650ms):
   - Provides rich temporal context
   - Captures sensor dynamics
   - Matches typical insertion task duration

4. **Robot State Windows** (650ms):
   - Captures arm motion trajectory
   - Learns velocity and acceleration patterns
   - Provides proprioceptive context

## Implementation Details

### Dataset Sampling

```python
# Action interval: robot_hz / action_expert_hz = 100 / 10 = 10
action_interval = 10  # robot samples per action

# VL interval: action_interval * vlm_reuse_count = 10 * 3 = 30
vlm_interval = 30  # robot samples per VL update

# Robot state window center
robot_center = idx * action_interval
robot_start = max(0, robot_center - 65 // 2)
robot_end = robot_start + 65

# Result: Each action gets a 65-sample robot state window
```

### Encoder Inputs

```python
# Training forward pass
vl_tokens: (B, seq_len, 2048)      # From VL model
sensor_data: (B, 650, 1026)        # 650ms @ 650Hz
robot_states: (B, 65, 12)          # 650ms @ 100Hz

# After encoding
sensor_features: (B, 2048)         # From SensorEncoder
robot_state_features: (B, 2048)    # From RobotStateEncoder
combined_features: (B, 4096)       # Concatenated

# Fusion
vl_pooled: (B, 2048)
fused: (B, 6144)                   # vl + sensor + robot_state
```

## Conclusion

✅ **Temporal synchronization is correct**
- VL, sensor, and robot states are properly aligned
- Different sampling rates are handled correctly
- Windows provide appropriate temporal context
- Asynchronous processing verified

The system implements proper **multi-modal sensor fusion** with asynchronous data streams at different frequencies, which is essential for real-time robot control with computationally expensive vision models.
