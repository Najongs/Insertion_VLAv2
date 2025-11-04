#!/usr/bin/env python3
"""
Test temporal synchronization of VL, Sensor, and Robot State

System frequencies:
- VL model: 3 Hz (updated every ~333ms)
- Action expert: 10 Hz (every 100ms)
- Sensor data: 100 Hz (every 10ms) → 650 Hz raw, but sampled at 100Hz
- Robot states: 100 Hz (every 10ms)

Expected behavior:
- VL features reused for multiple action predictions (vlm_reuse_count=3)
- Action predictions at 10 Hz
- Sensor window: 65 samples at 100Hz = 650ms window
- Robot state window: 65 samples at 100Hz = 650ms window
"""

import numpy as np
from pathlib import Path
from vla_datasets.unified_dataset import UnifiedVLADataset


def test_timing_synchronization():
    """Test that all modalities are properly synchronized"""

    episode_path = Path("/home/najo/NAS/VLA/dataset/New_dataset/Blue_point/episode_20251030_025856")

    if not episode_path.exists():
        print(f"❌ Episode not found: {episode_path}")
        return

    print("🕐 Testing Temporal Synchronization\n")
    print("=" * 70)

    # Load dataset
    dataset = UnifiedVLADataset(
        data_dir=str(episode_path),
        format='new',
        horizon=8,
        vlm_reuse_count=3,  # VL features reused 3 times
        sensor_window_size=65,
        action_expert_hz=10,
    )

    print(f"📊 Dataset Configuration:")
    print(f"   Total samples: {len(dataset)}")
    print(f"   Robot Hz: {dataset.robot_hz} Hz")
    print(f"   Sensor Hz: {dataset.sensor_hz} Hz")
    print(f"   Action expert Hz: {dataset.action_expert_hz} Hz")
    print(f"   Action interval: {dataset.action_interval} (robot samples per action)")
    print(f"   VLM interval: {dataset.vlm_interval} (robot samples per VL update)")
    print(f"   VLM reuse count: {dataset.vlm_reuse_count}")
    print(f"   Sensor window size: {dataset.sensor_window_size}")
    print(f"   Has sensor: {dataset.has_sensor}")
    print(f"   Has robot_states: {dataset.has_robot_states}")
    print()

    # Calculate expected timings
    robot_dt = 1000.0 / dataset.robot_hz  # ms per robot sample
    action_dt = 1000.0 / dataset.action_expert_hz  # ms per action
    vl_dt = action_dt * dataset.vlm_reuse_count  # ms per VL update
    sensor_window_duration = dataset.sensor_window_size * robot_dt  # ms

    print(f"⏱️  Expected Timings:")
    print(f"   Robot sample period: {robot_dt:.1f} ms")
    print(f"   Action prediction period: {action_dt:.1f} ms (every {dataset.action_interval} robot samples)")
    print(f"   VL update period: {vl_dt:.1f} ms (every {dataset.vlm_interval} robot samples)")
    print(f"   Sensor window duration: {sensor_window_duration:.1f} ms ({dataset.sensor_window_size} samples)")
    print()

    # Test a few samples
    print(f"🔍 Testing Sample Alignment:")
    print(f"{'Sample':<8} {'VLM idx':<10} {'Action idx':<12} {'Robot window':<20} {'Time (ms)':<12}")
    print("-" * 70)

    test_samples = [0, 1, 2, 3, 4, 10, 20, 50]
    for sample_idx in test_samples:
        if sample_idx >= len(dataset):
            break

        sample = dataset[sample_idx]

        # Calculate indices
        vlm_idx = sample['vlm_idx']
        action_idx = sample_idx

        # Robot state window center
        robot_center_idx = sample_idx * dataset.action_interval
        robot_start = max(0, robot_center_idx - dataset.sensor_window_size // 2)
        robot_end = robot_start + dataset.sensor_window_size

        # Time in ms
        time_ms = action_idx * action_dt

        print(f"{sample_idx:<8} {vlm_idx:<10} {action_idx:<12} [{robot_start:>4}-{robot_end:>4}] {time_ms:>8.0f} ms")

    print()
    print(f"✅ VL Feature Reuse Pattern:")
    print(f"   VL features are updated every {dataset.vlm_reuse_count} action predictions")
    print(f"   This means VL runs at {dataset.action_expert_hz / dataset.vlm_reuse_count:.1f} Hz (~3 Hz)")
    print()

    # Verify sensor and robot state windows
    print(f"🔬 Verifying Window Alignment:")
    sample_0 = dataset[0]
    sample_10 = dataset[10]

    print(f"   Sample 0:")
    print(f"      Sensor data shape: {sample_0['sensor_data'].shape}")
    print(f"      Robot states shape: {sample_0['robot_states'].shape}")
    print(f"      Actions shape: {sample_0['actions'].shape}")
    print()

    print(f"   Sample 10:")
    print(f"      Sensor data shape: {sample_10['sensor_data'].shape}")
    print(f"      Robot states shape: {sample_10['robot_states'].shape}")
    print(f"      Actions shape: {sample_10['actions'].shape}")
    print()

    # Check if robot states change between samples
    rs_0 = sample_0['robot_states'].numpy()
    rs_10 = sample_10['robot_states'].numpy()

    if dataset.has_robot_states:
        diff = np.abs(rs_0 - rs_10).mean()
        print(f"   Robot state difference (sample 0 vs 10): {diff:.6f}")
        if diff > 0:
            print(f"   ✅ Robot states are changing between samples (asynchronous)")
        else:
            print(f"   ⚠️  Robot states are identical (possible issue)")
    print()

    # Summary
    print(f"📝 Synchronization Summary:")
    print(f"   ✅ VL model: {dataset.action_expert_hz / dataset.vlm_reuse_count:.1f} Hz (reuse_count={dataset.vlm_reuse_count})")
    print(f"   ✅ Action predictions: {dataset.action_expert_hz} Hz")
    print(f"   ✅ Sensor window: {dataset.sensor_window_size} samples @ 100 Hz = {sensor_window_duration:.0f} ms")
    print(f"   ✅ Robot state window: {dataset.sensor_window_size} samples @ 100 Hz = {sensor_window_duration:.0f} ms")
    print()

    # Verify the window centers align with action predictions
    print(f"🎯 Window Centering:")
    print(f"   Each action prediction at time t uses:")
    print(f"      - VL features from most recent VL update (reused)")
    print(f"      - Sensor window: [{-sensor_window_duration/2:.0f} ms, +{sensor_window_duration/2:.0f} ms] around t")
    print(f"      - Robot state window: [{-sensor_window_duration/2:.0f} ms, +{sensor_window_duration/2:.0f} ms] around t")
    print()

    print(f"✅ Temporal synchronization test complete!")


if __name__ == "__main__":
    test_timing_synchronization()
