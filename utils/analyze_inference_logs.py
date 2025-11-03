#!/usr/bin/env python3
"""
Analyze inference logs from Async_inference_receiver.py

This script parses inference logs and provides performance statistics:
- VL update timing
- Action expert timing
- Data reception rates
- Sensor buffer status

Usage:
    # Analyze log file
    python analyze_inference_logs.py inference.log

    # Analyze saved inference results
    python analyze_inference_logs.py --results inference_results_20251029_143015.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


def parse_log_file(log_path):
    """Parse inference log file and extract metrics"""

    with open(log_path, 'r') as f:
        lines = f.readlines()

    # Extract VL update times
    vl_pattern = re.compile(r'🔄 \[VL Update #(\d+)\] Completed in (\d+)ms')
    vl_updates = []

    # Extract action expert times
    action_pattern = re.compile(
        r'\[ACTION #(\d+)\] VL_reuse=(\d+)/(\d+) \| '
        r'Actions\[0\]: \[([-\d., ]+)\] \| '
        r'Time: ([\d.]+)ms \| '
        r'Sensor: (\d+)/(\d+)'
    )
    action_times = []
    sensor_counts = []

    # Extract status lines
    status_pattern = re.compile(
        r'--- Status \([\d:]+\) ---\n'
        r'VL Updates: (\d+) \| VL avg: ([\d.]+)ms\n'
        r'Actions: (\d+) \| Action avg: ([\d.]+)ms'
    )

    for line in lines:
        # VL updates
        match = vl_pattern.search(line)
        if match:
            update_num = int(match.group(1))
            time_ms = int(match.group(2))
            vl_updates.append((update_num, time_ms))

        # Action expert
        match = action_pattern.search(line)
        if match:
            action_num = int(match.group(1))
            vl_reuse = int(match.group(2))
            time_ms = float(match.group(5))
            sensor_count = int(match.group(6))
            sensor_total = int(match.group(7))

            action_times.append(time_ms)
            sensor_counts.append((sensor_count, sensor_total))

    return {
        'vl_updates': vl_updates,
        'action_times': action_times,
        'sensor_counts': sensor_counts,
    }


def analyze_json_results(json_path):
    """Analyze inference results from JSON file"""

    with open(json_path, 'r') as f:
        results = json.load(f)

    timestamps = [r['timestamp'] for r in results]
    inference_times = [r['inference_time'] * 1000 for r in results]  # Convert to ms
    vl_update_numbers = [r.get('vl_update_number', 0) for r in results]

    # Calculate intervals
    intervals = np.diff(timestamps) * 1000  # Convert to ms

    # Extract action values
    actions_first = np.array([r['actions'][0] for r in results])  # First horizon

    return {
        'total_actions': len(results),
        'duration': timestamps[-1] - timestamps[0],
        'timestamps': timestamps,
        'inference_times': inference_times,
        'intervals': intervals,
        'vl_update_numbers': vl_update_numbers,
        'actions_first': actions_first,
    }


def print_header(text):
    """Print formatted header"""
    print(f"\n{'='*80}")
    print(f"{text}")
    print(f"{'='*80}")


def print_section(text):
    """Print section header"""
    print(f"\n{text}")
    print(f"{'-'*80}")


def print_stats(name, values, unit='ms'):
    """Print statistics for a metric"""
    if len(values) == 0:
        print(f"   {name}: No data")
        return

    arr = np.array(values)
    print(f"   {name}:")
    print(f"      Mean:   {arr.mean():.2f} {unit}")
    print(f"      Std:    {arr.std():.2f} {unit}")
    print(f"      Min:    {arr.min():.2f} {unit}")
    print(f"      Max:    {arr.max():.2f} {unit}")
    print(f"      Median: {np.median(arr):.2f} {unit}")

    # Add percentiles
    p95 = np.percentile(arr, 95)
    p99 = np.percentile(arr, 99)
    print(f"      P95:    {p95:.2f} {unit}")
    print(f"      P99:    {p99:.2f} {unit}")


def main():
    parser = argparse.ArgumentParser(description='Analyze async VLA inference logs')
    parser.add_argument('log_file', nargs='?', help='Log file to analyze')
    parser.add_argument('--results', help='JSON results file to analyze')
    args = parser.parse_args()

    if not args.log_file and not args.results:
        parser.error("Please provide either a log file or --results JSON file")

    print_header("Async VLA Inference Log Analysis")

    # ========================================
    # Analyze Log File
    # ========================================
    if args.log_file:
        log_path = Path(args.log_file)
        if not log_path.exists():
            print(f"❌ Error: Log file not found: {args.log_file}")
            return 1

        print(f"\nAnalyzing log file: {args.log_file}")

        data = parse_log_file(args.log_file)

        # VL Update Stats
        print_section("VL Update Performance")
        if data['vl_updates']:
            vl_times = [t for _, t in data['vl_updates']]
            print_stats("VL Update Time", vl_times, 'ms')

            # Check target
            target_vl_time = 381  # ms for 640x360
            avg_vl_time = np.mean(vl_times)
            if avg_vl_time < target_vl_time * 1.1:
                print(f"\n   ✅ VL update time is within target ({target_vl_time}ms)")
            else:
                print(f"\n   ⚠️  VL update time exceeds target ({target_vl_time}ms)")

            # Estimate VL update rate
            vl_rate = 1000 / avg_vl_time
            print(f"\n   VL Update Rate: ~{vl_rate:.2f} Hz")
        else:
            print("   No VL update data found in log")

        # Action Expert Stats
        print_section("Action Expert Performance")
        if data['action_times']:
            print_stats("Action Expert Time", data['action_times'], 'ms')

            # Check target
            target_action_time = 30  # ms
            avg_action_time = np.mean(data['action_times'])
            if avg_action_time < target_action_time:
                print(f"\n   ✅ Action expert time is within target ({target_action_time}ms)")
            else:
                print(f"\n   ⚠️  Action expert time exceeds target ({target_action_time}ms)")

            # Calculate actual action rate
            action_count = len(data['action_times'])
            print(f"\n   Total Actions: {action_count}")
        else:
            print("   No action data found in log")

        # Sensor Buffer Stats
        print_section("Sensor Buffer Status")
        if data['sensor_counts']:
            sensor_fill_rates = [count / total for count, total in data['sensor_counts']]
            print_stats("Sensor Fill Rate", sensor_fill_rates, '%')

            avg_fill = np.mean(sensor_fill_rates)
            if avg_fill > 0.9:
                print(f"\n   ✅ Sensor buffer is well filled (>{90}%)")
            elif avg_fill > 0.5:
                print(f"\n   ⚠️  Sensor buffer partially filled ({avg_fill*100:.1f}%)")
            else:
                print(f"\n   ❌ Sensor buffer is poorly filled ({avg_fill*100:.1f}%)")
        else:
            print("   No sensor data found in log")

    # ========================================
    # Analyze JSON Results
    # ========================================
    if args.results:
        results_path = Path(args.results)
        if not results_path.exists():
            print(f"❌ Error: Results file not found: {args.results}")
            return 1

        print(f"\nAnalyzing results file: {args.results}")

        data = analyze_json_results(args.results)

        # Overall Stats
        print_section("Overall Statistics")
        print(f"   Total Actions: {data['total_actions']}")
        print(f"   Duration: {data['duration']:.2f}s")
        action_rate = data['total_actions'] / data['duration']
        print(f"   Action Rate: {action_rate:.2f} Hz")

        # Check target action rate
        target_action_rate = 10.0  # Hz
        if abs(action_rate - target_action_rate) < 0.5:
            print(f"\n   ✅ Action rate is on target ({target_action_rate} Hz)")
        else:
            print(f"\n   ⚠️  Action rate deviates from target ({target_action_rate} Hz)")

        # Inference Timing
        print_section("Inference Timing")
        print_stats("Inference Time", data['inference_times'], 'ms')

        # Interval Stats
        print_section("Action Intervals")
        print_stats("Interval", data['intervals'], 'ms')

        # Check target interval
        target_interval = 100  # ms for 10Hz
        avg_interval = np.mean(data['intervals'])
        if abs(avg_interval - target_interval) < 10:
            print(f"\n   ✅ Action interval is on target ({target_interval}ms)")
        else:
            print(f"\n   ⚠️  Action interval deviates from target ({target_interval}ms)")

        # Action Value Stats
        print_section("Action Values (First Horizon)")
        actions = data['actions_first']
        print(f"   Shape: {actions.shape}")

        action_names = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6', 'Gripper']
        print(f"\n   Per-dimension statistics:")
        for i, name in enumerate(action_names):
            vals = actions[:, i]
            print(f"\n   {name}:")
            print(f"      Mean: {vals.mean():.4f}")
            print(f"      Std:  {vals.std():.4f}")
            print(f"      Min:  {vals.min():.4f}")
            print(f"      Max:  {vals.max():.4f}")

        # Check for stuck actions
        print(f"\n   Checking for stuck actions...")
        stuck_dims = []
        for i in range(actions.shape[1]):
            if actions[:, i].std() < 1e-6:
                stuck_dims.append(i)

        if stuck_dims:
            print(f"   ⚠️  Warning: Actions may be stuck in dimensions: {stuck_dims}")
            print(f"      These dimensions have near-zero variance")
        else:
            print(f"   ✅ All action dimensions show variation")

        # VL Update Analysis
        print_section("VL Update Pattern")
        unique_vl_updates = len(np.unique(data['vl_update_numbers']))
        actions_per_vl = data['total_actions'] / unique_vl_updates if unique_vl_updates > 0 else 0
        print(f"   Unique VL Updates: {unique_vl_updates}")
        print(f"   Actions per VL Update: {actions_per_vl:.2f}")

        # Expected: 4 actions per VL update (with VL_REUSE=4)
        if 3.5 < actions_per_vl < 4.5:
            print(f"\n   ✅ VL reuse pattern is as expected (~4 actions per VL update)")
        else:
            print(f"\n   ⚠️  VL reuse pattern may be unexpected")

    # ========================================
    # Summary
    # ========================================
    print_header("Summary")

    issues = []
    recommendations = []

    if args.log_file and data.get('vl_updates'):
        avg_vl = np.mean([t for _, t in data['vl_updates']])
        if avg_vl > 400:
            issues.append("VL update time is slow (>400ms)")
            recommendations.append("Check GPU usage, ensure images are 640x360")

    if args.log_file and data.get('action_times'):
        avg_action = np.mean(data['action_times'])
        if avg_action > 50:
            issues.append("Action expert time is slow (>50ms)")
            recommendations.append("Check GPU usage, verify sensor window is 65 samples")

    if args.log_file and data.get('sensor_counts'):
        avg_fill = np.mean([c/t for c, t in data['sensor_counts']])
        if avg_fill < 0.5:
            issues.append("Sensor buffer is poorly filled (<50%)")
            recommendations.append("Check sensor sender is running, verify UDP port 9999")

    if args.results:
        if abs(action_rate - 10.0) > 1.0:
            issues.append(f"Action rate deviates from target (got {action_rate:.2f} Hz, expected 10 Hz)")
            recommendations.append("Check system performance, verify timing loops")

    if not issues:
        print("\n   🎉 No issues detected! System is performing well.")
    else:
        print("\n   ⚠️  Issues detected:")
        for i, issue in enumerate(issues, 1):
            print(f"      {i}. {issue}")

        print("\n   💡 Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"      {i}. {rec}")

    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
