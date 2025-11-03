"""
Profile Diffusion Action Expert Inference Speed

Tests different DDIM step counts to find the optimal configuration for 10Hz operation.
Target: < 100ms for 10Hz action expert
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import time
import json
from tqdm import tqdm

# Import diffusion action expert
from models.model_with_sensor_diffusion import DiffusionActionExpert


def profile_diffusion_inference(
    vl_dim=3072,
    sensor_dim=3072,
    action_dim=7,
    horizon=8,
    hidden_dim=512,
    total_timesteps=100,
    test_ddim_steps=[5, 10, 15, 20, 50, 100],
    num_warmup=10,
    num_runs=50,
    batch_size=1,
):
    """
    Profile diffusion action expert with different DDIM steps

    Args:
        vl_dim: VL feature dimension
        sensor_dim: Sensor feature dimension
        action_dim: Action dimension
        horizon: Action horizon
        hidden_dim: Hidden dimension
        total_timesteps: Total diffusion timesteps
        test_ddim_steps: List of DDIM step counts to test
        num_warmup: Number of warmup runs
        num_runs: Number of profiling runs
        batch_size: Batch size for inference
    """
    print("=" * 80)
    print("🔬 Profiling Diffusion Action Expert Inference Speed")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  VL dim: {vl_dim}")
    print(f"  Sensor dim: {sensor_dim}")
    print(f"  Action: {action_dim}D × {horizon} horizon")
    print(f"  Total timesteps: {total_timesteps}")
    print(f"  Batch size: {batch_size}")
    print(f"\nTarget: < 100ms (for 10Hz action expert)")
    print("=" * 80)

    # Create diffusion action expert
    print(f"\n🚀 Loading Diffusion Action Expert...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DiffusionActionExpert(
        vl_dim=vl_dim,
        sensor_dim=sensor_dim,
        action_dim=action_dim,
        horizon=horizon,
        hidden_dim=hidden_dim,
        timesteps=total_timesteps,
        fusion_strategy='concat',
    ).to(device=device)

    model.eval()
    print(f"✅ Model loaded on {device}")
    print(f"   Note: Using float32 for profiling (dtype doesn't affect speed much)")

    # Create dummy inputs (float32 for compatibility)
    vl_tokens = torch.randn(batch_size, 1, vl_dim, device=device)
    sensor_features = torch.randn(batch_size, sensor_dim, device=device)

    results = {}

    # Test each DDIM step count
    for ddim_steps in test_ddim_steps:
        print(f"\n{'='*60}")
        print(f"Testing DDIM steps: {ddim_steps}")
        print(f"{'='*60}")

        # Warmup
        print(f"🔥 Warming up...")
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = model.sample(
                    vl_tokens=vl_tokens,
                    sensor_features=sensor_features,
                    batch_size=batch_size,
                    ddim_steps=ddim_steps if ddim_steps < total_timesteps else None
                )

        # Synchronize before profiling
        if device == "cuda":
            torch.cuda.synchronize()

        # Profile
        print(f"⏱️  Profiling {num_runs} runs...")
        times = []

        for i in range(num_runs):
            start = time.perf_counter()

            with torch.no_grad():
                _ = model.sample(
                    vl_tokens=vl_tokens,
                    sensor_features=sensor_features,
                    batch_size=batch_size,
                    ddim_steps=ddim_steps if ddim_steps < total_timesteps else None
                )

            if device == "cuda":
                torch.cuda.synchronize()

            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
            times.append(elapsed)

            if (i + 1) % 10 == 0:
                current_avg = sum(times) / len(times)
                print(f"  [{i+1}/{num_runs}] Current avg: {current_avg:.2f} ms")

        # Calculate statistics
        mean_ms = sum(times) / len(times)
        std_ms = (sum((t - mean_ms) ** 2 for t in times) / len(times)) ** 0.5
        mean_hz = 1000.0 / mean_ms

        results[ddim_steps] = {
            "mean_ms": mean_ms,
            "std_ms": std_ms,
            "mean_hz": mean_hz,
            "speedup_vs_full": (times[0] if ddim_steps == total_timesteps else results.get(total_timesteps, {}).get("mean_ms", mean_ms * (total_timesteps / ddim_steps))) / mean_ms,
        }

        print(f"\n📊 Results:")
        print(f"  Mean: {mean_ms:.2f} ms ({mean_hz:.1f} Hz)")
        print(f"  Std:  {std_ms:.2f} ms")

        # Check if meets 10Hz requirement
        if mean_ms <= 100:
            print(f"  ✅ PASSES 10Hz requirement (< 100ms)")
        else:
            print(f"  ❌ Too slow for 10Hz ({mean_ms:.1f}ms > 100ms)")

    # Summary
    print("\n" + "=" * 80)
    print("📊 DDIM Steps Comparison Summary")
    print("=" * 80)
    print()
    print(f"{'DDIM Steps':<12} {'Time (ms)':<12} {'Hz':<10} {'Status':<20} {'Note':<30}")
    print("-" * 80)

    for ddim_steps in sorted(results.keys()):
        r = results[ddim_steps]
        mean_ms = r["mean_ms"]
        mean_hz = r["mean_hz"]

        if mean_ms <= 100:
            status = "✅ OK for 10Hz"
        else:
            status = "❌ Too slow"

        # Calculate how many times Action Expert can reuse VL features
        vl_reuse = int(mean_ms / 100) + 1
        note = f"VL reuse: {vl_reuse}x"

        print(f"{ddim_steps:<12} {mean_ms:>8.1f} ms  {mean_hz:>6.1f} Hz  {status:<20} {note:<30}")

    print()
    print("💡 Recommendations:")
    print("-" * 80)

    # Find best DDIM steps for 10Hz
    valid_configs = [(steps, r["mean_ms"]) for steps, r in results.items() if r["mean_ms"] <= 100]

    if valid_configs:
        # Find highest quality (most steps) that still meets 10Hz requirement
        best_steps, best_time = max(valid_configs, key=lambda x: x[0])
        print(f"✅ Recommended DDIM steps: {best_steps}")
        print(f"   - Action Expert time: {best_time:.1f}ms")
        print(f"   - Frequency: {1000/best_time:.1f} Hz")
        print(f"   - VL feature reuse: {int(best_time/100) + 1}x")
    else:
        # Find minimum time and suggest VL reuse
        min_steps, min_time = min(results.items(), key=lambda x: x[1]["mean_ms"])
        vl_reuse = int(min_time / 100) + 1
        print(f"⚠️  No configuration meets 10Hz requirement")
        print(f"   Fastest: {min_steps} steps = {min_time:.1f}ms")
        print(f"   Suggested: Use {min_steps} steps with {vl_reuse}x VL feature reuse")
        print(f"   Effective Action Expert frequency: {1000/min_time:.1f} Hz")

    # Save results
    output_file = PROJECT_ROOT / "diffusion_profile_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {output_file}")

    return results


if __name__ == "__main__":
    results = profile_diffusion_inference(
        vl_dim=3072,
        sensor_dim=3072,
        action_dim=7,
        horizon=8,
        hidden_dim=512,
        total_timesteps=100,
        test_ddim_steps=[5, 10, 15, 20, 50, 100],
        num_warmup=10,
        num_runs=50,
        batch_size=1,
    )

    print("\n" + "=" * 80)
    print("✅ Profiling complete!")
    print("=" * 80)
