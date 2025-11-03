"""
VLM Inference Speed Profiling Utility

Measures the actual inference speed of Qwen-VL to determine
the optimal VLM update frequency for async training.

Usage:
    python utils/profile_vlm_speed.py --num-samples 100
"""

import argparse
import time
import torch
import numpy as np
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def profile_vlm_inference(
    model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    num_samples=100,
    image_paths=None,
    num_views=5,
    use_flash_attn=True,
):
    """
    Profile VLM inference speed with MULTI-VIEW images

    Args:
        model_name: Qwen-VL model name
        num_samples: Number of inference runs for profiling
        image_paths: List of test image paths (optional)
        num_views: Number of views to use (default: 5)
        use_flash_attn: Whether to use flash attention

    Returns:
        dict with profiling results
    """
    print(f"🚀 Loading {model_name}...")

    # Load model
    attn_impl = "flash_attention_2" if use_flash_attn else "sdpa"
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            device_map="cuda",
            low_cpu_mem_usage=True,
        )
        print(f"✅ Loaded with {attn_impl}")
    except Exception as e:
        print(f"⚠️ Flash attention failed: {e}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            low_cpu_mem_usage=True,
        )
        print("✅ Loaded with default attention")

    processor = AutoProcessor.from_pretrained(model_name)
    model.eval()

    # Find test images if not provided - collect multi-view images
    if image_paths is None:
        # Try to find multi-view images from actual dataset
        dataset_paths = [
            "/home/najo/NAS/VLA/dataset/White_silicone_white_circle/recv_all_20251027_170308",
            "/home/najo/NAS/VLA/dataset/Needle_insertion_eye_trocar",
        ]

        image_paths = []
        for ds_path in dataset_paths:
            ds_dir = Path(ds_path)
            if ds_dir.exists():
                # Try to find images from different views
                view_dirs = ["View1/left", "View2/left", "View3/left", "View4/left", "View5"]

                for view_dir in view_dirs:
                    view_path = ds_dir / view_dir
                    if view_path.exists():
                        image_files = sorted(view_path.glob("*.jpg"))
                        if not image_files:
                            image_files = sorted(view_path.glob("*.png"))
                        if image_files:
                            image_paths.append(str(image_files[0]))
                            if len(image_paths) >= num_views:
                                break

                if len(image_paths) >= num_views:
                    break

        if len(image_paths) < num_views:
            # If we couldn't find enough views, duplicate the first one
            if image_paths:
                while len(image_paths) < num_views:
                    image_paths.append(image_paths[0])
            else:
                raise FileNotFoundError("No test images found. Please provide --image-paths")

    # Ensure we have exactly num_views images
    image_paths = image_paths[:num_views]

    print(f"📸 Using {len(image_paths)} multi-view images:")
    for i, path in enumerate(image_paths):
        print(f"   View {i+1}: {Path(path).parent.name}/{Path(path).name}")

    # Prepare test input with MULTI-VIEW images
    test_text = "Observe the current state and predict the next action for needle insertion."
    message_content = []

    # Add all images
    for img_path in image_paths:
        message_content.append({"type": "image", "image": img_path})

    # Add text at the end
    message_content.append({"type": "text", "text": test_text})

    messages = [{
        "role": "user",
        "content": message_content
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    vision_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=vision_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device="cuda", dtype=torch.bfloat16)

    # Warmup runs
    print("🔥 Warming up...")
    for _ in range(10):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    torch.cuda.synchronize()

    # Profiling runs
    print(f"⏱️  Profiling {num_samples} inference runs...")
    latencies = []

    for i in range(num_samples):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)

        torch.cuda.synchronize()
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{num_samples}] Current avg: {np.mean(latencies):.2f} ms")

    # Calculate statistics
    latencies = np.array(latencies)

    results = {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
    }

    # Calculate achievable frequencies
    results["mean_hz"] = 1000.0 / results["mean_ms"]
    results["p95_hz"] = 1000.0 / results["p95_ms"]
    results["p99_hz"] = 1000.0 / results["p99_ms"]

    # Memory usage
    results["gpu_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9

    return results


def print_results(results):
    """Pretty print profiling results"""
    print("\n" + "="*60)
    print("📊 VLM Inference Profiling Results")
    print("="*60)

    print("\n⏱️  Latency Statistics:")
    print(f"  Mean:   {results['mean_ms']:.2f} ms")
    print(f"  Median: {results['median_ms']:.2f} ms")
    print(f"  Std:    {results['std_ms']:.2f} ms")
    print(f"  Min:    {results['min_ms']:.2f} ms")
    print(f"  Max:    {results['max_ms']:.2f} ms")
    print(f"  P95:    {results['p95_ms']:.2f} ms")
    print(f"  P99:    {results['p99_ms']:.2f} ms")

    print("\n🎯 Achievable Frequencies:")
    print(f"  Mean throughput: {results['mean_hz']:.2f} Hz")
    print(f"  P95 throughput:  {results['p95_hz']:.2f} Hz")
    print(f"  P99 throughput:  {results['p99_hz']:.2f} Hz")

    print("\n💡 Recommendations for Async Training:")

    # Calculate safe VLM update period
    safe_period_ms = results['p95_ms'] * 1.2  # 20% safety margin
    safe_hz = 1000.0 / safe_period_ms

    print(f"  Recommended VLM update frequency: {safe_hz:.2f} Hz")
    print(f"  Recommended VLM update period: {safe_period_ms:.0f} ms")

    # Calculate how many action expert steps per VLM update
    action_expert_hz = 10.0  # Target 10Hz
    action_expert_period_ms = 1000.0 / action_expert_hz

    vlm_reuse_count = int(safe_period_ms / action_expert_period_ms)
    actual_vlm_hz = action_expert_hz / vlm_reuse_count

    print(f"\n  With Action Expert @ {action_expert_hz} Hz:")
    print(f"    → VL features reused {vlm_reuse_count}x")
    print(f"    → Actual VLM update: {actual_vlm_hz:.2f} Hz ({1000/actual_vlm_hz:.0f} ms)")

    # Sensor window calculation
    sensor_sample_rate = 650  # 650Hz
    sensor_window_samples = int(action_expert_period_ms / 1000.0 * sensor_sample_rate)

    print(f"\n  Sensor configuration:")
    print(f"    → Window size: {sensor_window_samples} samples ({action_expert_period_ms:.0f} ms)")
    print(f"    → Sample rate: {sensor_sample_rate} Hz")

    print(f"\n💾 GPU Memory: {results['gpu_memory_gb']:.2f} GB")
    print("="*60 + "\n")

    # Return recommendations as dict
    return {
        "vlm_update_hz": actual_vlm_hz,
        "vlm_update_period_ms": 1000.0 / actual_vlm_hz,
        "vlm_reuse_count": vlm_reuse_count,
        "action_expert_hz": action_expert_hz,
        "sensor_window_samples": sensor_window_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Profile VLM inference speed with multi-view images")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="Qwen-VL model name")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of inference runs for profiling")
    parser.add_argument("--image-paths", type=str, nargs='+', default=None,
                        help="Paths to test images (multi-view, e.g., view1.jpg view2.jpg ...)")
    parser.add_argument("--num-views", type=int, default=5,
                        help="Number of views to use (default: 5)")
    parser.add_argument("--no-flash-attn", action="store_true",
                        help="Disable flash attention")
    parser.add_argument("--save-results", type=str, default=None,
                        help="Path to save results as JSON")

    args = parser.parse_args()

    results = profile_vlm_inference(
        model_name=args.model_name,
        num_samples=args.num_samples,
        image_paths=args.image_paths,
        num_views=args.num_views,
        use_flash_attn=not args.no_flash_attn,
    )

    recommendations = print_results(results)

    # Save results if requested
    if args.save_results:
        import json
        output = {
            "profiling_results": results,
            "recommendations": recommendations,
        }
        with open(args.save_results, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"✅ Results saved to {args.save_results}")


if __name__ == "__main__":
    main()
