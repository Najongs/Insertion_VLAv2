"""
Test impact of image resolution on VLM inference speed

This script resizes images to different resolutions and measures VLM speed
to find the optimal trade-off between speed and quality.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import time
import numpy as np
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def test_resolution_speed(image_paths, target_resolutions, num_samples=20):
    """
    Test VLM speed with different resolutions

    Args:
        image_paths: List of image paths (multi-view)
        target_resolutions: List of (width, height) tuples
        num_samples: Number of test runs per resolution
    """
    print("🚀 Loading Qwen2.5-VL-3B...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda",
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    model.eval()

    print(f"✅ Model loaded")
    print(f"📸 Testing with {len(image_paths)} views")

    results = {}

    for width, height in target_resolutions:
        print(f"\n{'='*60}")
        print(f"Testing resolution: {width}x{height}")
        print(f"{'='*60}")

        # Create temporary resized images
        temp_paths = []
        original_sizes = []

        for img_path in image_paths:
            img = Image.open(img_path)
            original_sizes.append(img.size)

            # Resize image
            resized = img.resize((width, height), Image.LANCZOS)

            # Save to temp file
            temp_path = f"/tmp/temp_resized_{width}x{height}_{Path(img_path).name}"
            resized.save(temp_path, quality=95)
            temp_paths.append(temp_path)

        print(f"Original sizes: {original_sizes[0]}")
        print(f"Resized to: {width}x{height}")
        print(f"Pixel reduction: {original_sizes[0][0]*original_sizes[0][1] / (width*height):.2f}x")

        # Prepare test input
        test_text = "Observe the current state and predict the next action."
        message_content = []
        for temp_path in temp_paths:
            message_content.append({"type": "image", "image": temp_path})
        message_content.append({"type": "text", "text": test_text})

        messages = [{"role": "user", "content": message_content}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        vision_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=vision_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(device="cuda", dtype=torch.bfloat16)

        # Warmup
        print("🔥 Warming up...")
        for _ in range(5):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                _ = model(**inputs, output_hidden_states=True, return_dict=True)

        torch.cuda.synchronize()

        # Actual profiling
        print(f"⏱️  Profiling {num_samples} runs...")
        latencies = []

        for i in range(num_samples):
            torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                _ = model(**inputs, output_hidden_states=True, return_dict=True)

            torch.cuda.synchronize()
            end = time.perf_counter()

            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)

            if (i + 1) % 5 == 0:
                print(f"  [{i+1}/{num_samples}] Current avg: {np.mean(latencies):.2f} ms")

        # Statistics
        latencies = np.array(latencies)
        mean_ms = float(np.mean(latencies))
        std_ms = float(np.std(latencies))

        results[f"{width}x{height}"] = {
            "mean_ms": mean_ms,
            "std_ms": std_ms,
            "mean_hz": 1000.0 / mean_ms,
        }

        print(f"\n📊 Results:")
        print(f"  Mean: {mean_ms:.2f} ms ({1000.0/mean_ms:.2f} Hz)")
        print(f"  Std:  {std_ms:.2f} ms")

        # Cleanup temp files
        import os
        for temp_path in temp_paths:
            try:
                os.remove(temp_path)
            except:
                pass

    return results


def print_comparison(results, original_resolution):
    """Print comparison table"""
    print("\n" + "="*80)
    print("📊 Resolution Comparison Summary")
    print("="*80)

    orig_key = f"{original_resolution[0]}x{original_resolution[1]}"
    if orig_key not in results:
        print(f"⚠️  Original resolution {orig_key} not tested")
        return

    orig_time = results[orig_key]["mean_ms"]

    print(f"\n{'Resolution':<15} {'Time (ms)':<12} {'Speedup':<10} {'Hz':<8} {'Action Expert Reuse':<20}")
    print("-" * 80)

    for res_name, res_data in sorted(results.items()):
        speedup = orig_time / res_data["mean_ms"]
        reuse_count = int(res_data["mean_ms"] / 100)  # 100ms per action expert

        print(f"{res_name:<15} {res_data['mean_ms']:<12.1f} {speedup:<10.2f}x {res_data['mean_hz']:<8.2f} {reuse_count}x")

    print("\n💡 Recommendations:")
    print("-" * 80)

    # Find best trade-offs
    best_speedup = max((k, v["mean_ms"]) for k, v in results.items() if k != orig_key)
    print(f"Best speedup: {best_speedup[0]} → {orig_time/best_speedup[1]:.2f}x faster")

    # Recommend based on reuse count
    for res_name, res_data in sorted(results.items(), key=lambda x: x[1]["mean_ms"]):
        reuse = int(res_data["mean_ms"] / 100)
        if reuse <= 6:  # Reasonable reuse count
            print(f"\n✅ Recommended: {res_name}")
            print(f"   - VLM time: {res_data['mean_ms']:.0f}ms")
            print(f"   - VL feature reuse: {reuse}x")
            print(f"   - Action Expert: 10 Hz (100ms period)")
            break


if __name__ == "__main__":
    import glob

    # Find multi-view images
    dataset_dir = Path("/home/najo/NAS/VLA/dataset/White_silicone_white_circle/recv_all_20251027_170308")

    image_paths = []
    view_dirs = ["View1/left", "View2/left", "View3/left", "View4/left", "View5"]

    for view_dir in view_dirs:
        view_path = dataset_dir / view_dir
        if view_path.exists():
            image_files = sorted(view_path.glob("*.jpg"))
            if image_files:
                image_paths.append(str(image_files[0]))
                if len(image_paths) >= 5:
                    break

    if len(image_paths) < 2:
        print("❌ Not enough images found")
        sys.exit(1)

    print(f"📸 Found {len(image_paths)} views")

    # Test different numbers of views with different resolutions
    num_views_to_test = [2, 3, 4, 5]

    # Resolutions to test
    resolutions = [
        (1280, 720),   # Original (HD)
        (960, 540),    # 75% (3/4 HD)
        (640, 360),    # 50% (nHD)
        (480, 270),    # 37.5%
    ]

    for num_views in num_views_to_test:
        print(f"\n{'#'*80}")
        print(f"# Testing with {num_views} views")
        print(f"{'#'*80}")

        test_images = image_paths[:num_views]
        results = test_resolution_speed(test_images, resolutions, num_samples=20)

        print_comparison(results, (1280, 720))

        # Save results
        import json
        output_file = f"resolution_test_{num_views}views.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to {output_file}")
