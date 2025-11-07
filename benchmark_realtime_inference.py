"""
Real-time Inference Benchmark for VLA Models

Tests inference speed for different model components:
- VL Model (Vision-Language encoding)
- Sensor Encoder
- Action Expert (Regression vs Flow Matching)

Usage:
    # Test both models with default settings
    python benchmark_realtime_inference.py --checkpoint-regression ./checkpoints/regression_best.pt --checkpoint-flow ./checkpoints/flow_matching_best.pt

    # Test with different configurations
    python benchmark_realtime_inference.py \
        --checkpoint-regression ./checkpoints/regression_best.pt \
        --num-iterations 20 \
        --num-views 3 \
        --disable-sensor

    # Compare different view counts
    python benchmark_realtime_inference.py \
        --checkpoint-regression ./checkpoints/regression_best.pt \
        --compare-views
"""

import argparse
import time
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm
import pandas as pd

# Add project root to path
import sys
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.unified_model import QwenVLAUnified
from vla_datasets.unified_dataset import UnifiedVLADataset
from transformers import AutoProcessor


@dataclass
class BenchmarkConfig:
    """Benchmark configuration"""
    num_iterations: int = 10
    num_views: int = 5  # Number of camera views to use (1-5)
    enable_sensor: bool = True
    enable_robot_states: bool = True
    horizon: int = 8
    warmup_iterations: int = 3
    device: str = "cuda:0"


@dataclass
class TimingResult:
    """Timing results for a single iteration"""
    vl_encoding_time: float  # Vision-Language encoding
    sensor_encoding_time: float  # Sensor encoding (if enabled)
    action_prediction_time: float  # Action expert (regression/flow)
    total_time: float  # End-to-end

    def to_dict(self):
        return asdict(self)


class ModelBenchmark:
    """Benchmark for VLA model inference"""

    def __init__(
        self,
        model: QwenVLAUnified,
        processor: AutoProcessor,
        config: BenchmarkConfig,
        model_name: str = "model"
    ):
        self.model = model
        self.processor = processor
        self.config = config
        self.model_name = model_name
        self.device = torch.device(config.device)

        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()

        # ✅ IMPORTANT: Disable VL cache for real-time inference benchmark
        self.model.cache_enabled = False
        print(f"⚠️ VL cache DISABLED for real-time inference benchmark")

        # Disable gradients for inference
        for param in self.model.parameters():
            param.requires_grad = False

    def prepare_sample(
        self,
        dataset: UnifiedVLADataset,
        sample_idx: int = 0
    ) -> Dict:
        """Prepare a sample for inference"""
        sample = dataset[sample_idx]

        # Get vlm_idx from sample
        vlm_idx = sample.get('vlm_idx', 0)

        # Load image paths from dataset.images dict
        image_paths = []
        if hasattr(dataset, 'images') and isinstance(dataset.images, dict):
            for view_name in sorted(dataset.images.keys()):
                view_images = dataset.images[view_name]
                if len(view_images) > 0:
                    img_idx = min(vlm_idx, len(view_images) - 1)
                    image_paths.append(view_images[img_idx])

        sample["images"] = image_paths if image_paths else None

        if sample["images"] is None or len(sample["images"]) == 0:
            print(f"⚠️ Warning: Could not find images in {dataset.data_dir}")
            print(f"   vlm_idx: {vlm_idx}")
            print(f"   Available camera views: {list(dataset.images.keys()) if hasattr(dataset, 'images') else 'N/A'}")
            print(f"   Will use text-only VL encoding")

        # Limit number of views
        if isinstance(sample["images"], list):
            sample["images"] = sample["images"][:self.config.num_views]

        # Disable sensor if needed
        if not self.config.enable_sensor:
            if "sensor_data" in sample:
                sample["sensor_data"] = torch.zeros_like(sample["sensor_data"])
            sample["has_sensor"] = False

        # Disable robot states if needed
        if not self.config.enable_robot_states:
            if "robot_states" in sample:
                sample["robot_states"] = torch.zeros_like(sample["robot_states"])
            sample["has_robot_states"] = False

        return sample

    @torch.no_grad() #나중에 View별로 주는거 좋다.
    def benchmark_vl_encoding(self, text_input: str, image_paths: List[str]):
        """Benchmark VL encoding using model's internal path (supports parallel view encoding)"""

        # 입력 형식 정리
        text_inputs = [text_input]
        image_inputs = [image_paths if image_paths else None]
        cache_keys = ["bench_0"]

        # 캐시 비활성화 (실시간 추론)
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        # ✅ 여기서 model._encode_vision_features() 호출
        vl_features = self.model._encode_vision_features(
            text_inputs=text_inputs,
            image_inputs=image_inputs,
            cache_keys=cache_keys,
            use_cache=False,           # 실시간이므로 캐시 사용 안 함
            device=self.device
        )

        torch.cuda.synchronize()
        elapsed_time = time.perf_counter() - start_time

        return vl_features, elapsed_time

    
    @torch.no_grad()
    def benchmark_sensor_encoding(
        self,
        sensor_data: torch.Tensor,
        robot_states: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], float]:
        """Benchmark sensor + robot state encoding time"""

        sensor_features = None
        robot_state_features = None

        # Add batch dimension if needed
        if sensor_data.ndim == 2:
            sensor_data = sensor_data.unsqueeze(0)
        if robot_states.ndim == 2:
            robot_states = robot_states.unsqueeze(0)

        # Move to device
        sensor_data = sensor_data.to(self.device, dtype=torch.bfloat16)
        robot_states = robot_states.to(self.device, dtype=torch.bfloat16)

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Encode sensor data
            if self.config.enable_sensor and self.model.sensor_enabled and sensor_data is not None:
                sensor_features = self.model.sensor_encoder(sensor_data)

            # Encode robot states
            if self.config.enable_robot_states and self.model.robot_state_enabled and robot_states is not None:
                robot_state_features = self.model.robot_state_encoder(robot_states)

        torch.cuda.synchronize()
        total_time = time.perf_counter() - start_time

        return sensor_features, robot_state_features, total_time

    @torch.no_grad()
    def benchmark_action_prediction(
        self,
        vl_features: torch.Tensor,
        sensor_features: Optional[torch.Tensor],
        robot_state_features: Optional[torch.Tensor],
        gt_actions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, float]:
        """Benchmark action prediction time"""

        # Prepare features
        batch_size = vl_features.shape[0]

        # Add batch dimension to gt_actions if needed
        if gt_actions is not None:
            gt_actions = gt_actions.to(self.device, dtype=torch.bfloat16)
            if gt_actions.ndim == 2:
                gt_actions = gt_actions.unsqueeze(0)
        else:
            # Create dummy actions for flow matching
            gt_actions = torch.zeros(
                batch_size, self.config.horizon, 7,
                device=self.device, dtype=torch.bfloat16
            )

        # Benchmark action prediction
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if self.model.model_type == "flow_matching":
                # Flow matching inference (sample trajectory)
                actions = self.model.action_expert.sample(
                    vl_features,
                    sensor_features,
                    robot_state_features,
                    batch_size=batch_size,
                    num_steps=self.model.flow_steps,
                    method=self.model.flow_solver
                )
            else:
                # Regression inference
                actions, _ = self.model.action_expert(
                    vl_features,
                    gt_actions,  # z_chunk for regression
                    sensor_features,
                    robot_state_features
                )

        torch.cuda.synchronize()
        elapsed_time = time.perf_counter() - start_time

        return actions, elapsed_time

    @torch.no_grad()
    def benchmark_end_to_end(
        self,
        sample: Dict
    ) -> TimingResult:
        """Benchmark end-to-end inference"""

        # Extract inputs
        text_input = sample["instruction"]
        image_paths = sample["images"]
        sensor_data = sample["sensor_data"]
        robot_states = sample["robot_states"]
        gt_actions = sample.get("actions", None)

        # 1. VL Encoding
        vl_features, vl_time = self.benchmark_vl_encoding(text_input, image_paths)

        # 2. Sensor + Robot State Encoding
        sensor_features, robot_state_features, sensor_time = self.benchmark_sensor_encoding(
            sensor_data, robot_states
        )

        # 3. Action Prediction
        actions, action_time = self.benchmark_action_prediction(
            vl_features, sensor_features, robot_state_features, gt_actions
        )

        # Total time
        total_time = vl_time + sensor_time + action_time

        return TimingResult(
            vl_encoding_time=vl_time,
            sensor_encoding_time=sensor_time,
            action_prediction_time=action_time,
            total_time=total_time
        )

    def run_benchmark(
        self,
        dataset: UnifiedVLADataset,
        sample_idx: int = 0
    ) -> Dict:
        """Run full benchmark with multiple iterations"""

        print(f"\n{'='*60}")
        print(f"Benchmarking: {self.model_name}")
        print(f"Model Type: {self.model.model_type}")
        print(f"VL Cache: {'DISABLED' if not self.model.cache_enabled else 'ENABLED'} (Real-time inference)")
        print(f"Sensor: {'ENABLED' if self.config.enable_sensor else 'DISABLED'}")
        print(f"Robot States: {'ENABLED' if self.config.enable_robot_states else 'DISABLED'}")
        print(f"Views: {self.config.num_views}")
        print(f"Iterations: {self.config.num_iterations}")
        print(f"{'='*60}\n")

        # Prepare sample
        sample = self.prepare_sample(dataset, sample_idx)

        # Warmup
        print(f"Warming up ({self.config.warmup_iterations} iterations)...")
        for _ in range(self.config.warmup_iterations):
            self.benchmark_end_to_end(sample)

        # Benchmark
        print(f"Running benchmark ({self.config.num_iterations} iterations)...")
        results = []

        for _ in tqdm(range(self.config.num_iterations), desc="Benchmarking"):
            result = self.benchmark_end_to_end(sample)
            results.append(result)

        # Compute statistics
        vl_times = [r.vl_encoding_time for r in results]
        sensor_times = [r.sensor_encoding_time for r in results]
        action_times = [r.action_prediction_time for r in results]
        total_times = [r.total_time for r in results]

        stats = {
            "model_name": self.model_name,
            "model_type": self.model.model_type,
            "config": asdict(self.config),
            "vl_encoding": {
                "mean": np.mean(vl_times),
                "std": np.std(vl_times),
                "min": np.min(vl_times),
                "max": np.max(vl_times),
            },
            "sensor_encoding": {
                "mean": np.mean(sensor_times),
                "std": np.std(sensor_times),
                "min": np.min(sensor_times),
                "max": np.max(sensor_times),
            },
            "action_prediction": {
                "mean": np.mean(action_times),
                "std": np.std(action_times),
                "min": np.min(action_times),
                "max": np.max(action_times),
            },
            "total": {
                "mean": np.mean(total_times),
                "std": np.std(total_times),
                "min": np.min(total_times),
                "max": np.max(total_times),
                "fps": 1.0 / np.mean(total_times),
            },
            "raw_results": [r.to_dict() for r in results]
        }

        return stats


def print_results(stats: Dict):
    """Print benchmark results in a formatted way"""
    print(f"\n{'='*60}")
    print(f"Results: {stats['model_name']} ({stats['model_type']})")
    print(f"{'='*60}")

    print(f"\n📊 Timing Breakdown:")
    print(f"  VL Encoding:       {stats['vl_encoding']['mean']*1000:.2f} ± {stats['vl_encoding']['std']*1000:.2f} ms")
    print(f"  Sensor Encoding:   {stats['sensor_encoding']['mean']*1000:.2f} ± {stats['sensor_encoding']['std']*1000:.2f} ms")
    print(f"  Action Prediction: {stats['action_prediction']['mean']*1000:.2f} ± {stats['action_prediction']['std']*1000:.2f} ms")
    print(f"  {'─'*40}")
    print(f"  Total (E2E):       {stats['total']['mean']*1000:.2f} ± {stats['total']['std']*1000:.2f} ms")
    print(f"  Throughput:        {stats['total']['fps']:.2f} FPS")

    print(f"\n📈 Component Breakdown:")
    total_mean = stats['total']['mean']
    vl_pct = (stats['vl_encoding']['mean'] / total_mean) * 100
    sensor_pct = (stats['sensor_encoding']['mean'] / total_mean) * 100
    action_pct = (stats['action_prediction']['mean'] / total_mean) * 100

    print(f"  VL Encoding:       {vl_pct:.1f}%")
    print(f"  Sensor Encoding:   {sensor_pct:.1f}%")
    print(f"  Action Prediction: {action_pct:.1f}%")


def compare_models(
    stats_list: List[Dict],
    output_dir: Path
):
    """Compare multiple model benchmarks"""

    print(f"\n{'='*60}")
    print("Model Comparison")
    print(f"{'='*60}\n")

    # Create comparison table
    rows = []
    for stats in stats_list:
        rows.append({
            "Model": stats['model_name'],
            "Type": stats['model_type'],
            "VL (ms)": f"{stats['vl_encoding']['mean']*1000:.2f}",
            "Sensor (ms)": f"{stats['sensor_encoding']['mean']*1000:.2f}",
            "Action (ms)": f"{stats['action_prediction']['mean']*1000:.2f}",
            "Total (ms)": f"{stats['total']['mean']*1000:.2f}",
            "FPS": f"{stats['total']['fps']:.2f}",
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # Save comparison
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "comparison.csv", index=False)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Timing breakdown
    ax = axes[0]
    model_names = [s['model_name'] for s in stats_list]
    vl_times = [s['vl_encoding']['mean'] * 1000 for s in stats_list]
    sensor_times = [s['sensor_encoding']['mean'] * 1000 for s in stats_list]
    action_times = [s['action_prediction']['mean'] * 1000 for s in stats_list]

    x = np.arange(len(model_names))
    width = 0.25

    ax.bar(x - width, vl_times, width, label='VL Encoding')
    ax.bar(x, sensor_times, width, label='Sensor Encoding')
    ax.bar(x + width, action_times, width, label='Action Prediction')

    ax.set_xlabel('Model')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Inference Time Breakdown')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Total time comparison
    ax = axes[1]
    total_times = [s['total']['mean'] * 1000 for s in stats_list]
    fps_values = [s['total']['fps'] for s in stats_list]

    ax2 = ax.twinx()
    bars = ax.bar(x, total_times, color='steelblue', alpha=0.7)
    line = ax2.plot(x, fps_values, 'ro-', linewidth=2, markersize=8, label='FPS')

    ax.set_xlabel('Model')
    ax.set_ylabel('Total Time (ms)', color='steelblue')
    ax2.set_ylabel('Throughput (FPS)', color='red')
    ax.set_title('End-to-End Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.tick_params(axis='y', labelcolor='steelblue')
    ax2.tick_params(axis='y', labelcolor='red')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "comparison.png", dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved comparison plot to {output_dir / 'comparison.png'}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark VLA model inference')

    # Model checkpoints
    parser.add_argument('--checkpoint-regression', type=str, help='Path to regression checkpoint')
    parser.add_argument('--checkpoint-flow', type=str, help='Path to flow matching checkpoint')

    # Dataset
    parser.add_argument('--dataset-dir', type=str,
                       default='/home/najo/NAS/VLA/dataset/New_dataset/Blue_point/episode_20251030_025856',
                       help='Path to test dataset')
    parser.add_argument('--sample-idx', type=int, default=0, help='Sample index to use')

    # Benchmark config
    parser.add_argument('--num-iterations', type=int, default=10, help='Number of benchmark iterations')
    parser.add_argument('--num-views', type=int, default=5, help='Number of camera views (1-5)')
    parser.add_argument('--disable-sensor', action='store_true', help='Disable sensor input')
    parser.add_argument('--disable-robot-states', action='store_true', help='Disable robot states')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')

    # VL optimization options
    parser.add_argument('--parallel-view-encoding', action='store_true',
                       help='Enable parallel multi-view encoding (2-3x speedup expected)')
    parser.add_argument('--view-aggregation', type=str, default='mean', choices=['mean', 'max', 'attention'],
                       help='View aggregation strategy for parallel encoding')

    # Comparison modes
    parser.add_argument('--compare-views', action='store_true',
                       help='Compare performance with different view counts')
    parser.add_argument('--compare-sensors', action='store_true',
                       help='Compare performance with/without sensors')
    parser.add_argument('--compare-parallel', action='store_true',
                       help='Compare sequential vs parallel view encoding')

    # Output
    parser.add_argument('--output-dir', type=str, default='./benchmark_results',
                       help='Output directory for results')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {args.dataset_dir}...")
    dataset = UnifiedVLADataset(
        data_dir=args.dataset_dir,
        format='auto',
        horizon=8,
        vlm_reuse_count=1,  # No caching for real-time test
        sensor_window_size=65,
    )

    print(f"Dataset loaded: {len(dataset)} samples")

    # Load processor
    vl_model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    processor = AutoProcessor.from_pretrained(vl_model_name)

    # Base config
    base_config = BenchmarkConfig(
        num_iterations=args.num_iterations,
        num_views=args.num_views,
        enable_sensor=not args.disable_sensor,
        enable_robot_states=not args.disable_robot_states,
        device=args.device,
    )

    all_stats = []

    # Benchmark regression model
    if args.checkpoint_regression:
        print(f"\n{'='*60}")
        print("Loading Regression Model")
        print(f"{'='*60}")

        model_reg = QwenVLAUnified(
            model_type='regression',
            vl_model_name=vl_model_name,
            action_dim=7,
            horizon=8,
            hidden_dim=1024,
            sensor_enabled=not args.disable_sensor,  # ✅ Respect --disable-sensor flag
            sensor_input_channels=1026,
            sensor_temporal_length=65,  # Robot state encoder uses this value
            sensor_output_dim=3072,  # Must match trained model
            robot_state_enabled=not args.disable_robot_states,  # ✅ Respect --disable-robot-states flag
            fusion_strategy='concat',
            finetune_vl='none',
            image_resize_height=360,
            image_resize_width=640,
            parallel_view_encoding=args.parallel_view_encoding,  # ⚡ VL optimization
            view_aggregation=args.view_aggregation,
            device_map=None,
        )

        # Load checkpoint (with compatibility handling)
        # Note: We skip checkpoint loading for benchmarking as we only care about speed
        # If you want to load checkpoint, ensure it was trained with same config
        print(f"⚠️ Using randomly initialized model for benchmarking")
        print(f"   (Checkpoint loading skipped - only speed matters for benchmarking)")

        # Uncomment below to try loading checkpoint (may fail due to architecture changes)
        # try:
        #     ckpt = torch.load(args.checkpoint_regression, map_location='cpu')
        #     model_reg.load_state_dict(ckpt['model_state_dict'], strict=False)
        #     print(f"✅ Loaded checkpoint from {args.checkpoint_regression}")
        # except Exception as e:
        #     print(f"⚠️ Could not load checkpoint: {e}")


        benchmark_reg = ModelBenchmark(
            model_reg, processor, base_config, model_name="Regression"
        )

        stats_reg = benchmark_reg.run_benchmark(dataset, args.sample_idx)
        print_results(stats_reg)
        all_stats.append(stats_reg)

        # Save results
        with open(output_dir / "regression_results.json", 'w') as f:
            json.dump(stats_reg, f, indent=2)

    # Benchmark flow matching model
    if args.checkpoint_flow:
        print(f"\n{'='*60}")
        print("Loading Flow Matching Model")
        print(f"{'='*60}")

        model_flow = QwenVLAUnified(
            model_type='flow_matching',
            vl_model_name=vl_model_name,
            action_dim=7,
            horizon=8,
            hidden_dim=1024,
            sensor_enabled=not args.disable_sensor,  # ✅ Respect --disable-sensor flag
            sensor_input_channels=1026,
            sensor_temporal_length=65,  # Robot state encoder uses this value
            sensor_output_dim=3072,  # Must match trained model
            robot_state_enabled=not args.disable_robot_states,  # ✅ Respect --disable-robot-states flag
            fusion_strategy='concat',
            finetune_vl='none',
            image_resize_height=360,
            image_resize_width=640,
            parallel_view_encoding=args.parallel_view_encoding,  # ⚡ VL optimization
            view_aggregation=args.view_aggregation,
            device_map=None,
        )

        # Load checkpoint (with compatibility handling)
        # Note: We skip checkpoint loading for benchmarking as we only care about speed
        # If you want to load checkpoint, ensure it was trained with same config
        print(f"⚠️ Using randomly initialized model for benchmarking")
        print(f"   (Checkpoint loading skipped - only speed matters for benchmarking)")

        # Uncomment below to try loading checkpoint (may fail due to architecture changes)
        # try:
        #     ckpt = torch.load(args.checkpoint_flow, map_location='cpu')
        #     model_flow.load_state_dict(ckpt['model_state_dict'], strict=False)
        #     print(f"✅ Loaded checkpoint from {args.checkpoint_flow}")
        # except Exception as e:
        #     print(f"⚠️ Could not load checkpoint: {e}")

        benchmark_flow = ModelBenchmark(
            model_flow, processor, base_config, model_name="Flow Matching"
        )

        stats_flow = benchmark_flow.run_benchmark(dataset, args.sample_idx)
        print_results(stats_flow)
        all_stats.append(stats_flow)

        # Save results
        with open(output_dir / "flow_matching_results.json", 'w') as f:
            json.dump(stats_flow, f, indent=2)

    # Compare models
    if len(all_stats) > 1:
        compare_models(all_stats, output_dir)

    # Additional comparisons
    if args.compare_views and args.checkpoint_regression:
        print(f"\n{'='*60}")
        print("Comparing Different View Counts")
        print(f"{'='*60}")

        view_stats = []
        for num_views in [1, 2, 3, 4, 5]:
            config = BenchmarkConfig(
                num_iterations=args.num_iterations,
                num_views=num_views,
                enable_sensor=not args.disable_sensor,
                enable_robot_states=not args.disable_robot_states,
                device=args.device,
            )

            benchmark = ModelBenchmark(
                model_reg, processor, config,
                model_name=f"Regression ({num_views} views)"
            )

            stats = benchmark.run_benchmark(dataset, args.sample_idx)
            print_results(stats)
            view_stats.append(stats)

        compare_models(view_stats, output_dir / "view_comparison")

    if args.compare_sensors and args.checkpoint_regression:
        print(f"\n{'='*60}")
        print("Comparing With/Without Sensors")
        print(f"{'='*60}")

        sensor_stats = []
        for enable_sensor in [True, False]:
            # ✅ CRITICAL: Reinitialize model with correct sensor setting
            print(f"\n  Initializing model with sensor_enabled={enable_sensor}...")

            model_temp = QwenVLAUnified(
                model_type='regression',
                vl_model_name=vl_model_name,
                action_dim=7,
                horizon=8,
                hidden_dim=1024,
                sensor_enabled=enable_sensor,  # ✅ Match config
                sensor_input_channels=1026,
                sensor_temporal_length=65,
                sensor_output_dim=3072,
                robot_state_enabled=not args.disable_robot_states,
                fusion_strategy='concat',
                finetune_vl='none',
                image_resize_height=360,
                image_resize_width=640,
                device_map=None,
            )
            print(f"  Model initialized with sensor_enabled={enable_sensor}")

            config = BenchmarkConfig(
                num_iterations=args.num_iterations,
                num_views=args.num_views,
                enable_sensor=enable_sensor,
                enable_robot_states=not args.disable_robot_states,
                device=args.device,
            )

            benchmark = ModelBenchmark(
                model_temp, processor, config,
                model_name=f"Regression ({'with' if enable_sensor else 'without'} sensor)"
            )

            stats = benchmark.run_benchmark(dataset, args.sample_idx)
            print_results(stats)
            sensor_stats.append(stats)

        compare_models(sensor_stats, output_dir / "sensor_comparison")

    # Compare parallel vs sequential encoding
    if args.compare_parallel and args.checkpoint_regression:
        print(f"\n{'='*60}")
        print("Comparing Sequential vs Parallel View Encoding")
        print(f"{'='*60}")

        parallel_stats = []
        for use_parallel in [False, True]:
            # ✅ CRITICAL: Reinitialize model with parallel encoding setting
            print(f"\n  Initializing model with parallel_view_encoding={use_parallel}...")

            model_temp = QwenVLAUnified(
                model_type='regression',
                vl_model_name=vl_model_name,
                action_dim=7,
                horizon=8,
                hidden_dim=1024,
                sensor_enabled=not args.disable_sensor,
                sensor_input_channels=1026,
                sensor_temporal_length=65,
                sensor_output_dim=3072,
                robot_state_enabled=not args.disable_robot_states,
                fusion_strategy='concat',
                finetune_vl='none',
                image_resize_height=360,
                image_resize_width=640,
                parallel_view_encoding=use_parallel,  # ⚡ Toggle parallel encoding
                view_aggregation=args.view_aggregation,
                device_map=None,
            )
            print(f"  Model initialized with parallel_view_encoding={use_parallel}")

            config = BenchmarkConfig(
                num_iterations=args.num_iterations,
                num_views=args.num_views,
                enable_sensor=not args.disable_sensor,
                enable_robot_states=not args.disable_robot_states,
                device=args.device,
            )

            benchmark = ModelBenchmark(
                model_temp, processor, config,
                model_name=f"Regression ({'Parallel' if use_parallel else 'Sequential'} Encoding)"
            )

            stats = benchmark.run_benchmark(dataset, args.sample_idx)
            print_results(stats)
            parallel_stats.append(stats)

        compare_models(parallel_stats, output_dir / "parallel_comparison")

    print(f"\n{'='*60}")
    print(f"✅ Benchmark complete! Results saved to {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
