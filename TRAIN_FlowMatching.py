"""
Flow Matching VLA Training Script with Sensor Integration

Specialized training script for flow matching-based action prediction.
Based on Pi0 paper: https://arxiv.org/pdf/2410.24164v1

Usage:
    # Build cache first
    torchrun --nproc_per_node=4 TRAIN_FlowMatching.py --mode cache

    # Then train
    torchrun --nproc_per_node=4 TRAIN_FlowMatching.py --mode train
"""

from pydantic import PydanticDeprecatedSince20
import warnings
warnings.filterwarnings("ignore", message=".*cudnnException.*")
warnings.filterwarnings("ignore", message=".*Deterministic behavior.*")
warnings.filterwarnings("ignore", message=".*Flash Attention.*")

import argparse
import wandb
import io, shutil, threading, queue, time
import os
import sys
import re
import math
import glob
import pickle
import atexit
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from contextlib import nullcontext

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset, DataLoader, random_split
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler, Subset
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR

# Set seeds
import random
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ✅ OPTIMIZATION: Enable cudnn.benchmark for faster training (non-deterministic)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.use_deterministic_algorithms(False, warn_only=True)
torch.set_float32_matmul_precision("high")

# Import unified models and datasets
from models.unified_model import QwenVLAUnified
from vla_datasets.unified_dataset import (
    UnifiedVLADataset,
    create_unified_dataloader,
    unified_collate_fn,
)

# Import cache builder
import importlib.util
cache_module_path = Path(__file__).parent / "Make_VL_cache.py"
spec = importlib.util.spec_from_file_location("Make_VL_cache", cache_module_path)
cache_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache_module)
build_vl_cache_distributed_optimized = cache_module.build_vl_cache_distributed_optimized

# ======== I/O & Checkpoint Utils ========
STAGING_DIR = Path("/home/najo/NAS/VLA/tmp_stage")
CKPT_DIR = Path("./checkpoints")
STAGING_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

def _atomic_move(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.move(src, tmp)
    os.replace(tmp, dst)

def copy_to_local_then_load(src_path: Path, map_location):
    """네트워크 파일을 로컬 스테이징으로 빠르게 복사 후 torch.load"""
    if not src_path.exists():
        raise FileNotFoundError(str(src_path))
    local_copy = STAGING_DIR / src_path.name
    shutil.copy2(src_path, local_copy)
    try:
        return torch.load(local_copy, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(local_copy, map_location=map_location)

class AsyncCheckpointWriter:
    """학습은 그대로 진행, 저장은 백그라운드 스레드가 처리"""
    def __init__(self, max_queue=2, sync_every=0):
        self.q = queue.Queue(maxsize=max_queue)
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.stop = False
        self.sync_every = sync_every
        self.thread.start()

    def _worker(self):
        last_sync = time.time()
        while not self.stop:
            try:
                payload = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            state_dict, final_dst = payload["state"], Path(payload["dst"])
            local_tmp = STAGING_DIR / (final_dst.name + f".{int(time.time())}.pt")
            torch.save(state_dict, local_tmp, _use_new_zipfile_serialization=True)
            if self.sync_every > 0 and (time.time() - last_sync) < self.sync_every:
                continue
            _atomic_move(local_tmp, final_dst)
            last_sync = time.time()

    def submit(self, state_dict, final_dst: Path):
        if self.q.full():
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
        self.q.put({"state": state_dict, "dst": str(final_dst)})

    def close(self):
        self.stop = True
        self.thread.join(timeout=5)

def build_trapezoid_scheduler(
    optimizer,
    total_steps: int,
    *,
    base_lr: float = 1e-4,
    min_lr: float = 1e-6,
    warmup_ratio: float = 0.03,
    hold_ratio: float = 0.02,
):
    """LLM 스타일: Warmup -> Hold -> Cosine Decay"""
    warmup_steps = int(total_steps * warmup_ratio)
    hold_steps = int(total_steps * hold_ratio)
    decay_steps = max(1, total_steps - warmup_steps - hold_steps)
    floor = min_lr / max(base_lr, 1e-12)

    def lr_lambda(step: int):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        elif step < warmup_steps + hold_steps:
            return 1.0
        else:
            t = (step - warmup_steps - hold_steps) / decay_steps
            cos_val = 0.5 * (1.0 + math.cos(math.pi * t))
            return floor + (1.0 - floor) * cos_val

    sched = LambdaLR(optimizer, lr_lambda=lr_lambda)
    prev_lr = base_lr * lr_lambda(0)
    for g in optimizer.param_groups:
        g["lr"] = prev_lr
    return sched

# ===========================================================
# 초기화
# ===========================================================
def setup_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # torchrun에서 LOCAL_RANK는 프로세스별 GPU ID임
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    print(f"[Rank {rank}] using device {device}")
    return rank, world_size, local_rank, device

# ============================================================
# Unified Dataloader Builder
# ============================================================
def build_dataloaders(args, rank, world_size, use_cache=True):
    """
    Build unified dataloaders combining:
      ① Old format datasets
      ② New format datasets
    """
    if rank == 0:
        print(f"[RANK {rank}] 🚀 Building Unified Async Dataloaders (world_size={world_size})")

    # Dataset directory patterns
    old_priority_patterns = []
    old_regular_patterns = []
    new_dataset_root = "/home/najo/NAS/VLA/dataset/New_dataset"

    # Weight configuration
    old_dataset_weight = getattr(args, "old_dataset_weight", 1.0)
    new_dataset_weight = getattr(args, "new_dataset_weight", 3.0)

    # Build TRAIN dataloader
    print("\n📦 Creating TRAIN dataloader (weighted mix of old/new)...")

    train_loader = create_unified_dataloader(
        old_dataset_patterns=old_priority_patterns + old_regular_patterns,
        new_dataset_path=new_dataset_root,
        old_weight=old_dataset_weight,
        new_weight=new_dataset_weight,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        horizon=args.horizon if hasattr(args, "horizon") else 8,
        vlm_reuse_count=args.vlm_reuse_count if hasattr(args, "vlm_reuse_count") else 3,
        sensor_window_size=args.sensor_window_size if hasattr(args, "sensor_window_size") else 65,
        action_expert_hz=args.action_expert_hz if hasattr(args, "action_expert_hz") else 10,
        distributed=True,
        rank=rank,
        world_size=world_size,
        use_cache=use_cache,  # Pass the flag here
    )

    # Build VAL dataloader
    print("\n📦 Creating VAL dataloader (validation subset)...")

    val_patterns = old_regular_patterns[:1]
    val_datasets = []
    for pattern in val_patterns:
        for traj_dir in sorted(glob.glob(pattern)):
            try:
                ds = UnifiedVLADataset(
                    data_dir=str(traj_dir),
                    format='old',
                    horizon=args.horizon if hasattr(args, "horizon") else 8,
                    vlm_reuse_count=args.vlm_reuse_count if hasattr(args, "vlm_reuse_count") else 3,
                    sensor_window_size=args.sensor_window_size if hasattr(args, "sensor_window_size") else 65,
                    action_expert_hz=args.action_expert_hz if hasattr(args, "action_expert_hz") else 10,
                )
                val_datasets.append(ds)
            except Exception as e:
                print(f"⚠️ Skipped val dataset {traj_dir}: {e}")

    from torch.utils.data import ConcatDataset
    if len(val_datasets) == 0:
        print("⚠️ No validation datasets found, using train subset instead.")
        val_datasets = [next(iter(train_loader.dataset.datasets))]

    val_dataset = ConcatDataset(val_datasets)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=val_sampler,
        collate_fn=unified_collate_fn,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    if rank == 0:
        print(f"✅ TRAIN loader: {len(train_loader)} batches | VAL loader: {len(val_loader)} batches")
        print(f"   Old dataset weight={old_dataset_weight}, New dataset weight={new_dataset_weight}")

    return train_loader, val_loader

# ===========================================================
# Flow Matching 학습 루프
# ===========================================================
def Train_FlowMatching(
    model,
    data_loader,
    optimizer,
    num_epochs=3,
    grad_accum_steps=8,
    device="cuda",
    scheduler=None,
    sched_on="step",
    val_loader=None,
    start_epoch=0,
    sensor_enabled=True,
    sensor_loss_weight=2.0,
):
    """Flow Matching training loop - OPTIMIZED"""
    rank = dist.get_rank()
    writer = AsyncCheckpointWriter(max_queue=2, sync_every=0) if rank == 0 else None

    model.train()
    if rank == 0:
        wandb.init(
            project="QwenVLA-FlowMatching",
            name=f"flow_matching_{time.strftime('%m%d_%H%M')}",
            resume="allow",
            id=f"qvla_flow_matching_{int(time.time())}",
            settings=wandb.Settings(start_method="thread", _disable_stats=True),
            config={
                "model_type": "flow_matching",
                "lr": optimizer.param_groups[0]["lr"],
                "grad_accum_steps": grad_accum_steps,
                "epochs": num_epochs,
                "scheduler": sched_on,
                "sensor_enabled": sensor_enabled,
                "sensor_loss_weight": sensor_loss_weight,
            }
        )

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(start_epoch, start_epoch + num_epochs):
        if isinstance(data_loader.sampler, DistributedSampler):
            data_loader.sampler.set_epoch(epoch)

        total_loss = 0.0
        total_sensor_samples = 0
        total_nonsensor_samples = 0

        optimizer.zero_grad(set_to_none=True)
        model.train()

        pbar = tqdm(enumerate(data_loader), total=len(data_loader),
                    desc=f"[Rank {rank}] Epoch {epoch+1}",
                    disable=(rank != 0))

        for step, batch in pbar:
            try:
                instructions = batch["instruction"]
                image_inputs = batch["images"]
                gt_actions = batch["actions"].to(device, dtype=torch.bfloat16, non_blocking=True)

                # ✅ FLOW MATCHING: Use all actions (B, 8, 7)
                sensor_data = (
                    batch["sensor_data"].to(device, dtype=torch.bfloat16, non_blocking=True)
                    if sensor_enabled else None
                )
                has_sensor_mask = (
                    batch["has_sensor_mask"].to(device, non_blocking=True)
                    if sensor_enabled else None
                )

                # Robot states
                robot_states = None
                if "robot_states" in batch and sensor_enabled:
                    try:
                        robot_states = batch["robot_states"].to(device, non_blocking=True)
                    except Exception as e:
                        if rank == 0 and step == 0:
                            print(f"⚠️ Failed to load robot_states: {e}")
                        robot_states = None

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # Flow matching: model returns loss directly
                    flow_loss, _, _ = model(
                        text_inputs=instructions,
                        image_inputs=image_inputs,
                        actions=gt_actions,  # ✅ Use 'actions' parameter
                        cache_keys=batch["cache_keys"],
                        sensor_data=sensor_data if sensor_enabled else None,
                        robot_states=robot_states,
                        vl_cache_tokens=batch.get("vl_cache"),
                    )

                    # ✅ Count sensor samples for logging
                    if sensor_enabled and has_sensor_mask is not None:
                        total_sensor_samples += has_sensor_mask.sum().item()
                        total_nonsensor_samples += (~has_sensor_mask).sum().item()

                    loss = flow_loss / grad_accum_steps

                sync_context = model.no_sync() if (step + 1) % grad_accum_steps != 0 else nullcontext()
                with sync_context:
                    loss.backward()

                total_loss += loss.item() * grad_accum_steps

                if (step + 1) % grad_accum_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    if scheduler is not None and sched_on == "step":
                        scheduler.step()

                    global_step += 1

                    lr = optimizer.param_groups[0]["lr"]
                    if rank == 0:
                        postfix_dict = {
                            "loss": f"{loss.item() * grad_accum_steps:.6f}",
                            "lr": f"{lr:.2e}",
                            "grad": f"{grad_norm:.2f}"
                        }
                        if sensor_enabled:
                            postfix_dict["sensor"] = f"{total_sensor_samples}/{total_sensor_samples+total_nonsensor_samples}"
                        pbar.set_postfix(postfix_dict)

                        log_dict = {
                            "train/loss_step": loss.item() * grad_accum_steps,
                            "train/lr": lr,
                            "train/grad_norm": grad_norm,
                            "global_step": global_step
                        }
                        if sensor_enabled:
                            log_dict["train/sensor_samples"] = total_sensor_samples
                            log_dict["train/nonsensor_samples"] = total_nonsensor_samples
                        wandb.log(log_dict)

            except FileNotFoundError as e:
                if rank == 0:
                    pbar.write(f"⚠️ [Rank {rank}] 캐시 파일 없음, Batch {step} 스킵. (오류: {e})")

                if (step + 1) % grad_accum_steps == 0:
                    optimizer.zero_grad(set_to_none=True)
                continue

        avg_loss_tensor = torch.tensor(total_loss / len(data_loader), device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = avg_loss_tensor.item()

        if scheduler is not None and sched_on == "epoch":
            scheduler.step()

        # Validation
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_loss_sum, val_count = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    try:
                        gt_actions = batch["actions"].to(device, dtype=torch.bfloat16, non_blocking=True)

                        sensor_data = (
                            batch["sensor_data"].to(device, dtype=torch.bfloat16, non_blocking=True)
                            if sensor_enabled else None
                        )
                        robot_states = (
                            batch["robot_states"].to(device, non_blocking=True)
                            if "robot_states" in batch and sensor_enabled else None
                        )

                        # Flow matching: model returns loss directly
                        loss, _, _ = model(
                            text_inputs=batch["instruction"],
                            image_inputs=batch["images"],
                            actions=gt_actions,
                            cache_keys=batch["cache_keys"],
                            sensor_data=sensor_data if sensor_enabled else None,
                            robot_states=robot_states,
                            vl_cache_tokens=batch.get("vl_cache"),
                        )
                        if loss.ndim > 0:
                            loss = loss.mean()
                        val_loss_sum += loss.item()
                        val_count += 1
                    except FileNotFoundError:
                        if rank == 0:
                            print(f"⚠️ [Rank {rank}] Validation 중 캐시 파일 없음, 스킵.")
                        continue

            val_loss = val_loss_sum / max(1, val_count)
            model.train()

        # Checkpoint saving
        if rank == 0:
            import psutil, gc
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            frozen = total_params - trainable

            gpu_mem = torch.cuda.memory_allocated()/1e9
            cpu_mem = psutil.virtual_memory().percent
            gc.collect()

            log_dict = {
                "epoch": epoch + 1,
                "train/loss_epoch": avg_loss,
                "val/loss_epoch": val_loss if val_loss else None,
                "params/trainable_M": trainable / 1e6,
                "params/frozen_M": frozen / 1e6,
                "params/frozen_ratio": frozen / total_params,
                "system/gpu_mem_GB": gpu_mem,
                "system/cpu_mem_%": cpu_mem,
                "lr/base_lr": optimizer.param_groups[0]["lr"],
            }

            if sensor_enabled:
                log_dict["train/epoch_sensor_samples"] = total_sensor_samples
                log_dict["train/epoch_nonsensor_samples"] = total_nonsensor_samples
                log_dict["train/sensor_ratio"] = total_sensor_samples / max(1, total_sensor_samples + total_nonsensor_samples)

            wandb.log(log_dict)
            print(f"\n📊 Epoch {epoch+1} Summary | Train: {avg_loss:.8f} | " +
                  (f"Val: {val_loss:.8f}" if val_loss else ""))

            model_module = model.module if hasattr(model, "module") else model
            ckpt_data = {
                "epoch": epoch + 1,
                "model_state_dict": model_module.state_dict(),
                "sensor_encoder": model_module.sensor_encoder.state_dict() if sensor_enabled else None,
                "action_expert": model_module.action_expert.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "val_loss": val_loss,
            }

            is_best = val_loss is not None and val_loss < best_val_loss

            if is_best:
                best_val_loss = val_loss
                best_path = CKPT_DIR / "flow_matching_best.pt"
                torch.save(ckpt_data, best_path)
                print(f"🏆 [Best] Validation improved → saved to {best_path}")
            else:
                latest_path = CKPT_DIR / "flow_matching_latest.pt"
                tmp_path = latest_path.with_suffix(".tmp")
                torch.save(ckpt_data, tmp_path)
                os.replace(tmp_path, latest_path)
                print(f"💾 Latest checkpoint updated: {latest_path}")

    if rank == 0 and writer is not None:
        atexit.register(writer.close)

    if rank == 0:
        wandb.finish()

# ===========================================================
# Main
# ===========================================================
def main():
    parser = argparse.ArgumentParser(description='Flow Matching VLA Training with Sensor')

    # Mode (for cache building)
    parser.add_argument('--mode', type=str, choices=['cache', 'train'], default='train',
                        help='Mode: cache (build VL cache) or train')

    # Dataset
    parser.add_argument('--dataset_dir', type=str, default='/home/najo/NAS/VLA/dataset',
                        help='Dataset directory')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--grad_accum', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--sensor_lr', type=float, default=5e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_ratio', type=float, default=0.03)
    parser.add_argument('--hold_ratio', type=float, default=0.02)
    parser.add_argument('--sched_on', type=str, choices=['step', 'epoch'], default='step')

    # Sensor options
    parser.add_argument('--sensor_enabled', action='store_true', default=True,
                        help='Enable sensor encoder training')
    parser.add_argument('--sensor_loss_weight', type=float, default=2.0)
    parser.add_argument('--fusion_strategy', type=str, default='concat',
                        choices=['concat', 'cross_attention', 'gated'])

    # Image resize
    parser.add_argument('--image_resize_height', type=int, default=360)
    parser.add_argument('--image_resize_width', type=int, default=640)

    # Pre-trained encoder loading
    parser.add_argument('--load_sensor_encoder_checkpoint', type=str, default='./checkpoints/sensor_clip_best.pth',
                        help='Path to pre-trained sensor encoder checkpoint.')
    parser.add_argument('--load_robot_state_encoder_checkpoint', type=str, default='./checkpoints/robot_state_mae_best.pth',
                        help='Path to pre-trained robot state encoder checkpoint.')
    parser.add_argument('--freeze_encoders', action='store_true', help='Freeze sensor and robot state encoders after loading weights.')

    # Other
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--use_cache', action='store_true', help='Enable VL feature caching')
    parser.add_argument('--finetune_vl', type=str, default='none', choices=['none', 'lora', 'full'], help='Fine-tuning mode for VL model')

    args = parser.parse_args()

    # Setup distributed
    rank, world_size, local_rank, device = setup_distributed()
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"🚀 Flow Matching VLA Training")
        print(f"   Mode: {args.mode.upper()}")
        print(f"   World Size: {world_size}")
        print(f"   Dataset: {args.dataset_dir}")

    vl_model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    cache_dir = Path("/home/najo/NAS/VLA/dataset/cache/qwen_vl_features")

    # Cache build mode
    if args.mode == 'cache':
        # ... (cache building logic remains the same)
        return

    # Training mode
    if rank == 0:
        print(f"⚠️ [주의] VL 캐시 검사를 건너뜁니다.")
        print(f"   캐시 경로: {cache_dir}")
        if not cache_dir.exists() or not any(cache_dir.iterdir()):
            print(f"   [경고!] 캐시 디렉토리가 비어있거나 존재하지 않습니다!")

    train_loader, val_loader = build_dataloaders(args, rank, world_size, use_cache=args.use_cache)

    if rank == 0: print("⏳ Initializing model for training...")

    model = QwenVLAUnified(
        model_type='flow_matching', vl_model_name=vl_model_name, action_dim=7, horizon=8,
        hidden_dim=1024, sensor_enabled=args.sensor_enabled, 
        sensor_encoder_type='force_aware', # Match pre-training architecture
        sensor_input_channels=1026,
        sensor_temporal_length=65, sensor_output_dim=3072, robot_state_enabled=args.sensor_enabled,
        fusion_strategy=args.fusion_strategy, finetune_vl=args.finetune_vl,
        image_resize_height=args.image_resize_height, image_resize_width=args.image_resize_width,
        device_map=None,
    )
    model = model.to(device)

    # Load pre-trained encoders on rank 0
    if rank == 0:
        # Load Sensor Encoder
        if args.sensor_enabled and args.load_sensor_encoder_checkpoint and os.path.exists(args.load_sensor_encoder_checkpoint):
            print(f"Loading SensorEncoder from: {args.load_sensor_encoder_checkpoint}")
            ckpt = torch.load(args.load_sensor_encoder_checkpoint, map_location='cpu')
            # Correctly strip the prefix from the unwrapped pre-trained model
            prefix = 'sensor_encoder.'
            sensor_encoder_state_dict = {k.replace(prefix, ''): v for k, v in ckpt['model_state_dict'].items() if k.startswith(prefix)}
            model.sensor_encoder.load_state_dict(sensor_encoder_state_dict, strict=True)
            print("✅ SensorEncoder weights loaded.")

        # Load Robot State Encoder
        if args.sensor_enabled and args.load_robot_state_encoder_checkpoint and os.path.exists(args.load_robot_state_encoder_checkpoint):
            print(f"Loading RobotStateEncoder from: {args.load_robot_state_encoder_checkpoint}")
            ckpt = torch.load(args.load_robot_state_encoder_checkpoint, map_location='cpu')
            prefix = 'encoder.'
            robot_state_encoder_state_dict = {k.replace(prefix, ''): v for k, v in ckpt['model_state_dict'].items() if k.startswith(prefix)}
            model.robot_state_encoder.load_state_dict(robot_state_encoder_state_dict, strict=False)
            print("✅ RobotStateEncoder weights loaded.")

    # DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # Freeze encoders after DDP wrapping and weight loading
    if args.freeze_encoders:
        if rank == 0: print("🧊 Freezing Sensor and Robot State Encoders...")
        for param in model.module.sensor_encoder.parameters():
            param.requires_grad = False
        for param in model.module.robot_state_encoder.parameters():
            param.requires_grad = False
        if rank == 0: print("✅ Encoders frozen.")

    # Optimizer (created AFTER freezing)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95),
    )

    # Scheduler
    total_steps = (len(train_loader) * args.epochs) // args.grad_accum
    scheduler = build_trapezoid_scheduler(
        optimizer, total_steps=total_steps, base_lr=args.lr, min_lr=args.min_lr,
        warmup_ratio=args.warmup_ratio, hold_ratio=args.hold_ratio,
    )

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        if rank == 0: print(f"Resuming from {args.resume}")
        ckpt = copy_to_local_then_load(Path(args.resume), map_location=device)
        # Load model state, but be careful about frozen parts
        model.module.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler and ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0)

    # Train
    Train_FlowMatching(
        model=model, data_loader=train_loader, optimizer=optimizer, num_epochs=args.epochs,
        grad_accum_steps=args.grad_accum, device=device, scheduler=scheduler, sched_on=args.sched_on,
        val_loader=val_loader, start_epoch=start_epoch, sensor_enabled=args.sensor_enabled,
        sensor_loss_weight=args.sensor_loss_weight,
    )

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
