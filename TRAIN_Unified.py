"""
Unified Training Script for QwenVLA with Sensor Integration

Supports two model types:
1. Diffusion-based action prediction (--model-type diffusion)
2. Regression-based action prediction (--model-type regression)

Usage:
    # Diffusion training
    torchrun --nproc_per_node=4 training/A5st_VLA_TRAIN_Unified.py \
        --model-type diffusion \
        --dataset_dir /path/to/dataset \
        --batch_size 4

    # Regression training (with cache building)
    torchrun --nproc_per_node=4 training/A5st_VLA_TRAIN_Unified.py \
        --model-type regression \
        --mode cache  # First build cache

    torchrun --nproc_per_node=4 training/A5st_VLA_TRAIN_Unified.py \
        --model-type regression \
        --mode train  # Then train
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
PROJECT_ROOT = Path(__file__).parent.parent
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
torch.backends.cudnn.deterministic = False  # Changed from True
torch.backends.cudnn.benchmark = True  # Changed from False - significant speedup for fixed input sizes
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

def build_rewarm_scheduler(
    optimizer,
    total_steps: int,
    *,
    prev_lr: float,
    target_lr: float = 1e-4,
    min_lr: float = 1e-6,
    warmup_ratio: float = 0.05,
    hold_ratio: float = 0.05,
):
    """ReWarm Scheduler"""
    assert target_lr > 0 and min_lr > 0
    warmup_steps = int(total_steps * warmup_ratio)
    hold_steps = int(total_steps * hold_ratio)
    decay_steps = max(1, total_steps - warmup_steps - hold_steps)

    floor = min_lr / target_lr
    start = max(1e-12, prev_lr / target_lr)

    def lr_lambda(step: int):
        if step < warmup_steps:
            prog = (step + 1) / max(1, warmup_steps)
            return start + (1.0 - start) * prog
        elif step < warmup_steps + hold_steps:
            return 1.0
        else:
            t = (step - warmup_steps - hold_steps) / decay_steps
            t = min(max(t, 0.0), 1.0)
            cos = 0.5 * (1 + math.cos(math.pi * t))
            return floor + (1.0 - floor) * cos

    for g in optimizer.param_groups:
        g["lr"] = target_lr

    sched = LambdaLR(optimizer, lr_lambda=lr_lambda)

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
# Unified Dataloader Builder (Old + New Async Dataset, compatible with main)
# ============================================================

import glob
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def build_dataloaders(args, rank, world_size):
    """
    Build unified dataloaders combining:
      ① Old format datasets
      ② New format datasets
    Uses UnifiedVLADataset for both formats.
    Compatible with main() expecting 4 return values.
    """
    if rank == 0:
        print(f"[RANK {rank}] 🚀 Building Unified Async Dataloaders (world_size={world_size})")

    # --------------------------
    # Dataset directory patterns
    # --------------------------
    old_priority_patterns = [
        # "/home/najo/NAS/VLA/dataset/White_silicone_white_circle/recv_all_*",
        # "/home/najo/NAS/VLA/dataset/Needle_insertion_eye_trocar/recv_all_*",
    ]
    old_regular_patterns = [
        # "/home/najo/NAS/VLA/dataset/OCT_insertion/Captures*",  # Commented out - no data.pkl
        # "/home/najo/NAS/VLA/dataset/part1/ZED_Captures_*th",  # Commented out - no data.pkl
    ]

    new_dataset_root = "/home/najo/NAS/VLA/dataset/New_dataset"

    # --------------------------
    # Weight configuration
    # --------------------------
    old_dataset_weight = getattr(args, "old_dataset_weight", 1.0)
    new_dataset_weight = getattr(args, "new_dataset_weight", 3.0)

    # --------------------------
    # Build TRAIN dataloader
    # --------------------------
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
    )

    # --------------------------
    # Build VAL dataloader
    # --------------------------
    print("\n📦 Creating VAL dataloader (validation subset)...")

    # Validation dataset: just reuse a subset of old datasets
    val_patterns = old_regular_patterns[:1]  # one representative old dataset
    val_datasets = []
    for pattern in val_patterns:
        for traj_dir in sorted(glob.glob(pattern)):
            try:
                ds = UnifiedVLADataset(
                    data_dir=str(traj_dir),
                    format='old',  # Old format for validation
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
        val_datasets = [next(iter(train_loader.dataset.datasets))]  # fallback to one dataset

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
    )

    if rank == 0:
        print(f"✅ TRAIN loader: {len(train_loader)} batches | VAL loader: {len(val_loader)} batches")
        print(f"   Old dataset weight={old_dataset_weight}, New dataset weight={new_dataset_weight}")

    return train_loader, val_loader

# ===========================================================
# Diffusion 학습 루프
# ===========================================================
def Train_Diffusion(
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
    """Diffusion-based VLA Training Loop"""
    # ✅ Using Smooth L1 Loss (Huber Loss) for robustness to outliers
    loss_fn = nn.SmoothL1Loss(beta=1.0, reduction='none')
    rank = dist.get_rank()
    writer = AsyncCheckpointWriter(max_queue=2, sync_every=0) if rank == 0 else None

    model.train()
    if rank == 0:
        wandb.init(
            project="QwenVLA-Unified-Diffusion",
            name=f"diffusion_{time.strftime('%m%d_%H%M')}",
            resume="allow",
            id=f"qvla_diffusion_{int(time.time())}",
            settings=wandb.Settings(start_method="thread", _disable_stats=True),
            config={
                "lr": optimizer.param_groups[0]["lr"],
                "grad_accum_steps": grad_accum_steps,
                "epochs": num_epochs,
                "scheduler": sched_on,
                "sensor_enabled": sensor_enabled,
                "sensor_loss_weight": sensor_loss_weight,
            }
        )

    best_val_loss = float("inf")
    global_step = 0
    log_interval = 10

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
            gt_actions = batch["actions"].to(device, dtype=torch.bfloat16, non_blocking=True)
            sensor_data = batch["sensor_data"].to(device, dtype=torch.bfloat16, non_blocking=True) if sensor_enabled else None
            has_sensor_mask = batch["has_sensor_mask"].to(device, non_blocking=True) if sensor_enabled else None

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                eps_pred, eps_target, timesteps = model(
                    text_inputs=batch["instruction"],
                    image_inputs=batch["images"],
                    actions=gt_actions,
                    cache_keys=batch["cache_keys"],
                    sensor_data=sensor_data if sensor_enabled else None,
                    vl_cache_tokens=batch.get("vl_cache"),
                )

            # loss_per_sample = loss_fn(eps_pred, eps_target).mean(dim=[1, 2])
            # ✅ Dual-Head aware Loss (Translation / Rotation / Gripper weighting)
            loss_trans = loss_fn(eps_pred[..., :3], eps_target[..., :3], reduction='none').mean(dim=[1, 2])
            loss_rot   = loss_fn(eps_pred[..., 3:6], eps_target[..., 3:6], reduction='none').mean(dim=[1, 2])
            loss_grip  = loss_fn(eps_pred[..., 6:],  eps_target[..., 6:],  reduction='none').mean(dim=[1, 2])

            # ✅ Updated weights: translation=1.0, rotation=1.0, gripper=0.1
            loss_per_sample = loss_trans + 1.0 * loss_rot + 0.1 * loss_grip

            weights = torch.tensor(batch["confidence"], device=device, dtype=torch.bfloat16)
            if sensor_enabled and has_sensor_mask is not None:
                sensor_weights = torch.where(
                    has_sensor_mask,
                    torch.tensor(sensor_loss_weight, device=device, dtype=torch.bfloat16),
                    torch.tensor(1.0, device=device, dtype=torch.bfloat16)
                )
                weights = weights * sensor_weights
                total_sensor_samples += has_sensor_mask.sum().item()
                total_nonsensor_samples += (~has_sensor_mask).sum().item()

            loss = (loss_per_sample * weights).mean() / grad_accum_steps

            sync_context = model.no_sync() if (step + 1) % grad_accum_steps != 0 else nullcontext()
            with sync_context:
                loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if scheduler is not None and sched_on == "step":
                    scheduler.step()
                global_step += 1

            total_loss += loss.item() * grad_accum_steps

            if rank == 0 and (step % log_interval == 0):
                lr = optimizer.param_groups[0]["lr"]
                pbar.set_postfix({"loss": f"{loss.item()*grad_accum_steps:.4f}", "lr": f"{lr:.2e}"})
                wandb.log({
                    "train/loss": loss.item() * grad_accum_steps,
                    "train/lr": lr,
                    "train/epoch": epoch,
                    "train/step": global_step,
                    "train/loss_total": loss.item() * grad_accum_steps,
                    "train/loss_trans": loss_trans.mean().item(),
                    "train/loss_rot": loss_rot.mean().item(),
                    "train/loss_grip": loss_grip.mean().item(),
                    "train/lr": optimizer.param_groups[0]["lr"]
                })

        if scheduler is not None and sched_on == "epoch":
            scheduler.step()

        avg_loss = total_loss / len(data_loader)

        # ----------------------------
        # Validation
        # ----------------------------
        val_loss = None
        if val_loader is not None:
            val_loss = validate_diffusion(model, val_loader, device, sensor_enabled, sensor_loss_weight)

        # ----------------------------
        # Checkpoint Saving
        # ----------------------------
        if rank == 0:
            model_module = model.module if hasattr(model, "module") else model
            state = {
                "epoch": epoch + 1,
                "model_state_dict": model_module.state_dict(),
                "sensor_encoder": model_module.sensor_encoder.state_dict() if model_module.sensor_enabled else None,
                "action_expert": model_module.action_expert.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "train_loss": avg_loss,
                "val_loss": val_loss,
            }

            # Save latest
            latest_ckpt = CKPT_DIR / "diffusion_latest.pt"
            writer.submit(state, latest_ckpt)

            # Save best (based on val_loss)
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt = CKPT_DIR / "diffusion_best.pt"
                writer.submit(state, best_ckpt)
                print(f"🏆 Validation improved ({val_loss:.4f}) → saved best checkpoint")

            print(f"💾 Saved latest checkpoint (epoch {epoch+1})")

            wandb.log({
                "epoch/train_loss": avg_loss,
                "epoch/val_loss": val_loss if val_loss else None,
                "epoch/sensor_samples": total_sensor_samples,
                "epoch/nonsensor_samples": total_nonsensor_samples,
            })

    if rank == 0 and writer:
        atexit.register(writer.close)
        wandb.finish()


def validate_diffusion(model, val_loader, device, sensor_enabled, sensor_loss_weight):
    """Validation for diffusion model"""
    was_training = model.training
    model.train()

    # ✅ Using Smooth L1 Loss (Huber Loss) for robustness to outliers
    loss_fn = nn.SmoothL1Loss(beta=1.0, reduction='none')
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", disable=(dist.get_rank() != 0)):
            instructions = batch["instruction"]
            image_inputs = batch["images"]
            gt_actions = batch["actions"].to(device, dtype=torch.bfloat16)
            sensor_data = batch["sensor_data"].to(device, dtype=torch.bfloat16) if sensor_enabled else None
            has_sensor_mask = batch["has_sensor_mask"].to(device) if sensor_enabled else None

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                eps_pred, eps_target, timesteps = model(
                    text_inputs=instructions,
                    image_inputs=image_inputs,
                    actions=gt_actions,
                    cache_keys=batch["cache_keys"],
                    sensor_data=sensor_data,
                    vl_cache_tokens=batch.get("vl_cache"),
                )

                # loss_per_sample = loss_fn(eps_pred, eps_target).mean(dim=[1, 2])
                loss_trans = loss_fn(eps_pred[..., :3], eps_target[..., :3], reduction='none').mean(dim=[1, 2])
                loss_rot   = loss_fn(eps_pred[..., 3:6], eps_target[..., 3:6], reduction='none').mean(dim=[1, 2])
                loss_grip  = loss_fn(eps_pred[..., 6:],  eps_target[..., 6:],  reduction='none').mean(dim=[1, 2])

                # ✅ Updated weights: translation=1.0, rotation=1.0, gripper=0.1
                loss_per_sample = loss_trans + 1.0 * loss_rot + 0.1 * loss_grip
                
                weights = torch.tensor(batch["confidence"], device=device, dtype=torch.bfloat16)

                if sensor_enabled and has_sensor_mask is not None:
                    sensor_weights = torch.where(has_sensor_mask,
                                                 torch.tensor(sensor_loss_weight, device=device),
                                                 torch.tensor(1.0, device=device))
                    weights = weights * sensor_weights

                loss = (loss_per_sample * weights).mean()
                total_loss += loss.item()

    if was_training:
        model.train()
    else:
        model.eval()

    return total_loss / len(val_loader)

# ===========================================================
# Regression 학습 루프
# ===========================================================
def Train_Regression(
    model,
    data_loader,
    optimizer,
    num_epochs=3,
    grad_accum_steps=8,
    device="cuda",
    save_path="./checkpoints/qwen_vla_regression.pt",
    scheduler=None,
    sched_on="step",
    val_loader=None,
    start_epoch=0,
    sensor_enabled=True,
    sensor_loss_weight=2.0,
    model_type="regression",  # 'regression' or 'flow_matching'
):
    """Regression/Flow Matching training loop - OPTIMIZED"""
    # ✅ Using F.smooth_l1_loss directly in training loop (no need for loss_fn here)
    # loss_fn = nn.MSELoss()  # Removed - using Smooth L1 Loss instead
    rank = dist.get_rank()
    writer = AsyncCheckpointWriter(max_queue=2, sync_every=0) if rank == 0 else None

    # ✅ OPTIMIZATION: GradScaler not needed for BFloat16 (already overflow-resistant)
    # BFloat16 has wider dynamic range than FP16, so GradScaler is unnecessary
    # scaler = torch.cuda.amp.GradScaler(enabled=False)  # Disabled for bfloat16

    model.train()
    if rank == 0:
        project_name = "QwenVLA-Unified-FlowMatching" if model_type == "flow_matching" else "QwenVLA-Unified-Regression"
        run_name = f"{model_type}_{time.strftime('%m%d_%H%M')}"
        wandb.init(
            project=project_name,
            name=run_name,
            resume="allow",
            id=f"qvla_{model_type}_{int(time.time())}",
            settings=wandb.Settings(start_method="thread", _disable_stats=True),
            config={
                "model_type": model_type,
                "lr": optimizer.param_groups[0]["lr"],
                "grad_accum_steps": grad_accum_steps,
                "epochs": num_epochs,
                "scheduler": sched_on,
                "sensor_enabled": sensor_enabled,
                "sensor_loss_weight": sensor_loss_weight,
            }
        )

    global_step = 0
    for epoch in range(start_epoch, start_epoch + num_epochs):
        if isinstance(data_loader.sampler, DistributedSampler):
            data_loader.sampler.set_epoch(epoch)

        total_loss = 0.0
        total_sensor_samples = 0
        total_nonsensor_samples = 0

        optimizer.zero_grad(set_to_none=True)   # ✅ 메모리 최적화
        model.train()

        pbar = tqdm(enumerate(data_loader), total=len(data_loader),
                    desc=f"[Rank {rank}] Epoch {epoch+1}",
                    disable=(rank != 0))

        # ✅ Initialize loss components for epoch-end logging
        last_loss_trans = 0.0
        last_loss_rot = 0.0
        last_loss_grip = 0.0

        for step, batch in pbar:
            # ⬇️ [수정] try...except 블록 추가
            try:
                # 🔍 Debug: Check what's in the batch
                if rank == 0 and step == 0:
                    print(f"\n🔍 DEBUG - Batch keys: {batch.keys()}")
                    if "robot_states" in batch:
                        print(f"   robot_states in batch: {batch['robot_states'].shape if batch['robot_states'] is not None else 'None'}")
                    else:
                        print(f"   ❌ robot_states NOT in batch!")

                # ✅ 모든 텐서 GPU로 비동기 전송
                instructions = batch["instruction"]
                image_inputs = batch["images"]
                gt_actions_full = batch["actions"].to(device, dtype=torch.bfloat16, non_blocking=True)

                # ✅ REGRESSION: Use only first action (B, 8, 7) -> (B, 1, 7)
                # ✅ FLOW_MATCHING: Use all actions (B, 8, 7)
                if model_type == "regression":
                    gt_actions = gt_actions_full[:, 0:1, :]  # Only first action
                else:
                    gt_actions = gt_actions_full  # All actions

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
                        if rank == 0 and step == 0:
                            print(f"   ✅ robot_states loaded: {robot_states.shape}")
                    except Exception as e:
                        if rank == 0:
                            print(f"⚠️ Failed to load robot_states: {e}")
                        robot_states = None
                elif rank == 0 and step == 0:
                    print(f"   ⚠️ robot_states not loaded: in_batch={('robot_states' in batch)}, sensor_enabled={sensor_enabled}")

                # ✅ OPTIMIZATION: Use autocast with GradScaler
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if model_type == "flow_matching":
                        # Flow matching: model returns loss directly
                        flow_loss, _, _ = model(
                            text_inputs=instructions,
                            image_inputs=image_inputs,
                            actions=gt_actions,  # Use 'actions' for flow matching
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
                    else:
                        # Regression: compute MSE loss
                        pred_actions, _ = model(
                            text_inputs=instructions,
                            image_inputs=image_inputs,
                            z_chunk=gt_actions,
                            cache_keys=batch["cache_keys"],
                            sensor_data=sensor_data if sensor_enabled else None,
                            robot_states=robot_states,
                            vl_cache_tokens=batch.get("vl_cache"),
                        )

                        # ✅ confidence도 GPU에서 바로 weight tensor 생성
                        weights = torch.tensor(batch["confidence"], device=device, dtype=torch.bfloat16)

                        if sensor_enabled and has_sensor_mask is not None:
                            sensor_weights = torch.where(
                                has_sensor_mask,
                                torch.tensor(sensor_loss_weight, device=device, dtype=torch.bfloat16),
                                torch.tensor(1.0, device=device, dtype=torch.bfloat16)
                            )
                            weights = weights * sensor_weights
                            total_sensor_samples += has_sensor_mask.sum().item()
                            total_nonsensor_samples += (~has_sensor_mask).sum().item()

                        weights = weights / weights.mean()
                        # loss_each = (pred_actions.float() - gt_actions.float()).pow(2).mean(dim=[1, 2])
                        # loss = (loss_each * weights).mean() / grad_accum_steps
                        # ✅ Dual-Head Loss with Smooth L1 (Huber Loss) for outlier robustness
                        pred = pred_actions.float()
                        gt = gt_actions.float()

                        # Using Smooth L1 Loss (beta=1.0 for standard Huber loss)
                        loss_trans = F.smooth_l1_loss(pred[..., :3], gt[..., :3], beta=1.0, reduction='none').mean(dim=[1, 2])
                        loss_rot   = F.smooth_l1_loss(pred[..., 3:6], gt[..., 3:6], beta=1.0, reduction='none').mean(dim=[1, 2])
                        loss_grip  = F.smooth_l1_loss(pred[..., 6:],  gt[..., 6:],  beta=1.0, reduction='none').mean(dim=[1, 2])

                        # Store for epoch-end logging
                        last_loss_trans = loss_trans.mean().item()
                        last_loss_rot = loss_rot.mean().item()
                        last_loss_grip = loss_grip.mean().item()

                        # ✅ Updated weights: translation=1.0, rotation=1.0, gripper=0.1
                        loss_each = loss_trans + 1.0 * loss_rot + 0.1 * loss_grip
                        loss = (loss_each * weights).mean() / grad_accum_steps

                sync_context = model.no_sync() if (step + 1) % grad_accum_steps != 0 else nullcontext()
                with sync_context:
                    # ✅ BFloat16 doesn't need GradScaler (already overflow-resistant)
                    loss.backward()
                total_loss += loss.item() * grad_accum_steps

                if (step + 1) % grad_accum_steps == 0:
                    # ✅ Gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    # ✅ Optimizer step
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
                
                # 그라디언트 축적 단계였다면, 스킵했으므로 그라디언트 초기화
                if (step + 1) % grad_accum_steps == 0:
                    optimizer.zero_grad(set_to_none=True)
                continue # 다음 스텝으로 넘어감

        avg_loss_tensor = torch.tensor(total_loss / len(data_loader), device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = avg_loss_tensor.item()

        if scheduler is not None and sched_on == "epoch":
            scheduler.step()

        # ✅ Validation 동일 로직
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_loss_sum, val_count = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    # ⬇️ [수정] Validation 루프에도 try...except 추가
                    try:
                        gt_actions_full = batch["actions"].to(device, dtype=torch.bfloat16, non_blocking=True)

                        # ✅ REGRESSION: Use only first action
                        # ✅ FLOW_MATCHING: Use all actions
                        if model_type == "regression":
                            gt_actions = gt_actions_full[:, 0:1, :]  # Only first action
                        else:
                            gt_actions = gt_actions_full  # All actions

                        sensor_data = (
                            batch["sensor_data"].to(device, dtype=torch.bfloat16, non_blocking=True)
                            if sensor_enabled else None
                        )
                        robot_states = (
                            batch["robot_states"].to(device, non_blocking=True)
                            if "robot_states" in batch and sensor_enabled else None
                        )

                        if model_type == "flow_matching":
                            # Flow matching: model returns loss directly (not 3 values like diffusion)
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
                        else:
                            # Regression: compute MSE loss with dual-head
                            pred_actions, _ = model(
                                text_inputs=batch["instruction"],
                                image_inputs=batch["images"],
                                z_chunk=gt_actions,
                                cache_keys=batch["cache_keys"],
                                sensor_data=sensor_data if sensor_enabled else None,
                                robot_states=robot_states,
                                vl_cache_tokens=batch.get("vl_cache"),
                            )

                            weights = torch.tensor(batch["confidence"], device=device, dtype=torch.bfloat16)
                            weights = weights / weights.mean()

                            # ✅ Use dual-head loss with Smooth L1 for validation (consistent with training)
                            pred = pred_actions.float()
                            gt = gt_actions.float()

                            loss_trans = F.smooth_l1_loss(pred[..., :3], gt[..., :3], beta=1.0, reduction='none').mean(dim=[1, 2])
                            loss_rot   = F.smooth_l1_loss(pred[..., 3:6], gt[..., 3:6], beta=1.0, reduction='none').mean(dim=[1, 2])
                            loss_grip  = F.smooth_l1_loss(pred[..., 6:],  gt[..., 6:],  beta=1.0, reduction='none').mean(dim=[1, 2])

                            # ✅ Updated weights: translation=1.0, rotation=1.0, gripper=0.1
                            loss_each = loss_trans + 1.0 * loss_rot + 0.1 * loss_grip
                            loss = (loss_each * weights).mean()
                        val_loss_sum += loss.item()
                        val_count += 1
                    except FileNotFoundError:
                        if rank == 0:
                            print(f"⚠️ [Rank {rank}] Validation 중 캐시 파일 없음, 스킵.")
                        continue # Validation 배치 스킵
                    # ⬆️ [수정] 여기까지

            val_loss = val_loss_sum / max(1, val_count)
            model.train()

        # ✅ 로깅 및 체크포인트 저장
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
                "train/loss_trans": last_loss_trans,
                "train/loss_rot": last_loss_rot,
                "train/loss_grip": last_loss_grip,
                "train/lr": optimizer.param_groups[0]["lr"]
            }

            if sensor_enabled:
                log_dict["train/epoch_sensor_samples"] = total_sensor_samples
                log_dict["train/epoch_nonsensor_samples"] = total_nonsensor_samples
                log_dict["train/sensor_ratio"] = total_sensor_samples / max(1, total_sensor_samples + total_nonsensor_samples)

            wandb.log(log_dict)
            print(f"\n📊 Epoch {epoch+1} Summary | Train: {avg_loss:.8f} | " +
                  (f"Val: {val_loss:.8f}" if val_loss else ""))
            
            # ✅ Checkpoint에 model_state_dict 추가
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

            # ✅ Best / Latest 저장 로직 그대로
            if not hasattr(Train_Regression, "_best_loss"):
                Train_Regression._best_loss = float("inf")

            is_best = val_loss is not None and val_loss < Train_Regression._best_loss
            
            ckpt_prefix = "flow_matching" if model_type == "flow_matching" else "regression"
            
            if is_best:
                Train_Regression._best_loss = val_loss
                best_path = CKPT_DIR / f"{ckpt_prefix}_best.pt"
                torch.save(ckpt_data, best_path)
                print(f"🏆 [Best] Validation improved → saved to {best_path}")
            else:
                latest_path = CKPT_DIR / f"{ckpt_prefix}_latest.pt"
                tmp_path = latest_path.with_suffix(".tmp")
                torch.save(ckpt_data, tmp_path)
                os.replace(tmp_path, latest_path)
                print(f"💾 Latest checkpoint updated: {latest_path}")

    if rank == 0 and writer is not None:
        atexit.register(writer.close)

    if rank == 0:
        wandb.finish()


# ===========================================================
# Dataset Building
# ===========================================================
def build_datasets(args, rank):
    """Build integrated async datasets"""
    datasets = []
    dataset_weights = []

    # Priority old datasets (2x weight)
    priority_old_dataset_dirs = [
        # "/home/najo/NAS/VLA/dataset/White_silicone_white_circle/recv_all_*",
        # "/home/najo/NAS/VLA/dataset/Needle_insertion_eye_trocar/recv_all_*",
    ]

    # Regular old datasets (1x weight)
    regular_old_dataset_dirs = [
        # "/home/najo/NAS/VLA/dataset/OCT_insertion/Captures*",
        # "/home/najo/NAS/VLA/dataset/part1/ZED_Captures_*th",
    ]

    # New dataset path (3x weight)
    new_dataset_path = Path("/home/najo/NAS/VLA/dataset/New_dataset")

    # Check for deprecated model type
    if args.model_type == 'diffusion':
        raise ValueError(
            "❌ Diffusion model is deprecated!\n"
            "Please use 'flow_matching' or 'regression' instead.\n"
            "Flow matching provides faster inference and better performance."
        )

    # Determine VLM reuse count and sensor window size
    # Note: All datasets use sensor_window_size=650 (pre-processed)
    sensor_window_size = 650
    vlm_reuse_count = 3  # VL features reused for 3 action predictions

    # Load priority old datasets (2x weight)
    # if rank == 0:
        # print("\n📦 Loading priority old datasets (2x weight)...")
    for pattern in priority_old_dataset_dirs:
        expanded_paths = glob.glob(pattern)
        for traj_dir in expanded_paths:
            try:
                ds = UnifiedVLADataset(
                    data_dir=str(traj_dir),
                    format='old',  # Explicitly specify old format
                    horizon=8,
                    vlm_reuse_count=vlm_reuse_count,
                    sensor_window_size=sensor_window_size,
                    action_expert_hz=10,
                )
                datasets.append(ds)
                dataset_weights.extend([2.0] * len(ds))
                sensor_status = "WITH sensor" if ds.has_sensor else "NO sensor"
                if rank == 0:
                    print(f"✅ [2x] {Path(traj_dir).name}: {len(ds)} samples ({sensor_status})")
            except Exception as e:
                if rank == 0:
                    print(f"⚠️ Failed to load {traj_dir}: {e}")

    # Load regular old datasets (1x weight)
    # if rank == 0:
    #     print("\n📦 Loading regular old datasets (1x weight)...")
    for pattern in regular_old_dataset_dirs:
        expanded_paths = glob.glob(pattern)
        for traj_dir in expanded_paths:
            try:
                ds = UnifiedVLADataset(
                    data_dir=str(traj_dir),
                    format='old',  # Explicitly specify old format
                    horizon=8,
                    vlm_reuse_count=vlm_reuse_count,
                    sensor_window_size=sensor_window_size,
                    action_expert_hz=10,
                )
                datasets.append(ds)
                dataset_weights.extend([1.0] * len(ds))
                sensor_status = "WITH sensor" if ds.has_sensor else "NO sensor"
                if rank == 0:
                    print(f"✅ [1x] {Path(traj_dir).name}: {len(ds)} samples ({sensor_status})")
            except Exception as e:
                if rank == 0:
                    print(f"⚠️ Failed to load {traj_dir}: {e}")

    # Load new datasets (3x weight) - OPTIMIZED with parallel loading
    if rank == 0:
        t_loop_start = time.time()
        episode_count = 0

    if new_dataset_path.exists():
        all_task_dirs = list(new_dataset_path.iterdir())
        if rank == 0:
            print(f"    Found {len(all_task_dirs)} task directories.")

        # ✅ OPTIMIZATION: Collect all episode paths first (fast)
        episode_paths = []
        for task_dir in all_task_dirs:
            if not task_dir.is_dir():
                continue
            task_name = task_dir.name.replace('_', ' ')
            instruction = f"Perform {task_name} insertion task"

            for episode_dir in task_dir.iterdir():
                if episode_dir.is_dir() and episode_dir.name.startswith('episode_'):
                    episode_paths.append((str(episode_dir), instruction))

        if rank == 0:
            print(f"    Found {len(episode_paths)} episodes to load.")

        # ✅ OPTIMIZATION: Load datasets with progress tracking
        pbar = tqdm(episode_paths, desc="Loading datasets", disable=(rank != 0))

        for episode_dir_str, instruction in pbar:
            try:
                ds = UnifiedVLADataset(
                    data_dir=episode_dir_str,
                    format='new',
                    horizon=8,
                    vlm_reuse_count=vlm_reuse_count,
                    sensor_window_size=sensor_window_size,
                    action_expert_hz=10,
                    instruction=instruction,
                )
                datasets.append(ds)
                dataset_weights.extend([3.0] * len(ds))
                episode_count += 1

                if rank == 0:
                    pbar.set_postfix({"loaded": episode_count, "samples": len(datasets[-1])})

            except Exception as e:
                if rank == 0:
                    pbar.write(f"⚠️ Failed to load {Path(episode_dir_str).name}: {e}")
    else:
        if rank == 0:
            print(f"⚠️ New dataset path not found: {new_dataset_path}")

    if rank == 0:
        print(f"    ✅ New dataset loading finished in {time.time() - t_loop_start:.2f}s (Total {episode_count} episodes)")


    if not datasets:
        raise ValueError("No datasets loaded!")

    full_dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    if rank == 0:
        print(f"\n📊 Total dataset statistics:")
        print(f"   Total samples: {len(full_dataset)}")
        print(f"   Old priority (2x): {sum(1 for w in dataset_weights if w == 2.0)}")
        print(f"   Old regular (1x): {sum(1 for w in dataset_weights if w == 1.0)}")
        print(f"   New datasets (3x): {sum(1 for w in dataset_weights if w == 3.0)}")

    return full_dataset, dataset_weights

# ===========================================================
# Main
# ===========================================================
def main():
    parser = argparse.ArgumentParser(description='Unified VLA Training with Sensor')

    # Model selection
    parser.add_argument('--model-type', type=str, choices=['diffusion', 'regression', 'flow_matching'], required=True,
                        help='Model type: diffusion, regression, or flow_matching')

    # Mode (for regression with cache)
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

    # Diffusion-specific
    parser.add_argument('--diffusion_timesteps', type=int, default=100,
                        help='Number of diffusion steps (diffusion only)')

    # Regression-specific
    parser.add_argument('--image_resize_height', type=int, default=360)
    parser.add_argument('--image_resize_width', type=int, default=640)

    # Other
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')

    args = parser.parse_args()

    # Setup distributed
    rank, world_size, local_rank, device = setup_distributed()
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"🚀 Unified VLA Training")
        print(f"   Model Type: {args.model_type.upper()}")
        print(f"   Mode: {args.mode.upper()}")
        print(f"   World Size: {world_size}")
        print(f"   Dataset: {args.dataset_dir}")

    vl_model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    cache_dir = Path("/home/najo/NAS/VLA/dataset/cache/qwen_vl_features")

    # ===========================================================
    # Cache build mode
    # ===========================================================
    if args.mode == "cache" and args.model_type in ["regression", "diffusion", "flow_matching"]:
        # Build datasets only for cache mode
        if rank == 0:
            print("📦 Building integrated dataset for caching...")

        full_dataset, dataset_weights = build_datasets(args, rank)
        if rank == 0:
            print("⏳ Building VL cache (shared for regression & diffusion)...")

        processor = AutoProcessor.from_pretrained(vl_model_name)
        target_pixels = args.image_resize_height * args.image_resize_width
        processor.image_processor.min_pixels = target_pixels
        processor.image_processor.max_pixels = target_pixels

        vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            vl_model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda",
            low_cpu_mem_usage=True,
        )

        class DummyVLA:
            def __init__(self, vl_model, processor):
                self.vl_model = vl_model
                self.processor = processor
                self.cache_dir = cache_dir
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self._cache_path = QwenVLAUnified._cache_path.__get__(self)
                self._enforce_cache_limit = QwenVLAUnified._enforce_cache_limit.__get__(self)
                self._atomic_save = QwenVLAUnified._atomic_save
            def eval(self):
                self.vl_model.eval()
                return self

        dummy_model = DummyVLA(vl_model, processor)
        build_vl_cache_distributed_optimized(dummy_model, full_dataset, device=device)

        dist.barrier()
        if rank == 0:
            print("✅ Cache build complete. You can now run training with --mode train.")
        dist.destroy_process_group()
        return
    # ===========================================================
    # (제거됨) Cache validation & rebuild missing
    # ===========================================================
    
    # 캐시 검사 로직을 제거했습니다.
    if rank == 0:
        print(f"⚠️ [주의] VL 캐시 검사를 건너뜁니다.")
        print(f"   캐시 경로: {cache_dir}")
        print(f"   만약 캐시 파일이 없으면 학습이 즉시 실패합니다.")
        if not cache_dir.exists() or not any(cache_dir.iterdir()):
             print(f"   [경고!] 캐시 디렉토리가 비어있거나 존재하지 않습니다!")

    # ===========================================================
    # Training mode
    # ===========================================================
    # Build datasets for training mode
    if rank == 0:
        print("📦 Building integrated dataset for training...")

    train_loader, val_loader = build_dataloaders(args, rank, world_size)
    
    if rank == 0:
        print("⏳ Initializing model for training...")

    # Initialize unified model based on type
    # ✅ REGRESSION: Use horizon=1 (predict only first action)
    # ✅ FLOW_MATCHING: Keep horizon=8 (original behavior)
    model_horizon = 1 if args.model_type == 'regression' else 8

    model = QwenVLAUnified(
        model_type=args.model_type,  # 'diffusion' or 'regression'
        vl_model_name=vl_model_name,
        action_dim=7,
        horizon=model_horizon,  # ✅ Changed: 1 for regression, 8 for flow_matching
        hidden_dim=1024,
        sensor_enabled=args.sensor_enabled,
        sensor_input_channels=1026,
        sensor_temporal_length=65,
        sensor_output_dim=2048,  # Adjusted to match actual sensor encoder output
        robot_state_enabled=args.sensor_enabled,  # Enable robot state together with sensor
        fusion_strategy=args.fusion_strategy,
        diffusion_timesteps=args.diffusion_timesteps if args.model_type == 'diffusion' else 100,
        finetune_vl='none',
        image_resize_height=args.image_resize_height,
        image_resize_width=args.image_resize_width,
        device_map=None,  # Don't use device_map with DDP
    )

    if rank == 0:
        print(f"   ✅ Model horizon: {model_horizon} ({'single action' if model_horizon == 1 else 'action chunk'})")

    # Manually move to device (required for DDP)
    model = model.to(device)

    if rank == 0:
        print(f"   Model: QwenVLAUnified ({args.model_type})")
        print(f"   Model moved to device: {device}")
        if args.model_type == 'diffusion':
            print(f"   Diffusion timesteps: {args.diffusion_timesteps}")
        else:
            print(f"   Image resize: {args.image_resize_width}x{args.image_resize_height}")

    # DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=True,          # ✅ branch/조건부 경로 대응
                gradient_as_bucket_view=True,         # ✅ 성능
                broadcast_buffers=False               # (선택) BatchNorm 버퍼 동기 최소화
            )  

    
    if rank == 0:
        print(f"   Train loader: {len(train_loader)} batches")
        print(f"   Val loader: {len(val_loader)} batches")


    # Optimizer
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )

    # Scheduler
    total_steps = (len(train_loader) * args.epochs) // args.grad_accum
    scheduler = build_trapezoid_scheduler(
        optimizer,
        total_steps=total_steps,
        base_lr=args.lr,
        min_lr=args.min_lr,
        warmup_ratio=args.warmup_ratio,
        hold_ratio=args.hold_ratio,
    )

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        if rank == 0:
            print(f"Resuming from {args.resume}")
        ckpt = copy_to_local_then_load(Path(args.resume), map_location=device)
        model.module.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler and ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            for param_group in optimizer.param_groups:
                param_group['lr'] = scheduler.get_last_lr()[0]
        start_epoch = ckpt.get("epoch", 0)

    # Train
    if args.model_type == 'diffusion':
        Train_Diffusion(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            num_epochs=args.epochs,
            grad_accum_steps=args.grad_accum,
            device=device,
            scheduler=scheduler,
            sched_on=args.sched_on,
            val_loader=val_loader,
            start_epoch=start_epoch,
            sensor_enabled=args.sensor_enabled,
            sensor_loss_weight=args.sensor_loss_weight,
        )
    else:  # regression or flow_matching
        Train_Regression(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            num_epochs=args.epochs,
            grad_accum_steps=args.grad_accum,
            device=device,
            scheduler=scheduler,
            sched_on=args.sched_on,
            val_loader=val_loader,
            start_epoch=start_epoch,
            sensor_enabled=args.sensor_enabled,
            sensor_loss_weight=args.sensor_loss_weight,
            model_type=args.model_type,  # Pass model type for flow matching
        )

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
