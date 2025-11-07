"""
Robot State Encoder Pre-training using Masked Auto-Encoding (MAE)

This script pre-trains the RobotStateEncoder on the task of reconstructing
masked portions of a robot state sequence.

Methodology:
1. A window of robot state data (joints + pose) is loaded.
2. A significant portion of the timesteps in the window are randomly masked.
3. The RobotStateEncoder processes the corrupted sequence.
4. A simple decoder head predicts the values of the original masked timesteps.
5. An MSE loss between the prediction and the ground truth is used for training.
6. This forces the encoder to learn the underlying dynamics and correlations
   of the robot's movement.
"""

import argparse
import os
import glob
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import math
from torch.optim.lr_scheduler import LambdaLR

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Add project root to import custom modules
import sys
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.unified_model import RobotStateEncoder

# =====================================
# 1. MAE Model Definition
# =====================================

class MAERobotStateModel(nn.Module):
    """
    Masked Auto-Encoder model built around the RobotStateEncoder.
    """
    def __init__(self, encoder: RobotStateEncoder, decoder_dim: int = 128):
        super().__init__()
        self.encoder = encoder
        self.encoder_dim = encoder.model_dim
        self.decoder_dim = decoder_dim
        self.output_dim = encoder.input_dim # Should be 12

        # Decoder head
        self.decoder = nn.Sequential(
            nn.Linear(self.encoder_dim, self.decoder_dim),
            nn.GELU(),
            nn.Linear(self.decoder_dim, self.output_dim)
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src (torch.Tensor): Masked robot state sequence, shape (B, T, D_in)
        
        Returns:
            torch.Tensor: Reconstructed sequence, shape (B, T, D_in)
        """
        # Encode the masked sequence. The encoder for MAE should output a sequence.
        # The imported RobotStateEncoder is modified for this script's purpose.
        encoded_sequence = self.encoder(src, return_sequence=True) # (B, T, D_encoder)
        
        # Decode each timestep to reconstruct the original sequence
        reconstructed_sequence = self.decoder(encoded_sequence) # (B, T, D_in)
        return reconstructed_sequence

# =====================================
# 2. Robot State Dataset
# =====================================

class RobotStateDataset(Dataset):
    """
    Dataset that provides windows of robot state data from .npz files.
    """
    def __init__(self, root_dir: str, window_size: int = 60, step: int = 10):
        self.window_size = window_size
        self.step = step
        self.data_files = []
        
        print(f"Scanning for robot_states.npz in {root_dir}...")
        # Find all robot_states.npz files recursively
        for path in Path(root_dir).rglob('robot_states.npz'):
            self.data_files.append(path)
        
        print(f"Found {len(self.data_files)} robot_states.npz files.")
        
        self.windows = []
        for file_path in tqdm(self.data_files, desc="Creating windows"):
            try:
                # Load the data using mmap_mode for memory efficiency
                data = np.load(file_path, mmap_mode='r')['robot_states']
                
                # Create sliding windows
                for i in range(0, len(data) - self.window_size + 1, self.step):
                    self.windows.append((file_path, i))
            except Exception as e:
                print(f"Warning: Could not load or process {file_path}: {e}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        file_path, start_idx = self.windows[idx]
        
        # The file is re-opened here, which is fine with mmap
        data = np.load(file_path, mmap_mode='r')['robot_states']
        
        window = data[start_idx : start_idx + self.window_size]
        
        return torch.from_numpy(window.astype(np.float32))

# =====================================
# 3. Main Training Function
# =====================================

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


def main(args):
    # DDP Setup
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    is_main_process = local_rank == 0

    if is_main_process:
        print(f"Using {torch.cuda.device_count()} GPUs for MAE pre-training.")

    # Dataset and DataLoader
    if is_main_process:
        print("Creating dataset...")
    dataset = RobotStateDataset(root_dir=args.dataset_path, window_size=args.window_size)
    
    # Split dataset into training and validation
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    # Validation loader runs on all processes, but we only need results from the main one
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2, # Can use larger batch for validation
        num_workers=args.num_workers,
        pin_memory=True
    )

    if is_main_process:
        print(f"Dataset created with {len(train_dataset)} training windows and {len(val_dataset)} validation windows.")

    # Model Setup
    encoder = RobotStateEncoder(
        temporal_length=args.window_size,
        model_dim=args.model_dim,
        output_dim=args.output_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers
    )
    model = MAERobotStateModel(encoder).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Dataset and DataLoader
    if is_main_process:
        print("Creating dataset...")
    dataset = RobotStateDataset(root_dir=args.dataset_path, window_size=args.window_size)
    
    # Split dataset into training and validation
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    # Validation loader runs on all processes, but we only need results from the main one
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2, # Can use larger batch for validation
        num_workers=args.num_workers,
        pin_memory=True
    )

    if is_main_process:
        print(f"Dataset created with {len(train_dataset)} training windows and {len(val_dataset)} validation windows.")

    # Scheduler
    total_steps = (len(train_loader) * args.epochs) // args.grad_accum
    scheduler = build_trapezoid_scheduler(
        optimizer, total_steps=total_steps, base_lr=args.learning_rate, min_lr=args.min_lr,
        warmup_ratio=args.warmup_ratio, hold_ratio=args.hold_ratio,
    )

    # Checkpoint Loading
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume_from and os.path.exists(args.resume_from):
        if is_main_process:
            print(f"Resuming from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']

    # Training Loop
    if is_main_process:
        print("Starting MAE pre-training...")
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        print(f"Checkpoints will be saved to: {args.checkpoint_dir}")

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]", disable=not is_main_process)

        for batch in train_progress_bar:
            original_data = batch.to(device, non_blocking=True)
            B, T, D = original_data.shape

            optimizer.zero_grad()

            # Create mask
            num_masked = int(args.mask_ratio * T)
            masked_indices = torch.rand(original_data.shape[:2], device=device).topk(k=num_masked, dim=-1).indices
            
            masked_input = original_data.clone()
            loss_mask = torch.zeros_like(original_data, dtype=torch.bool)

            batch_indices = torch.arange(B, device=device).unsqueeze(-1)
            masked_input[batch_indices, masked_indices] = 0.0
            loss_mask[batch_indices, masked_indices] = True

            # Forward pass
            reconstructed_data = model(masked_input)
            loss = F.smooth_l1_loss(reconstructed_data[loss_mask], original_data[loss_mask])

            loss.backward()
            optimizer.step()

            if scheduler is not None and args.sched_on == "step":
                scheduler.step()

            if is_main_process:
                train_progress_bar.set_postfix(loss=loss.item())

        if scheduler is not None and args.sched_on == "epoch":
            scheduler.step()

        # Validation Loop
        model.eval()
        total_val_loss = 0
        val_count = 0
        with torch.no_grad():
            val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Val]", disable=not is_main_process)
            for batch in val_progress_bar:
                original_data = batch.to(device, non_blocking=True)
                B, T, D = original_data.shape

                # In validation, we can reconstruct the whole sequence to check general performance
                # or stick to the same masking strategy. Sticking to masking is a better test.
                num_masked = int(args.mask_ratio * T)
                masked_indices = torch.rand(original_data.shape[:2], device=device).topk(k=num_masked, dim=-1).indices
                
                masked_input = original_data.clone()
                loss_mask = torch.zeros_like(original_data, dtype=torch.bool)

                batch_indices = torch.arange(B, device=device).unsqueeze(-1)
                masked_input[batch_indices, masked_indices] = 0.0
                loss_mask[batch_indices, masked_indices] = True

                reconstructed_data = model(masked_input)
                val_loss = F.smooth_l1_loss(reconstructed_data[loss_mask], original_data[loss_mask])
                
                total_val_loss += val_loss.item() * B
                val_count += B

        avg_val_loss = total_val_loss / val_count
        if is_main_process:
            print(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.4f}")

            # Save checkpoint
            latest_checkpoint_path = os.path.join(args.checkpoint_dir, "robot_state_mae_latest.pth")
            best_checkpoint_path = os.path.join(args.checkpoint_dir, "robot_state_mae_best.pth")
            
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'loss': loss.item(),
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss
            }

            # Save latest checkpoint
            torch.save(checkpoint_data, latest_checkpoint_path)

            # Save best checkpoint if validation loss improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint_data['best_val_loss'] = best_val_loss
                torch.save(checkpoint_data, best_checkpoint_path)
                print(f"✨ New best model saved with validation loss: {avg_val_loss:.4f}")

    if is_main_process:
        print("MAE Pre-training finished.")
    
    dist.destroy_process_group()

# =====================================
# 4. Argparse and Entrypoint
# =====================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train RobotStateEncoder with MAE.")

    # Dataset & Dataloader
    parser.add_argument('--dataset_path', type=str, default="/home/najo/NAS/VLA/dataset/New_dataset", help='Path to the root dataset directory.')
    parser.add_argument('--window_size', type=int, default=65, help='Temporal window size for robot states.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size per GPU.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of dataloader workers per GPU.')
    parser.add_argument('--val_split', type=float, default=0.05, help='Proportion of the dataset to use for validation.')

    # Model & Architecture
    parser.add_argument('--model_dim', type=int, default=256, help='Dimension of the Transformer model.')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of heads in the Transformer.')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of layers in the Transformer.')
    parser.add_argument('--output_dim', type=int, default=3072, help='Output dimension of the encoder projection head.')
    parser.add_argument('--mask_ratio', type=float, default=0.5, help='Ratio of timesteps to mask.')

    # Training & Optimization
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--grad_accum', type=int, default=1, help='Number of gradient accumulation steps.')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate.')
    parser.add_argument('--warmup_ratio', type=float, default=0.03, help='Ratio of total steps for learning rate warmup.')
    parser.add_argument('--hold_ratio', type=float, default=0.02, help='Ratio of total steps for holding learning rate after warmup.')
    parser.add_argument('--sched_on', type=str, choices=['step', 'epoch'], default='step', help="When to step the scheduler: 'step' or 'epoch'.")

    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='/home/najo/NAS/VLA/Insertion_VLAv2/checkpoints', help='Directory to save checkpoints.')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to a checkpoint to resume training from.')

    args = parser.parse_args()
    main(args)
