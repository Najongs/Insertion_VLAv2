"""
Sensor Encoder Pre-training Script using CLIP-style Contrastive Learning

This script pre-trains the SensorEncoder by matching its output with the
features from a Vision-Language Model (VLM) that sees the corresponding
hand-eye camera view.

Methodology:
1. A batch consists of (sensor_data, hand_eye_image) pairs.
2. Sensor data is encoded by the SensorEncoder.
3. The hand-eye image is encoded by a VLM (e.g., Qwen2.5-VL) with a
   specific prompt asking it to identify contact events.
4. Both outputs are projected into a shared embedding space.
5. A contrastive loss (CLIP loss) pulls the embeddings of matching
   (sensor, image) pairs together and pushes non-matching pairs apart.
6. The trained SensorEncoder weights are saved for downstream tasks.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image
from tqdm import tqdm
import re # Added for timestamp extraction

from pathlib import Path
from functools import partial
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import math
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root to import custom modules
import sys
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.unified_model import ForceAwareSensorEncoder, force_bn_fp32_
from vla_datasets.unified_dataset import UnifiedVLADataset, create_unified_dataloader
from qwen_vl_utils import process_vision_info

# =====================================
# 1. CLIP-Style Dataset
# =====================================

class SensorImageCLIPDataset(Dataset):
    """
    A wrapper dataset that provides (sensor_data, hand_eye_image) pairs
    for contrastive pre-training.
    """
    def __init__(self, unified_dataset: UnifiedVLADataset, vlm_annotations: dict = None):
        self.unified_dataset = unified_dataset
        self.hand_eye_view_keyword = "View5" # Or "Oak"
        # vlm_annotations now maps episode_id to a description string
        self.vlm_annotations = vlm_annotations if vlm_annotations is not None else {}

    def __len__(self):
        return len(self.unified_dataset)

    def __getitem__(self, idx):
        # Get the full data sample from the underlying dataset
        try:
            sample = self.unified_dataset[idx]
        except (FileNotFoundError, IndexError) as e:
            print(f"Warning: Skipping index {idx} due to error: {e}")
            # Return a dummy sample or the first sample
            sample = self.unified_dataset[0]
        
        episode_id = sample.get("episode_id")

        # Find the hand-eye camera image path
        hand_eye_image_path = None
        if sample["images"]:
            for img_path in sample["images"]:
                if self.hand_eye_view_keyword in img_path:
                    hand_eye_image_path = img_path
                    break
            # Fallback to the last image if no specific keyword match
            if not hand_eye_image_path:
                hand_eye_image_path = sample["images"][-1]

        # Default description and timestamp
        vlm_description = "no_vlm_description"
        timestamp = None

        if not hand_eye_image_path or not os.path.exists(hand_eye_image_path):
             # Return a blank image if no valid path is found
            hand_eye_image = Image.new('RGB', (224, 224), color = 'black')
        else:
            hand_eye_image = Image.open(hand_eye_image_path).convert("RGB")
            # Extract timestamp and get per-frame description
            timestamp_match = re.search(r'(\d{10,}\.\d+)\.jpg', hand_eye_image_path)
            if episode_id and timestamp_match:
                timestamp = timestamp_match.group(1)
                vlm_description = self.vlm_annotations.get(episode_id, {}).get(timestamp, "no_vlm_description")

        return {
            "sensor_data": sample["sensor_data"],
            "hand_eye_image": hand_eye_image,
            "vlm_description": vlm_description,
            "episode_id": episode_id,
            "timestamp": timestamp
        }
def clip_collate_fn(batch, window_size):
    """Robust collate function that pads or truncates sensor data to a fixed size."""
    sensor_tensors = []
    image_list = []
    vlm_descriptions = []
    episode_ids = []
    timestamps = []

    for sample in batch:
        sensor = sample["sensor_data"]
        # Truncate or pad the sensor data to the fixed window size
        if sensor.shape[0] > window_size:
            sensor = sensor[:window_size, :]
        elif sensor.shape[0] < window_size:
            pad = torch.zeros((window_size - sensor.shape[0], sensor.shape[1]), dtype=sensor.dtype)
            sensor = torch.cat([sensor, pad], dim=0)
        sensor_tensors.append(sensor)
        image_list.append(sample["hand_eye_image"])
        vlm_descriptions.append(sample["vlm_description"])
        episode_ids.append(sample["episode_id"])
        timestamps.append(sample["timestamp"])

    return {
        "sensor_data": torch.stack(sensor_tensors),
        "hand_eye_image": image_list,
        "vlm_description": vlm_descriptions,
        "episode_ids": episode_ids,
        "timestamps": timestamps
    }


# =====================================
# 2. CLIP Model Definition
# =====================================

class CLIPModel(nn.Module):
    """
    A model that holds the image and sensor encoders for CLIP training.
    """
    def __init__(self, sensor_encoder, image_encoder, sensor_output_dim, embedding_dim=512):
        super().__init__()
        self.sensor_encoder = sensor_encoder
        self.image_encoder = image_encoder # This is the Qwen-VL model

        image_output_dim = self.image_encoder.config.hidden_size

        # Projection heads to map both to the same embedding space
        self.sensor_projection = nn.Linear(sensor_output_dim, embedding_dim)
        self.image_projection = nn.Linear(image_output_dim, embedding_dim)

    def forward(self, sensor_data, images, image_prompts, processor):
        # Encode sensor data
        sensor_features = self.sensor_encoder(sensor_data)
        sensor_embedding = self.sensor_projection(sensor_features)

        # The VLM's chat template requires specific formatting. We process each
        # image and prompt pair individually before batching them for the processor.
        text_inputs_for_processor = []
        vision_inputs_for_processor = []

        for img, prompt_text in zip(images, image_prompts):
            # Construct messages for process_vision_info
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt_text}
                ]
            }]
            
            # Apply chat template to get the text input for the VLM
            text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            text_inputs_for_processor.append(text_input)

            # Process vision info to get the image input for the VLM
            vision_input, _ = process_vision_info(messages)
            vision_inputs_for_processor.append(vision_input)

        # Now, pass to processor
        inputs = processor(
            text=text_inputs_for_processor, # List of text strings
            images=vision_inputs_for_processor, # List of image inputs (which are lists of PIL images)
            padding=True,
            return_tensors="pt"
        ).to(device=self.image_encoder.device, dtype=self.image_encoder.dtype)

        # The VLM is frozen, so no gradients are needed here
        with torch.no_grad():
            outputs = self.image_encoder(**inputs, output_hidden_states=True, return_dict=True)
        
        # Use the embedding of the last token as the image representation
        image_features = outputs.hidden_states[-1][:, -1, :]
        image_embedding = self.image_projection(image_features)

        # Normalize embeddings for cosine similarity
        sensor_embedding = F.normalize(sensor_embedding, p=2, dim=-1)
        image_embedding = F.normalize(image_embedding, p=2, dim=-1)

        return sensor_embedding, image_embedding

# =====================================
# 3. Contrastive Loss with Focal Loss + Temporal Smoothness
# =====================================

def focal_contrastive_loss(sensor_embeddings, image_embeddings, episode_ids, timestamps, vlm_annotations,
                          temperature=0.07, important_weight=10.0, focal_gamma=2.0):
    """
    Focal Loss variant of contrastive loss that focuses more on hard samples.
    Applies higher weight to samples after target_found_timestamp.

    Args:
        focal_gamma: Focusing parameter (gamma). Higher values focus more on hard samples.
                     gamma=0 is equivalent to standard cross-entropy.
    """
    logits = torch.matmul(sensor_embeddings, image_embeddings.T) / temperature
    labels = torch.arange(len(logits)).to(logits.device)

    # Create weights for samples based on timestamp vs target_found_timestamp
    weights = torch.ones(len(episode_ids), device=logits.device, dtype=torch.float)

    for i, (episode_id, timestamp) in enumerate(zip(episode_ids, timestamps)):
        if episode_id is None or timestamp is None:
            continue

        # Get the target_found_timestamp for this episode
        episode_data = vlm_annotations.get(episode_id, {})
        target_found_timestamp = episode_data.get("target_found_timestamp")

        if target_found_timestamp is not None:
            # Convert timestamps to float for comparison
            try:
                current_ts = float(timestamp)
                target_ts = float(target_found_timestamp)

                # If current timestamp is >= target_found_timestamp, apply higher weight
                if current_ts >= target_ts:
                    weights[i] = important_weight
            except (ValueError, TypeError):
                # If conversion fails, skip this sample
                continue

    # Focal Loss implementation
    # Calculate probabilities
    probs_sensor = F.softmax(logits, dim=1)
    probs_image = F.softmax(logits.T, dim=1)

    # Get probabilities of correct classes
    probs_sensor_correct = probs_sensor[torch.arange(len(labels)), labels]
    probs_image_correct = probs_image[torch.arange(len(labels)), labels]

    # Focal loss modulation factor: (1 - p_t)^gamma
    focal_weight_sensor = (1.0 - probs_sensor_correct) ** focal_gamma
    focal_weight_image = (1.0 - probs_image_correct) ** focal_gamma

    # Calculate cross entropy loss without reduction
    loss_sensor_unreduced = F.cross_entropy(logits, labels, reduction='none')
    loss_image_unreduced = F.cross_entropy(logits.T, labels, reduction='none')

    # Apply focal weights and timestamp-based weights
    loss_sensor = (loss_sensor_unreduced * focal_weight_sensor * weights).mean()
    loss_image = (loss_image_unreduced * focal_weight_image * weights).mean()

    return (loss_sensor + loss_image) / 2


def temporal_smoothness_loss(sensor_embeddings, episode_ids, timestamps, margin=0.1):
    """
    Encourages temporal smoothness: embeddings from temporally close frames should be similar.

    Args:
        sensor_embeddings: [B, D] sensor embeddings
        episode_ids: List of episode IDs
        timestamps: List of timestamps
        margin: Margin for temporal distance

    Returns:
        Temporal smoothness loss
    """
    batch_size = len(episode_ids)
    if batch_size < 2:
        return torch.tensor(0.0, device=sensor_embeddings.device)

    total_loss = 0.0
    num_pairs = 0

    # Convert timestamps to float for comparison
    timestamps_float = []
    for ts in timestamps:
        try:
            if ts is not None:
                timestamps_float.append(float(ts))
            else:
                timestamps_float.append(None)
        except (ValueError, TypeError):
            timestamps_float.append(None)

    # For each sample, find temporally close samples from the same episode
    for i in range(batch_size):
        if episode_ids[i] is None or timestamps_float[i] is None:
            continue

        for j in range(i + 1, batch_size):
            if episode_ids[j] is None or timestamps_float[j] is None:
                continue

            # Only compare samples from the same episode
            if episode_ids[i] == episode_ids[j]:
                time_diff = abs(timestamps_float[i] - timestamps_float[j])

                # If frames are close in time (within 0.5 seconds), encourage similarity
                if time_diff < 0.5:
                    # L2 distance between embeddings
                    distance = torch.norm(sensor_embeddings[i] - sensor_embeddings[j], p=2)

                    # Encourage distance to be small
                    # Scale loss by how close they are in time (closer = more penalty)
                    temporal_weight = 1.0 - (time_diff / 0.5)  # 1.0 for same frame, 0.0 for 0.5s apart
                    total_loss += distance * temporal_weight
                    num_pairs += 1

    if num_pairs == 0:
        return torch.tensor(0.0, device=sensor_embeddings.device)

    return total_loss / num_pairs


def combined_loss(sensor_embeddings, image_embeddings, episode_ids, timestamps, vlm_annotations,
                 temperature=0.07, important_weight=10.0, focal_gamma=2.0,
                 temporal_weight=0.1):
    """
    Combined loss: Focal Contrastive Loss + Temporal Smoothness Loss

    Args:
        temporal_weight: Weight for temporal smoothness loss (typically 0.05-0.2)
    """
    # Focal contrastive loss (main objective)
    focal_loss = focal_contrastive_loss(
        sensor_embeddings, image_embeddings, episode_ids, timestamps, vlm_annotations,
        temperature=temperature, important_weight=important_weight, focal_gamma=focal_gamma
    )

    # Temporal smoothness loss (regularization)
    temporal_loss = temporal_smoothness_loss(sensor_embeddings, episode_ids, timestamps)

    # Combine losses
    total_loss = focal_loss + temporal_weight * temporal_loss

    return total_loss, focal_loss, temporal_loss

# =====================================
# 4. Main Training Function
# =====================================




def main(args):
    # DDP Setup
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    is_main_process = local_rank == 0

    if is_main_process:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")

    # VLM (Image Encoder) Setup
    if is_main_process: print("Loading VLM for Image Encoder...")
    vlm_processor = AutoProcessor.from_pretrained(args.vlm_model, trust_remote_code=True)
    vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.vlm_model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    vlm_model.eval()
    for param in vlm_model.parameters():
        param.requires_grad = False
    if is_main_process: print("VLM loaded and frozen.")

    # Sensor Encoder Setup
    if is_main_process: print("Initializing Sensor Encoder...")
    sensor_encoder = ForceAwareSensorEncoder(
        temporal_length=args.sensor_window_size, output_dim=args.sensor_output_dim,
        use_transformer=True, num_transformer_layers=2
    )
    sensor_encoder.to(device, dtype=torch.bfloat16)
    force_bn_fp32_(sensor_encoder)
    if is_main_process: print("Sensor Encoder initialized.")

    # CLIP Model & Optimizer
    clip_model = CLIPModel(
        sensor_encoder=sensor_encoder, image_encoder=vlm_model,
        sensor_output_dim=args.sensor_output_dim, embedding_dim=args.embedding_dim
    ).to(device)
    clip_model = DDP(clip_model, device_ids=[local_rank], find_unused_parameters=True)
    optimizer = torch.optim.AdamW(clip_model.parameters(), lr=args.learning_rate)

    # Dataset and DataLoader
    if is_main_process: 
        print("Creating dataset...")

    # Load VLM annotations (on all processes)
    annotations = {}
    annotation_path = Path(args.annotation_path)
    if annotation_path.exists():
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
        if is_main_process:
            print(f"Loaded {len(annotations)} VLM annotations from {annotation_path}")
    elif is_main_process:
        print(f"Warning: Annotation file not found at {annotation_path}. Proceeding without VLM-based weighting.")

    unified_dataset = create_unified_dataloader(
        new_dataset_path=args.new_dataset_path, old_dataset_patterns=args.old_dataset_patterns,
        new_weight=args.new_weight, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False, return_dataset=True
    )
    clip_dataset = SensorImageCLIPDataset(unified_dataset, vlm_annotations=annotations)

    # Split dataset
    val_size = int(len(clip_dataset) * args.val_split)
    train_size = len(clip_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(clip_dataset, [train_size, val_size])

    if is_main_process:
        print(f"Dataset created with {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    collate_fn_with_window = partial(clip_collate_fn, window_size=args.sensor_window_size)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, collate_fn=collate_fn_with_window, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, num_workers=args.num_workers,
        collate_fn=collate_fn_with_window, pin_memory=True
    )

    # Scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.min_lr)

    # Checkpoint Loading
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume_from and os.path.exists(args.resume_from):
        if is_main_process: print(f"Resuming from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        clip_model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if 'best_val_loss' in checkpoint: best_val_loss = checkpoint['best_val_loss']

    # Training Loop
    if is_main_process:
        print("Starting training...")
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        print(f"Checkpoints will be saved to: {args.checkpoint_dir}")

    prompt_text = "This is a robot hand-eye view. A sensor is attached to the tip of the needle. Analyze the image to understand the surrounding objects and, most importantly, determine if the needle tip is currently making contact with any object."

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        clip_model.train()
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]", disable=not is_main_process)

        for batch in train_progress_bar:
            sensor_data = batch["sensor_data"].to(device, non_blocking=True)
            images = batch["hand_eye_image"]
            episode_ids = batch["episode_ids"]
            timestamps = batch["timestamps"]
            prompts_for_vlm_encoding = [prompt_text] * len(images)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                sensor_embed, image_embed = clip_model(sensor_data, images, prompts_for_vlm_encoding, vlm_processor)
                total_loss, focal_loss, temporal_loss = combined_loss(
                    sensor_embed, image_embed, episode_ids, timestamps, annotations,
                    temperature=args.temperature,
                    important_weight=args.important_weight,
                    focal_gamma=args.focal_gamma,
                    temporal_weight=args.temporal_weight
                )

            total_loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            if is_main_process:
                train_progress_bar.set_postfix(
                    loss=total_loss.item(),
                    focal=focal_loss.item(),
                    temporal=temporal_loss.item(),
                    lr=optimizer.param_groups[0]['lr']
                )

        # Validation Loop
        clip_model.eval()
        total_val_loss, total_focal_loss, total_temporal_loss = 0, 0, 0
        val_count = 0
        with torch.no_grad():
            val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Val]", disable=not is_main_process)
            for batch in val_progress_bar:
                sensor_data = batch["sensor_data"].to(device, non_blocking=True)
                images = batch["hand_eye_image"]
                episode_ids = batch["episode_ids"]
                timestamps = batch["timestamps"]
                prompts_for_vlm_encoding = [prompt_text] * len(images)

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    sensor_embed, image_embed = clip_model(sensor_data, images, prompts_for_vlm_encoding, vlm_processor)
                    val_total, val_focal, val_temporal = combined_loss(
                        sensor_embed, image_embed, episode_ids, timestamps, annotations,
                        temperature=args.temperature,
                        important_weight=args.important_weight,
                        focal_gamma=args.focal_gamma,
                        temporal_weight=args.temporal_weight
                    )

                batch_size = len(images)
                total_val_loss += val_total.item() * batch_size
                total_focal_loss += val_focal.item() * batch_size
                total_temporal_loss += val_temporal.item() * batch_size
                val_count += batch_size
                val_progress_bar.set_postfix(loss=val_total.item())

        avg_val_loss = total_val_loss / val_count if val_count > 0 else 0
        avg_focal_loss = total_focal_loss / val_count if val_count > 0 else 0
        avg_temporal_loss = total_temporal_loss / val_count if val_count > 0 else 0
        
        if is_main_process:
            print(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.4f} (Focal: {avg_focal_loss:.4f}, Temporal: {avg_temporal_loss:.4f})")

            # Save Checkpoints
            latest_checkpoint_path = os.path.join(args.checkpoint_dir, "sensor_clip_latest.pth")
            best_checkpoint_path = os.path.join(args.checkpoint_dir, "sensor_clip_best.pth")
            
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': clip_model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'loss': avg_val_loss, # Save average validation loss
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss
            }

            torch.save(checkpoint_data, latest_checkpoint_path)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint_data['best_val_loss'] = best_val_loss
                torch.save(checkpoint_data, best_checkpoint_path)
                print(f"✨ New best model saved with validation loss: {avg_val_loss:.4f}")

    if is_main_process: print("Training finished.")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train Sensor Encoder with CLIP-style loss.")

    # Dataset & Dataloader
    parser.add_argument('--new_dataset_path', type=str, default="/home/najo/NAS/VLA/dataset/New_dataset", help='Path to the new format dataset directory.')
    parser.add_argument('--old_dataset_patterns', type=str, nargs='*', default=[])
    parser.add_argument('--new_weight', type=float, default=3.0, help='Weight for new datasets in weighted sampling.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of dataloader workers per GPU.')
    parser.add_argument('--val_split', type=float, default=0.05, help='Proportion of the dataset to use for validation.')
    parser.add_argument('--annotation_path', type=str, default="vlm_annotations.json", help='Path to the VLM annotations file.')

    # Model & Architecture
    parser.add_argument('--vlm_model', type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help='Path to the VLM model for image encoding.')
    parser.add_argument('--sensor_window_size', type=int, default=60, help='Temporal window size for sensor data.')
    parser.add_argument('--sensor_output_dim', type=int, default=3072, help='Output dimension of the sensor encoder.')
    parser.add_argument('--embedding_dim', type=int, default=512, help='Dimension of the shared embedding space.')

    # Training & Optimization
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature for contrastive loss.')
    parser.add_argument('--important_weight', type=float, default=10.0, help='Weight for important samples (e.g., post-contact) in the loss function.')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focusing parameter for Focal Loss. Higher values focus more on hard samples.')
    parser.add_argument('--temporal_weight', type=float, default=0.1, help='Weight for the temporal smoothness regularization loss.')
    parser.add_argument('--grad_accum', type=int, default=1, help='Number of gradient accumulation steps.')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate for cosine scheduler.')

    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='/home/najo/NAS/VLA/Insertion_VLAv2/checkpoints', help='Directory to save checkpoints.')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to a checkpoint to resume training from.')

    args = parser.parse_args()
    main(args)