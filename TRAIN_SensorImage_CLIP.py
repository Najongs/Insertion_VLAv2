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
import hashlib

from pathlib import Path
from functools import partial
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import math
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from torchvision import transforms

# Add project root to import custom modules
import sys
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.unified_model import ForceAwareSensorEncoder, force_bn_fp32_
from vla_datasets.unified_dataset import UnifiedVLADataset, create_unified_dataloader
from qwen_vl_utils import process_vision_info
from vla_cache_manager import VLACacheManager

# =====================================
# 0. Data Augmentation
# =====================================

class SensorAugmentation:
    """
    Sensor data augmentation for CLIP training.
    All augmentations have ≤30% probability.
    """
    def __init__(self,
                 time_mask_ratio=0.1,
                 noise_std=0.005,
                 scale_range=(0.97, 1.03)):
        self.time_mask_ratio = time_mask_ratio
        self.noise_std = noise_std
        self.scale_range = scale_range

    def __call__(self, sensor_data):
        """
        Args:
            sensor_data: (T, C=1026) - distance features (1-1025) + force (1026)
        """
        augmented = sensor_data.clone()

        # 1. Time masking (20% probability)
        if np.random.random() < 0.20:
            T = augmented.shape[0]
            num_mask = int(T * self.time_mask_ratio)
            if num_mask > 0:
                mask_indices = torch.randperm(T)[:num_mask]
                augmented[mask_indices] = 0.0

        # 2. Gaussian noise (25% probability)
        if np.random.random() < 0.25:
            noise = torch.randn_like(augmented) * self.noise_std
            # Force channel (last) gets slightly more noise
            noise[:, -1] *= 1.5
            augmented += noise

        # 3. Magnitude scaling (30% probability)
        if np.random.random() < 0.30:
            scale = np.random.uniform(*self.scale_range)
            augmented *= scale

        return augmented


class ImageAugmentation:
    """
    Hand-eye camera image augmentation for CLIP training.
    All augmentations have ≤30% probability.
    """
    def __init__(self):
        self.gaussian_blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

    def __call__(self, image):
        """
        Args:
            image: PIL Image
        Returns:
            Augmented PIL Image
        """
        try:
            # 1. Gaussian blur (15% probability)
            if np.random.random() < 0.15:
                # Convert to tensor for blur operation
                tensor_img = self.to_tensor(image)
                blurred = self.gaussian_blur(tensor_img)
                image = self.to_pil(blurred)

            # 2. Small rotation (20% probability)
            if np.random.random() < 0.20:
                angle = np.random.uniform(-3, 3)
                image = transforms.functional.rotate(image, angle)
        except Exception as e:
            # If any augmentation fails, return the original image
            print(f"Warning: Image augmentation failed with error: {e}. Returning original image.")
            pass

        return image


# =====================================
# 1. CLIP-Style Dataset
# =====================================

class SensorImageCLIPDataset(Dataset):
    """
    A wrapper dataset that provides (sensor_data, hand_eye_image) pairs
    for contrastive pre-training.

    Only includes samples that are >= target_found_timestamp for efficiency.
    Caches the filtered indices to speed up subsequent runs.
    """
    def __init__(self, unified_dataset: UnifiedVLADataset, vlm_annotations: dict = None,
                 use_augmentation: bool = True, cache_path: str = None, clip_cache_root: str = None):
        self.unified_dataset = unified_dataset
        self.hand_eye_view_keyword = "View5" # Or "Oak"
        self.vlm_annotations = vlm_annotations if vlm_annotations is not None else {}
        self.cache_path = cache_path # Cache for filtered indices

        # Cache for VLM features
        self.clip_cache_manager = None
        if clip_cache_root:
            self.clip_cache_manager = VLACacheManager(cache_dir=clip_cache_root)
            print(f"   ... Using CLIP VLM feature cache at: {clip_cache_root}")

        # Data augmentation
        self.use_augmentation = use_augmentation
        self.is_training = True  # Training mode by default
        if self.use_augmentation:
            self.sensor_aug = SensorAugmentation()
            self.image_aug = ImageAugmentation()

        # Pre-filter valid indices (samples >= target_found_timestamp)
        is_main_process = not dist.is_initialized() or dist.get_rank() == 0
        if is_main_process:
            print("📋 Filtering dataset for CLIP training (only samples >= target_found_timestamp)...")
        
        self.valid_indices = self._filter_valid_samples()
        
        if is_main_process:
            print(f"✓ Filtered: {len(self.valid_indices)}/{len(self.unified_dataset)} samples are valid ({len(self.valid_indices)/len(self.unified_dataset)*100:.1f}%)")

    def _perform_filtering(self):
        """The actual filtering logic, run by the main process if cache is missed."""
        valid_indices = []
        global_idx_offset = 0  # To track the start index of the current sub_dataset
        
        is_main_process = not dist.is_initialized() or dist.get_rank() == 0
        
        # Iterate through each sub-dataset (each is one episode)
        pbar_datasets = tqdm(self.unified_dataset.datasets, desc="   Filtering episodes", disable=not is_main_process)
        
        for sub_dataset in pbar_datasets:
            num_samples_in_episode = len(sub_dataset)
            if num_samples_in_episode == 0:
                continue

            episode_id = sub_dataset.data_dir.name
            pbar_datasets.set_postfix(episode=episode_id)

            episode_data = self.vlm_annotations.get(episode_id, {})
            target_found_timestamp = episode_data.get("target_found_timestamp")

            # 1. Skip episode if no valid target timestamp
            if target_found_timestamp is None:
                global_idx_offset += num_samples_in_episode
                continue
            try:
                target_ts = float(target_found_timestamp)
            except (ValueError, TypeError):
                global_idx_offset += num_samples_in_episode
                continue

            # 2. Get hand-eye camera images for this episode (very fast)
            hand_eye_images = []
            if isinstance(sub_dataset.images, dict):
                for view_name, image_list in sub_dataset.images.items():
                    if self.hand_eye_view_keyword in view_name:
                        hand_eye_images = image_list
                        break
                # Fallback logic from __getitem__
                if not hand_eye_images and sub_dataset.images:
                    # Look for the last view as a fallback
                    last_view_name = sorted(sub_dataset.images.keys())[-1]
                    hand_eye_images = sub_dataset.images[last_view_name]
            
            if not hand_eye_images:
                global_idx_offset += num_samples_in_episode
                continue
                
            # Create a pre-parsed list of timestamps for speed
            path_timestamps = {} # Cache parsed timestamps for the current episode
            def get_timestamp_from_path(path):
                if path in path_timestamps:
                    return path_timestamps[path]
                match = re.search(r'(\d{10,}\.\d+)\.jpg', path)
                ts = float(match.group(1)) if match else None
                path_timestamps[path] = ts
                return ts

            # 3. Iterate through samples of this episode and check timestamp
            for local_idx in range(num_samples_in_episode):
                # Find the correct image index by proportionally mapping the sample's
                # position in the pose sequence to the image sequence.
                # This is more robust than relying on vlm_idx, which was buggy.
                
                # The "time" of the sample corresponds to its pose index
                pose_idx = local_idx * sub_dataset.action_interval
                
                num_images = len(hand_eye_images)
                num_poses = sub_dataset.num_poses

                if num_poses == 0:
                    continue

                # Estimate the corresponding image index based on proportional time
                img_fraction = pose_idx / num_poses
                img_idx_in_view = min(int(img_fraction * num_images), num_images - 1)
                
                img_path = hand_eye_images[img_idx_in_view]
                
                # Get timestamp and compare
                current_ts = get_timestamp_from_path(img_path)
                if current_ts is not None and current_ts >= target_ts:
                    valid_indices.append(global_idx_offset + local_idx)
        
            global_idx_offset += num_samples_in_episode

        return valid_indices

    def _filter_valid_samples(self):
        """Filter samples, using a cache if available in a DDP-safe manner."""
        is_main_process = not dist.is_initialized() or dist.get_rank() == 0

        if self.cache_path:
            # If cache doesn't exist, main process creates it
            if is_main_process and not os.path.exists(self.cache_path):
                print("   No cache found. Filtering from scratch (this will be slow the first time)...")
                valid_indices = self._perform_filtering()
                print(f"   Saving {len(valid_indices)} valid indices to cache: {self.cache_path}")
                os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
                with open(self.cache_path, 'w') as f:
                    json.dump(valid_indices, f)

            # All processes wait for the main process to finish creating the cache
            if dist.is_initialized():
                dist.barrier()

            # All processes load from cache
            try:
                if is_main_process:
                    print(f"   Loading valid indices from cache: {self.cache_path}")
                with open(self.cache_path, 'r') as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                # This should not happen in DDP if barrier works, but is a fallback
                if is_main_process:
                    print(f"   Warning: Cache file not found or corrupted after barrier ({e}). Filtering again.")
                return self._perform_filtering()
        else:
            # No cache path provided, filter normally
            return self._perform_filtering()

    def train(self):
        """Enable augmentation for training"""
        self.is_training = True

    def eval(self):
        """Disable augmentation for validation"""
        self.is_training = False

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Map to actual index in unified_dataset
        actual_idx = self.valid_indices[idx]

        # Get the full data sample from the underlying dataset
        try:
            sample = self.unified_dataset[actual_idx]
        except (FileNotFoundError, IndexError) as e:
            print(f"Warning: Skipping index {actual_idx} due to error: {e}")
            # Return a dummy sample or the first valid sample
            sample = self.unified_dataset[self.valid_indices[0]]
        
        episode_id = sample.get("episode_id")
        vlm_idx = sample.get("vlm_idx")
        vlm_feature = None
        hand_eye_image = None

        # 1. Try to load from CLIP VLM feature cache
        if self.clip_cache_manager and episode_id is not None and vlm_idx is not None:
            vlm_feature = self.clip_cache_manager.load_cache(dataset_name=episode_id, vlm_idx=vlm_idx)

        vlm_feature_cached = vlm_feature is not None

        if vlm_feature_cached:
            # Use the cached feature, squeeze out batch and sequence dimensions
            hand_eye_image = vlm_feature.squeeze(0).squeeze(0)
            timestamp = sample.get("timestamp") # Timestamp is still needed for loss calculation
            vlm_description = self.vlm_annotations.get(episode_id, {}).get(str(timestamp), "no_vlm_description")
        else:
            # 2. Fallback to loading the image if cache misses
            hand_eye_image_path = None
            if sample["images"]:
                for img_path in sample["images"]:
                    if self.hand_eye_view_keyword in img_path:
                        hand_eye_image_path = img_path
                        break
                if not hand_eye_image_path:
                    hand_eye_image_path = sample["images"][-1]

            timestamp = None
            vlm_description = "no_vlm_description"

            if not hand_eye_image_path or not os.path.exists(hand_eye_image_path):
                hand_eye_image = Image.new('RGB', (224, 224), color='black')
            else:
                hand_eye_image = Image.open(hand_eye_image_path).convert("RGB")
                timestamp_match = re.search(r'(\d{10,}\.\d+)\.jpg', hand_eye_image_path)
                if episode_id and timestamp_match:
                    timestamp = timestamp_match.group(1)
                    vlm_description = self.vlm_annotations.get(episode_id, {}).get(timestamp, "no_vlm_description")

        # Apply sensor augmentation (only during training)
        sensor_data = sample["sensor_data"]
        if self.use_augmentation and self.is_training:
            sensor_data = self.sensor_aug(sensor_data)
            # Image augmentation is only applied if we are not using a cached feature
            if not vlm_feature_cached and isinstance(hand_eye_image, Image.Image):
                hand_eye_image = self.image_aug(hand_eye_image)

        return {
            "sensor_data": sensor_data,
            "hand_eye_image": hand_eye_image, # Can be a PIL Image or a Tensor
            "vlm_feature_cached": vlm_feature_cached,
            "vlm_description": vlm_description,
            "episode_id": episode_id,
            "timestamp": timestamp,
            "vlm_idx": vlm_idx,
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

    vlm_feature_cached_list = [sample["vlm_feature_cached"] for sample in batch]

    return {
        "sensor_data": torch.stack(sensor_tensors),
        "hand_eye_image": image_list,
        "vlm_feature_cached": vlm_feature_cached_list,
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

    def forward(self, sensor_data, images, vlm_feature_cached, image_prompts, processor):
        # Encode sensor data
        sensor_features = self.sensor_encoder(sensor_data)
        sensor_embedding = self.sensor_projection(sensor_features)

        # Process images and cached features
        live_images_batch = []
        live_prompts_batch = []
        live_indices = []
        
        image_features_list = [None] * len(images)

        for i, (img_or_feat, is_cached) in enumerate(zip(images, vlm_feature_cached)):
            if is_cached:
                image_features_list[i] = img_or_feat.to(device=self.image_encoder.device, dtype=self.image_encoder.dtype)
            else:
                live_images_batch.append(img_or_feat)
                live_prompts_batch.append(image_prompts[i])
                live_indices.append(i)

        # If there are any live images to process, run the VLM
        if live_images_batch:
            text_inputs_for_processor = []
            vision_inputs_for_processor = []
            for img, prompt_text in zip(live_images_batch, live_prompts_batch):
                messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt_text}]}]
                text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                text_inputs_for_processor.append(text_input)
                vision_input, _ = process_vision_info(messages)
                vision_inputs_for_processor.append(vision_input)

            inputs = processor(
                text=text_inputs_for_processor,
                images=vision_inputs_for_processor,
                padding=True,
                return_tensors="pt"
            ).to(device=self.image_encoder.device, dtype=self.image_encoder.dtype)

            with torch.no_grad():
                outputs = self.image_encoder(**inputs, output_hidden_states=True, return_dict=True)
            
            live_image_features = outputs.hidden_states[-1][:, -1, :]

            for i, feat in zip(live_indices, live_image_features):
                image_features_list[i] = feat

        # Stack all features into a single tensor
        image_features = torch.stack(image_features_list, dim=0)
        
        # Project and normalize
        image_embedding = self.image_projection(image_features)
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
    Only trains on samples after target_found_timestamp (earlier samples are excluded).

    Args:
        focal_gamma: Focusing parameter (gamma). Higher values focus more on hard samples.
                     gamma=0 is equivalent to standard cross-entropy.
    """
    logits = torch.matmul(sensor_embeddings, image_embeddings.T) / temperature
    labels = torch.arange(len(logits)).to(logits.device)

    # Create weights for samples based on timestamp vs target_found_timestamp
    # Default weight is 1.0 (include all samples by default)
    weights = torch.ones(len(episode_ids), device=logits.device, dtype=torch.float)

    for i, (episode_id, timestamp) in enumerate(zip(episode_ids, timestamps)):
        if episode_id is None or timestamp is None:
            weights[i] = 0.0  # Exclude invalid samples
            continue

        # Get the target_found_timestamp for this episode (not per-frame)
        episode_data = vlm_annotations.get(episode_id, {})
        target_found_timestamp = episode_data.get("target_found_timestamp")

        if target_found_timestamp is not None:
            # Convert timestamps to float for comparison
            try:
                current_ts = float(timestamp)
                target_ts = float(target_found_timestamp)

                # Exclude samples BEFORE target_found_timestamp
                # Include all samples >= target_found_timestamp
                if current_ts < target_ts:
                    weights[i] = 0.0
            except (ValueError, TypeError):
                # If conversion fails, exclude this sample
                weights[i] = 0.0
        # If no target_found_timestamp, keep weight=1.0 (include in training)

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


def temporal_smoothness_loss(sensor_embeddings, episode_ids, timestamps, vlm_annotations, margin=0.1):
    """
    Encourages temporal smoothness: embeddings from temporally close frames should be similar.
    Only considers samples after target_found_timestamp.

    Args:
        sensor_embeddings: [B, D] sensor embeddings
        episode_ids: List of episode IDs
        timestamps: List of timestamps
        vlm_annotations: Dictionary of VLM annotations with target_found_timestamp
        margin: Margin for temporal distance

    Returns:
        Temporal smoothness loss
    """
    batch_size = len(episode_ids)
    if batch_size < 2:
        return torch.tensor(0.0, device=sensor_embeddings.device)

    total_loss = 0.0
    num_pairs = 0

    # Convert timestamps to float and check if they are after target_found_timestamp
    timestamps_float = []
    valid_samples = []

    for i, (episode_id, ts) in enumerate(zip(episode_ids, timestamps)):
        try:
            if episode_id is not None and ts is not None:
                current_ts = float(ts)
                timestamps_float.append(current_ts)

                # Check if this sample is after target_found_timestamp (not per-frame)
                episode_data = vlm_annotations.get(episode_id, {})
                target_found_timestamp = episode_data.get("target_found_timestamp")

                if target_found_timestamp is not None:
                    target_ts = float(target_found_timestamp)
                    # Only include samples >= target_found_timestamp
                    valid_samples.append(current_ts >= target_ts)
                else:
                    # If no target_found_timestamp, include this sample
                    valid_samples.append(True)
            else:
                timestamps_float.append(None)
                valid_samples.append(False)
        except (ValueError, TypeError):
            timestamps_float.append(None)
            valid_samples.append(False)

    # For each sample, find temporally close samples from the same episode
    for i in range(batch_size):
        if not valid_samples[i] or episode_ids[i] is None or timestamps_float[i] is None:
            continue

        for j in range(i + 1, batch_size):
            if not valid_samples[j] or episode_ids[j] is None or timestamps_float[j] is None:
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
    Only trains on samples after target_found_timestamp.

    Args:
        temporal_weight: Weight for temporal smoothness loss (typically 0.05-0.2)
    """
    # Focal contrastive loss (main objective)
    focal_loss = focal_contrastive_loss(
        sensor_embeddings, image_embeddings, episode_ids, timestamps, vlm_annotations,
        temperature=temperature, important_weight=important_weight, focal_gamma=focal_gamma
    )

    # Temporal smoothness loss (regularization)
    temporal_loss = temporal_smoothness_loss(sensor_embeddings, episode_ids, timestamps, vlm_annotations)

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
        new_dataset_paths=args.new_dataset_paths, old_dataset_patterns=args.old_dataset_patterns,
        new_weight=args.new_weight, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False, return_dataset=True, use_cache=False  # CLIP needs actual images, not cached VL features
    )
    
    # Define a cache path for filtered indices to speed up subsequent runs
    cache_dir = Path(__file__).parent / "cache"
    cache_dir.mkdir(exist_ok=True)
    # Create a hash based on dataset paths to ensure cache validity
    dataset_paths_str = "".join(sorted(args.new_dataset_paths)) + "".join(sorted(args.old_dataset_patterns))
    dataset_hash = hashlib.md5(dataset_paths_str.encode()).hexdigest()[:8]
    cache_path = cache_dir / f"sensor_clip_indices_{dataset_hash}.json"

    # Define cache path for CLIP VLM features
    clip_cache_root = Path(args.cache_root) / "clip_vlm_features"

    clip_dataset = SensorImageCLIPDataset(
        unified_dataset,
        vlm_annotations=annotations,
        cache_path=str(cache_path),
        clip_cache_root=str(clip_cache_root)
    )

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
        if is_main_process:
            print(f"Resuming from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        
        model_to_load = clip_model.module if isinstance(clip_model, DDP) else clip_model
        
        # Check for architecture mismatch by comparing state dicts
        ckpt_state_dict = checkpoint.get('model_state_dict', {})
        current_state_dict = model_to_load.state_dict()
        
        # Filter out layers that have different shapes
        new_state_dict = {}
        incompatible_keys = []
        for key, ckpt_param in ckpt_state_dict.items():
            if key in current_state_dict:
                current_param = current_state_dict[key]
                if ckpt_param.shape == current_param.shape:
                    new_state_dict[key] = ckpt_param
                else:
                    incompatible_keys.append(key)
            else:
                # Key from checkpoint not in current model
                incompatible_keys.append(key)

        model_to_load.load_state_dict(new_state_dict, strict=False)
        
        if is_main_process:
            if incompatible_keys:
                print("⚠️ WARNING: Architecture mismatch detected. Some layers were not loaded from checkpoint:")
                for key in incompatible_keys:
                    ckpt_shape = ckpt_state_dict[key].shape if key in ckpt_state_dict else 'N/A'
                    curr_shape = current_state_dict[key].shape if key in current_state_dict else 'N/A'
                    print(f"  - Layer '{key}': Checkpoint shape {ckpt_shape}, Model shape {curr_shape}")
            else:
                print("   Model weights loaded successfully.")

        # IMPORTANT: If there was any incompatibility, do NOT load optimizer and scheduler state.
        if not incompatible_keys and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint and scheduler is not None:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if is_main_process:
                    print("   Optimizer and scheduler states loaded successfully.")
            except ValueError as e:
                if is_main_process:
                    print(f"⚠️ WARNING: Could not load optimizer state, possibly due to parameter shape mismatch. Starting with a fresh optimizer. Error: {e}")
        elif incompatible_keys:
            if is_main_process:
                print("   Skipping optimizer and scheduler state loading due to model architecture mismatch.")
        
        start_epoch = checkpoint.get('epoch', -1) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        if is_main_process:
            print(f"   Resuming training from epoch {start_epoch}.")

    # Training Loop
    if is_main_process:
        print("Starting training...")
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        print(f"Checkpoints will be saved to: {args.checkpoint_dir}")

    prompt_text = (
        "This is a robot hand-eye view with a sensorized needle. Focus on the needle tip and its interaction with the environment. "
        "Analyze the following aspects:\n"
        "1. Proximity: How close is the needle tip to the intended target and other nearby objects? (e.g., far, near, touching).\n"
        "2. Contact State: Is the needle tip making contact with any surface? Describe the nature of the contact (e.g., no contact, light touch, firm press, inserting).\n"
        "3. Certainty: How certain are you about the contact state? (High, Medium, Low)."
    )

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        clip_model.train()
        clip_dataset.train()  # Enable augmentation for training
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]", disable=not is_main_process)

        for batch in train_progress_bar:
            sensor_data = batch["sensor_data"].to(device, non_blocking=True)
            images = batch["hand_eye_image"]
            vlm_feature_cached = batch["vlm_feature_cached"]
            episode_ids = batch["episode_ids"]
            timestamps = batch["timestamps"]
            prompts_for_vlm_encoding = [prompt_text] * len(images)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                sensor_embed, image_embed = clip_model(
                    sensor_data, images, vlm_feature_cached, prompts_for_vlm_encoding, vlm_processor
                )
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
        clip_dataset.eval()  # Disable augmentation for validation
        total_val_loss, total_focal_loss, total_temporal_loss = 0, 0, 0
        val_count = 0
        with torch.no_grad():
            val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Val]", disable=not is_main_process)
            for batch in val_progress_bar:
                sensor_data = batch["sensor_data"].to(device, non_blocking=True)
                images = batch["hand_eye_image"]
                vlm_feature_cached = batch["vlm_feature_cached"]
                episode_ids = batch["episode_ids"]
                timestamps = batch["timestamps"]
                prompts_for_vlm_encoding = [prompt_text] * len(images)

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    sensor_embed, image_embed = clip_model(
                        sensor_data, images, vlm_feature_cached, prompts_for_vlm_encoding, vlm_processor
                    )
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
    parser.add_argument('--new_dataset_paths', type=str, nargs='*',
                        default=["/home/najo/NAS/VLA/dataset/New_dataset", "/home/najo/NAS/VLA/dataset/New_dataset2"],
                        help='Paths to the new format dataset directories.')
    parser.add_argument('--old_dataset_patterns', type=str, nargs='*', default=[])
    parser.add_argument('--new_weight', type=float, default=3.0, help='Weight for new datasets in weighted sampling.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of dataloader workers per GPU.')
    parser.add_argument('--val_split', type=float, default=0.05, help='Proportion of the dataset to use for validation.')
    parser.add_argument('--annotation_path', type=str, default="vlm_annotations.json", help='Path to the VLM annotations file.')
    parser.add_argument('--cache_root', type=str, default="/home/najo/NAS/VLA/dataset/cache", help='Root directory for all caches.')

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