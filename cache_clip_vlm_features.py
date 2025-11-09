"""
Cache VLM Features for CLIP Pre-training

This script pre-computes and caches the image features from the VLM 
for the samples that will be used in the `TRAIN_SensorImage_CLIP.py` script.

This avoids running the large VLM model during the training loop,
significantly speeding up the pre-training process.

This version is parallelized to run on multiple GPUs and reports skip/generation stats.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
from pathlib import Path
import hashlib

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image

# Add project root to import custom modules
import sys
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from vla_datasets.unified_dataset import create_unified_dataloader
from TRAIN_SensorImage_CLIP import SensorImageCLIPDataset, process_vision_info
from vla_cache_manager import VLACacheManager

def simple_collate_fn(batch):
    """
    Custom collate function to handle PIL Images.
    It converts a list of sample dictionaries into a single dictionary where each value
    is a list of the values from the samples. This avoids the default_collate error.
    e.g., [{'img': <A>, 'id': 1}, {'img': <B>, 'id': 2}] -> {'img': [<A>, <B>], 'id': [1, 2]}
    """
    keys = batch[0].keys()
    collated_batch = {key: [d[key] for d in batch] for key in keys}
    return collated_batch

def main(args):
    # DDP Setup
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    is_main_process = local_rank == 0

    # Use a dedicated cache directory for CLIP features
    clip_cache_dir = Path(args.cache_root) / "clip_vlm_features"
    if is_main_process:
        if not clip_cache_dir.exists():
            print(f"Creating CLIP VLM feature cache directory: {clip_cache_dir}")
            clip_cache_dir.mkdir(parents=True)

    cache_manager = VLACacheManager(cache_dir=str(clip_cache_dir))
    
    if is_main_process:
        print(f"Using device: {device}")

    # 1. Load VLM
    if is_main_process: print("Loading VLM...")
    vlm_processor = AutoProcessor.from_pretrained(args.vlm_model, trust_remote_code=True)
    vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.vlm_model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    vlm_model.eval()
    if is_main_process: print("VLM loaded.")

    # 2. Create the dataset to find the valid samples
    if is_main_process: print("Creating dataset to identify valid samples for caching...")
    annotations = {}
    annotation_path = Path(args.annotation_path)
    if annotation_path.exists():
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
        if is_main_process:
            print(f"Loaded {len(annotations)} VLM annotations from {annotation_path}")

    index_cache_dir = Path(__file__).parent / "cache"
    index_cache_dir.mkdir(exist_ok=True)
    dataset_paths_str = "".join(sorted(args.new_dataset_paths)) + "".join(sorted(args.old_dataset_patterns))
    dataset_hash = hashlib.md5(dataset_paths_str.encode()).hexdigest()[:8]
    index_cache_path = index_cache_dir / f"sensor_clip_indices_{dataset_hash}.json"

    unified_dataset = create_unified_dataloader(
        new_dataset_paths=args.new_dataset_paths,
        old_dataset_patterns=args.old_dataset_patterns,
        return_dataset=True,
        use_cache=False,
    )

    clip_dataset = SensorImageCLIPDataset(
        unified_dataset,
        vlm_annotations=annotations,
        use_augmentation=False,
        cache_path=str(index_cache_path)
    )
    if is_main_process:
        print(f"Found {len(clip_dataset)} valid samples to cache.")

    sampler = DistributedSampler(clip_dataset, shuffle=False)
    dataloader = DataLoader(
        clip_dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=simple_collate_fn  # Use the custom collate function
    )

    # 3. Iterate and cache features
    prompt_text = (
        "This is a robot hand-eye view with a sensorized needle. Focus on the needle tip and its interaction with the environment. "
        "Analyze the following aspects:\n"
        "1. Proximity: How close is the needle tip to the intended target and other nearby objects? (e.g., far, near, touching).\n"
        "2. Contact State: Is the needle tip making contact with any surface? Describe the nature of the contact (e.g., no contact, light touch, firm press, inserting).\n"
        "3. Certainty: How certain are you about the contact state? (High, Medium, Low)."
    )
    
    # Counters for this specific process
    local_skipped_count = 0
    local_generated_count = 0

    pbar = None
    if is_main_process:
        pbar = tqdm(total=len(clip_dataset), desc="Caching VLM Features")
    
    for i, sample in enumerate(dataloader):
        image = sample["hand_eye_image"][0]
        episode_id = sample["episode_id"][0]
        vlm_idx = sample["vlm_idx"][0]

        if vlm_idx is None:
            if is_main_process: pbar.update(1)
            continue

        if cache_manager.cache_exists(dataset_name=episode_id, vlm_idx=vlm_idx):
            local_skipped_count += 1
            if is_main_process: pbar.update(1)
            continue

        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]}]
        text_input = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        vision_input, _ = process_vision_info(messages)
        
        inputs = vlm_processor(
            text=[text_input],
            images=[vision_input],
            padding=True,
            return_tensors="pt"
        ).to(device=vlm_model.device, dtype=vlm_model.dtype)

        with torch.no_grad():
            outputs = vlm_model(**inputs, output_hidden_states=True, return_dict=True)
            image_features = outputs.hidden_states[-1][:, -1, :]

        cache_manager.save_cache(
            dataset_name=episode_id,
            vlm_idx=vlm_idx,
            vl_features=image_features
        )
        local_generated_count += 1
        
        if is_main_process:
            pbar.update(1)
            pbar.set_postfix({"last_cached": f"{episode_id}_vlm{vlm_idx}"})

    # Synchronize and gather stats
    counts = torch.tensor([local_skipped_count, local_generated_count], dtype=torch.long, device=device)
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)

    if is_main_process:
        if pbar: pbar.close()
        
        total_skipped = counts[0].item()
        total_generated = counts[1].item()
        total_processed = total_skipped + total_generated
        
        print("\n✅ VLM feature caching complete.")
        
        if total_processed > 0:
            skip_ratio = (total_skipped / total_processed) * 100 if total_processed > 0 else 0
            print(f"   Summary: {total_processed}/{len(clip_dataset)} samples processed.")
            print(f"   - Generated: {total_generated}")
            print(f"   - Skipped:   {total_skipped} ({skip_ratio:.2f}%)")
        else:
            print("   No samples were processed.")

        stats = cache_manager.get_cache_stats()
        print("Cache stats:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
            
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache VLM features for CLIP pre-training (Parallel Version).")

    # Dataset paths
    parser.add_argument('--new_dataset_paths', type=str, nargs='*',
                        default=["/home/najo/NAS/VLA/dataset/New_dataset", "/home/najo/NAS/VLA/dataset/New_dataset2"],
                        help='Paths to the new format dataset directories.')
    parser.add_argument('--old_dataset_patterns', type=str, nargs='*', default=[])
    parser.add_argument('--annotation_path', type=str, default="vlm_annotations.json", help='Path to the VLM annotations file.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers per GPU.')

    # Model
    parser.add_argument('--vlm_model', type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help='Path to the VLM model for image encoding.')

    # Cache settings
    parser.add_argument('--cache_root', type=str, default="/home/najo/NAS/VLA/dataset/cache", help='Root directory for all caches.')
    
    # DDP
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for DDP. Passed by torchrun.')

    args = parser.parse_args()
    main(args)