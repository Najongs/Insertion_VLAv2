import os
import sys
import math

from pathlib import Path
import hashlib, fcntl

from tqdm import tqdm

import torch
import torch.distributed as dist

from qwen_vl_utils import process_vision_info
from torch.utils.data import DataLoader, DistributedSampler

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vla_datasets.unified_dataset import unified_collate_fn
from vla_cache_manager import get_cache_manager

# =====================================
# 1️⃣ Action Expert (Temporal Decoder)
# =====================================
def build_vl_cache_distributed_optimized(
    model,
    dataset,
    device="cuda",
    *,
    batch_size=16,          # DataLoader 배치 (VRAM 24GB면 2~4 권장)
    num_workers=8,
    prefetch_factor=4,
    micro_bs=1,            # 마이크로 배치 (OOM 시 자동 백오프)
    cache_dir_fallback="/home/najo/NAS/VLA/dataset/cache/qwen_vl_features",
):
    """
    완전 고정 캐싱 시스템 (VLACacheManager 사용):
      - 마이크로배칭 + OOM 백오프
      - use_cache=False (KV cache 비활성화)
      - 캐시 경로: {dataset_name}_vlm{vlm_idx}.pt (instruction/image 변경에 영향 없음)
      - Atomic save + 캐시 용량 제한 자동 관리
      - tqdm 진행률, miss/skipped 통계 표시

    model 요구사항:
      - model.vl_model, model.processor 필요
      - (선택) model.cache_dir 있으면 사용, 없으면 cache_dir_fallback 사용
    """

    distributed = dist.is_available() and dist.is_initialized()
    if distributed:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    # base cache dir
    base_cache_dir = getattr(model, "cache_dir", None)
    if base_cache_dir is None:
        base_cache_dir = Path(cache_dir_fallback)
    else:
        base_cache_dir = Path(base_cache_dir)

    # VLACacheManager 초기화
    cache_mgr = get_cache_manager(
        cache_dir=str(base_cache_dir),
        cache_limit_gb=50.0
    )

    # ---------------------------
    # DataLoader (샘플 분배 보장)
    # ---------------------------
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        shuffle=False if sampler else False,
        collate_fn=unified_collate_fn,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=False,
        persistent_workers=False,
    )

    total_local = math.ceil(len(dataset) / world_size)
    print(f"[Rank {rank}] Assigned ~{total_local} samples for caching.")
    current_device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    print(f"[Rank {rank}] CUDA ready: {torch.cuda.is_available()}, device={current_device}")

    # ---------------------------
    # 캐싱 루프
    # ---------------------------
    if hasattr(model, "eval"):
        model.eval()

    total_cached, total_skipped, total_processed = 0, 0, 0
    pbar = tqdm(
        total=total_local,
        desc=f"[Rank {rank}] Caching progress",
        dynamic_ncols=True,
        disable=(rank != 0)
    )

    with torch.inference_mode():
        for batch_idx, batch in enumerate(data_loader):
            texts = batch["instruction"]
            image_paths_list = batch["images"]
            cache_keys = batch["cache_keys"]
            vlm_indices = batch["vlm_indices"]

            # --- 미스/스킵 분리 (VLACacheManager 사용) ---
            miss_items = []
            for cache_key, vlm_idx, txt, views in zip(cache_keys, vlm_indices, texts, image_paths_list):
                # cache_key format: "{dataset_name}_vlm{vlm_idx}"
                # Extract dataset_name
                dataset_name = cache_key.rsplit("_vlm", 1)[0]

                if not cache_mgr.cache_exists(dataset_name, vlm_idx):
                    miss_items.append({
                        "text": txt,
                        "views": views,
                        "dataset_name": dataset_name,
                        "vlm_idx": vlm_idx
                    })
                else:
                    total_skipped += 1

            total_processed += len(cache_keys)
            if not miss_items:
                pbar.update(len(cache_keys))
                if rank == 0:
                    cached_ratio = (total_cached / max(1, total_processed)) * 100
                    pbar.set_postfix({
                        "cached": total_cached,
                        "skipped": total_skipped,
                        "miss%": f"{100 - cached_ratio:.1f}%",
                        "GPU": f"{torch.cuda.memory_allocated(device)/1e9:.1f}GB"
                    })
                continue

            # --- 메시지 전처리 (CPU) ---
            messages_list = []
            for item in miss_items:
                txt, views = item["text"], item["views"]
                msg_content = [{"type": "image", "image": v} for v in views if v is not None]
                msg_content.append({"type": "text", "text": txt})
                messages_list.append([{"role": "user", "content": msg_content}])

            processed_texts, vision_inputs_list = [], []
            for messages in messages_list:
                text = model.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                vision_inputs, _ = process_vision_info(messages)
                processed_texts.append(text)
                vision_inputs_list.append(vision_inputs)

            # --- 마이크로배칭 + OOM 백오프 ---
            start = 0
            _micro_bs = max(1, micro_bs)
            while start < len(miss_items):
                end = min(start + _micro_bs, len(miss_items))
                sub_items  = miss_items[start:end]
                sub_texts  = processed_texts[start:end]
                sub_vision = vision_inputs_list[start:end]

                try:
                    inputs = model.processor(
                        text=sub_texts,
                        images=sub_vision,
                        padding=True,
                        return_tensors="pt"
                    ).to(device=device, dtype=torch.bfloat16, non_blocking=True)

                    outputs = model.vl_model(
                        **inputs,
                        output_hidden_states=True,
                        use_cache=False,          # ✅ 메모리 절감
                        return_dict=True
                    )
                    vl_tokens_batch = outputs.hidden_states[-1]
                    pooled_batch = vl_tokens_batch.mean(dim=1, keepdim=True)

                    for j, item in enumerate(sub_items):
                        pooled_single = pooled_batch[j:j+1]
                        # VLACacheManager로 저장
                        cache_mgr.save_cache(
                            dataset_name=item["dataset_name"],
                            vlm_idx=item["vlm_idx"],
                            vl_features=pooled_single
                        )
                        total_cached += 1

                    # 정리
                    del inputs, outputs, vl_tokens_batch, pooled_batch
                    torch.cuda.empty_cache()

                    start = end  # 다음 마이크로 배치로 진행

                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        torch.cuda.empty_cache()
                        if _micro_bs == 1:
                            raise  # 더 줄일 수 없음
                        _micro_bs = max(1, _micro_bs // 2)
                        if rank == 0:
                            print(f"[OOM] Lowering micro_bs to #{_micro_bs} and retrying...")
                        continue
                    else:
                        raise

            # --- 진행률 업데이트 ---
            pbar.update(len(cache_keys))
            if rank == 0:
                cached_ratio = (total_cached / max(1, total_processed)) * 100
                pbar.set_postfix({
                    "cached": total_cached,
                    "skipped": total_skipped,
                    "miss%": f"{100 - cached_ratio:.1f}%",
                    "GPU": f"{torch.cuda.memory_allocated(device)/1e9:.1f}GB"
                })

            # Note: Cache limit is automatically enforced by VLACacheManager.save_cache()

    pbar.close()
    print(f"[Rank {rank}] ✅ Finished. Cached {total_cached} / Skipped {total_skipped}")
    dist.barrier()
    if rank == 0:
        print("🚀 All ranks finished caching. Cache is ready for training.")
