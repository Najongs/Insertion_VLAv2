# VLA Cache Migration Guide

## 📊 현재 상태

### 기존 캐시 시스템
- **파일 수**: 1,085,457개
- **총 용량**: 8.4GB
- **파일명 형식**: Hash 기반 (예: `00002a20a20e3399f3c7d146.pt`)
- **문제점**: Instruction이나 image path가 바뀌면 캐시를 못 찾음

### 새로운 캐시 시스템
- **파일명 형식**: `{dataset_name}_vlm{vlm_idx}.pt`
- **예시**:
  - `recv_all_20251027_170308_vlm0.pt`
  - `episode_20251030_025119_vlm150.pt`
- **장점**: 완전히 결정론적, instruction/image path 변경에 강건

---

## 🔄 마이그레이션 옵션

### Option 1: 캐시 재생성 (권장)

**장점**:
- ✅ 깨끗한 시작
- ✅ 새로운 시스템 완전 활용
- ✅ 디스크 공간 절약 (중복 제거)

**단점**:
- ⏱️ 시간 소요 (GPU 사용)

**절차**:
```bash
# 1. 기존 캐시 백업 (선택사항)
mv /home/najo/NAS/VLA/dataset/cache/qwen_vl_features \
   /home/najo/NAS/VLA/dataset/cache/qwen_vl_features_old_hash_backup

# 2. 새 캐시 디렉토리 생성
mkdir -p /home/najo/NAS/VLA/dataset/cache/qwen_vl_features

# 3. 새로운 캐시 생성 (Make_VL_cache.py 사용)
# 방법은 아래 "캐시 재생성 방법" 참조
```

---

### Option 2: 두 시스템 병행 (임시)

**방법**: 새로운 캐시를 별도 디렉토리에 생성
```bash
# 새 캐시를 다른 위치에 생성
mkdir -p /home/najo/NAS/VLA/dataset/cache/qwen_vl_features_new

# TRAIN_Unified.py에서 cache_dir 변경
# 또는 환경 변수 사용
```

**장점**:
- ✅ 기존 캐시 보존
- ✅ 점진적 전환 가능

**단점**:
- ❌ 디스크 공간 2배 사용
- ❌ 관리 복잡

---

### Option 3: 기존 캐시 삭제 후 재생성 (간단)

**방법**:
```bash
# 경고: 기존 캐시를 모두 삭제합니다!
rm -rf /home/najo/NAS/VLA/dataset/cache/qwen_vl_features/*

# 캐시 재생성
# 아래 "캐시 재생성 방법" 참조
```

**장점**:
- ✅ 가장 간단
- ✅ 디스크 공간 즉시 확보

**단점**:
- ❌ 기존 캐시 완전 손실
- ⏱️ 전체 재생성 필요

---

## 🚀 캐시 재생성 방법

### 준비사항
1. GPU가 있는 환경
2. `Make_VL_cache.py` 스크립트
3. 학습 데이터셋 경로 확인

### 단일 GPU로 캐시 생성
```python
# make_cache_single_gpu.py
import torch
import torch.distributed as dist
from pathlib import Path
from models.unified_model import QwenVLAUnified
from vla_datasets.unified_dataset import UnifiedVLADataset, create_unified_dataloader
from Make_VL_cache import build_vl_cache_distributed_optimized

# Initialize distributed (single process)
dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:29500', world_size=1, rank=0)

# Load model
model = QwenVLAUnified(
    model_type='regression',
    vl_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
).cuda()
model.eval()

# Create dataset (example for new format)
dataset = UnifiedVLADataset(
    data_dir="/home/najo/NAS/VLA/dataset/New_dataset/Yellow_point/episode_20251030_025119",
    format='new',
    horizon=8,
    vlm_reuse_count=3,
)

# Build cache
build_vl_cache_distributed_optimized(
    model=model,
    dataset=dataset,
    device="cuda",
    batch_size=4,  # GPU 메모리에 맞게 조정
    num_workers=4,
    micro_bs=1,
)

dist.destroy_process_group()
print("✅ Cache generation complete!")
```

실행:
```bash
python make_cache_single_gpu.py
```

---

### Multi-GPU로 캐시 생성 (더 빠름)

```bash
# 4 GPUs 사용 예시
torchrun --nproc_per_node=4 make_cache_multi_gpu.py
```

`make_cache_multi_gpu.py`:
```python
import torch
import torch.distributed as dist
from pathlib import Path
from models.unified_model import QwenVLAUnified
from vla_datasets.unified_dataset import UnifiedVLADataset
from Make_VL_cache import build_vl_cache_distributed_optimized

# Initialize distributed
dist.init_process_group(backend='nccl')
rank = dist.get_rank()
device = torch.device(f"cuda:{rank}")

# Load model
model = QwenVLAUnified(
    model_type='regression',
    vl_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
).to(device)
model.eval()

# Create dataset
dataset = UnifiedVLADataset(
    data_dir="/home/najo/NAS/VLA/dataset/New_dataset/Yellow_point/episode_20251030_025119",
    format='new',
    horizon=8,
    vlm_reuse_count=3,
)

# Build cache (distributed)
build_vl_cache_distributed_optimized(
    model=model,
    dataset=dataset,
    device=device,
    batch_size=4,
    num_workers=4,
    micro_bs=1,
)

dist.destroy_process_group()
if rank == 0:
    print("✅ All ranks finished. Cache generation complete!")
```

---

### 모든 데이터셋에 대해 캐시 생성

```python
# make_all_caches.py
import torch
import torch.distributed as dist
from pathlib import Path
from models.unified_model import QwenVLAUnified
from vla_datasets.unified_dataset import UnifiedVLADataset
from Make_VL_cache import build_vl_cache_distributed_optimized

dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:29500', world_size=1, rank=0)

model = QwenVLAUnified(
    model_type='regression',
    vl_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
).cuda()
model.eval()

# Old format datasets
old_dataset_root = Path("/home/najo/NAS/VLA/dataset/dataset")
for traj_dir in sorted(old_dataset_root.glob("*")):
    if not traj_dir.is_dir():
        continue

    print(f"\n{'='*80}")
    print(f"Processing OLD format: {traj_dir.name}")
    print(f"{'='*80}")

    try:
        dataset = UnifiedVLADataset(
            data_dir=str(traj_dir),
            format='old',
            horizon=8,
            vlm_reuse_count=3,
        )

        build_vl_cache_distributed_optimized(
            model=model,
            dataset=dataset,
            device="cuda",
            batch_size=4,
            num_workers=4,
        )
    except Exception as e:
        print(f"⚠️ Failed: {e}")
        continue

# New format datasets
new_dataset_root = Path("/home/najo/NAS/VLA/dataset/New_dataset")
for color_dir in new_dataset_root.glob("*"):
    if not color_dir.is_dir():
        continue

    for episode_dir in sorted(color_dir.glob("episode_*")):
        print(f"\n{'='*80}")
        print(f"Processing NEW format: {episode_dir.name}")
        print(f"{'='*80}")

        try:
            dataset = UnifiedVLADataset(
                data_dir=str(episode_dir),
                format='new',
                horizon=8,
                vlm_reuse_count=3,
            )

            build_vl_cache_distributed_optimized(
                model=model,
                dataset=dataset,
                device="cuda",
                batch_size=4,
                num_workers=4,
            )
        except Exception as e:
            print(f"⚠️ Failed: {e}")
            continue

dist.destroy_process_group()
print("\n✅ All datasets cached!")
```

실행:
```bash
# Single GPU
python make_all_caches.py

# Multi-GPU (더 빠름)
torchrun --nproc_per_node=4 make_all_caches.py
```

---

## 📊 예상 소요 시간

### 단일 데이터셋 (episode 1개)
- **샘플 수**: ~200
- **VLM 호출**: ~67회 (vlm_reuse_count=3)
- **GPU**: RTX 3090 기준
- **예상 시간**: 2-5분

### 전체 데이터셋
- **데이터셋 수**: 수십~수백 개
- **Single GPU**: 수 시간 ~ 하루
- **4 GPUs**: 1/4 시간

---

## ✅ 마이그레이션 체크리스트

### 1. 백업 (선택)
- [ ] 기존 캐시 디렉토리 백업
```bash
mv /home/najo/NAS/VLA/dataset/cache/qwen_vl_features \
   /home/najo/NAS/VLA/dataset/cache/qwen_vl_features_backup
```

### 2. 새 캐시 디렉토리 준비
- [ ] 디렉토리 생성
```bash
mkdir -p /home/najo/NAS/VLA/dataset/cache/qwen_vl_features
```

### 3. 캐시 생성
- [ ] 테스트 데이터셋으로 먼저 테스트
- [ ] 전체 데이터셋 캐시 생성 실행

### 4. 검증
- [ ] 캐시 파일 확인
```bash
ls /home/najo/NAS/VLA/dataset/cache/qwen_vl_features/ | head -20
```
- [ ] 파일명 형식 확인 (dataset_name_vlmN.pt)
- [ ] Dataset 로딩 테스트
```bash
python test_cache_system.py
```

### 5. 학습 시작
- [ ] TRAIN_Unified.py 실행
- [ ] "VL Cache: N/N" 로그 확인 (100% 적중 확인)

---

## 🔍 Troubleshooting

### 문제: OOM during cache generation

**해결**:
1. `batch_size` 줄이기 (4 → 2 → 1)
2. `micro_bs` 줄이기 (자동 백오프 있음)
3. 모델 precision 낮추기 (bfloat16 사용 중)

### 문제: 캐시 생성이 너무 느림

**해결**:
1. Multi-GPU 사용
2. `num_workers` 증가
3. `prefetch_factor` 증가

### 문제: 디스크 공간 부족

**해결**:
1. 기존 hash 기반 캐시 삭제
```bash
rm /home/najo/NAS/VLA/dataset/cache/qwen_vl_features/*[0-9a-f]*.pt
```
2. Cache limit 조정:
```python
cache_mgr = get_cache_manager(cache_limit_gb=30.0)  # 기본 50GB
```

### 문제: 캐시 생성 중단되었을 때

**해결**:
- VLACacheManager는 이미 생성된 캐시를 자동으로 스킵합니다
- 다시 실행하면 중단된 부분부터 계속됩니다
- "skipped" 카운트로 확인 가능

---

## 📈 기대 효과

### 이전 시스템
- ❌ Instruction 변경 → 캐시 미스
- ❌ Path 변경 → 캐시 미스
- ❌ 캐시 적중률: 불확실

### 새 시스템
- ✅ Instruction 변경 → 캐시 유지
- ✅ Path 변경 → 캐시 유지
- ✅ 캐시 적중률: ~100%

### 학습 시작 시간 개선
- **이전**: 매번 VLM 실행 (느림)
- **새로운 시스템**: 캐시 로드 (매우 빠름)
- **예상 개선**: 10-50배 빠른 데이터 로딩

---

## 🎉 권장 마이그레이션 플랜

### Phase 1: 테스트 (1시간)
1. 테스트 데이터셋 1개로 캐시 생성
2. Dataset 로딩 테스트
3. TRAIN_Unified.py로 짧은 학습 테스트

### Phase 2: 점진적 마이그레이션 (선택)
1. 중요한 데이터셋부터 캐시 생성
2. 학습하면서 나머지 캐시 생성

### Phase 3: 전체 마이그레이션 (권장)
1. 기존 캐시 백업 또는 삭제
2. 모든 데이터셋 캐시 생성 (Multi-GPU)
3. 완료 후 학습 시작

---

**마이그레이션 날짜**: 2025-11-03
**작성자**: Claude Code
