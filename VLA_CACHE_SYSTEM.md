# VLA Cache System - 완전 고정 캐싱

## 🎯 문제 해결

### 이전 문제
```
📦 Loaded episode_20251030_025209 (new format)
   Samples: 169, Sensor: True, VL Cache: 0/57
```

**원인**: 캐시 경로가 다음 정보로 생성되었기 때문에 불안정했습니다:
- Hash of: `{trajectory_key} + "||" + {instruction} + "||" + {image_paths}`
- instruction이나 image_paths가 조금만 바뀌어도 캐시를 못 찾음

### 새로운 솔루션: 완전 고정 캐싱

**캐시 경로**: `{dataset_name}_vlm{vlm_idx}.pt` 만 사용

```python
# 예시:
# recv_all_20251027_170308_vlm0.pt
# episode_20251030_025119_vlm150.pt
```

**장점**:
- ✅ Instruction 변경에도 캐시 유지
- ✅ Image path 변경에도 캐시 유지
- ✅ 데이터셋 이름 + VLM index만으로 완전 결정
- ✅ 안정적이고 예측 가능한 캐싱

---

## 📁 수정된 파일들

### 1. vla_cache_manager.py (신규 생성)

**VLACacheManager 클래스**:
```python
class VLACacheManager:
    def get_cache_path(self, dataset_name: str, vlm_idx: int) -> Path:
        """완전 고정 캐시 경로"""
        return self.cache_dir / f"{dataset_name}_vlm{vlm_idx}.pt"

    def cache_exists(self, dataset_name: str, vlm_idx: int) -> bool:
        """캐시 존재 확인"""

    def load_cache(self, dataset_name: str, vlm_idx: int, device="cpu"):
        """캐시 로드"""

    def save_cache(self, dataset_name: str, vlm_idx: int, vl_features):
        """캐시 저장 (atomic + cache limit 자동 적용)"""
```

**특징**:
- Atomic save with file locking (동시 접근 안전)
- 자동 캐시 용량 제한 (기본 50GB)
- 통계 및 관리 기능 제공

---

### 2. vla_datasets/unified_dataset.py (수정)

#### Line 293-321: `_scan_vl_cache()` 메서드
```python
def _scan_vl_cache(self):
    """Pre-scan VL cache files using VLACacheManager"""
    from vla_cache_manager import get_cache_manager

    cache_mgr = get_cache_manager(cache_dir=str(self.cache_root))
    self.vl_cache_files = {}
    dataset_name = self.data_dir.name

    if self.format == 'old':
        for action_step in range(self.max_action_steps):
            vlm_idx = min(action_step * self.action_step_size, len(self.actions) - 1)
            if vlm_idx not in self.vl_cache_files:
                if cache_mgr.cache_exists(dataset_name, vlm_idx):
                    self.vl_cache_files[vlm_idx] = cache_mgr.get_cache_path(dataset_name, vlm_idx)
                else:
                    self.vl_cache_files[vlm_idx] = None

    else:  # new format
        num_vlm_steps = (self._total_samples + self.vlm_reuse_count - 1) // self.vlm_reuse_count
        for i in range(num_vlm_steps):
            vlm_idx = i * self.vlm_interval
            if cache_mgr.cache_exists(dataset_name, vlm_idx):
                self.vl_cache_files[vlm_idx] = cache_mgr.get_cache_path(dataset_name, vlm_idx)
            else:
                self.vl_cache_files[vlm_idx] = None

    self.cache_found_count = sum(1 for p in self.vl_cache_files.values() if p is not None)
```

#### Line 412-444: `_load_vl_or_images()` 메서드
```python
def _load_vl_or_images(self, vlm_idx):
    """Load VL cache or return image paths using VLACacheManager"""
    from vla_cache_manager import get_cache_manager

    vl_cache = None
    image_paths = []

    cache_path = self.vl_cache_files.get(vlm_idx)

    if cache_path:
        # Use cache manager for loading
        cache_mgr = get_cache_manager(cache_dir=str(self.cache_root))
        vl_cache = cache_mgr.load_cache(
            dataset_name=self.data_dir.name,
            vlm_idx=vlm_idx,
            device="cpu"
        )
        if vl_cache is not None:
            return vl_cache, None

    # Fallback to image paths
    ...
```

**변경 사항**:
- Hash 기반 경로 → VLACacheManager 사용
- `cache_mgr.cache_exists()` 및 `cache_mgr.load_cache()` 사용
- 완전히 결정론적인 캐시 탐색

---

### 3. Make_VL_cache.py (대폭 수정)

#### Import 추가
```python
from vla_cache_manager import get_cache_manager
```

#### Line 37-64: VLACacheManager 초기화
```python
# VLACacheManager 초기화
cache_mgr = get_cache_manager(
    cache_dir=str(base_cache_dir),
    cache_limit_gb=50.0
)
```

#### Line 108-136: 캐시 체크 로직
```python
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
```

#### Line 179-187: 캐시 저장 로직
```python
for j, item in enumerate(sub_items):
    pooled_single = pooled_batch[j:j+1]
    # VLACacheManager로 저장
    cache_mgr.save_cache(
        dataset_name=item["dataset_name"],
        vlm_idx=item["vlm_idx"],
        vl_features=pooled_single
    )
    total_cached += 1
```

**변경 사항**:
- `key_mode`, `rank_sharded_cache` 파라미터 제거 (더 이상 필요 없음)
- Hash 기반 `_cache_path_for()` 함수 제거
- `_local_atomic_save()`, `_local_enforce_cache_limit()` 제거 (VLACacheManager가 처리)
- 완전히 VLACacheManager 기반으로 전환

---

## 🔍 캐시 키 생성 방식

### Dataset에서 생성 (unified_dataset.py)

#### Old Format (Line 367):
```python
cache_key = f"{self.data_dir.name}_vlm{vlm_idx}"
# 예: recv_all_20251027_170308_vlm0
```

#### New Format (Line 397):
```python
cache_key = f"{self.data_dir.name}_vlm{vlm_idx}"
# 예: episode_20251030_025119_vlm150
```

**중요**:
- `self.data_dir.name`은 데이터셋 폴더 이름 (예: `recv_all_20251027_170308`, `episode_20251030_025119`)
- `vlm_idx`는 VLM이 실행되는 인덱스 (0, 3, 6, ... 또는 0, 10, 20, ...)

---

## 🧪 테스트 방법

### VLACacheManager 단독 테스트
```bash
python vla_cache_manager.py
```

**출력 예시**:
```
🧪 Testing VLA Cache Manager...

📁 Cache path generation:
   Old format: recv_all_20251027_170308_vlm0.pt
   New format: episode_20251030_025119_vlm150.pt

💾 Save and load test:
   Saved: test_dataset_vlm0.pt
   Loaded: torch.Size([1, 1, 3072])
   Match: True

📊 Cache statistics:
   cache_dir: /tmp/test_vla_cache
   total_files: 1
   total_size_gb: 0.000012
   limit_gb: 1.0
   usage_percent: 0.0012

📋 Cached datasets:
   test_dataset: 1 cached VLM features

✅ All tests passed!
```

### Dataset 로딩 테스트
```bash
# Old format dataset
python -c "
from vla_datasets.unified_dataset import UnifiedVLADataset
ds = UnifiedVLADataset(
    data_dir='/home/najo/NAS/VLA/dataset/dataset/recv_all_20251027_170308',
    format='old'
)
print(f'Total samples: {len(ds)}')
print(f'Cached VL features: {ds.cache_found_count}/{len(ds.vl_cache_files)}')
"

# New format dataset
python -c "
from vla_datasets.unified_dataset import UnifiedVLADataset
ds = UnifiedVLADataset(
    data_dir='/home/najo/NAS/VLA/dataset/New_dataset/Yellow_point/episode_20251030_025119',
    format='new'
)
print(f'Total samples: {len(ds)}')
print(f'Cached VL features: {ds.cache_found_count}/{len(ds.vl_cache_files)}')
"
```

### 전체 Training 테스트
```bash
# Single GPU
python TRAIN_Unified.py --mode train --model-type regression

# Multi-GPU
torchrun --nproc_per_node=4 TRAIN_Unified.py --mode train --model-type regression
```

**기대 출력**:
```
📦 Loaded episode_20251030_025209 (new format)
   Samples: 169, Sensor: True, VL Cache: 57/57  ✅
```

---

## 📊 캐시 디렉토리 구조

```
/home/najo/NAS/VLA/dataset/cache/qwen_vl_features/
├── recv_all_20251027_170308_vlm0.pt
├── recv_all_20251027_170308_vlm3.pt
├── recv_all_20251027_170308_vlm6.pt
├── episode_20251030_025119_vlm0.pt
├── episode_20251030_025119_vlm10.pt
├── episode_20251030_025119_vlm20.pt
└── ...
```

**특징**:
- 파일 이름만 봐도 어떤 데이터셋의 어떤 VLM 인덱스인지 명확
- Instruction이나 image path가 바뀌어도 파일 이름 동일
- 디버깅 및 관리 용이

---

## 🎛️ VLACacheManager 설정

### 기본 설정
```python
from vla_cache_manager import get_cache_manager

cache_mgr = get_cache_manager(
    cache_dir="/home/najo/NAS/VLA/dataset/cache/qwen_vl_features",
    cache_limit_gb=50.0  # 50GB 제한
)
```

### 캐시 통계 확인
```python
stats = cache_mgr.get_cache_stats()
print(stats)
# {
#     'cache_dir': '/home/najo/NAS/VLA/dataset/cache/qwen_vl_features',
#     'total_files': 1234,
#     'total_size_gb': 45.2,
#     'limit_gb': 50.0,
#     'usage_percent': 90.4
# }
```

### 캐시된 데이터셋 목록
```python
datasets = cache_mgr.list_cached_datasets()
print(datasets)
# {
#     'recv_all_20251027_170308': [0, 3, 6, 9, ...],
#     'episode_20251030_025119': [0, 10, 20, 30, ...]
# }
```

### 캐시 삭제 (주의!)
```python
cache_mgr.clear_cache(confirm=True)
```

---

## ⚙️ 주요 동작 원리

### 1. Dataset 초기화 시
```python
def __init__(self, data_dir, ...):
    ...
    self._scan_vl_cache()  # 캐시 미리 스캔
```

`_scan_vl_cache()`는:
1. VLACacheManager 초기화
2. 모든 예상 VLM 인덱스에 대해 캐시 존재 확인
3. `self.vl_cache_files` 딕셔너리 구성:
   - Key: vlm_idx
   - Value: Path 또는 None

### 2. `__getitem__()` 호출 시
```python
def _load_vl_or_images(self, vlm_idx):
    cache_path = self.vl_cache_files.get(vlm_idx)

    if cache_path:
        vl_cache = cache_mgr.load_cache(dataset_name, vlm_idx, device="cpu")
        if vl_cache is not None:
            return vl_cache, None  # 캐시 반환

    # 캐시 없으면 image paths 반환
    return None, image_paths
```

### 3. VLM 실행 후 저장 (Make_VL_cache.py)
```python
pooled_batch = vl_tokens_batch.mean(dim=1, keepdim=True)

for j, item in enumerate(sub_items):
    cache_mgr.save_cache(
        dataset_name=item["dataset_name"],
        vlm_idx=item["vlm_idx"],
        vl_features=pooled_batch[j:j+1]
    )
```

`save_cache()` 내부:
1. Atomic save with file locking (race condition 방지)
2. 이미 존재하면 스킵
3. 저장 후 자동으로 `_enforce_cache_limit()` 호출
4. 용량 초과 시 오래된 파일부터 삭제

---

## 🚀 성능 및 안정성

### 이전 시스템 문제점
- ❌ Instruction 변경 → 캐시 미스
- ❌ Image path 변경 → 캐시 미스
- ❌ Hash collision 가능성
- ❌ 디버깅 어려움 (파일 이름이 hash)

### 새로운 시스템 장점
- ✅ 완전히 결정론적 (dataset name + vlm_idx만 사용)
- ✅ Instruction/Image path 변경에 강건
- ✅ 파일 이름만 봐도 내용 파악 가능
- ✅ Atomic save로 동시 접근 안전
- ✅ 자동 캐시 용량 관리
- ✅ 통계 및 관리 기능 제공

### 예상 캐시 적중률
- **기존 시스템**: 0% (경로 변경 시)
- **새로운 시스템**: ~100% (데이터셋이 동일하면)

---

## 🔧 Troubleshooting

### 문제: 캐시를 못 찾음 (VL Cache: 0/N)

**체크리스트**:
1. 캐시 디렉토리 확인:
```bash
ls /home/najo/NAS/VLA/dataset/cache/qwen_vl_features/
```

2. Dataset 이름 확인:
```python
from pathlib import Path
data_dir = Path("/home/najo/NAS/VLA/dataset/New_dataset/Yellow_point/episode_20251030_025119")
print(data_dir.name)  # episode_20251030_025119
```

3. 예상 캐시 파일 이름:
```
episode_20251030_025119_vlm0.pt
episode_20251030_025119_vlm10.pt
episode_20251030_025119_vlm20.pt
...
```

4. 실제 캐시 파일과 비교

### 문제: 캐시 생성이 너무 느림

**해결책**:
1. `batch_size` 증가 (GPU 메모리가 충분하면)
2. `num_workers` 조정
3. `micro_bs` 증가 (OOM이 안 나면)

### 문제: 캐시 용량 초과

**해결책**:
1. `cache_limit_gb` 증가:
```python
cache_mgr = get_cache_manager(cache_limit_gb=100.0)
```

2. 또는 오래된 캐시 수동 삭제:
```python
cache_mgr.clear_cache(confirm=True)
```

---

## 📝 마이그레이션 가이드

### 기존 Hash 기반 캐시가 있는 경우

**Option 1**: 캐시 재생성 (권장)
```bash
# 기존 캐시 백업
mv /home/najo/NAS/VLA/dataset/cache/qwen_vl_features \
   /home/najo/NAS/VLA/dataset/cache/qwen_vl_features_old

# 새로운 캐시 생성
python Make_VL_cache.py
```

**Option 2**: 캐시 변환 스크립트 (필요시 작성 가능)

---

**수정 완료 날짜**: 2025-11-03
**작성자**: Claude Code
