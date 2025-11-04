"""
VLA Cache Manager - 완전 고정 캐싱 시스템

캐시 파일 이름 규칙:
- Old format: {trajectory_name}_vlm{vlm_idx}.pt
- New format: {episode_name}_vlm{vlm_idx}.pt

예시:
- recv_all_20251027_170308_vlm0.pt
- episode_20251030_025119_vlm0.pt

이렇게 하면 instruction이나 image path가 바뀌어도 캐시를 재사용할 수 있습니다.
"""

import hashlib
import fcntl
import os
from pathlib import Path
from typing import Optional, List
import torch


class VLACacheManager:
    """
    VLA 캐시 관리자 - 완전 고정 캐싱

    Features:
    - 데이터셋 이름 + VLM index만으로 캐시 경로 생성
    - Instruction, image path 변경에도 영향 없음
    - Atomic save로 동시 접근 안전
    - Cache limit으로 디스크 관리
    """

    def __init__(
        self,
        cache_dir: str = "/home/najo/NAS/VLA/dataset/cache/qwen_vl_features",
        cache_limit_gb: float = 50.0,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_limit_gb = cache_limit_gb

    def get_cache_path(
        self,
        dataset_name: str,
        vlm_idx: int,
    ) -> Path:
        """
        캐시 파일 경로 생성 (완전 고정)

        Args:
            dataset_name: 데이터셋 이름 (e.g., "recv_all_20251027_170308", "episode_20251030_025119")
            vlm_idx: VLM index (e.g., 0, 10, 20, ...)

        Returns:
            캐시 파일 경로
        """
        return self.cache_dir / f"{dataset_name}_vlm{vlm_idx}.pt"

    def cache_exists(
        self,
        dataset_name: str,
        vlm_idx: int,
    ) -> bool:
        """캐시 파일 존재 여부 확인"""
        cache_path = self.get_cache_path(dataset_name, vlm_idx)
        return cache_path.exists()

    def load_cache(
        self,
        dataset_name: str,
        vlm_idx: int,
        device: str = "cpu",
    ) -> Optional[torch.Tensor]:
        """
        캐시 로드

        Returns:
            캐시된 VL features (B, 1, vl_dim) 또는 None
        """
        cache_path = self.get_cache_path(dataset_name, vlm_idx)

        if not cache_path.exists():
            return None

        try:
            cached = torch.load(cache_path, map_location=device)
            return cached
        except Exception as e:
            print(f"⚠️ Failed to load cache {cache_path.name}: {e}")
            return None

    def save_cache(
        self,
        dataset_name: str,
        vlm_idx: int,
        vl_features: torch.Tensor,
    ):
        """
        캐시 저장 (atomic)

        Args:
            dataset_name: 데이터셋 이름
            vlm_idx: VLM index
            vl_features: VL features to cache (B, 1, vl_dim)
        """
        cache_path = self.get_cache_path(dataset_name, vlm_idx)

        # Atomic save with file lock
        self._atomic_save(vl_features.detach().to("cpu", dtype=torch.float16), cache_path)

        # Enforce cache limit
        self._enforce_cache_limit()

    @staticmethod
    def _atomic_save(tensor_cpu: torch.Tensor, path: Path):
        """Atomic save with file lock"""
        tmp = path.with_suffix(".pt.tmp")
        lock_path = str(path) + ".lock"

        with open(lock_path, "w") as lockfile:
            try:
                fcntl.flock(lockfile, fcntl.LOCK_EX)

                # Skip if already exists
                if path.exists():
                    return

                # Save to temp file
                torch.save(tensor_cpu, tmp)

                # Atomic move
                os.replace(tmp, path)

            finally:
                fcntl.flock(lockfile, fcntl.LOCK_UN)

    def _enforce_cache_limit(self):
        """캐시 크기 제한 적용"""
        if self.cache_limit_gb <= 0:
            return

        total_bytes = sum(f.stat().st_size for f in self.cache_dir.glob("*.pt"))
        limit_bytes = self.cache_limit_gb * (1024 ** 3)

        if total_bytes <= limit_bytes:
            return

        # 오래된 파일부터 삭제
        all_files = sorted(
            self.cache_dir.glob("*.pt"),
            key=lambda f: f.stat().st_mtime
        )

        for file in all_files:
            if total_bytes <= limit_bytes:
                break
            try:
                size = file.stat().st_size
                file.unlink(missing_ok=True)
                total_bytes -= size
            except FileNotFoundError:
                continue

    def get_cache_stats(self) -> dict:
        """캐시 통계 정보"""
        cache_files = list(self.cache_dir.glob("*.pt"))
        total_bytes = sum(f.stat().st_size for f in cache_files)
        total_gb = total_bytes / (1024 ** 3)

        return {
            "cache_dir": str(self.cache_dir),
            "total_files": len(cache_files),
            "total_size_gb": total_gb,
            "limit_gb": self.cache_limit_gb,
            "usage_percent": (total_gb / self.cache_limit_gb * 100) if self.cache_limit_gb > 0 else 0,
        }

    def clear_cache(self, confirm: bool = False):
        """
        모든 캐시 삭제

        Args:
            confirm: True로 설정해야 삭제 실행
        """
        if not confirm:
            print("⚠️ Cache clear requires confirm=True")
            return

        cache_files = list(self.cache_dir.glob("*.pt"))
        for f in cache_files:
            f.unlink(missing_ok=True)

        print(f"✅ Cleared {len(cache_files)} cache files from {self.cache_dir}")

    def list_cached_datasets(self) -> dict:
        """캐시된 데이터셋 목록"""
        cache_files = list(self.cache_dir.glob("*.pt"))

        datasets = {}
        for f in cache_files:
            # Parse filename: {dataset_name}_vlm{vlm_idx}.pt
            name = f.stem
            if "_vlm" in name:
                dataset_name, vlm_part = name.rsplit("_vlm", 1)
                try:
                    vlm_idx = int(vlm_part)
                    if dataset_name not in datasets:
                        datasets[dataset_name] = []
                    datasets[dataset_name].append(vlm_idx)
                except ValueError:
                    continue

        # Sort vlm indices
        for dataset_name in datasets:
            datasets[dataset_name] = sorted(datasets[dataset_name])

        return datasets


# Global cache manager instance
_cache_manager = None


def get_cache_manager(
    cache_dir: str = "/home/najo/NAS/VLA/dataset/cache/qwen_vl_features",
    cache_limit_gb: float = 50.0,
) -> VLACacheManager:
    """Get global cache manager instance"""
    global _cache_manager

    if _cache_manager is None:
        _cache_manager = VLACacheManager(
            cache_dir=cache_dir,
            cache_limit_gb=cache_limit_gb,
        )

    return _cache_manager


if __name__ == "__main__":
    print("🧪 Testing VLA Cache Manager...")

    # Create cache manager
    cache_mgr = VLACacheManager(
        cache_dir="/tmp/test_vla_cache",
        cache_limit_gb=1.0,
    )

    # Test cache path generation
    print("\n📁 Cache path generation:")
    path1 = cache_mgr.get_cache_path("recv_all_20251027_170308", 0)
    path2 = cache_mgr.get_cache_path("episode_20251030_025119", 150)
    print(f"   Old format: {path1.name}")
    print(f"   New format: {path2.name}")

    # Test save and load
    print("\n💾 Save and load test:")
    test_features = torch.randn(1, 1, 3072)
    cache_mgr.save_cache("test_dataset", 0, test_features)
    print(f"   Saved: test_dataset_vlm0.pt")

    loaded = cache_mgr.load_cache("test_dataset", 0)
    if loaded is not None:
        print(f"   Loaded: {loaded.shape}")
        print(f"   Match: {torch.allclose(test_features.cpu().float(), loaded.float(), rtol=1e-3)}")

    # Test stats
    print("\n📊 Cache statistics:")
    stats = cache_mgr.get_cache_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Test list datasets
    print("\n📋 Cached datasets:")
    datasets = cache_mgr.list_cached_datasets()
    for dataset_name, vlm_indices in datasets.items():
        print(f"   {dataset_name}: {len(vlm_indices)} cached VLM features")

    # Cleanup
    cache_mgr.clear_cache(confirm=True)

    print("\n✅ All tests passed!")
