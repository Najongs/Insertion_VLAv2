#!/usr/bin/env python
"""
Sensor Encoder Embedding Visualization (t-SNE / UMAP)
----------------------------------------------------

Usage example:
    python analyze_sensor_embeddings.py \
        --sensor-checkpoint checkpoints/sensor_clip_best.pth \
        --dataset-paths /home/najo/NAS/VLA/dataset/New_dataset2/Yellow_point/episode_20240910_123000 \
                         /home/najo/NAS/VLA/dataset/New_dataset2/Blue_point/episode_20240910_124500 \
        --output-dir analysis/sensor_tsne \
        --max-samples-per-episode 200 \
        --method both \
        --device cuda:0

This script:
  1. Loads the pretrained ForceAwareSensorEncoder used in TRAIN_SensorImage_CLIP.py.
  2. Iterates over the specified episodes (new-format datasets) via UnifiedVLADataset.
  3. Extracts sensor embeddings for uniformly sampled steps per episode.
  4. Projects the embeddings into 2D via t-SNE and/or UMAP.
  5. Saves scatter plots + CSV with coordinates and metadata.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from inspect import signature as _inspect_signature

from models.unified_model import ForceAwareSensorEncoder, force_bn_fp32_
from vla_datasets.unified_dataset import UnifiedVLADataset

try:
    import umap
    HAS_UMAP = True
except Exception:  # pragma: no cover - optional dependency
    HAS_UMAP = False


def load_sensor_encoder(checkpoint_path: str, device: torch.device) -> ForceAwareSensorEncoder:
    """Instantiate ForceAwareSensorEncoder and load weights from the CLIP checkpoint."""
    encoder = ForceAwareSensorEncoder(
        temporal_length=65,
        output_dim=3072,
        use_transformer=True,
        num_transformer_layers=2,
    )
    encoder.to(device=device, dtype=torch.bfloat16)
    force_bn_fp32_(encoder)
    encoder.eval()

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Extract only the sensor_encoder.* weights
    filtered = {}
    for key, value in state_dict.items():
        if key.startswith("module.sensor_encoder."):
            filtered[key.replace("module.sensor_encoder.", "", 1)] = value
        elif key.startswith("sensor_encoder."):
            filtered[key.replace("sensor_encoder.", "", 1)] = value
    missing, unexpected = encoder.load_state_dict(filtered, strict=False)

    print(f"✅ Loaded sensor encoder from {checkpoint_path}")
    if missing:
        print(f"   · Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"   · Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    return encoder


def load_dataset(path: str, cache_root: Path) -> UnifiedVLADataset:
    """Create a UnifiedVLADataset for a single episode directory."""
    dataset = UnifiedVLADataset(
        data_dir=path,
        format="auto",
        horizon=8,
        vlm_reuse_count=1,
        sensor_window_size=65,
        use_cache=False,
        cache_root=str(cache_root),
    )
    if not dataset.has_sensor:
        raise RuntimeError(f"Dataset at {path} has no sensor data.")
    return dataset


def sample_indices(total: int, max_samples: int) -> np.ndarray:
    """Uniformly sample up to max_samples indices without replacement."""
    if total <= max_samples:
        return np.arange(total, dtype=int)
    return np.unique(np.linspace(0, total - 1, max_samples, dtype=int))


def collect_embeddings(
    dataset_paths: List[str],
    encoder: ForceAwareSensorEncoder,
    device: torch.device,
    cache_root: Path,
    max_samples_per_episode: int,
) -> Tuple[np.ndarray, List[Dict]]:
    """Run the sensor encoder over sampled steps and gather embeddings + metadata."""
    embeddings = []
    metadata = []

    with torch.no_grad():
        for dataset_path in dataset_paths:
            dataset = load_dataset(dataset_path, cache_root)
            indices = sample_indices(len(dataset), max_samples_per_episode)

            print(f"➡️  {dataset.data_dir.name}: sampling {len(indices)} / {len(dataset)} steps")

            for idx in indices:
                sample = dataset[idx]
                sensor = sample["sensor_data"]  # Tensor (T,C)
                if not torch.is_tensor(sensor):
                    sensor = torch.from_numpy(sensor)

                sensor = sensor.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
                feat = encoder(sensor).float().cpu().numpy()[0]
                embeddings.append(feat)

                metadata.append({
                    "task": dataset.data_dir.parent.name,
                    "episode": dataset.data_dir.name,
                    "index": int(idx),
                    "timestamp": float(sample.get("timestamp", 0.0)),
                    "has_sensor": bool(sample.get("has_sensor", True)),
                })

    if not embeddings:
        raise RuntimeError("No sensor embeddings were collected. Check dataset paths and sampling settings.")

    return np.stack(embeddings, axis=0), metadata


_TSNE_PARAMS = _inspect_signature(TSNE.__init__).parameters
_TSNE_USE_MAX_ITER = "max_iter" in _TSNE_PARAMS


def run_tsne(embeddings: np.ndarray, perplexity: float, seed: int) -> np.ndarray:
    common_kwargs = dict(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        random_state=seed,
        learning_rate="auto",
    )
    if _TSNE_USE_MAX_ITER:
        common_kwargs["max_iter"] = 1500
    else:
        common_kwargs["n_iter"] = 1500

    tsne = TSNE(**common_kwargs)
    return tsne.fit_transform(embeddings)


def run_umap(embeddings: np.ndarray, seed: int, n_neighbors: int, min_dist: float) -> np.ndarray:
    if not HAS_UMAP:
        raise RuntimeError("umap-learn is not installed. pip install umap-learn to enable this method.")
    reducer = umap.UMAP(
        n_components=2,
        random_state=seed,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
    )
    return reducer.fit_transform(embeddings)


def plot_embedding(coords: np.ndarray, metadata: List[Dict], title: str, output_path: Path):
    """Create a scatter plot grouped by task (color) and episode (marker)."""
    tasks = sorted({m["task"] for m in metadata})
    task_to_color = {task: plt.cm.tab10(i % 10) for i, task in enumerate(tasks)}

    episodes = sorted({m["episode"] for m in metadata})
    markers = ["o", "s", "^", "D", "P", "X", "*", "v", "<", ">"]
    episode_to_marker = {ep: markers[i % len(markers)] for i, ep in enumerate(episodes)}

    plt.figure(figsize=(8, 6))
    for coord, meta in zip(coords, metadata):
        plt.scatter(
            coord[0],
            coord[1],
            c=[task_to_color[meta["task"]]],
            marker=episode_to_marker[meta["episode"]],
            alpha=0.75,
            edgecolors="none",
            s=24,
        )

    handles = []
    for task in tasks:
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=task_to_color[task], markersize=8,
                                  label=task))
    plt.legend(handles=handles, title="Task (color)", loc="best", fontsize=8)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"📈 Saved plot: {output_path}")


def save_coords(coords: np.ndarray, metadata: List[Dict], output_path: Path):
    rows = []
    for coord, meta in zip(coords, metadata):
        rows.append({
            "x": coord[0],
            "y": coord[1],
            **meta,
        })
    import csv
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"💾 Saved coordinates: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize sensor encoder embeddings via t-SNE/UMAP.")
    parser.add_argument("--sensor-checkpoint", type=str, required=True,
                        help="Path to sensor_clip_best.pth (or latest).")
    parser.add_argument("--dataset-paths", type=str, nargs="+", required=True,
                        help="List of episode directories (new-format) to analyze.")
    parser.add_argument("--output-dir", type=str, default="analysis/sensor_embeddings",
                        help="Directory to store plots and CSVs.")
    parser.add_argument("--max-samples-per-episode", type=int, default=200,
                        help="Uniformly sample up to N steps per episode.")
    parser.add_argument("--perplexity", type=float, default=30.0,
                        help="t-SNE perplexity.")
    parser.add_argument("--method", type=str, choices=["tsne", "umap", "both"], default="tsne",
                        help="Projection method(s) to run.")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for the sensor encoder.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for projections.")
    parser.add_argument("--umap-neighbors", type=int, default=30, help="UMAP n_neighbors.")
    parser.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP min_dist.")
    parser.add_argument("--cache-root", type=str, default="./cache/qwen_vl_features",
                        help="Writable cache root for datasets (images not used but metadata scanning needs access).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    encoder = load_sensor_encoder(args.sensor_checkpoint, device)

    embeddings, metadata = collect_embeddings(
        dataset_paths=args.dataset_paths,
        encoder=encoder,
        device=device,
        cache_root=cache_root,
        max_samples_per_episode=args.max_samples_per_episode,
    )

    # Normalize embeddings before projection for stability
    embeddings = (embeddings - embeddings.mean(axis=0)) / (embeddings.std(axis=0) + 1e-6)

    results = {}

    if args.method in ("tsne", "both"):
        coords = run_tsne(embeddings, perplexity=args.perplexity, seed=args.seed)
        save_coords(coords, metadata, output_dir / "sensor_tsne.csv")
        plot_embedding(coords, metadata, "Sensor Encoder t-SNE", output_dir / "sensor_tsne.png")
        results["tsne"] = {
            "csv": str((output_dir / "sensor_tsne.csv").resolve()),
            "plot": str((output_dir / "sensor_tsne.png").resolve()),
        }

    if args.method in ("umap", "both"):
        coords = run_umap(
            embeddings,
            seed=args.seed,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
        )
        save_coords(coords, metadata, output_dir / "sensor_umap.csv")
        plot_embedding(coords, metadata, "Sensor Encoder UMAP", output_dir / "sensor_umap.png")
        results["umap"] = {
            "csv": str((output_dir / "sensor_umap.csv").resolve()),
            "plot": str((output_dir / "sensor_umap.png").resolve()),
        }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"📝 Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
