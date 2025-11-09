#!/usr/bin/env python3
"""
Reconstruct robot state trajectories using a trained MAE RobotStateEncoder.

Example:
    python reconstruct_robot_states.py \
        --episode-roots /home/najo/NAS/VLA/dataset/New_dataset2/Yellow_point \
        --checkpoint checkpoints/robot_state_mae_best.pth \
        --window-size 100 \
        --stride 20 \
        --mask-ratio 0.0 \
        --output-name robot_states_recon.npz \
        --device cuda:0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
from torch import nn

from models.unified_model import RobotStateEncoder


class MAERobotStateModel(nn.Module):
    """Minimal MAE wrapper matching TRAIN_RobotState_MAE."""

    def __init__(self, encoder: RobotStateEncoder, decoder_dim: int = 128):
        super().__init__()
        self.encoder = encoder
        self.encoder_dim = encoder.model_dim
        self.decoder_dim = decoder_dim
        self.output_dim = encoder.input_dim
        self.decoder = nn.Sequential(
            nn.Linear(self.encoder_dim, self.decoder_dim),
            nn.GELU(),
            nn.Linear(self.decoder_dim, self.output_dim),
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        encoded_sequence = self.encoder(src, return_sequence=True)
        return self.decoder(encoded_sequence)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct robot states with MAE.")
    parser.add_argument(
        "--episode-roots",
        nargs="+",
        required=True,
        help="Episode directories or task directories containing data_collection_* folders.",
    )
    parser.add_argument(
        "--exclude-episodes",
        nargs="*",
        default=[],
        help="Full paths to specific episodes that should be skipped.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to robot_state_mae checkpoint (.pth).",
    )
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=20, help="Stride between inference windows.")
    parser.add_argument("--mask-ratio", type=float, default=0.0, help="Optional masking ratio for inputs.")
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--output-dim", type=int, default=3072)
    parser.add_argument("--decoder-dim", type=int, default=128)
    parser.add_argument("--output-name", default="robot_states_recon.npz")
    parser.add_argument(
        "--output-root",
        default=None,
        help="If set, save recon files under this root using <task>/<episode>/<output-name> structure.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--episodes-limit", type=int, default=None, help="Optional cap on episodes processed per task.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def discover_episodes(root: Path) -> List[Path]:
    if (root / "robot_states.npz").exists():
        return [root]
    episodes = []
    for sub in sorted(root.iterdir()):
        if sub.is_dir() and sub.name.startswith("data_collection_") and (sub / "robot_states.npz").exists():
            episodes.append(sub)
    return episodes


def load_model(args: argparse.Namespace) -> MAERobotStateModel:
    device = torch.device(args.device)
    encoder = RobotStateEncoder(
        temporal_length=args.window_size,
        model_dim=args.model_dim,
        output_dim=args.output_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    )
    model = MAERobotStateModel(encoder, decoder_dim=args.decoder_dim).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {missing[:6]}{'...' if len(missing) > 6 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected[:6]}{'...' if len(unexpected) > 6 else ''}")
    model.eval()
    if args.dtype == "bfloat16":
        model = model.to(dtype=torch.bfloat16)
    return model


def reconstruct_episode(
    model: MAERobotStateModel,
    data: np.ndarray,
    window_size: int,
    stride: int,
    mask_ratio: float,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    T = data.shape[0]
    if T == 0:
        raise ValueError("Empty robot state sequence.")
    sums = np.zeros_like(data, dtype=np.float64)
    counts = np.zeros((T, 1), dtype=np.float64)

    if T <= window_size:
        padded = np.zeros((window_size, data.shape[1]), dtype=np.float32)
        padded[:T] = data
        starts = [0]
        tail_trim = T
    else:
        starts = list(range(0, T - window_size + 1, stride))
        if starts[-1] != T - window_size:
            starts.append(T - window_size)
        tail_trim = None

    with torch.no_grad():
        for start in starts:
            chunk = data[start : start + window_size]
            if chunk.shape[0] < window_size:
                pad = np.repeat(chunk[-1:], window_size - chunk.shape[0], axis=0)
                chunk = np.concatenate([chunk, pad], axis=0)
            tensor = torch.from_numpy(chunk).to(device=device, dtype=torch.float32)
            tensor = tensor.unsqueeze(0)
            if dtype == torch.bfloat16:
                tensor = tensor.to(dtype=torch.bfloat16)
            masked = tensor.clone()
            if mask_ratio > 0:
                num_mask = max(1, int(mask_ratio * window_size))
                mask_idx = torch.randperm(window_size, device=device)[:num_mask]
                masked[:, mask_idx, :] = 0.0
            recon = model(masked).to(dtype=torch.float32).squeeze(0).cpu().numpy()
            valid_len = min(window_size, T - start)
            sums[start : start + valid_len] += recon[:valid_len]
            counts[start : start + valid_len] += 1

    counts[counts == 0] = 1
    recon_seq = sums / counts
    if tail_trim:
        recon_seq = recon_seq[:tail_trim]
    return recon_seq.astype(np.float32)


def save_reconstruction(path: Path, recon: np.ndarray):
    poses = recon[:, 6:12]
    np.savez_compressed(
        path,
        robot_states=recon,
        poses=poses,
    )


def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model = load_model(args)
    model.to(device)

    exclude = {str(Path(p).resolve()) for p in args.exclude_episodes}
    episodes: List[Path] = []
    for root_str in args.episode_roots:
        root = Path(root_str).resolve()
        if not root.exists():
            print(f"[WARN] Path does not exist: {root}")
            continue
        eps = [ep for ep in discover_episodes(root) if str(ep.resolve()) not in exclude]
        if args.episodes_limit:
            eps = eps[: args.episodes_limit]
        episodes.extend(eps)

    if not episodes:
        raise RuntimeError("No episodes found to process.")

    print(f"Processing {len(episodes)} episodes...")
    output_root = Path(args.output_root).resolve() if args.output_root else None
    for ep_dir in episodes:
        if output_root:
            rel = Path(ep_dir.parent.name) / ep_dir.name
            target_dir = output_root / rel
            target_dir.mkdir(parents=True, exist_ok=True)
            out_path = target_dir / args.output_name
        else:
            out_path = ep_dir / args.output_name
        if out_path.exists():
            print(f"[SKIP] {out_path} already exists.")
            continue
        data_npz = np.load(ep_dir / "robot_states.npz")
        states = data_npz["robot_states"].astype(np.float32)
        recon = reconstruct_episode(
            model,
            states,
            window_size=args.window_size,
            stride=args.stride,
            mask_ratio=args.mask_ratio,
            device=device,
            dtype=dtype,
        )
        save_reconstruction(out_path, recon)
        if args.verbose:
            err = np.sqrt(np.mean((recon - states) ** 2))
            print(f"Saved {out_path} | RMS diff: {err:.4f}")


if __name__ == "__main__":
    main()
