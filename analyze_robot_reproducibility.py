#!/usr/bin/env python3
"""
Analyze robot end-effector reproducibility (position & orientation) across episodes.

Example:
    python analyze_robot_reproducibility.py \
        --task-dirs /home/najo/NAS/VLA/dataset/New_dataset2/Yellow_point \
        --exclude-episodes /home/najo/NAS/VLA/dataset/New_dataset2/Yellow_point/data_collection_20251108_043306 \
        --output-dir analysis/reproducibility \
        --target-length 200
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


@dataclass
class EpisodeTraj:
    name: str
    steps: int
    position: np.ndarray  # (L, 3)
    orientation: np.ndarray  # (L, 3)
    position_recon: np.ndarray | None = None
    orientation_recon: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute EE position/orientation reproducibility metrics per task."
    )
    parser.add_argument(
        "--task-dirs",
        nargs="+",
        required=True,
        help="Task directories that contain data_collection_*/robot_states.npz.",
    )
    parser.add_argument(
        "--exclude-episodes",
        nargs="*",
        default=[],
        help="Full paths to episode folders that should be ignored.",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/reproducibility",
        help="Directory to store plots + metrics.",
    )
    parser.add_argument(
        "--target-length",
        type=int,
        default=200,
        help="Resample each trajectory to this many steps for fair comparison.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap per task (after exclusions).",
    )
    parser.add_argument(
        "--use-median",
        action="store_true",
        help="Use median trajectory as reference (mean is default).",
    )
    parser.add_argument(
        "--recon-file-name",
        default=None,
        help="If set, load reconstructed poses from <episode>/<recon-file-name> "
             "and overlay them with the original trajectories.",
    )
    parser.add_argument(
        "--recon-root",
        default=None,
        help="If reconstruction files are stored separately, provide the root directory "
             "containing <task>/<episode>/<recon-file-name>.",
    )
    parser.add_argument(
        "--recon-key",
        default="poses",
        help="Key inside the reconstruction npz that contains pose trajectories.",
    )
    return parser.parse_args()


def find_episode_dirs(task_dir: Path) -> List[Path]:
    episodes = []
    for child in sorted(task_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("data_collection_") and (child / "robot_states.npz").exists():
            episodes.append(child)
    return episodes


def resample_traj(data: np.ndarray, target_len: int) -> np.ndarray:
    if data.shape[0] < 2:
        raise ValueError("Need at least 2 timesteps to resample.")
    original = np.linspace(0.0, 1.0, data.shape[0])
    target = np.linspace(0.0, 1.0, target_len)
    resampled = np.stack([np.interp(target, original, data[:, i]) for i in range(data.shape[1])], axis=-1)
    return resampled


def load_episode(
    path: Path,
    target_len: int,
    recon_file: str | None,
    recon_key: str,
    recon_root: Path | None,
) -> EpisodeTraj:
    npz = np.load(path / "robot_states.npz")
    poses = np.array(npz["poses"], dtype=np.float64)
    position = resample_traj(poses[:, :3], target_len)
    orientation = resample_traj(poses[:, 3:6], target_len)
    position_recon = None
    orientation_recon = None
    if recon_file:
        if recon_root:
            recon_path = recon_root / path.parent.name / path.name / recon_file
        else:
            recon_path = path / recon_file
        if recon_path.exists():
            recon_npz = np.load(recon_path)
            if recon_key not in recon_npz:
                raise KeyError(f"{recon_path} missing key '{recon_key}'")
            recon_poses = np.array(recon_npz[recon_key], dtype=np.float64)
            if recon_poses.ndim != 2 or recon_poses.shape[1] < 6:
                raise ValueError(f"{recon_path} has invalid pose shape {recon_poses.shape}")
            position_recon = resample_traj(recon_poses[:, :3], target_len)
            orientation_recon = resample_traj(recon_poses[:, 3:6], target_len)
        else:
            print(f"[WARN] Reconstruction file not found: {recon_path}")
    return EpisodeTraj(
        name=path.name,
        steps=target_len,
        position=position,
        orientation=orientation,
        position_recon=position_recon,
        orientation_recon=orientation_recon,
    )


def compute_reference(stack: np.ndarray, use_median: bool) -> np.ndarray:
    return np.median(stack, axis=0) if use_median else np.mean(stack, axis=0)


def l2_deviation(stack: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.linalg.norm(stack - reference[None, ...], axis=-1)


def summarize(task_name: str, episodes: List[EpisodeTraj], use_median: bool, out_dir: Path) -> Dict:
    pos_stack = np.stack([ep.position for ep in episodes], axis=0)
    rot_stack = np.stack([ep.orientation for ep in episodes], axis=0)

    pos_ref = compute_reference(pos_stack, use_median)
    rot_ref = compute_reference(rot_stack, use_median)

    pos_dev = l2_deviation(pos_stack, pos_ref)  # (N, L)
    rot_dev = l2_deviation(rot_stack, rot_ref)

    pos_rms = np.sqrt(np.mean(pos_dev**2, axis=1))
    rot_rms = np.sqrt(np.mean(rot_dev**2, axis=1))

    pos_max = np.max(pos_dev, axis=1)
    rot_max = np.max(rot_dev, axis=1)

    per_step_pos_std = np.std(pos_stack, axis=0)  # (L, 3)
    per_step_rot_std = np.std(rot_stack, axis=0)

    recon_entries = [
        (
            ep.name,
            ep.position_recon,
            ep.orientation_recon,
            ep.position,
            ep.orientation,
        )
        for ep in episodes
        if ep.position_recon is not None
    ]

    summary = {
        "task": task_name,
        "num_episodes": len(episodes),
        "target_length": episodes[0].steps,
        "reference": "median" if use_median else "mean",
        "position": {
            "mean_rms_m": float(pos_rms.mean()),
            "std_rms_m": float(pos_rms.std()),
            "mean_max_m": float(pos_max.mean()),
            "std_max_m": float(pos_max.std()),
            "per_axis_std_m": per_step_pos_std.mean(axis=0).tolist(),
        },
        "orientation": {
            "mean_rms_rad": float(rot_rms.mean()),
            "std_rms_rad": float(rot_rms.std()),
            "mean_max_rad": float(rot_max.mean()),
            "std_max_rad": float(rot_max.std()),
            "per_axis_std_rad": per_step_rot_std.mean(axis=0).tolist(),
        },
        "episodes": [
            {
                "name": ep.name,
                "rms_position_m": float(pos_rms[i]),
                "max_position_m": float(pos_max[i]),
                "rms_orientation_rad": float(rot_rms[i]),
                "max_orientation_rad": float(rot_max[i]),
            }
            for i, ep in enumerate(episodes)
        ]
    }

    if recon_entries:
        recon_pos = np.stack([item[1] for item in recon_entries], axis=0)
        recon_rot = np.stack([item[2] for item in recon_entries], axis=0)
        orig_pos = np.stack([item[3] for item in recon_entries], axis=0)
        orig_rot = np.stack([item[4] for item in recon_entries], axis=0)

        pos_err = np.sqrt(np.mean((recon_pos - orig_pos) ** 2, axis=(1, 2)))
        rot_err = np.sqrt(np.mean((recon_rot - orig_rot) ** 2, axis=(1, 2)))

        summary["reconstruction"] = {
            "num_pairs": len(recon_entries),
            "position_rms_m": {
                "mean": float(pos_err.mean()),
                "std": float(pos_err.std()),
            },
            "orientation_rms_rad": {
                "mean": float(rot_err.mean()),
                "std": float(rot_err.std()),
            },
            "episodes": [
                {
                    "name": name,
                    "position_rms_m": float(pos_err[i]),
                    "orientation_rms_rad": float(rot_err[i]),
                }
                for i, (name, *_rest) in enumerate(recon_entries)
            ],
        }

    save_plots(task_name, episodes, pos_stack, rot_stack, pos_ref, rot_ref, out_dir)

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2))
    return summary


def save_plots(
    task_name: str,
    episodes: Sequence[EpisodeTraj],
    pos_stack: np.ndarray,
    rot_stack: np.ndarray,
    pos_ref: np.ndarray,
    rot_ref: np.ndarray,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = np.arange(pos_stack.shape[1])
    dim_names = ["x", "y", "z", "roll", "pitch", "yaw"]
    has_recon = any(ep.position_recon is not None for ep in episodes)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for row in range(2):
        for col in range(3):
            ax = axes[row, col]
            dim_idx = col
            ref = pos_ref if row == 0 else rot_ref
            for ep in episodes:
                series = ep.position if row == 0 else ep.orientation
                ax.plot(steps, series[:, dim_idx], color="C0", alpha=0.25, linewidth=1)
                if has_recon:
                    recon_series = ep.position_recon if row == 0 else ep.orientation_recon
                    if recon_series is not None:
                        ax.plot(
                            steps,
                            recon_series[:, dim_idx],
                            color="C1",
                            alpha=0.6,
                            linewidth=1.2,
                            linestyle="--",
                        )
            ax.plot(steps, ref[:, dim_idx], color="black", linewidth=2, label="reference")
            ax.set_title(f"{task_name} — {dim_names[row * 3 + col]}")
            ax.set_ylabel("m" if row == 0 else "rad")
            ax.grid(True, alpha=0.2)
    legend_items = [plt.Line2D([0], [0], color="C0", lw=2, label="original")]
    if has_recon:
        legend_items.append(plt.Line2D([0], [0], color="C1", lw=2, linestyle="--", label="reconstruction"))
    legend_items.append(plt.Line2D([0], [0], color="black", lw=2, label="reference"))
    axes[-1, -1].legend(handles=legend_items, loc="upper right")
    axes[-1, 1].set_xlabel("Normalized step")
    fig.tight_layout()
    fig.savefig(out_dir / "trajectories.png", dpi=200)
    plt.close(fig)

    pos_dev = np.sqrt(np.mean((pos_stack - pos_ref[None, ...]) ** 2, axis=(1, 2)))
    rot_dev = np.sqrt(np.mean((rot_stack - rot_ref[None, ...]) ** 2, axis=(1, 2)))

    fig, ax = plt.subplots(figsize=(8, 4))
    indices = np.arange(len(episodes))
    width = 0.35
    ax.bar(indices - width / 2, pos_dev, width, label="Position RMS (m)")
    ax.bar(indices + width / 2, rot_dev, width, label="Orientation RMS (rad)")
    ax.set_xticks(indices)
    ax.set_xticklabels([ep.name for ep in episodes], rotation=45, ha="right")
    ax.set_ylabel("Deviation")
    ax.set_title(f"{task_name} reproducibility")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "deviation_bars.png", dpi=200)
    plt.close(fig)

    recon_eps = [ep for ep in episodes if ep.position_recon is not None]
    if recon_eps:
        pos_err = [
            math.sqrt(np.mean((ep.position_recon - ep.position) ** 2))
            for ep in recon_eps
        ]
        rot_err = [
            math.sqrt(np.mean((ep.orientation_recon - ep.orientation) ** 2))
            for ep in recon_eps
        ]
        fig, ax = plt.subplots(figsize=(8, 4))
        indices = np.arange(len(recon_eps))
        width = 0.35
        ax.bar(indices - width / 2, pos_err, width, label="Reconstruction Position RMS (m)", color="C1")
        ax.bar(indices + width / 2, rot_err, width, label="Reconstruction Orientation RMS (rad)", color="C2")
        ax.set_xticks(indices)
        ax.set_xticklabels([ep.name for ep in recon_eps], rotation=45, ha="right")
        ax.set_ylabel("Deviation")
        ax.set_title(f"{task_name} reconstruction error")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "reconstruction_bars.png", dpi=200)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    exclude_set = {str(Path(p).resolve()) for p in args.exclude_episodes}
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    recon_root = Path(args.recon_root).resolve() if args.recon_root else None

    summaries = []
    for task_dir_str in args.task_dirs:
        task_dir = Path(task_dir_str).resolve()
        if not task_dir.exists():
            raise FileNotFoundError(f"Task dir not found: {task_dir}")

        episode_dirs = [
            ep for ep in find_episode_dirs(task_dir) if str(ep.resolve()) not in exclude_set
        ]
        if not episode_dirs:
            print(f"[WARN] No episodes found for {task_dir.name}")
            continue
        if args.max_episodes:
            episode_dirs = episode_dirs[: args.max_episodes]

        episodes = []
        for ep_dir in episode_dirs:
            try:
                episodes.append(
                    load_episode(
                        ep_dir,
                        args.target_length,
                        args.recon_file_name,
                        args.recon_key,
                        recon_root,
                    )
                )
            except Exception as exc:
                print(f"[WARN] Skipping {ep_dir.name}: {exc}")

        if len(episodes) < 2:
            print(f"[WARN] Need >=2 valid episodes for {task_dir.name}, got {len(episodes)}")
            continue

        task_out = output_root / task_dir.name
        summary = summarize(task_dir.name, episodes, args.use_median, task_out)
        summaries.append(summary)
        print(
            f"{task_dir.name}: position RMS {summary['position']['mean_rms_m']:.4f} m "
            f"(±{summary['position']['std_rms_m']:.4f}), "
            f"orientation RMS {summary['orientation']['mean_rms_rad']:.4f} rad "
            f"(±{summary['orientation']['std_rms_rad']:.4f})"
        )

    if summaries:
        aggregate_path = output_root / "summary.json"
        aggregate_path.write_text(json.dumps(summaries, indent=2))
        print(f"\nSaved detailed metrics to {aggregate_path}")
    else:
        print("No valid tasks processed.")


if __name__ == "__main__":
    main()
