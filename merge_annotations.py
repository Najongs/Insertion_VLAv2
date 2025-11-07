import json
import os
from pathlib import Path
import argparse

def merge_annotations(output_file="vlm_annotations.json"):
    """
    Merge all task-specific annotation files into a single file.
    """
    merged_annotations = {}

    # Find all task-specific annotation files
    pattern = "vlm_annotations_*.json"
    annotation_files = list(Path.cwd().glob(pattern))

    if not annotation_files:
        print(f"No annotation files found matching pattern: {pattern}")
        return

    print(f"Found {len(annotation_files)} annotation files to merge:")
    for file_path in annotation_files:
        print(f"  - {file_path.name}")

    # Merge all files
    for file_path in annotation_files:
        task_name = file_path.stem.replace("vlm_annotations_", "")
        print(f"\nMerging {task_name}...")

        with open(file_path, "r") as f:
            task_annotations = json.load(f)

        # Check for duplicate episode IDs
        for episode_id in task_annotations.keys():
            if episode_id in merged_annotations:
                print(f"  Warning: Duplicate episode ID found: {episode_id}")
            else:
                merged_annotations[episode_id] = task_annotations[episode_id]

    # Save merged annotations
    output_path = Path.cwd() / output_file
    with open(output_path, "w") as f:
        json.dump(merged_annotations, f, indent=4)

    print(f"\n✅ Merged {len(merged_annotations)} episodes")
    print(f"Saved to {output_path}")

    # Print statistics
    episodes_with_target = sum(1 for ep_data in merged_annotations.values()
                               if isinstance(ep_data, dict) and ep_data.get("target_found_timestamp") is not None)
    print(f"Episodes with target found: {episodes_with_target}/{len(merged_annotations)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge task-specific annotation files")
    parser.add_argument("--output", type=str, default="vlm_annotations.json",
                       help="Output filename (default: vlm_annotations.json)")
    args = parser.parse_args()

    merge_annotations(args.output)
