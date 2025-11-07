#!/bin/bash

# ==============================
# Parallel VLM Annotation Generation
# ==============================
# This script runs annotation generation for each task in parallel
# across multiple GPUs to speed up processing.

echo "Starting parallel VLM annotation generation..."
echo "================================================"

# Create log directory
mkdir -p logs

# Define tasks
TASKS=("Blue_point" "Eye_trocar" "Green_point" "Red_point" "White_point" "Yellow_point")

# Map tasks to GPUs (4 GPUs available)
# We'll distribute 6 tasks across 4 GPUs
GPU_MAPPING=(
    "0"  # Blue_point -> GPU 0
    "1"  # Eye_trocar -> GPU 1
    "2"  # Green_point -> GPU 2
    "3"  # Red_point -> GPU 3
    "0"  # White_point -> GPU 0 (shared with Blue_point)
    "1"  # Yellow_point -> GPU 1 (shared with Eye_trocar)
)

# Store process IDs for waiting
PIDS=()

# Launch each task in background
for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"
    GPU="${GPU_MAPPING[$i]}"
    LOG_FILE="logs/${TASK}.log"

    echo "Launching $TASK on GPU $GPU (log: $LOG_FILE)"

    # Run in background and redirect output to log file
    python generate_vlm_annotations.py \
        --task_name "$TASK" \
        --gpu_id "$GPU" \
        > "$LOG_FILE" 2>&1 &

    # Store the process ID
    PIDS+=($!)

    # Small delay to avoid race conditions during model loading
    sleep 5
done

echo ""
echo "All tasks launched. Waiting for completion..."
echo "You can monitor progress with: tail -f logs/<task_name>.log"
echo ""

# Wait for all processes to complete
for i in "${!PIDS[@]}"; do
    PID="${PIDS[$i]}"
    TASK="${TASKS[$i]}"

    echo "Waiting for $TASK (PID: $PID)..."
    wait "$PID"
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ $TASK completed successfully"
    else
        echo "❌ $TASK failed with exit code $EXIT_CODE"
    fi
done

echo ""
echo "================================================"
echo "All tasks completed. Merging annotations..."
echo "================================================"

# Merge all task-specific annotation files
python merge_annotations.py

echo ""
echo "✅ All done! Check vlm_annotations.json for the final result."
