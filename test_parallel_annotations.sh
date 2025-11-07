#!/bin/bash

# ==============================
# Test Parallel VLM Annotation Generation
# ==============================
# This script runs annotation generation in test mode (1 episode per task)
# to verify the setup before running the full dataset.

echo "Starting TEST parallel VLM annotation generation..."
echo "================================================"
echo "This will process only 1 episode per task for quick validation."
echo ""

# Create log directory
mkdir -p logs

# Define tasks
TASKS=("Blue_point" "Eye_trocar" "Green_point" "Red_point" "White_point" "Yellow_point")

# Map tasks to GPUs (4 GPUs available)
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

# Launch each task in background with --test_mode flag
for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"
    GPU="${GPU_MAPPING[$i]}"
    LOG_FILE="logs/${TASK}_test.log"

    echo "Launching $TASK on GPU $GPU (log: $LOG_FILE)"

    # Run in background with --test_mode flag
    python generate_vlm_annotations.py \
        --task_name "$TASK" \
        --gpu_id "$GPU" \
        --test_mode \
        > "$LOG_FILE" 2>&1 &

    # Store the process ID
    PIDS+=($!)

    # Small delay to avoid race conditions during model loading
    sleep 3
done

echo ""
echo "All test tasks launched. Waiting for completion..."
echo "You can monitor progress with: tail -f logs/<task_name>_test.log"
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
echo "All test tasks completed. Merging annotations..."
echo "================================================"

# Merge all task-specific annotation files
python merge_annotations.py --output vlm_annotations_test.json

echo ""
echo "✅ Test complete! Check vlm_annotations_test.json for the result."
echo "If everything looks good, run ./run_parallel_annotations.sh for the full dataset."
