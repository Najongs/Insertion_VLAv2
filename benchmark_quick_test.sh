#!/bin/bash

# Quick Benchmark Test
# Fast test with fewer iterations for rapid prototyping

echo "⚡ Quick Benchmark Test"
echo "======================="
echo ""

# Configuration
REGRESSION_CKPT="./checkpoints/regression_best.pt"
FLOW_CKPT="./checkpoints/flow_matching_best.pt"
DATASET_DIR="/home/najo/NAS/VLA/dataset/New_dataset/Blue_point/episode_20251030_025856"
OUTPUT_DIR="./benchmark_results/quick_test"
NUM_ITERATIONS=3  # Quick test with only 3 iterations

# Check if checkpoints exist
if [ ! -f "$REGRESSION_CKPT" ] && [ ! -f "$FLOW_CKPT" ]; then
    echo "❌ Error: No checkpoint files found!"
    echo "   Please specify at least one checkpoint:"
    echo "   - Regression: $REGRESSION_CKPT"
    echo "   - Flow Matching: $FLOW_CKPT"
    exit 1
fi

# Determine which models to test
MODELS=""
if [ -f "$REGRESSION_CKPT" ]; then
    MODELS="$MODELS --checkpoint-regression $REGRESSION_CKPT"
    echo "✅ Found regression checkpoint"
fi

if [ -f "$FLOW_CKPT" ]; then
    MODELS="$MODELS --checkpoint-flow $FLOW_CKPT"
    echo "✅ Found flow matching checkpoint"
fi

echo ""
echo "Running quick benchmark..."
echo "Iterations: $NUM_ITERATIONS (faster but less accurate)"
echo ""

python benchmark_realtime_inference.py \
    $MODELS \
    --dataset-dir ${DATASET_DIR} \
    --num-iterations ${NUM_ITERATIONS} \
    --num-views 5 \
    --device cuda:0 \
    --output-dir ${OUTPUT_DIR}

echo ""
echo "=================================="
echo "✅ Quick test complete!"
echo "=================================="
echo ""
echo "Results saved in: ${OUTPUT_DIR}/"
echo ""
echo "For more accurate results, run:"
echo "  bash run_benchmark.sh"
echo ""
