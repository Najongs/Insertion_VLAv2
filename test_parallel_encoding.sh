#!/bin/bash

# Test Parallel View Encoding Optimization
# This script compares sequential vs parallel view encoding performance

set -e  # Exit on error

DATASET_DIR="/home/najo/NAS/VLA/dataset/New_dataset/Blue_point/episode_20251030_025856"
REGRESSION_CKPT="./checkpoints/regression_best.pt"
OUTPUT_DIR="./benchmark_results/parallel_encoding_test"
ITERATIONS=3  # Quick test

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   Parallel View Encoding Test                             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check checkpoint exists
if [ ! -f "$REGRESSION_CKPT" ]; then
    echo "❌ Error: Checkpoint not found at $REGRESSION_CKPT"
    exit 1
fi

echo "📦 Using dataset: $DATASET_DIR"
echo "🏋️  Using checkpoint: $REGRESSION_CKPT"
echo "🔄 Iterations: $ITERATIONS (quick test)"
echo ""

# ============================================================
# Test 1: Sequential Encoding (Baseline)
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 1: Sequential Encoding (Baseline)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python benchmark_realtime_inference.py \
    --checkpoint-regression ${REGRESSION_CKPT} \
    --dataset-dir ${DATASET_DIR} \
    --num-iterations ${ITERATIONS} \
    --num-views 5 \
    --device cuda:0 \
    --output-dir ${OUTPUT_DIR}/sequential

echo "✅ Sequential test complete!"
echo ""

# ============================================================
# Test 2: Parallel Encoding (Optimized)
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 2: Parallel Encoding (Optimized)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python benchmark_realtime_inference.py \
    --checkpoint-regression ${REGRESSION_CKPT} \
    --dataset-dir ${DATASET_DIR} \
    --num-iterations ${ITERATIONS} \
    --num-views 5 \
    --device cuda:0 \
    --parallel-view-encoding \
    --view-aggregation mean \
    --output-dir ${OUTPUT_DIR}/parallel

echo "✅ Parallel test complete!"
echo ""

# ============================================================
# Test 3: Direct Comparison Mode
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 3: Sequential vs Parallel (Direct Comparison)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python benchmark_realtime_inference.py \
    --checkpoint-regression ${REGRESSION_CKPT} \
    --dataset-dir ${DATASET_DIR} \
    --num-iterations ${ITERATIONS} \
    --num-views 5 \
    --device cuda:0 \
    --compare-parallel \
    --output-dir ${OUTPUT_DIR}/comparison

echo "✅ Comparison complete!"
echo ""

# ============================================================
# Results Summary
# ============================================================
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║             Performance Summary                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Extract VL encoding times
echo "📊 VL Encoding Times:"
echo ""

for mode in sequential parallel; do
    result_file="${OUTPUT_DIR}/${mode}/regression_results.json"
    if [ -f "$result_file" ]; then
        vl_time=$(python -c "import json; data=json.load(open('$result_file')); print(f\"{data['timing']['vl_encoding']['mean']:.1f}\")")
        total_time=$(python -c "import json; data=json.load(open('$result_file')); print(f\"{data['timing']['total']['mean']:.1f}\")")
        fps=$(python -c "import json; data=json.load(open('$result_file')); print(f\"{data['fps']:.2f}\")")
        echo "  ${mode^}:"
        echo "    VL Encoding: ${vl_time} ms"
        echo "    Total Time:  ${total_time} ms"
        echo "    FPS:         ${fps}"
        echo ""
    fi
done

# Calculate speedup
if [ -f "${OUTPUT_DIR}/sequential/regression_results.json" ] && [ -f "${OUTPUT_DIR}/parallel/regression_results.json" ]; then
    speedup=$(python -c "
import json
seq = json.load(open('${OUTPUT_DIR}/sequential/regression_results.json'))
par = json.load(open('${OUTPUT_DIR}/parallel/regression_results.json'))
seq_time = seq['timing']['vl_encoding']['mean']
par_time = par['timing']['vl_encoding']['mean']
speedup = seq_time / par_time
print(f'{speedup:.2f}x')
")
    echo "⚡ Speedup: ${speedup}"
    echo ""
fi

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   ✅ All Tests Passed!                                     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved in: ${OUTPUT_DIR}/"
echo ""
echo "To view comparison plot:"
echo "  open ${OUTPUT_DIR}/comparison/comparison.png"
echo ""
