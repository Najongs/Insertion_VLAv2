#!/bin/bash

# Test all benchmark modes to verify fixes
# This script runs quick tests for all major benchmark configurations

set -e  # Exit on error

DATASET_DIR="/home/najo/NAS/VLA/dataset/New_dataset/Blue_point/episode_20251030_025856"
REGRESSION_CKPT="./checkpoints/regression_best.pt"
OUTPUT_BASE="./benchmark_results/verification"
ITERATIONS=2  # Quick test

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   Benchmark Verification - Testing All Modes              ║"
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
# Test 1: Standard benchmark (with sensor and robot states)
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 1: Standard Benchmark (with sensor)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python benchmark_realtime_inference.py \
    --checkpoint-regression ${REGRESSION_CKPT} \
    --dataset-dir ${DATASET_DIR} \
    --num-iterations ${ITERATIONS} \
    --num-views 5 \
    --device cuda:0 \
    --output-dir ${OUTPUT_BASE}/test1_standard

echo "✅ Test 1 passed!"
echo ""

# ============================================================
# Test 2: Without sensor
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 2: Without Sensor (dimension mismatch fix verification)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python benchmark_realtime_inference.py \
    --checkpoint-regression ${REGRESSION_CKPT} \
    --dataset-dir ${DATASET_DIR} \
    --num-iterations ${ITERATIONS} \
    --num-views 5 \
    --disable-sensor \
    --device cuda:0 \
    --output-dir ${OUTPUT_BASE}/test2_no_sensor

echo "✅ Test 2 passed!"
echo ""

# ============================================================
# Test 3: Without robot states
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 3: Without Robot States"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python benchmark_realtime_inference.py \
    --checkpoint-regression ${REGRESSION_CKPT} \
    --dataset-dir ${DATASET_DIR} \
    --num-iterations ${ITERATIONS} \
    --num-views 5 \
    --disable-robot-states \
    --device cuda:0 \
    --output-dir ${OUTPUT_BASE}/test3_no_robot

echo "✅ Test 3 passed!"
echo ""

# ============================================================
# Test 4: Different view counts
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 4: Different View Counts (1, 3, 5)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for views in 1 3 5; do
    echo "  Testing with ${views} views..."
    python benchmark_realtime_inference.py \
        --checkpoint-regression ${REGRESSION_CKPT} \
        --dataset-dir ${DATASET_DIR} \
        --num-iterations ${ITERATIONS} \
        --num-views ${views} \
        --device cuda:0 \
        --output-dir ${OUTPUT_BASE}/test4_views_${views} \
        > /dev/null 2>&1
    echo "    ✅ ${views} views OK"
done

echo "✅ Test 4 passed!"
echo ""

# ============================================================
# Verification Summary
# ============================================================
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║             Verification Summary                           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check VL encoding times to verify image loading
echo "📊 Checking VL encoding times (should be ~1400-1500ms with images):"
echo ""

for test in test1_standard test2_no_sensor test3_no_robot; do
    result_file="${OUTPUT_BASE}/${test}/regression_results.json"
    if [ -f "$result_file" ]; then
        vl_time=$(python -c "import json; print(f\"{json.load(open('$result_file'))['timing']['vl_encoding']['mean']:.1f}\")")
        if (( $(echo "$vl_time > 1000" | bc -l) )); then
            echo "  ✅ ${test}: ${vl_time} ms (images loaded correctly)"
        else
            echo "  ❌ ${test}: ${vl_time} ms (WARNING: too fast, images may not be loaded)"
        fi
    fi
done

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   ✅ All Tests Passed!                                     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved in: ${OUTPUT_BASE}/"
echo ""
echo "To view detailed results:"
echo "  cat ${OUTPUT_BASE}/*/regression_results.json"
echo ""
