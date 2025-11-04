#!/bin/bash

# Real-time Inference Benchmark Script
# Tests model inference speed for real-world deployment

echo "🚀 VLA Model Inference Benchmark"
echo "=================================="
echo ""

# Configuration
REGRESSION_CKPT="./checkpoints/regression_best.pt"
FLOW_CKPT="./checkpoints/flow_matching_best.pt"
DATASET_DIR="/home/najo/NAS/VLA/dataset/New_dataset/Blue_point/episode_20251030_025856"
OUTPUT_DIR="./benchmark_results"
DEVICE="cuda:0"
NUM_ITERATIONS=10

# ==========================================
# 1. Compare Regression vs Flow Matching
# ==========================================
echo "📊 Test 1: Regression vs Flow Matching"
echo "========================================"

python benchmark_realtime_inference.py \
    --checkpoint-regression ${REGRESSION_CKPT} \
    --checkpoint-flow ${FLOW_CKPT} \
    --dataset-dir ${DATASET_DIR} \
    --num-iterations ${NUM_ITERATIONS} \
    --num-views 5 \
    --device ${DEVICE} \
    --output-dir ${OUTPUT_DIR}/model_comparison

echo ""
echo "✅ Test 1 complete!"
echo ""

# ==========================================
# 2. Compare Different View Counts (1-5)
# ==========================================
echo "📊 Test 2: View Count Comparison"
echo "========================================"

python benchmark_realtime_inference.py \
    --checkpoint-regression ${REGRESSION_CKPT} \
    --dataset-dir ${DATASET_DIR} \
    --num-iterations ${NUM_ITERATIONS} \
    --device ${DEVICE} \
    --compare-views \
    --output-dir ${OUTPUT_DIR}/view_comparison

echo ""
echo "✅ Test 2 complete!"
echo ""

# ==========================================
# 3. Compare With/Without Sensors
# ==========================================
echo "📊 Test 3: Sensor Impact Analysis"
echo "========================================"

python benchmark_realtime_inference.py \
    --checkpoint-regression ${REGRESSION_CKPT} \
    --dataset-dir ${DATASET_DIR} \
    --num-iterations ${NUM_ITERATIONS} \
    --num-views 5 \
    --device ${DEVICE} \
    --compare-sensors \
    --output-dir ${OUTPUT_DIR}/sensor_comparison

echo ""
echo "✅ Test 3 complete!"
echo ""

# ==========================================
# 4. With Sensor vs Without Sensor
# ==========================================
echo "📊 Test 4: Direct Sensor Comparison"
echo "========================================"

# With sensor
python benchmark_realtime_inference.py \
    --checkpoint-regression ${REGRESSION_CKPT} \
    --dataset-dir ${DATASET_DIR} \
    --num-iterations ${NUM_ITERATIONS} \
    --num-views 5 \
    --device ${DEVICE} \
    --output-dir ${OUTPUT_DIR}/with_sensor

# Without sensor
python benchmark_realtime_inference.py \
    --checkpoint-regression ${REGRESSION_CKPT} \
    --dataset-dir ${DATASET_DIR} \
    --num-iterations ${NUM_ITERATIONS} \
    --num-views 5 \
    --disable-sensor \
    --device ${DEVICE} \
    --output-dir ${OUTPUT_DIR}/without_sensor

echo ""
echo "✅ Test 4 complete!"
echo ""

# ==========================================
# Summary
# ==========================================
echo "=================================="
echo "✅ All benchmarks complete!"
echo "=================================="
echo ""
echo "Results saved in: ${OUTPUT_DIR}/"
echo ""
echo "Available results:"
echo "  - model_comparison/     : Regression vs Flow Matching"
echo "  - view_comparison/      : 1-5 camera views comparison"
echo "  - sensor_comparison/    : With/without sensor comparison"
echo "  - with_sensor/          : Results with sensor enabled"
echo "  - without_sensor/       : Results with sensor disabled"
echo ""
echo "Visualizations:"
echo "  - */comparison.png      : Performance comparison plots"
echo "  - */comparison.csv      : Detailed comparison table"
echo ""
