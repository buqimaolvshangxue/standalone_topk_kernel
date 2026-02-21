#!/bin/bash
# Compare Baseline Analysis Script for DL Platform
#
# This script runs baseline kernel tests and topk kernel tests,
# then calculates and outputs analysis results.
#
# Model: Qwen3-30B-A3B (num_experts=128, topk=8)
# Test tokens: 1, 4, 256, 1024
#
# Prerequisites:
#   1. source /LocalRun/xiaolong.zhu/artifactory/sdk/testenv.sh
#   2. source /LocalRun/xiaolong.zhu/artifactory/venv/bin/activate
#   3. Run ./build.sh dl first to compile the binaries
#
# Usage: ./compare_baseline_dl.sh

set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

# ============================================
# Hardware Configuration (Hardcoded for DL)
# ============================================
GPU_NAME="ks38"
GPU_MEM="64 GB"
PEAK_BANDWIDTH_GBPS=409.6
BF16_TFLOPS_WITHOUT_SPARSITY=70
BF16_TFLOPS_WITH_SPARSITY=140

# Test Configuration
ITERS=100
DTYPE="bf16"

# Qwen3-30B-A3B MoE Configuration
EXPERTS=128
TOPK=8

echo "=============================================="
echo "  DL Kernel Baseline Comparison Analysis"
echo "  Model: Qwen3-30B-A3B (experts=$EXPERTS, topk=$TOPK)"
echo "=============================================="
echo ""

echo "[Hardware Info]"
echo "  Device: $GPU_NAME"
echo "  Memory: $GPU_MEM"
echo "  Peak Memory Bandwidth: ${PEAK_BANDWIDTH_GBPS} GB/s"
echo "  BF16 Tensor TFLOPs: ${BF16_TFLOPS_WITHOUT_SPARSITY} (W/O Sparsity) / ${BF16_TFLOPS_WITH_SPARSITY} (W/ Sparsity)"
echo ""

# Check if binaries exist
if [ ! -f "build/bench_baseline" ]; then
    echo "Error: build/bench_baseline not found."
    echo "Please run ./build.sh dl first."
    exit 1
fi

if [ ! -f "build/bench_perf" ]; then
    echo "Error: build/bench_perf not found."
    echo "Please run ./build.sh dl first."
    exit 1
fi

# ============================================
# Step 1: Run Baseline Kernel Tests
# ============================================
echo "[Step 1/4] Running baseline kernel tests..."
echo ""

# Empty kernel
echo "--- Empty Kernel ---"
EMPTY_OUTPUT=$(./build/bench_baseline empty $ITERS 2>&1)
EMPTY_TIME_US=$(echo "$EMPTY_OUTPUT" | grep "avg_time_us=" | cut -d'=' -f2)
echo "Average time: $EMPTY_TIME_US us"
echo ""

# Minimal RW kernel
echo "--- Minimal RW Kernel ---"
RW_OUTPUT=$(./build/bench_baseline rw $ITERS 2>&1)
RW_TIME_US=$(echo "$RW_OUTPUT" | grep "avg_time_us=" | cut -d'=' -f2)
echo "Average time: $RW_TIME_US us"
echo ""

# ============================================
# Step 2: Run TopK Kernel Tests
# ============================================
echo "[Step 2/4] Running TopK kernel tests..."
echo ""

# Token = 1
echo "--- TopK (tokens=1) ---"
TOPK_1_OUTPUT=$(./build/bench_perf $EXPERTS 1 $DTYPE $TOPK $ITERS 2>&1)
TOPK_1_TIME_US=$(echo "$TOPK_1_OUTPUT" | grep "Average kernel time:" | sed 's/.*: \([0-9.]*\) us.*/\1/')
echo "Average time: $TOPK_1_TIME_US us"
echo ""

# Token = 4
echo "--- TopK (tokens=4) ---"
TOPK_4_OUTPUT=$(./build/bench_perf $EXPERTS 4 $DTYPE $TOPK $ITERS 2>&1)
TOPK_4_TIME_US=$(echo "$TOPK_4_OUTPUT" | grep "Average kernel time:" | sed 's/.*: \([0-9.]*\) us.*/\1/')
echo "Average time: $TOPK_4_TIME_US us"
echo ""

# Token = 256
echo "--- TopK (tokens=256) ---"
TOPK_256_OUTPUT=$(./build/bench_perf $EXPERTS 256 $DTYPE $TOPK $ITERS 2>&1)
TOPK_256_TIME_US=$(echo "$TOPK_256_OUTPUT" | grep "Average kernel time:" | sed 's/.*: \([0-9.]*\) us.*/\1/')
echo "Average time: $TOPK_256_TIME_US us"
echo ""

# Token = 1024
echo "--- TopK (tokens=1024) ---"
TOPK_1024_OUTPUT=$(./build/bench_perf $EXPERTS 1024 $DTYPE $TOPK $ITERS 2>&1)
TOPK_1024_TIME_US=$(echo "$TOPK_1024_OUTPUT" | grep "Average kernel time:" | sed 's/.*: \([0-9.]*\) us.*/\1/')
echo "Average time: $TOPK_1024_TIME_US us"
echo ""

# ============================================
# Step 3: Calculate Theoretical Values
# ============================================
echo "[Step 3/4] Calculating theoretical values..."
echo ""

# Data sizes (bytes)
# Input: tokens * experts * 2 (bf16)
# Output: tokens * topk * 4 (float) + tokens * topk * 4 (int) + tokens * topk * 4 (source_rows)

calc_data_size() {
    local tokens=$1
    local experts=$EXPERTS
    local topk=$TOPK
    local input_bytes=$(echo "$tokens * $experts * 2" | bc)
    local output_bytes=$(echo "$tokens * $topk * 4 * 3" | bc)
    local total_bytes=$(echo "$input_bytes + $output_bytes" | bc)
    echo $total_bytes
}

# Calculate for each token count
DATA_1_BYTES=$(calc_data_size 1)
DATA_4_BYTES=$(calc_data_size 4)
DATA_256_BYTES=$(calc_data_size 256)
DATA_1024_BYTES=$(calc_data_size 1024)

# Theoretical transfer time (nanoseconds) with formatting
calc_theoretical_ns() {
    local bytes=$1
    local bandwidth=$PEAK_BANDWIDTH_GBPS
    local raw=$(echo "scale=3; $bytes / $bandwidth" | bc)
    # Add leading zero if needed
    if [[ "$raw" == .* ]]; then
        raw="0$raw"
    fi
    echo "$raw"
}

THEORY_1_NS=$(calc_theoretical_ns $DATA_1_BYTES)
THEORY_4_NS=$(calc_theoretical_ns $DATA_4_BYTES)
THEORY_256_NS=$(calc_theoretical_ns $DATA_256_BYTES)
THEORY_1024_NS=$(calc_theoretical_ns $DATA_1024_BYTES)

echo "Token=1:    Data size = $DATA_1_BYTES bytes, Theoretical transfer = $THEORY_1_NS ns"
echo "Token=4:    Data size = $DATA_4_BYTES bytes, Theoretical transfer = $THEORY_4_NS ns"
echo "Token=256:  Data size = $DATA_256_BYTES bytes, Theoretical transfer = $THEORY_256_NS ns"
echo "Token=1024: Data size = $DATA_1024_BYTES bytes, Theoretical transfer = $THEORY_1024_NS ns"
echo ""

# Calculate bandwidth utilization with proper formatting
calc_bw_util() {
    local data_bytes=$1
    local time_us=$2
    local peak_bw=$PEAK_BANDWIDTH_GBPS
    # Effective BW (GB/s) = data_bytes / (time_us * 1000)
    # Utilization = effective_bw / peak_bw * 100
    local raw=$(echo "scale=4; $data_bytes / $time_us / 1000 / $peak_bw * 100" | bc)
    # Format: add leading zero if needed, limit to 2 decimal places
    if [[ "$raw" == .* ]]; then
        raw="0$raw"
    fi
    # Round to 2 decimal places using printf
    printf "%.2f" "$raw"
}

BW_UTIL_1=$(calc_bw_util $DATA_1_BYTES $TOPK_1_TIME_US)
BW_UTIL_4=$(calc_bw_util $DATA_4_BYTES $TOPK_4_TIME_US)
BW_UTIL_256=$(calc_bw_util $DATA_256_BYTES $TOPK_256_TIME_US)
BW_UTIL_1024=$(calc_bw_util $DATA_1024_BYTES $TOPK_1024_TIME_US)

# Calculate percentages with proper scale
FIXED_PCT_1=$(echo "scale=1; ${RW_TIME_US} / ${TOPK_1_TIME_US} * 100" | bc)
COMPUTE_PCT_1=$(echo "scale=1; (${TOPK_1_TIME_US} - ${RW_TIME_US}) / ${TOPK_1_TIME_US} * 100" | bc)
COMPUTE_TIME_1=$(echo "scale=2; ${TOPK_1_TIME_US} - ${RW_TIME_US}" | bc)

FIXED_PCT_4=$(echo "scale=1; ${RW_TIME_US} / ${TOPK_4_TIME_US} * 100" | bc)
COMPUTE_PCT_4=$(echo "scale=1; (${TOPK_4_TIME_US} - ${RW_TIME_US}) / ${TOPK_4_TIME_US} * 100" | bc)
COMPUTE_TIME_4=$(echo "scale=2; ${TOPK_4_TIME_US} - ${RW_TIME_US}" | bc)

FIXED_PCT_256=$(echo "scale=1; ${RW_TIME_US} / ${TOPK_256_TIME_US} * 100" | bc)
COMPUTE_PCT_256=$(echo "scale=1; (${TOPK_256_TIME_US} - ${RW_TIME_US}) / ${TOPK_256_TIME_US} * 100" | bc)
COMPUTE_TIME_256=$(echo "scale=2; ${TOPK_256_TIME_US} - ${RW_TIME_US}" | bc)

FIXED_PCT_1024=$(echo "scale=1; ${RW_TIME_US} / ${TOPK_1024_TIME_US} * 100" | bc)
COMPUTE_PCT_1024=$(echo "scale=1; (${TOPK_1024_TIME_US} - ${RW_TIME_US}) / ${TOPK_1024_TIME_US} * 100" | bc)
COMPUTE_TIME_1024=$(echo "scale=2; ${TOPK_1024_TIME_US} - ${RW_TIME_US}" | bc)

# Time ratios
TIME_RATIO_4=$(echo "scale=2; ${TOPK_4_TIME_US} / ${TOPK_1_TIME_US}" | bc)
TIME_RATIO_256=$(echo "scale=2; ${TOPK_256_TIME_US} / ${TOPK_1_TIME_US}" | bc)
TIME_RATIO_1024=$(echo "scale=2; ${TOPK_1024_TIME_US} / ${TOPK_1_TIME_US}" | bc)

# For "差了 X 倍" calculation: target_data / current_data
TARGET_DATA_BYTES=$(echo "scale=0; $PEAK_BANDWIDTH_GBPS / 2 * $RW_TIME_US * 1000" | bc)  # bytes needed for 50% BW
RATIO_TO_TARGET=$(echo "scale=1; $TARGET_DATA_BYTES / $DATA_1024_BYTES" | bc)

# Calculate effective bandwidth (GB/s) with formatting
calc_effective_bw() {
    local data_bytes=$1
    local time_us=$2
    local raw=$(echo "scale=2; $data_bytes / $time_us / 1000" | bc)
    if [[ "$raw" == .* ]]; then
        raw="0$raw"
    fi
    echo "$raw"
}

EFF_BW_1=$(calc_effective_bw $DATA_1_BYTES $TOPK_1_TIME_US)
EFF_BW_4=$(calc_effective_bw $DATA_4_BYTES $TOPK_4_TIME_US)
EFF_BW_256=$(calc_effective_bw $DATA_256_BYTES $TOPK_256_TIME_US)
EFF_BW_1024=$(calc_effective_bw $DATA_1024_BYTES $TOPK_1024_TIME_US)

# ============================================
# Step 4: Output Summary
# ============================================
echo "[Step 4/4] Summary..."
echo ""

echo "=============================================="
echo "  Analysis Summary"
echo "=============================================="
echo ""

echo "=== Hardware ==="
echo "  Device: $GPU_NAME"
echo "  Memory: $GPU_MEM"
echo "  Peak Bandwidth: ${PEAK_BANDWIDTH_GBPS} GB/s"
echo ""

echo "=== Baseline Kernel Results ==="
echo "  | Kernel             | Time (us) |"
echo "  |--------------------|-----------|"
echo "  | empty_kernel       | ${EMPTY_TIME_US} |"
echo "  | minimal_rw_kernel  | ${RW_TIME_US} |"
echo ""
echo "  Fixed Overhead Baseline: ~${RW_TIME_US} us"
echo ""

echo "=== TopK Kernel Results ==="
echo "  | Tokens | Data (bytes) | Theory (ns) | Actual (us) | BW Util |"
echo "  |--------|--------------|-------------|-------------|---------|"
echo "  | 1      | ${DATA_1_BYTES}        | ${THEORY_1_NS}      | ${TOPK_1_TIME_US}     | ${BW_UTIL_1}% |"
echo "  | 4      | ${DATA_4_BYTES}       | ${THEORY_4_NS}      | ${TOPK_4_TIME_US}     | ${BW_UTIL_4}% |"
echo "  | 256    | ${DATA_256_BYTES}      | ${THEORY_256_NS}     | ${TOPK_256_TIME_US}     | ${BW_UTIL_256}% |"
echo "  | 1024   | ${DATA_1024_BYTES}     | ${THEORY_1024_NS}    | ${TOPK_1024_TIME_US}   | ${BW_UTIL_1024}% |"
echo ""

echo "=== Time Breakdown ==="
echo "  tokens=1:"
echo "    Total:      ${TOPK_1_TIME_US} us"
echo "    Fixed:      ${RW_TIME_US} us (${FIXED_PCT_1}%)"
echo "    Compute:    ${COMPUTE_TIME_1} us (${COMPUTE_PCT_1}%)"
echo ""
echo "  tokens=4:"
echo "    Total:      ${TOPK_4_TIME_US} us"
echo "    Fixed:      ${RW_TIME_US} us (${FIXED_PCT_4}%)"
echo "    Compute:    ${COMPUTE_TIME_4} us (${COMPUTE_PCT_4}%)"
echo ""
echo "  tokens=256:"
echo "    Total:      ${TOPK_256_TIME_US} us"
echo "    Fixed:      ${RW_TIME_US} us (${FIXED_PCT_256}%)"
echo "    Compute:    ${COMPUTE_TIME_256} us (${COMPUTE_PCT_256}%)"
echo ""
echo "  tokens=1024:"
echo "    Total:      ${TOPK_1024_TIME_US} us"
echo "    Fixed:      ${RW_TIME_US} us (${FIXED_PCT_1024}%)"
echo "    Compute:    ${COMPUTE_TIME_1024} us (${COMPUTE_PCT_1024}%)"
echo ""

echo "=== Bandwidth Analysis ==="
echo "  tokens=1:    ${EFF_BW_1} GB/s (util: ${BW_UTIL_1}%)"
echo "  tokens=4:    ${EFF_BW_4} GB/s (util: ${BW_UTIL_4}%)"
echo "  tokens=256:  ${EFF_BW_256} GB/s (util: ${BW_UTIL_256}%)"
echo "  tokens=1024: ${EFF_BW_1024} GB/s (util: ${BW_UTIL_1024}%)"
echo ""

echo "=== Time Ratio (relative to tokens=1) ==="
echo "  Data ratio:   1 : 4 : 256 : 1024"
echo "  Time ratio:   1.00 : ${TIME_RATIO_4} : ${TIME_RATIO_256} : ${TIME_RATIO_1024}"
echo ""

echo "=== To Achieve 50% Bandwidth Utilization ==="
echo "  Need data size: $(echo "scale=2; $TARGET_DATA_BYTES / 1024 / 1024" | bc) MB"
echo "  Current tokens=1024 data: $(echo "scale=1; $DATA_1024_BYTES / 1024" | bc) KB"
echo "  Need ${RATIO_TO_TARGET}x more data"
echo ""

echo "=============================================="
echo "  Analysis Complete"
echo "=============================================="
