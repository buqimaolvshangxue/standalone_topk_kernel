#!/bin/bash
# Compare Baseline Analysis Script (Unified for NV/DL Platforms)
#
# This script runs baseline kernel tests and topk kernel tests,
# outputs raw data for later analysis.
#
# Model: Qwen3-30B-A3B (num_experts=128, topk=8)
# Test tokens: 1, 4, 256, 1024
#
# Prerequisites:
#   - NV: Run ./build.sh nv first
#   - DL: source SDK env, then run ./build.sh dl first
#
# Usage: ./compare_baseline.sh [warmup] [iters]
#   warmup: Number of warmup iterations (default: 100)
#   iters:  Number of test iterations (default: 100)

set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

# ============================================
# Parse Arguments
# ============================================
WARMUP=${1:-100}
ITERS=${2:-100}

# ============================================
# Detect Platform
# ============================================
if command -v nvidia-smi &> /dev/null; then
    PLATFORM="nv"
elif command -v dlcc &> /dev/null || [ -n "${SDK_DIR:-}" ]; then
    PLATFORM="dl"
else
    echo "Error: Cannot detect platform (neither nvidia-smi nor dlcc found)"
    echo "For DL platform, please source the SDK environment first"
    exit 1
fi

# ============================================
# Configuration
# ============================================
DTYPE="bf16"
EXPERTS=128
TOPK=8

echo "=============================================="
echo "  GPU Kernel Baseline Comparison"
echo "  Platform: $PLATFORM"
echo "  Model: Qwen3-30B-A3B (experts=$EXPERTS, topk=$TOPK)"
echo "  Warmup: $WARMUP, Iterations: $ITERS"
echo "=============================================="
echo ""

# Check if binaries exist
if [ ! -f "build/bench_factor" ]; then
    echo "Error: build/bench_factor not found."
    echo "Please run ./build.sh $PLATFORM first."
    exit 1
fi

if [ ! -f "build/bench_perf" ]; then
    echo "Error: build/bench_perf not found."
    echo "Please run ./build.sh $PLATFORM first."
    exit 1
fi

# ============================================
# Step 1: Collect GPU Information
# ============================================
echo "[Step 1/4] Collecting GPU information..."
echo ""

if [ "$PLATFORM" = "nv" ]; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader | head -1)
    GPU_NAME=$(echo $GPU_INFO | cut -d',' -f1 | sed 's/^ *//;s/ *$//')
    GPU_MEM=$(echo $GPU_INFO | cut -d',' -f2 | tr -d ' ')
    GPU_CC=$(echo $GPU_INFO | cut -d',' -f3 | tr -d ' ')

    if [[ "$GPU_NAME" == *"A40"* ]]; then
        PEAK_BANDWIDTH_GBPS=696
    elif [[ "$GPU_NAME" == *"A100"* ]]; then
        PEAK_BANDWIDTH_GBPS=1555
    else
        PEAK_BANDWIDTH_GBPS=500
    fi
    echo "GPU: $GPU_NAME"
    echo "Memory: $GPU_MEM"
    echo "Compute Capability: $GPU_CC"
    echo "Peak Bandwidth: ${PEAK_BANDWIDTH_GBPS} GB/s"

elif [ "$PLATFORM" = "dl" ]; then
    GPU_NAME="ks20"
    GPU_MEM="32 GB"
    PEAK_BANDWIDTH_GBPS=102.4
    BF16_TFLOPS_WITHOUT_SPARSITY=18
    BF16_TFLOPS_WITH_SPARSITY=36

    echo "Device: $GPU_NAME"
    echo "Memory: $GPU_MEM"
    echo "Peak Bandwidth: ${PEAK_BANDWIDTH_GBPS} GB/s"
    echo "BF16 TFLOPs: ${BF16_TFLOPS_WITHOUT_SPARSITY} (W/O Sparsity) / ${BF16_TFLOPS_WITH_SPARSITY} (W/ Sparsity)"
fi
echo ""

# ============================================
# Step 2: Run Baseline Kernel Tests
# ============================================
echo "[Step 2/4] Running baseline kernel tests..."
echo ""

EMPTY_OUTPUT=$(./build/bench_factor empty $WARMUP $ITERS 2>&1)
EMPTY_TIME_US=$(echo "$EMPTY_OUTPUT" | grep "avg_time_us=" | cut -d'=' -f2)

RW_OUTPUT=$(./build/bench_factor minimal_rw $WARMUP $ITERS 2>&1)
RW_TIME_US=$(echo "$RW_OUTPUT" | grep "avg_time_us=" | cut -d'=' -f2)

echo "Baseline Results:"
printf "  %-20s %10s\n" "Kernel" "Time (us)"
printf "  %-20s %10s\n" "--------------------" "----------"
printf "  %-20s %10s\n" "empty_kernel" "$EMPTY_TIME_US"
printf "  %-20s %10s\n" "minimal_rw_kernel" "$RW_TIME_US"
echo ""

# ============================================
# Step 3: Run TopK Kernel Tests
# ============================================
echo "[Step 3/4] Running TopK kernel tests..."
echo ""

TOPK_1_OUTPUT=$(./build/bench_perf $EXPERTS 1 $DTYPE $TOPK $WARMUP $ITERS 2>&1)
TOPK_1_TIME_US=$(echo "$TOPK_1_OUTPUT" | grep "Average kernel time:" | sed 's/.*: \([0-9.]*\) us.*/\1/')

TOPK_4_OUTPUT=$(./build/bench_perf $EXPERTS 4 $DTYPE $TOPK $WARMUP $ITERS 2>&1)
TOPK_4_TIME_US=$(echo "$TOPK_4_OUTPUT" | grep "Average kernel time:" | sed 's/.*: \([0-9.]*\) us.*/\1/')

TOPK_256_OUTPUT=$(./build/bench_perf $EXPERTS 256 $DTYPE $TOPK $WARMUP $ITERS 2>&1)
TOPK_256_TIME_US=$(echo "$TOPK_256_OUTPUT" | grep "Average kernel time:" | sed 's/.*: \([0-9.]*\) us.*/\1/')

TOPK_1024_OUTPUT=$(./build/bench_perf $EXPERTS 1024 $DTYPE $TOPK $WARMUP $ITERS 2>&1)
TOPK_1024_TIME_US=$(echo "$TOPK_1024_OUTPUT" | grep "Average kernel time:" | sed 's/.*: \([0-9.]*\) us.*/\1/')

# ============================================
# Step 4: Output Results Table
# ============================================
echo "[Step 4/4] Results Summary"
echo ""

# Calculate data sizes
calc_data_size() {
    local tokens=$1
    local input_bytes=$(echo "$tokens * $EXPERTS * 2" | bc)
    local output_bytes=$(echo "$tokens * $TOPK * 4 * 3" | bc)
    echo $(echo "$input_bytes + $output_bytes" | bc)
}

DATA_1_BYTES=$(calc_data_size 1)
DATA_4_BYTES=$(calc_data_size 4)
DATA_256_BYTES=$(calc_data_size 256)
DATA_1024_BYTES=$(calc_data_size 1024)

# Calculate bandwidth utilization
calc_bw_util() {
    local data_bytes=$1
    local time_us=$2
    local raw=$(echo "scale=2; $data_bytes / $time_us / 1000 / $PEAK_BANDWIDTH_GBPS * 100" | bc)
    if [[ "$raw" == .* ]]; then raw="0$raw"; fi
    echo "$raw"
}

BW_UTIL_1=$(calc_bw_util $DATA_1_BYTES $TOPK_1_TIME_US)
BW_UTIL_4=$(calc_bw_util $DATA_4_BYTES $TOPK_4_TIME_US)
BW_UTIL_256=$(calc_bw_util $DATA_256_BYTES $TOPK_256_TIME_US)
BW_UTIL_1024=$(calc_bw_util $DATA_1024_BYTES $TOPK_1024_TIME_US)

# Calculate effective bandwidth
calc_eff_bw() {
    local data_bytes=$1
    local time_us=$2
    local raw=$(echo "scale=2; $data_bytes / $time_us / 1000" | bc)
    if [[ "$raw" == .* ]]; then raw="0$raw"; fi
    echo "$raw"
}

EFF_BW_1=$(calc_eff_bw $DATA_1_BYTES $TOPK_1_TIME_US)
EFF_BW_4=$(calc_eff_bw $DATA_4_BYTES $TOPK_4_TIME_US)
EFF_BW_256=$(calc_eff_bw $DATA_256_BYTES $TOPK_256_TIME_US)
EFF_BW_1024=$(calc_eff_bw $DATA_1024_BYTES $TOPK_1024_TIME_US)

# Output table
printf "%-8s %12s %12s %12s %12s\n" "Tokens" "Data(B)" "Time(us)" "BW(GB/s)" "BW Util%"
printf "%-8s %12s %12s %12s %12s\n" "--------" "------------" "------------" "------------" "------------"
printf "%-8s %12s %12s %12s %12s\n" "1" "$DATA_1_BYTES" "$TOPK_1_TIME_US" "$EFF_BW_1" "${BW_UTIL_1}%"
printf "%-8s %12s %12s %12s %12s\n" "4" "$DATA_4_BYTES" "$TOPK_4_TIME_US" "$EFF_BW_4" "${BW_UTIL_4}%"
printf "%-8s %12s %12s %12s %12s\n" "256" "$DATA_256_BYTES" "$TOPK_256_TIME_US" "$EFF_BW_256" "${BW_UTIL_256}%"
printf "%-8s %12s %12s %12s %12s\n" "1024" "$DATA_1024_BYTES" "$TOPK_1024_TIME_US" "$EFF_BW_1024" "${BW_UTIL_1024}%"
echo ""

# Time ratio
TIME_RATIO_4=$(echo "scale=2; ${TOPK_4_TIME_US} / ${TOPK_1_TIME_US}" | bc)
TIME_RATIO_256=$(echo "scale=2; ${TOPK_256_TIME_US} / ${TOPK_1_TIME_US}" | bc)
TIME_RATIO_1024=$(echo "scale=2; ${TOPK_1024_TIME_US} / ${TOPK_1_TIME_US}" | bc)

echo "Time Ratio (relative to tokens=1):"
echo "  Data ratio:   1 : 4 : 256 : 1024"
echo "  Time ratio:   1.00 : ${TIME_RATIO_4} : ${TIME_RATIO_256} : ${TIME_RATIO_1024}"
echo ""

echo "=============================================="
echo "  Test Complete"
echo "=============================================="
