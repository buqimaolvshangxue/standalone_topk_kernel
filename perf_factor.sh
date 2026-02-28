#!/bin/bash
# Hardware Performance Factor Test Script
#
# Run hardware performance factor tests on current platform.
# Outputs performance metrics for comparison across platforms.
#
# Prerequisites for DL platform:
#   source /LocalRun/xiaolong.zhu/artifactory/sdk/testenv.sh
#   source /LocalRun/xiaolong.zhu/artifactory/venv/bin/activate
#
# Usage: ./hardware_perf_factor.sh [warmup] [iters]
#   warmup: Number of warmup iterations (default: 100)
#   iters:  Number of test iterations (default: 100)

set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

# Default parameters
WARMUP=${1:-100}
ITERS=${2:-100}

# Detect platform
if command -v dlcc &> /dev/null; then
    PLATFORM=dl
elif command -v nvcc &> /dev/null; then
    PLATFORM=nv
else
    echo "Error: Neither dlcc nor nvcc found in PATH"
    echo "For DL platform, please source the SDK environment first:"
    echo "  source /LocalRun/xiaolong.zhu/artifactory/sdk/testenv.sh"
    echo "  source /LocalRun/xiaolong.zhu/artifactory/venv/bin/activate"
    exit 1
fi

# All test factors
FACTORS=(
    # Launch Overhead
    "empty"
    "minimal_rw"
    # Memory Bandwidth
    "coalesced_read"
    "coalesced_write"
    "coalesced_copy"
    "strided_read"
    "random_read"
    # Warp Operations
    "warp_shfl_xor"
    "warp_shfl_up"
    "warp_broadcast"
    "warp_reduce_sum"
    "warp_reduce_max"
    # Synchronization
    "syncthreads"
    "syncwarp"
    # Compute
    "expf"
    "compare_select"
    # Block Size
    "block_32"
    "block_64"
    "block_128"
    "block_256"
    "block_512"
)

echo "=============================================="
echo "  Hardware Performance Factor Test"
echo "  Platform: $PLATFORM"
echo "  Warmup: $WARMUP iterations"
echo "  Test: $ITERS iterations"
echo "=============================================="
echo ""

# Check if binary exists, build if not
if [ ! -f "build/bench_factor" ]; then
    echo "Binary not found. Building..."
    ./build.sh $PLATFORM
    echo ""
fi

# Print table header
printf "%-20s %12s %12s %12s\n" "Factor" "Time(us)" "BW(GB/s)" "Per-op(us)"
printf "%-20s %12s %12s %12s\n" "--------------------" "------------" "------------" "------------"

# Run each factor test
for factor in "${FACTORS[@]}"; do
    OUTPUT=$(./build/bench_factor $factor $WARMUP $ITERS 2>&1)
    TIME_US=$(echo "$OUTPUT" | grep "avg_time_us=" | cut -d'=' -f2)
    BW_GBPS=$(echo "$OUTPUT" | grep "bandwidth_gbps=" | cut -d'=' -f2)
    PER_OP_US=$(echo "$OUTPUT" | grep "per_op_us=" | cut -d'=' -f2)

    # Format output based on what metrics are available
    if [ -n "$BW_GBPS" ] && [ "$BW_GBPS" != "0.00" ]; then
        printf "%-20s %12s %12s %12s\n" "$factor" "$TIME_US" "$BW_GBPS" "-"
    elif [ -n "$PER_OP_US" ] && [ "$PER_OP_US" != "0.0000" ]; then
        printf "%-20s %12s %12s %12s\n" "$factor" "$TIME_US" "-" "$PER_OP_US"
    else
        printf "%-20s %12s %12s %12s\n" "$factor" "$TIME_US" "-" "-"
    fi
done

echo ""
echo "=============================================="
echo "  Test Complete"
echo "=============================================="
