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
# Usage: ./hardware_perf_factor.sh [warmup_iters] [test_iters]

set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

# Default parameters
WARMUP=${1:-20}
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

echo "=============================================="
echo "  Hardware Performance Factor Test"
echo "  Platform: $PLATFORM"
echo "  Warmup: $WARMUP iterations"
echo "  Test: $ITERS iterations"
echo "=============================================="
echo ""

# Check if binary exists, build if not
if [ ! -f "build/bench_hardware_perf_factor" ]; then
    echo "Binary not found. Building..."
    ./build.sh $PLATFORM
    echo ""
fi

# Run benchmark
echo "Running benchmark..."
echo ""
./build/bench_hardware_perf_factor $WARMUP $ITERS
