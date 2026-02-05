#!/bin/bash
# Performance benchmark for all configurations
# Usage: ./perf_all.sh [iters] [max_tests]

set -e

BENCH="./build/bench_perf"
GOLDEN_DIR="golden"
ITERS=${1:-100}      # Default 100 iterations per test
MAX_TESTS=${2:-0}    # Default 0 = run all tests

# Check if benchmark exists
if [ ! -f "$BENCH" ]; then
    echo "[ERROR] Benchmark not found: $BENCH"
    echo "        Run ./build.sh first"
    exit 1
fi

# Check golden data
if [ ! -d "$GOLDEN_DIR" ] || [ -z "$(ls -A $GOLDEN_DIR/*.bin 2>/dev/null)" ]; then
    echo "[ERROR] No golden data in $GOLDEN_DIR"
    echo "        Run: python tools/generate_golden.py"
    exit 1
fi

echo "========================================"
echo " TopK Kernel Performance Benchmark"
echo "========================================"
echo "Iterations per test: $ITERS"
echo ""

TOTAL=$(ls -1 ${GOLDEN_DIR}/*.bin | wc -l)
if [ "$MAX_TESTS" -gt 0 ] && [ "$MAX_TESTS" -lt "$TOTAL" ]; then
    TOTAL=$MAX_TESTS
    echo "Found $(ls -1 ${GOLDEN_DIR}/*.bin | wc -l) golden data files, testing first $TOTAL"
else
    echo "Found $TOTAL golden data files"
fi
echo ""

# Results table
printf "%-8s %-8s %-6s %-5s %-12s %-15s\n" \
    "Experts" "Tokens" "Dtype" "TopK" "Time(us)" "Throughput"
printf "%s\n" "----------------------------------------------------------------------------------------"

COUNT=0
for f in ${GOLDEN_DIR}/*.bin; do
    COUNT=$((COUNT + 1))
    if [ "$MAX_TESTS" -gt 0 ] && [ "$COUNT" -gt "$MAX_TESTS" ]; then
        break
    fi
    filename=$(basename "$f" .bin)
    tokens=$(echo "$filename" | sed 's/tokens_\([0-9]*\)_.*/\1/')
    experts=$(echo "$filename" | sed 's/.*experts_\([0-9]*\)_.*/\1/')
    dtype=$(echo "$filename" | sed 's/.*experts_[0-9]*_\([a-z0-9]*\)_topk.*/\1/')
    topk=$(echo "$filename" | sed 's/.*topk\([0-9]*\)/\1/')

    printf "%-8d %-8d %-6s %-5s " "$experts" "$tokens" "$dtype" "$topk"

    # Run benchmark and capture output
    output=$($BENCH "$experts" "$tokens" "$dtype" "$topk" "$ITERS" 2>&1)

    # Parse results
    time_us=$(echo "$output" | grep "Average kernel time:" | awk '{print $4}')
    throughput=$(echo "$output" | grep "Throughput:" | awk '{print $2}')

    if [ -n "$time_us" ] && [ -n "$throughput" ]; then
        printf "%-12s %-15s\n" "$time_us" "$throughput"
    else
        printf "[ERROR]\n"
        echo "$output" | tail -5 | sed 's/^/    /'
    fi
done

echo ""
echo "========================================"
echo " Benchmark Complete!"
echo "========================================"
