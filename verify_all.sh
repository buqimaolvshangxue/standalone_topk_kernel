#!/bin/bash
# Verify all golden data configurations
# Usage: ./verify_all.sh

set -e

BENCHMARK="./build/verify_topk"
GOLDEN_DIR="golden"

# Check if benchmark exists
if [ ! -f "$BENCHMARK" ]; then
    echo "[ERROR] Benchmark not found: $BENCHMARK"
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
echo " Verifying Standalone TopK Kernel"
echo "========================================"

TOTAL=$(ls -1 ${GOLDEN_DIR}/*.bin | wc -l)
echo "Found $TOTAL golden data files"
echo ""

PASSED=0
FAILED=0

for f in ${GOLDEN_DIR}/*.bin; do
    filename=$(basename "$f" .bin)
    tokens=$(echo "$filename" | sed 's/tokens_\([0-9]*\)_.*/\1/')
    experts=$(echo "$filename" | sed 's/.*experts_\([0-9]*\)_.*/\1/')
    dtype=$(echo "$filename" | sed 's/.*experts_[0-9]*_\([a-z0-9]*\)_topk.*/\1/')
    topk=$(echo "$filename" | sed 's/.*topk\([0-9]*\)/\1/')

    printf "[%3d/%3d] experts=%-3d tokens=%-3d %s topk=%d ... " \
           $((PASSED + FAILED + 1)) "$TOTAL" "$experts" "$tokens" "$dtype" "$topk"
    
    if $BENCHMARK "$experts" "$tokens" "$dtype" "$topk" --verify > /tmp/verify.log 2>&1; then
        echo "[PASS]"
        PASSED=$((PASSED + 1))
    else
        echo "[FAIL]"
        FAILED=$((FAILED + 1))
        tail -3 /tmp/verify.log | sed 's/^/    /'
    fi
done

echo ""
echo "========================================"
echo " Total: $TOTAL | Passed: $PASSED | Failed: $FAILED"
echo "========================================"

[ "$FAILED" -eq 0 ] && echo "[SUCCESS] All tests PASSED!" && exit 0
echo "[FAILURE] $FAILED tests failed" && exit 1
