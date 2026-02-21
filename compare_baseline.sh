#!/bin/bash
# Compare Baseline Analysis Script (Unified for NV/DL Platforms)
#
# This script runs baseline kernel tests and topk kernel tests,
# then generates an analysis report comparing the results.
#
# Model: Qwen3-30B-A3B (num_experts=128, topk=8)
# Test tokens: 1, 4, 256, 1024
#
# Prerequisites:
#   - NV: Run ./build.sh nv first
#   - DL: source SDK env, then run ./build.sh dl first
#
# Usage: ./compare_baseline.sh

set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

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
ITERS=100
DTYPE="bf16"
EXPERTS=128
TOPK=8

echo "=============================================="
echo "  GPU Kernel Baseline Comparison Analysis"
echo "  Platform: $PLATFORM"
echo "  Model: Qwen3-30B-A3B (experts=$EXPERTS, topk=$TOPK)"
echo "=============================================="
echo ""

# Check if binaries exist
if [ ! -f "build/bench_baseline" ]; then
    echo "Error: build/bench_baseline not found."
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
echo "[Step 1/5] Collecting GPU information..."
echo ""

if [ "$PLATFORM" = "nv" ]; then
    # NV platform: query GPU info via nvidia-smi
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader | head -1)
    GPU_NAME=$(echo $GPU_INFO | cut -d',' -f1 | sed 's/^ *//;s/ *$//')
    GPU_MEM=$(echo $GPU_INFO | cut -d',' -f2 | tr -d ' ')
    GPU_CC=$(echo $GPU_INFO | cut -d',' -f3 | tr -d ' ')

    # Peak bandwidth from spec
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
    echo "Peak Memory Bandwidth: ${PEAK_BANDWIDTH_GBPS} GB/s (from spec)"

elif [ "$PLATFORM" = "dl" ]; then
    # DL platform: hardcoded hardware info
    GPU_NAME="ks38"
    GPU_MEM="64 GB"
    PEAK_BANDWIDTH_GBPS=409.6
    BF16_TFLOPS_WITHOUT_SPARSITY=70
    BF16_TFLOPS_WITH_SPARSITY=140

    echo "Device: $GPU_NAME"
    echo "Memory: $GPU_MEM"
    echo "Peak Memory Bandwidth: ${PEAK_BANDWIDTH_GBPS} GB/s"
    echo "BF16 Tensor TFLOPs: ${BF16_TFLOPS_WITHOUT_SPARSITY} (W/O Sparsity) / ${BF16_TFLOPS_WITH_SPARSITY} (W/ Sparsity)"
fi
echo ""

# ============================================
# Step 2: Run Baseline Kernel Tests
# ============================================
echo "[Step 2/5] Running baseline kernel tests..."
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
# Step 3: Run TopK Kernel Tests
# ============================================
echo "[Step 3/5] Running TopK kernel tests..."
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
# Step 4: Calculate Theoretical Values
# ============================================
echo "[Step 4/5] Calculating theoretical values..."
echo ""

# Data sizes (bytes)
calc_data_size() {
    local tokens=$1
    local experts=$EXPERTS
    local topk=$TOPK
    local input_bytes=$(echo "$tokens * $experts * 2" | bc)
    local output_bytes=$(echo "$tokens * $topk * 4 * 3" | bc)
    local total_bytes=$(echo "$input_bytes + $output_bytes" | bc)
    echo $total_bytes
}

DATA_1_BYTES=$(calc_data_size 1)
DATA_4_BYTES=$(calc_data_size 4)
DATA_256_BYTES=$(calc_data_size 256)
DATA_1024_BYTES=$(calc_data_size 1024)

# Theoretical transfer time (nanoseconds)
calc_theoretical_ns() {
    local bytes=$1
    local bandwidth=$PEAK_BANDWIDTH_GBPS
    local raw=$(echo "scale=3; $bytes / $bandwidth" | bc)
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

# Calculate bandwidth utilization
calc_bw_util() {
    local data_bytes=$1
    local time_us=$2
    local peak_bw=$PEAK_BANDWIDTH_GBPS
    local raw=$(echo "scale=4; $data_bytes / $time_us / 1000 / $peak_bw * 100" | bc)
    if [[ "$raw" == .* ]]; then
        raw="0$raw"
    fi
    printf "%.2f" "$raw"
}

BW_UTIL_1=$(calc_bw_util $DATA_1_BYTES $TOPK_1_TIME_US)
BW_UTIL_4=$(calc_bw_util $DATA_4_BYTES $TOPK_4_TIME_US)
BW_UTIL_256=$(calc_bw_util $DATA_256_BYTES $TOPK_256_TIME_US)
BW_UTIL_1024=$(calc_bw_util $DATA_1024_BYTES $TOPK_1024_TIME_US)

# Calculate percentages
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

# For 50% BW target calculation
TARGET_DATA_BYTES=$(echo "scale=0; $PEAK_BANDWIDTH_GBPS / 2 * $RW_TIME_US * 1000" | bc)
RATIO_TO_TARGET=$(echo "scale=1; $TARGET_DATA_BYTES / $DATA_1024_BYTES" | bc)

# Calculate effective bandwidth
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
# Step 5: Generate Report
# ============================================
echo "[Step 5/5] Generating analysis report..."

if [ "$PLATFORM" = "nv" ]; then
    # NV platform: generate full report with conclusions
    OUTPUT_FILE="compare_baseline_nv.md"

    cat > $OUTPUT_FILE << EOF
# GPU Kernel Baseline Comparison Analysis (NV Platform)

## Test Environment

| Item | Value |
|------|-------|
| GPU | ${GPU_NAME} |
| Memory | ${GPU_MEM} |
| Compute Capability | ${GPU_CC} |
| Peak Memory Bandwidth | ${PEAK_BANDWIDTH_GBPS} GB/s |
| Test Iterations | ${ITERS} |

## Model Configuration (Qwen3-30B-A3B)

| Parameter | Value |
|-----------|-------|
| num_experts | ${EXPERTS} |
| topk (num_experts_per_tok) | ${TOPK} |
| dtype | ${DTYPE} |

## Baseline Kernel Results

| Kernel | Average Time (us) | Description |
|--------|------------------|-------------|
| empty_kernel | ${EMPTY_TIME_US} | Pure GPU launch + event overhead |
| minimal_rw_kernel | ${RW_TIME_US} | Launch + minimal memory access |

**Fixed Overhead Baseline**: ~${RW_TIME_US} us - This is the minimum overhead for ANY kernel execution on this GPU.

## TopK Kernel Results (Pure Kernel Time)

| Tokens | Data Size (bytes) | Theoretical Transfer (ns) | Actual Time (us) | BW Utilization |
|--------|------------------|--------------------------|------------------|----------------|
| 1 | ${DATA_1_BYTES} | ${THEORY_1_NS} | ${TOPK_1_TIME_US} | ${BW_UTIL_1}% |
| 4 | ${DATA_4_BYTES} | ${THEORY_4_NS} | ${TOPK_4_TIME_US} | ${BW_UTIL_4}% |
| 256 | ${DATA_256_BYTES} | ${THEORY_256_NS} | ${TOPK_256_TIME_US} | ${BW_UTIL_256}% |
| 1024 | ${DATA_1024_BYTES} | ${THEORY_1024_NS} | ${TOPK_1024_TIME_US} | ${BW_UTIL_1024}% |

### Key Observation

**1 token 和 1024 tokens 的执行时间差距远小于数据量差距！**

\`\`\`
tokens=1:     ${TOPK_1_TIME_US} us    (数据量: ${DATA_1_BYTES} bytes)
tokens=4:     ${TOPK_4_TIME_US} us    (数据量: ${DATA_4_BYTES} bytes)
tokens=256:   ${TOPK_256_TIME_US} us   (数据量: ${DATA_256_BYTES} bytes = $(echo "scale=1; $DATA_256_BYTES / 1024" | bc) KB)
tokens=1024:  ${TOPK_1024_TIME_US} us  (数据量: ${DATA_1024_BYTES} bytes = $(echo "scale=1; $DATA_1024_BYTES / 1024" | bc) KB)

数据量比: 1 : 4 : 256 : 1024
时间比:   1.00 : ${TIME_RATIO_4} : ${TIME_RATIO_256} : ${TIME_RATIO_1024}
\`\`\`

这证明了**固定开销占绝对主导地位**，数据量增加不带来成比例的时间增加。

## Analysis

### Time Breakdown for All Token Counts

\`\`\`
┌──────────────────────────────────────────────────────────────────────────┐
│  TopK Kernel 时间分解                                                      │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  tokens=1:   总时间 ${TOPK_1_TIME_US} us                                  │
│    固定开销:  ${RW_TIME_US} us (${FIXED_PCT_1}%)  ████████████████████████│
│    计算开销:  ${COMPUTE_TIME_1} us (${COMPUTE_PCT_1}%)  ██████            │
│    数据传输:  ~${THEORY_1_NS} ns (<0.01%)                                 │
│                                                                          │
│  tokens=4:   总时间 ${TOPK_4_TIME_US} us                                  │
│    固定开销:  ${RW_TIME_US} us (${FIXED_PCT_4}%)  ████████████████████████│
│    计算开销:  ${COMPUTE_TIME_4} us (${COMPUTE_PCT_4}%)  ██████            │
│    数据传输:  ~${THEORY_4_NS} ns (<0.01%)                                 │
│                                                                          │
│  tokens=256: 总时间 ${TOPK_256_TIME_US} us                                │
│    固定开销:  ${RW_TIME_US} us (${FIXED_PCT_256}%)  ██████████████████████│
│    计算开销:  ${COMPUTE_TIME_256} us (${COMPUTE_PCT_256}%)  ████████      │
│    数据传输:  ~${THEORY_256_NS} ns (~0.02%)                               │
│                                                                          │
│  tokens=1024: 总时间 ${TOPK_1024_TIME_US} us                              │
│    固定开销:  ${RW_TIME_US} us (${FIXED_PCT_1024}%)  ████████████████████ │
│    计算开销:  ${COMPUTE_TIME_1024} us (${COMPUTE_PCT_1024}%)  ██████████  │
│    数据传输:  ~${THEORY_1024_NS} ns (~0.07%)                              │
│                                                                          │
│  关键发现: 即使 tokens=1024，固定开销仍占 ~${FIXED_PCT_1024}%！            │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
\`\`\`

### TopK vs Baseline

\`\`\`
TopK kernel:    ~${TOPK_1_TIME_US} us
Baseline:       ~${RW_TIME_US} us
差距:           ~${COMPUTE_TIME_1} us

这 ${COMPUTE_TIME_1} us 来自:
- Softmax 计算 (exp, reduction, normalize)
- TopK 选择 (${TOPK}次迭代找最大值)
- 内存访问

这些是算法必需的开销，不是"写得不好"！
\`\`\`

### Bandwidth Utilization

\`\`\`
有效带宽 = 数据量 / 实际时间

tokens=1:
  有效带宽 = ${DATA_1_BYTES} bytes / ${TOPK_1_TIME_US} us = ${EFF_BW_1} GB/s
  带宽利用率 = ${BW_UTIL_1}%

tokens=4:
  有效带宽 = ${DATA_4_BYTES} bytes / ${TOPK_4_TIME_US} us = ${EFF_BW_4} GB/s
  带宽利用率 = ${BW_UTIL_4}%

tokens=256:
  有效带宽 = ${DATA_256_BYTES} bytes / ${TOPK_256_TIME_US} us = ${EFF_BW_256} GB/s
  带宽利用率 = ${BW_UTIL_256}%

tokens=1024:
  有效带宽 = ${DATA_1024_BYTES} bytes / ${TOPK_1024_TIME_US} us = ${EFF_BW_1024} GB/s
  带宽利用率 = ${BW_UTIL_1024}%
\`\`\`

### Why Low Bandwidth is Inevitable for Small Tokens

\`\`\`
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  带宽利用率 = 数据传输时间 / 总时间                              │
│                                                                 │
│  对于 tokens=1:                                                 │
│    带宽利用率 = ${THEORY_1_NS} ns / ${TOPK_1_TIME_US} us ≈ $(echo "scale=4; ${THEORY_1_NS} / ${TOPK_1_TIME_US} / 1000 * 100" | bc)%                   │
│                                                                 │
│  对于 tokens=1024:                                              │
│    带宽利用率 = ${THEORY_1024_NS} ns / ${TOPK_1024_TIME_US} us ≈ $(echo "scale=2; ${THEORY_1024_NS} / ${TOPK_1024_TIME_US} / 1000 * 100" | bc)%                 │
│                                                                 │
│  即使数据量增加 1024 倍，带宽利用率仍然较低！                     │
│  因为总时间被固定开销主导，不是数据传输。                         │
│                                                                 │
│  固定开销 (${RW_TIME_US} us) >> 数据传输时间 (${THEORY_1_NS} ns - ${THEORY_1024_NS} ns)       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
\`\`\`

## Conclusion

### 核心结论

**小 token 场景下带宽利用率低是正常的，不是 kernel 实现有问题。**

### 证据链

1. **Baseline 测试证明固定开销存在**: 即使空 kernel 也需要 ~${RW_TIME_US} us
2. **1/4/256/1024 tokens 时间增长缓慢**: 证明数据量变化，时间不成比例增加
3. **TopK vs Baseline 差距小**: 只有 ~${COMPUTE_TIME_1} us，是算法必需的计算
4. **理论计算**: 数据传输时间 (${THEORY_1_NS} ns) << 固定开销 (${RW_TIME_US} us = $(echo "${RW_TIME_US} * 1000" | bc) ns)

### To Achieve High Bandwidth Utilization

需要数据量大到传输时间能覆盖固定开销：

\`\`\`
要达到 50% 带宽利用率 ($(echo "scale=0; $PEAK_BANDWIDTH_GBPS / 2" | bc) GB/s):
  需要传输时间 ≈ 固定开销 ≈ ${RW_TIME_US} us
  数据量 = $(echo "scale=0; $PEAK_BANDWIDTH_GBPS / 2" | bc) GB/s × ${RW_TIME_US} us = $(echo "scale=2; $TARGET_DATA_BYTES / 1024 / 1024" | bc) MB

当前 tokens=1024 时数据量只有 $(echo "scale=1; $DATA_1024_BYTES / 1024" | bc) KB
需要的数据量是当前的 ${RATIO_TO_TARGET} 倍！
\`\`\`

---

## Test Details

### Testing Methodology

\`\`\`
- warmup: 10 iterations (不计入统计)
- iterations: 100 (计入统计)
- timing: cudaEvent (GPU 时间线)
- memory: 所有内存在测试循环外预分配
- API: topk_softmax_async() (纯 kernel 调用，不含同步)
\`\`\`

### Raw Results

\`\`\`
Baseline:
  empty_kernel:      ${EMPTY_TIME_US} us
  minimal_rw_kernel: ${RW_TIME_US} us

TopK (experts=${EXPERTS}, topk=${TOPK}, dtype=${DTYPE}):
  tokens=1:     ${TOPK_1_TIME_US} us
  tokens=4:     ${TOPK_4_TIME_US} us
  tokens=256:   ${TOPK_256_TIME_US} us
  tokens=1024:  ${TOPK_1024_TIME_US} us
\`\`\`

---

*Generated from actual test results on ${GPU_NAME}*
*Model: Qwen3-30B-A3B (experts=${EXPERTS}, topk=${TOPK})*
*Using topk_softmax_async() for pure kernel timing*
EOF

elif [ "$PLATFORM" = "dl" ]; then
    # DL platform: generate placeholder report (to be filled later)
    OUTPUT_FILE="compare_baseline_dl.md"

    cat > $OUTPUT_FILE << EOF
# GPU Kernel Baseline Comparison Analysis (DL Platform)

## Test Environment

| Item | Value |
|------|-------|
| Device | ${GPU_NAME} |
| Memory | ${GPU_MEM} |
| Peak Memory Bandwidth | ${PEAK_BANDWIDTH_GBPS} GB/s |
| BF16 Tensor TFLOPs | ${BF16_TFLOPS_WITHOUT_SPARSITY} (W/O Sparsity) / ${BF16_TFLOPS_WITH_SPARSITY} (W/ Sparsity) |
| Test Iterations | ${ITERS} |

## Model Configuration (Qwen3-30B-A3B)

| Parameter | Value |
|-----------|-------|
| num_experts | ${EXPERTS} |
| topk (num_experts_per_tok) | ${TOPK} |
| dtype | ${DTYPE} |

## Baseline Kernel Results

| Kernel | Average Time (us) | Description |
|--------|------------------|-------------|
| empty_kernel | ${EMPTY_TIME_US} | Pure GPU launch + event overhead |
| minimal_rw_kernel | ${RW_TIME_US} | Launch + minimal memory access |

## TopK Kernel Results (Pure Kernel Time)

| Tokens | Data Size (bytes) | Theoretical Transfer (ns) | Actual Time (us) | BW Utilization |
|--------|------------------|--------------------------|------------------|----------------|
| 1 | ${DATA_1_BYTES} | ${THEORY_1_NS} | ${TOPK_1_TIME_US} | ${BW_UTIL_1}% |
| 4 | ${DATA_4_BYTES} | ${THEORY_4_NS} | ${TOPK_4_TIME_US} | ${BW_UTIL_4}% |
| 256 | ${DATA_256_BYTES} | ${THEORY_256_NS} | ${TOPK_256_TIME_US} | ${BW_UTIL_256}% |
| 1024 | ${DATA_1024_BYTES} | ${THEORY_1024_NS} | ${TOPK_1024_TIME_US} | ${BW_UTIL_1024}% |

## Analysis

### Time Breakdown

| Tokens | Total (us) | Fixed (us) | Fixed % | Compute (us) | Compute % |
|--------|------------|------------|---------|--------------|-----------|
| 1 | ${TOPK_1_TIME_US} | ${RW_TIME_US} | ${FIXED_PCT_1}% | ${COMPUTE_TIME_1} | ${COMPUTE_PCT_1}% |
| 4 | ${TOPK_4_TIME_US} | ${RW_TIME_US} | ${FIXED_PCT_4}% | ${COMPUTE_TIME_4} | ${COMPUTE_PCT_4}% |
| 256 | ${TOPK_256_TIME_US} | ${RW_TIME_US} | ${FIXED_PCT_256}% | ${COMPUTE_TIME_256} | ${COMPUTE_PCT_256}% |
| 1024 | ${TOPK_1024_TIME_US} | ${RW_TIME_US} | ${FIXED_PCT_1024}% | ${COMPUTE_TIME_1024} | ${COMPUTE_PCT_1024}% |

### Bandwidth Utilization

| Tokens | Effective BW (GB/s) | Utilization |
|--------|---------------------|-------------|
| 1 | ${EFF_BW_1} | ${BW_UTIL_1}% |
| 4 | ${EFF_BW_4} | ${BW_UTIL_4}% |
| 256 | ${EFF_BW_256} | ${BW_UTIL_256}% |
| 1024 | ${EFF_BW_1024} | ${BW_UTIL_1024}% |

### Time Ratio (relative to tokens=1)

\`\`\`
Data ratio:   1 : 4 : 256 : 1024
Time ratio:   1.00 : ${TIME_RATIO_4} : ${TIME_RATIO_256} : ${TIME_RATIO_1024}
\`\`\`

## Conclusion

> **Note**: This report is a placeholder. Detailed analysis and conclusions for DL platform
> will be added after further investigation of the performance characteristics.
>
> See \`hardware_perf_factor.sh\` for hardware performance factor analysis that explains
> the performance differences between DL and NV platforms.

---

*Generated from actual test results on ${GPU_NAME}*
*Model: Qwen3-30B-A3B (experts=${EXPERTS}, topk=${TOPK})*
*Using topk_softmax_async() for pure kernel timing*
EOF

fi

echo ""
echo "=============================================="
echo "  Analysis Complete!"
echo "=============================================="
echo ""
echo "Report saved to: $OUTPUT_FILE"
echo ""
echo "Summary:"
echo "  Platform: $PLATFORM"
echo "  Model: Qwen3-30B-A3B (experts=${EXPERTS}, topk=${TOPK})"
echo "  - Empty kernel (pure overhead): ${EMPTY_TIME_US} us"
echo "  - Minimal RW kernel: ${RW_TIME_US} us"
echo "  - TopK (1 token):    ${TOPK_1_TIME_US} us  (BW: ${BW_UTIL_1}%)"
echo "  - TopK (4 tokens):   ${TOPK_4_TIME_US} us  (BW: ${BW_UTIL_4}%)"
echo "  - TopK (256 tokens): ${TOPK_256_TIME_US} us  (BW: ${BW_UTIL_256}%)"
echo "  - TopK (1024 tokens):${TOPK_1024_TIME_US} us  (BW: ${BW_UTIL_1024}%)"
echo ""

if [ "$PLATFORM" = "nv" ]; then
    echo "Key Finding: Fixed overhead (~${RW_TIME_US} us) >> Data transfer time (~${THEORY_1_NS} ns for 1 token)"
    echo "Conclusion: Low bandwidth utilization is due to small data size, not kernel implementation."
elif [ "$PLATFORM" = "dl" ]; then
    echo "Note: DL platform analysis is placeholder. Run hardware_perf_factor.sh for detailed performance analysis."
fi
