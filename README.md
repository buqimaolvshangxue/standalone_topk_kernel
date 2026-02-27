# Standalone TopK Kernel

从vLLM中抓取的独立TopK Softmax kernel，用于性能分析和验证。

## 目录结构

```
standalone_topk_kernel/
├── src/
│   ├── topk_softmax.cu     # 从vLLM抓取并改造的kernel
│   ├── topk_softmax.h      # 纯CUDA接口声明
│   ├── cuda_compat.h       # CUDA兼容性宏
│   └── cub_helpers.h       # CUB辅助类型
├── baseline_src/
│   ├── baseline_kernels.cu # 基准测试kernel
│   └── baseline_kernels.h  # 基准测试接口
├── perf_factor_src/
│   ├── hardware_perf_factor_kernels.cu  # 硬件性能因子测试kernel
│   └── hardware_perf_factor_kernels.h   # 硬件性能因子接口
├── tests/
│   ├── verify_topk.cpp              # Kernel验证程序
│   ├── bench_perf.cpp               # TopK性能测试
│   ├── bench_baseline.cpp           # 基准性能测试
│   └── bench_hardware_perf_factor.cpp  # 硬件性能因子测试
├── tools/
│   └── generate_golden.py  # 用vLLM生成golden数据
├── golden/                  # golden数据存储目录
├── build.sh                # 编译脚本
├── perf_all.sh             # 全量性能测试脚本
├── compare_baseline.sh     # 基准对比分析脚本
├── hardware_perf_factor.sh # 硬件性能因子测试脚本
└── verify_all.sh           # 全量验证脚本
```

## 快速开始

### 1. 生成Golden数据（需要vLLM环境）

```bash
cd standalone_topk_kernel

# 激活vLLM环境
source /LocalRun/xiaolong.zhu/artifactory/sdk/testenv.sh && \
source /LocalRun/xiaolong.zhu/artifactory/venv/bin/activate && \
export VLLM_USE_LOCAL_MODEL_PATH=/mars/aebox/LLM/model && \
export XFORMERS_FORCE_DISABLE_TRITON=1 && \
export CUDA_VISIBLE_DEVICES=10

# 生成golden数据（选择Phase 1快速验证）
python tools/generate_golden.py
```

### 2. 编译

```bash
# 自动检测平台（DL或NV）
./build.sh

# 或指定平台
./build.sh dl   # DL平台
./build.sh nv   # NVIDIA平台
```

### 3. 验证

```bash
# 验证kernel抓取是否正确
./build/verify_topk 64 4 fp16 4 --verify
./build/verify_topk 64 16 fp16 4 --verify

# 成功标准：输出 [PASS] Kernel extraction correct!
```

## 性能测试

### 统一参数说明

所有性能测试脚本和程序使用统一的参数格式：
- `warmup`: 预热迭代次数（默认: 100）
- `iters`: 测试迭代次数（默认: 100）

### 全量性能测试

```bash
# 默认: warmup=100, iters=100
./perf_all.sh

# 自定义参数
./perf_all.sh 50 200   # warmup=50, iters=200
```

### 基准对比分析

```bash
# 默认: warmup=100, iters=100
./compare_baseline.sh

# 自定义参数
./compare_baseline.sh 50 200   # warmup=50, iters=200
```

输出报告：`compare_baseline_nv.md` 或 `compare_baseline_dl.md`

### 硬件性能因子测试

```bash
# 默认: warmup=100, iters=100
./hardware_perf_factor.sh

# 自定义参数
./hardware_perf_factor.sh 50 200   # warmup=50, iters=200
```

测试项目包括：
- 启动开销（Empty Kernel, Minimal RW）
- 内存带宽（Coalesced/Strided/Random Read/Write）
- Warp操作（Shuffle, Broadcast, Reduce）
- 同步（__syncthreads, __syncwarp）
- 计算（expf, Compare/Select）
- 并行度（不同Block Size）

### 单独运行测试程序

```bash
# TopK性能测试
./build/bench_perf <experts> <tokens> <dtype> <topk> [warmup] [iters]
./build/bench_perf 128 1024 bf16 8 100 100

# 基准性能测试
./build/bench_baseline <kernel_type> [warmup] [iters]
./build/bench_baseline empty 100 100
./build/bench_baseline rw 100 100
```

## 分析报告

运行测试后会生成以下报告：

| 报告文件 | 说明 |
|----------|------|
| `compare_baseline_nv.md` | NV平台基准对比分析 |
| `compare_baseline_dl.md` | DL平台基准对比分析 |
| `hardware_performance_comparison.md` | 硬件性能因子对比分析 |
| `CUDA_Kernel_Execution_Analysis.md` | CUDA Kernel执行流程分析 |
| `NVIDIA_Profiling_Guide_2026.md` | NVIDIA Profiling指南 |

## 验证原理

```
vLLM (正确基准)
  ↓ 生成
Golden数据 (input + 正确output)
  ↓ 加载
Standalone Kernel
  ↓ 对比
结果一致？ → 是：kernel抓对了 → 否：需要排查
```

## 接口说明

```cpp
template<typename T>
void topk_softmax_launch(
    const T* gating_output,     // [num_tokens, num_experts] 输入
    float* topk_weights,        // [num_tokens, topk] 输出权重
    int* topk_indices,          // [num_tokens, topk] 输出索引
    int num_tokens,
    int num_experts,
    int topk,
    bool renormalize = false,   // 默认false
    cudaStream_t stream = 0
);
```

支持的数据类型：`float`, `__half`, `__nv_bfloat16`

## 支持的配置

- 专家数：1, 2, 4, 8, 16, 32, 64, 128, 256, 512
- TopK：2, 4, 6, 8
- 数据类型：fp16, bf16, fp32

## 编译选项

| 平台 | 编译器 | 必需选项 |
|------|--------|----------|
| DL | dlcc | `-mllvm -dlgpu-lower-ptx=true -soft-spill-allocator -DUSE_DLIN` |
| NV | nvcc | `-std=c++17` |

## 问题排查

| 问题 | 解决方法 |
|------|----------|
| 找不到dlcc | 执行 `source /LocalRun/xiaolong.zhu/artifactory/sdk/testenv.sh` |
| Golden数据不存在 | 先运行 `python tools/generate_golden.py` |
| 验证失败 | 检查kernel代码或编译选项是否与vLLM一致 |
