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
├── tests/
│   └── benchmark_topk.cpp  # 测试和验证程序
├── tools/
│   └── generate_golden.py  # 用vLLM生成golden数据
├── golden/                  # golden数据存储目录
│   └── Phase1/             # Phase1快速验证数据
└── build.sh                # 编译脚本
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
./build/benchmark_topk 64 4 fp16 4 --verify
./build/benchmark_topk 64 16 fp16 4 --verify

# 成功标准：输出 [PASS] Kernel extraction correct!
```

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
