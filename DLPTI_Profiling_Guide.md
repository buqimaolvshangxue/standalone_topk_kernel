# DLPTI Kernel Profiling Guide

## 前置环境

```bash
source /LocalRun/xiaolong.zhu/artifactory/sdk/testenv.sh && \
source /LocalRun/xiaolong.zhu/artifactory/venv/bin/activate && \
export VLLM_USE_LOCAL_MODEL_PATH=/mars/aebox/LLM/model && \
export XFORMERS_FORCE_DISABLE_TRITON=1
```

## Step 1: 采集数据

```bash
dlpti_tools kprof --targets cu,tu --data-file ./topk.db -- ./build/bench_perf 128 1 bf16 8 2 1
```

**参数说明：**
| 参数 | 说明 |
|------|------|
| `--targets cu,tu` | 采集目标：cu (compute unit), tu (tensor unit) |
| `--data-file ./topk.db` | 输出的原始数据文件 |
| `--` | 分隔符，后面是要运行的程序及参数 |
| `./build/bench_perf 128 1 bf16 8 2 1` | 被测程序及其参数 |

**其他常用参数：**
- `-k, --kernel-name <REGEX>`: 过滤特定 kernel
- `--devices <ID>`: 指定 GPU 设备
- `--metrics <QUERY>`: 指定要收集的指标
- `--list-metrics`: 列出所有可用指标

## Step 2: 导出为 UI 可读格式

```bash
dlpti_tools export --format metric ./topk.db
```

这会生成 `./topk.metric.db` 文件。

## Step 3: 用 dlkprof-ui 查看

```bash
dlkprof-ui ./topk.metric.db
```

## 对比 NV 命令

| NV | DL |
|----|-----|
| `ncu -s 10 -c 1 ./app` | `dlpti_tools kprof --targets cu,tu --data-file out.db -- ./app` |
| 直接查看输出 | `dlpti_tools export --format metric out.db` 后用 dlkprof-ui 查看 |
