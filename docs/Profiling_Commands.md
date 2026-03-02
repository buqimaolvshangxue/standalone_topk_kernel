# GPU Profiling 命令速查表

> NV (NVIDIA) 和 DL (登临) 两边的 Profiling 命令，方便不用每次重查。

---

## 一、NV (NVIDIA)

### 1.1 Application Profiling (nsys)

**采集命令：**
```bash
# 基本用法
sudo /usr/local/cuda/bin/nsys profile -o ./nsys_out --force-overwrite true ./your_app args

# 示例
sudo /usr/local/cuda/bin/nsys profile -o ./nsys_topk --force-overwrite true ./build/bench_perf 128 1 bf16 8 10 10
sudo /usr/local/cuda/bin/nsys profile -o ./nsys_empty --force-overwrite true ./build/bench_factor empty 10 10
sudo /usr/local/cuda/bin/nsys profile -o ./nsys_minimal_rw --force-overwrite true ./build/bench_factor minimal_rw 10 10
```

**查看结果：**
```bash
nsys-ui ./nsys_out.qdrep
```

### 1.2 Kernel Profiling (ncu)

**采集命令：**
```bash
# 基本用法
sudo /usr/local/cuda/bin/ncu --set full -o ./ncu_out -f ./your_app args

# 示例
sudo /usr/local/cuda/bin/ncu --set full -o ./ncu_topk -f ./build/bench_perf 128 1 bf16 8 10 10
sudo /usr/local/cuda/bin/ncu --set full -o ./ncu_empty -f ./build/bench_factor empty 10 10
sudo /usr/local/cuda/bin/ncu --set full -o ./ncu_minimal_rw -f ./build/bench_factor minimal_rw 10 10
```

**查看结果：**
```bash
ncu-ui ./ncu_out.ncu-rep
```

---

## 二、DL (登临)

### 前置环境

```bash
source /LocalRun/xiaolong.zhu/artifactory/sdk/testenv.sh && \
source /LocalRun/xiaolong.zhu/artifactory/venv/bin/activate
```

### 2.1 Application Profiling (capture)

**采集命令：**
```bash
# 基本用法
dlpti_tools capture --data-file ./capture_out.db -- ./your_app args

# 示例
dlpti_tools capture --data-file ./capture_topk.db -- ./build/bench_perf 128 1 bf16 8 10 10
dlpti_tools capture --data-file ./capture_empty.db -- ./build/bench_factor empty 10 10
dlpti_tools capture --data-file ./capture_minimal_rw.db -- ./build/bench_factor minimal_rw 10 10
```

**高级选项：**
```bash
# 添加更多 activity 类型
dlpti_tools capture --activity-mask +cublas,cudnn --data-file ./out.db -- ./your_app

# 只采集 cudaProfilerStart/Stop 之间的内容
dlpti_tools capture --capture-range cudaProfilerApi --data-file ./out.db -- ./your_app

# 导出到 Perfetto JSON（可选）
dlpti_tools export --format perfetto-json ./capture_out.db
```

**查看结果：**
```bash
dlsys-ui ./capture_out.db

# 如果文件太大，只加载部分
dlsys-ui --import-range 10%:50% ./capture_out.db
```

### 2.2 Kernel Profiling (kprof)

**采集命令：**
```bash
# Step 1: 采集硬件计数器
dlpti_tools kprof --targets cu,tu --data-file ./kprof_out.db -- ./your_app args

# Step 2: 导出为 metric 格式（--skip-obfusion 导出未加密信息，可用 Python 解析）
dlpti_tools export --format metric --output ./kprof_out.metric.db --skip-obfusion ./kprof_out.db

# 示例
dlpti_tools kprof --targets cu,tu --data-file ./kprof_topk.db -- ./build/bench_perf 128 1 bf16 8 10 10
dlpti_tools export --format metric --output ./kprof_topk.metric.db --skip-obfusion ./kprof_topk.db

dlpti_tools kprof --targets cu,tu --data-file ./kprof_empty.db -- ./build/bench_factor empty 10 10
dlpti_tools export --format metric --output ./kprof_empty.metric.db --skip-obfusion ./kprof_empty.db

dlpti_tools kprof --targets cu,tu --data-file ./kprof_minimal_rw.db -- ./build/bench_factor minimal_rw 10 10
dlpti_tools export --format metric --output ./kprof_minimal_rw.metric.db --skip-obfusion ./kprof_minimal_rw.db
```

**高级选项：**
```bash
# 只采集特定 kernel
dlpti_tools kprof --targets cu --kernel-name "my_kernel" --data-file ./out.db -- ./your_app

# 查看可用的 metrics
dlpti_tools kprof --targets cu --list-metrics -- ./your_app

# 指定设备
dlpti_tools kprof --targets cu,tu --devices 0 --data-file ./out.db -- ./your_app
```

**查看结果：**
```bash
dlkprof-ui ./kprof_out.metric.db
```

---

## 三、NV vs DL 对照表

| 用途 | NV | DL |
|------|----|----|
| **时间线采集** | `nsys profile -o out ./app` | `capture --data-file out.db -- ./app` |
| **时间线查看** | `nsys-ui out.qdrep` | `dlsys-ui out.db` |
| **Kernel 采集** | `ncu --set full -o out ./app` | `kprof --targets cu,tu --data-file out.db -- ./app` |
| **Kernel 导出** | (不需要) | `export --format metric --output out.metric.db --skip-obfusion out.db` |
| **Kernel 查看** | `ncu-ui out.ncu-rep` | `dlkprof-ui out.metric.db` |

---

## 四、帮助命令

```bash
# DL
dlpti_tools --help
dlpti_tools capture --help
dlpti_tools kprof --help
dlpti_tools export --help
dlsys-ui --help
dlkprof-ui --help
```

---

## 五、更多文档

- 详细使用说明见 [dlPTI_User_Guide.md](dlPTI_User_Guide.md)
