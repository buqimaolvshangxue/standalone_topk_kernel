# NVIDIA GPU 性能分析实战手册 (2025/2026 版)

> **目标读者**：CUDA 初学者至中级开发者
> **版本**：v2.0 (已针对 Ampere/Hopper 架构更新)
> **关键词**：`nsys`, `ncu`, 性能分析, 调优, 瓶颈定位

---

## 📚 1. 核心工具概览

现在的 NVIDIA 性能分析遵循 **“宏观 -> 微观”** 的标准工作流。请务必抛弃老旧的 `nvprof`。

| 特性 | **Nsight Systems (`nsys`)** | **Nsight Compute (`ncu`)** |
| :--- | :--- | :--- |
| **定位** | 🦅 **宏观视角 (Bird's Eye)** | 🔬 **微观视角 (Microscope)** |
| **解决问题** | 程序慢在 CPU 还是 GPU？有气泡吗？哪里卡住了？ | **这个** Kernel 内部发生了什么？为何带宽没跑满？ |
| **关注点** | 时间线 (Timeline)、系统交互、调度、API 延迟 | 指令流水线、显存层级 (L1/L2/DRAM)、Warp 行为 |
| **使用时机** | **第一步**：全局扫盲，定位 Top 1 耗时函数 | **第二步**：针对上述 Top 1 函数进行“手术级”分析 |
| **输出文件** | `.nsys-rep` | `.ncu-rep` |
| **官方文档** | [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) | [Nsight Compute Kernel Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) |

---

## 🦅 2. Nsight Systems (`nsys`)：全景分析

当你想知道程序慢在哪，先用 `nsys` 看全局。

---

### 2.1 nsys 是什么？

**nsys 是一个”宏观”分析工具，用于观察程序的整体运行情况。**

| 问题 | nsys 能回答 |
|------|------------|
| 程序慢在 CPU 还是 GPU？ | 看时间轴上 CPU/GPU 的比例 |
| 有没有”气泡”（GPU 空闲）？ | 看 GPU 行有没有空隙 |
| 内存拷贝花了多少时间？ | 看 HtoD/DtoH 条的长度 |
| 哪个 kernel 最耗时？ | 看 kernel 条的长度排序 |

**什么时候用 nsys？**
1. **第一步**：用 `nsys` 看全局，找到瓶颈在哪
2. **第二步**：用 `ncu` 深入分析那个瓶颈 kernel

---

### 2.2 nsys 的工作原理

#### Trace 机制（不是 Profile！）

**nsys 是”旁观者”，被动记录事件，不干预程序运行。**

```
程序正常运行 → nsys 在旁边记录 → 生成时间轴
```

#### 与 ncu 的核心区别

| | nsys | ncu |
|--|------|-----|
| **机制** | Trace（追踪） | Profile（采样） |
| **是否重放** | ❌ 不重放 | ✅ 多次重放 |
| **开销** | 2-5% | 10-100 倍 |
| **共享环境** | ✅ 完全可用 | ⚠️ 需谨慎 |
| **关注点** | CPU/GPU 交互 | kernel 内部 |

**nsys 没有重放机制！程序跑一次就完成采集。**

---

### 2.3 核心概念：Trace

#### 什么是 Trace？

**Trace 是”追踪哪些事件”的配置。**

```
nsys profile --trace=cuda,osrt,nvtx ./app
                        └─────┬─────┘
                          追踪哪些事件
```

#### 可追踪的事件类型

| Trace 类型 | 含义 | 包含内容 |
|-----------|------|---------|
| `cuda` | CUDA API | cudaMalloc, cudaMemcpy, kernel launch 等 |
| `osrt` | OS Runtime | CPU 线程调度、系统调用 |
| `nvtx` | NVTX 标签 | 用户自定义的标记（推荐） |
| `cudnn` | cuDNN | 卷积等深度学习操作 |
| `cublas` | cuBLAS | 矩阵乘法等线性代数 |
| `nvmpi` | NVML | GPU 功耗、温度等 |

#### 常用组合

```bash
# 最常用：看 CUDA + CPU + 自定义标签
nsys profile --trace=cuda,osrt,nvtx ./app

# 只看 CUDA（文件最小）
nsys profile --trace=cuda ./app

# 深度学习训练
nsys profile --trace=cuda,osrt,nvtx,cudnn,cublas ./app
```

---

### 2.4 核心概念：Timeline

#### 什么是 Timeline？

**Timeline 是时间轴，显示程序运行过程中各组件的活动。**

```
时间轴结构：
├── Processes（进程）
│   └── Threads（线程）
│       ├── OS Runtime（CPU 系统调用）
│       ├── CUDA API（CUDA 函数调用）
│       └── CUDA HW（GPU 硬件活动）
│           ├── Compute（kernel 执行）
│           └── Memory（内存拷贝）
```

#### 时间轴各行含义

| 行 | 含义 | 关注点 |
|----|------|--------|
| **OS Runtime** | CPU 在做什么 | 系统调用、线程调度 |
| **CUDA API** | CUDA 函数调用 | API 耗时、调用频率 |
| **Compute** | GPU 计算 | kernel 执行时间 |
| **Memory** | 内存传输 | HtoD/DtoH 拷贝时间 |

#### NVTX：给时间轴加”书签”

在茫茫时间轴中找自己的函数就像大海捞针。**NVTX** 是官方提供的标记工具。

```cpp
#include <nvtx3/nvToolsExt.h>

void train_step() {
    nvtxRangePushA(“Train Step”);  // <--- 在时间轴上画一个色块
    kernel_A<<<...>>>();
    kernel_B<<<...>>>();
    nvtxRangePop();                // <--- 色块结束
}
```
*编译时需链接库：`-lnvToolsExt`（部分新版 CUDA 只需 include 头文件）*

---

### 2.5 实战：bench_perf 例子

#### 程序执行流程

```bash
./build/bench_perf 128 1 bf16 8 1
# 参数含义: experts=128, tokens=1, dtype=bf16, topk=8, iters=1
```

**程序内部执行流程：**
```
1. 初始化数据
2. Warmup: kernel 调用 10 次（预热，不计入时间）
3. 正式测试: kernel 调用 1 次（计时，由第 5 个参数控制）
4. 输出平均时间
```

**总共调用 kernel：10 + 1 = 11 次**

#### nsys 的行为

```bash
nsys profile --trace=cuda,osrt -o report ./build/bench_perf 128 1 bf16 8 1
```

**nsys 行为：**
1. 程序正常运行（几乎无额外开销）
2. nsys 记录所有 CUDA API 和 OS 事件
3. 生成 `report.nsys-rep` 文件

**重要：nsys 不重放，程序只跑一次！**

#### 用 `--stats` 快速查看

```bash
nsys profile --stats=true ./build/bench_perf 128 1 bf16 8 1
```

**终端输出示例：**
```
Time (%)  Total Time   Instances  Avg         Min         Max         Name
--------  -----------  ---------  ----------  ----------  ----------  ----------
   85.2%    15.234 us         11    1.385 us    1.234 us    1.567 us  topkGatingSoftmax
    8.3%     1.485 us          3  495.000 ns  234.000 ns  890.000 ns  cudaMalloc
    6.5%     1.162 us          2  581.000 ns  456.000 ns  706.000 ns  cudaMemcpy
```

---

### 2.6 参数详解

#### `--trace`：追踪哪些事件

```bash
--trace=cuda              # 只追踪 CUDA
--trace=cuda,osrt         # CUDA + OS
--trace=cuda,osrt,nvtx    # CUDA + OS + NVTX（推荐）
```

#### `--delay` / `--duration`：时间窗口控制

```bash
--delay=10      # 程序启动后 10 秒开始记录
--duration=5    # 只记录 5 秒
```

**适用场景：**
- 程序很长，只想分析某个阶段
- 避免 `.nsys-rep` 文件过大

#### `-o`：输出文件名

```bash
-o report      # 生成 report.nsys-rep
```

#### `--stats`：终端输出统计

```bash
--stats=true   # 在终端打印统计信息
```

#### `--cuda-devices`：指定 GPU

```bash
--cuda-devices=0     # 只追踪 0 号卡
--cuda-devices=0,1   # 追踪 0 和 1 号卡
```

---

### 2.7 输出解读

#### 时间轴常见模式

| 现象 | 诊断 | 原因 | 解决方案 |
|------|------|------|----------|
| **大量空隙 (Gaps)** | GPU 饥饿 | CPU 准备数据太慢 | 多流、CUDA Graphs |
| **HtoD 传输条极长** | PCIe 瓶颈 | 频繁拷贝数据 | Pinned Memory、批处理 |
| **细碎的 Kernel** | 启动开销大 | kernel 太小 (<10us) | Kernel Fusion |
| **OS Runtime 条很长** | 调度延迟 | 系统上下文切换 | 绑核、检查干扰 |

#### 实战案例

**案例 1：显存拷贝地狱**
- **现象**: Compute 条很短，但 `MemCpy (HtoD)` 条极长
- **诊断**: 频繁在 CPU 和 GPU 之间搬运数据
- **后果**: GPU 计算 1ms，搬数据花了 10ms
- **图示**: `[HtoD] [Kernel] [DtoH] ... [HtoD] [Kernel] [DtoH]`

**案例 2：CPU 拖后腿**
- **现象**: GPU 时间轴上有巨大的空隙
- **诊断**: CPU 准备数据太慢（如图片解码）
- **结论**: 优化 GPU 代码无意义，先优化 CPU/数据加载

**案例 3：API 启动延迟**
- **现象**: kernel 运行 5us，但 `cudaLaunchKernel` 花了 10us
- **诊断**: 任务太碎了
- **解法**: Kernel Fusion 或 CUDA Graph

> **结论**: 如果 GPU 只有 20% 的时间在干活，kernel 优化快 10 倍，整体也只能快 18%。

---

### 2.8 命令速查表

| 场景 | 命令 |
|------|------|
| **标准体检** | `nsys profile --trace=cuda,osrt,nvtx -o report ./app` |
| **只看统计** | `nsys profile --stats=true ./app` |
| **定点抓取** | `nsys profile --delay=10 --duration=5 -o report ./app` |
| **指定 GPU** | `nsys profile --cuda-devices=0 -o report ./app` |
| **Python 分析** | `nsys profile --trace=cuda,nvtx --python-backtrace=cuda python train.py` |

---

### 2.9 常见问题

**Q: nsys 和 ncu 先用哪个？**
A: **先用 nsys**。nsys 看全局，找到瓶颈；再用 ncu 深入分析。

**Q: `.nsys-rep` 文件太大？**
A:
1. 用 `--duration` 限制时长
2. 用 `--delay` 跳过初始化
3. 用 `--trace=cuda` 只追踪核心事件

**Q: 如何在 GUI 中查看？**
A:
```bash
# 生成文件
nsys profile -o report ./app

# 本地 GUI 打开
nsys-ui report.nsys-rep
```

**Q: nsys 会影响程序性能吗？**
A: 几乎不会。开销只有 2-5%，可以在生产环境使用。

**Q: 如何对比优化前后？**
A:
```bash
# 生成 baseline
nsys profile -o baseline ./app_before

# 生成优化后
nsys profile -o optimized ./app_after

# 在 GUI 中对比
nsys-ui baseline.nsys-rep
# 然后导入 optimized.nsys-rep
```

---

## 🔬 3. Nsight Compute (`ncu`)：深度解剖

当你确定了某个 kernel 是瓶颈，用 `ncu` 分析它内部的执行细节。

---

### 3.1 ncu 是什么？

**ncu 是一个”微观”分析工具，用于深入分析单个 kernel 的执行情况。**

| 问题 | ncu 能回答 |
|------|-----------|
| kernel 执行时间是多少？ | Duration |
| 内存带宽跑满了吗？ | Memory Throughput |
| 计算单元跑满了吗？ | Compute Throughput |
| 为什么线程在等待？ | Warp State Statistics |

**什么时候用 ncu？**
1. 先用 `nsys` 找到瓶颈 kernel
2. 再用 `ncu` 深入分析这个 kernel

---

### 3.2 ncu 的工作原理

#### 两个独立的概念

**理解 ncu 必须区分两个概念：**

| 概念 | 含义 | 控制方式 |
|------|------|----------|
| **抓取次数** | 抓取多少次不同的 kernel 调用 | `-c` 参数 |
| **重放次数 (passes)** | 对每次调用重放多少次来采集指标 | 由 Section 数量决定 |

#### 重放机制 (passes)

**ncu 不能一次采集所有指标，需要多次运行同一个 kernel。**

```
第 1 次重放 kernel → 采集指标组 A
第 2 次重放 kernel → 采集指标组 B
第 3 次重放 kernel → 采集指标组 C
...
```

你看到的输出：
```
==PROF== Profiling “topkGatingSoftmax”: 0%....50%....100% - 8 passes
```

**`8 passes` = 这 1 次 kernel 调用被重放了 8 次！**

#### 为什么需要重放？

GPU 的硬件性能计数器数量有限，不能同时采集所有指标。ncu 需要”分批”采集，每次重放只采集一部分。

#### 实际执行次数计算

```
抓取次数 × 重放次数 = 实际执行次数

例 1：-c 1（抓 1 次），8 passes
      1 × 8 = 8 次实际执行

例 2：-c 5（抓 5 次），8 passes
      5 × 8 = 40 次实际执行

例 3：不加参数（抓 11 次），8 passes
      11 × 8 = 88 次实际执行
```

#### 如何减少重放次数？

**`-c` 无法控制重放次数！重放次数由 Section 数量决定。**

```bash
# 默认 4 个 Section → 约 8 passes
ncu -s 10 -c 1 ./app

# 只看 1 个 Section → 更少 passes（更快）
ncu -s 10 -c 1 --section SpeedOfLight ./app
```

**Section 越少 → passes 越少 → 越快！**

---

### 3.3 核心概念：Section

#### 什么是 Section？

**Section 是一组相关指标的集合，类似”体检套餐”。**

```
ncu 采集的数据
├── Section: SpeedOfLight（利用率概览）
│   ├── Duration
│   ├── Memory Throughput
│   ├── Compute Throughput
│   └── ...
├── Section: Occupancy（占用率）
│   ├── Theoretical Occupancy
│   ├── Achieved Occupancy
│   └── ...
└── Section: LaunchStats（启动参数）
    ├── Grid Size
    ├── Block Size
    └── ...
```

#### 默认启用的 Section

**不加 `--section` 参数时，ncu 默认只采集 4 个 Section：**

| Section | 含义 | 启用 |
|---------|------|------|
| `SpeedOfLight` | GPU 利用率概览 | ✅ 默认 |
| `LaunchStats` | 启动参数（grid/block） | ✅ 默认 |
| `Occupancy` | 占用率 | ✅ 默认 |
| `WorkloadDistribution` | 负载分布 | ✅ 默认 |

#### 所有可用的 Section

```bash
# 查看所有 Section
ncu --list-sections
```

| Section | 含义 | 默认启用 | 用途 |
|---------|------|----------|------|
| `SpeedOfLight` | GPU 利用率概览 | ✅ | 快速看瓶颈 |
| `LaunchStats` | 启动参数 | ✅ | grid/block 大小 |
| `Occupancy` | 占用率 | ✅ | 线程是否跑满 |
| `WorkloadDistribution` | 负载分布 | ✅ | SM/内存活跃度 |
| `MemoryWorkloadAnalysis` | 内存负载分析 | ❌ | 详细带宽分析 |
| `ComputeWorkloadAnalysis` | 计算负载分析 | ❌ | 详细计算分析 |
| `WarpStateStats` | Warp 状态统计 | ❌ | 为什么线程等待 |
| `InstructionStats` | 指令统计 | ❌ | 指令分布 |
| `SourceCounters` | 源码级计数器 | ❌ | 定位热点代码 |
| `SchedulerStats` | 调度统计 | ❌ | 调度效率 |

#### 不同场景选哪些 Section？

| 想分析什么 | 推荐的 Section |
|-----------|---------------|
| 快速看整体瓶颈 | 默认 4 个（不加参数） |
| 内存带宽问题 | `SpeedOfLight,MemoryWorkloadAnalysis` |
| 计算利用率问题 | `SpeedOfLight,ComputeWorkloadAnalysis` |
| 线程等待问题 | `Occupancy,WarpStateStats` |
| 定位热点代码 | `SourceCounters`（需编译时加 `-lineinfo`） |

---

### 3.4 核心概念：Metrics

#### 什么是 Metrics？

**Metrics 是单个具体指标，Section 是 Metrics 的集合。**

```
Section: SpeedOfLight
├── Metric: DRAM Frequency      ← 单个指标
├── Metric: SM Frequency        ← 单个指标
├── Metric: Elapsed Cycles      ← 单个指标
├── Metric: Duration            ← 单个指标
├── Metric: Memory Throughput   ← 单个指标
└── ...
```

#### Section vs Metrics

| | Section | Metrics |
|--|---------|---------|
| **粒度** | 粗（一组指标） | 细（单个指标） |
| **用法** | `--section SpeedOfLight` | `--metrics gpu__dram_throughput...` |
| **场景** | 日常分析 | 脚本自动化、精确控制 |

**日常用 `--section` 就够了！**

---

### 3.5 实战：bench_perf 例子

#### 程序执行流程

假设你有一个 benchmark 程序：

```bash
./build/bench_perf 128 1 bf16 8 1
# 参数含义: experts=128, tokens=1, dtype=bf16, topk=8, iters=1
```

**程序内部执行流程：**
```
1. 初始化数据
2. Warmup: kernel 调用 10 次（预热，不计入时间）
3. 正式测试: kernel 调用 1 次（计时，由第 5 个参数控制）
4. 输出平均时间
```

**总共调用 kernel：10 + 1 = 11 次**

#### ncu 的行为

如果不加任何参数：
```
ncu ./build/bench_perf 128 1 bf16 8 1

行为：
1. 程序调用 kernel 11 次
2. ncu 对每次调用都进行 profile
3. 输出 11 次结果（刷屏！）
```

#### 用 `-s` 和 `-c` 控制

```bash
ncu -s 10 -c 1 ./build/bench_perf 128 1 bf16 8 1
```

**参数含义：**
- `-s 10`：**跳过**前 10 次 kernel 调用（不分析）
- `-c 1`：**只抓取 1 次**（抓取后就结束，不再分析后面的）

**重要：`-c 1` 是"总共只抓 1 次"，不是"每次都抓"！**

**图示：**
```
程序调用:  K1  K2  ...  K10 | K11 |
           └─── 跳过 ───┘   └抓1次┘
              warmup         ↑
                         只抓这一次
```

**常用组合：**

| 命令 | 效果 | 适用场景 |
|------|------|----------|
| `-s 10 -c 1` | 跳过 10 次 warmup，只抓第 11 次 | 跳过 warmup（推荐） |
| `-s 0 -c 1` | 不跳过，只抓第 1 次 | 抓第一次调用 |
| `-s 10 -c 5` | 跳过 10 次，抓第 11-15 次（共 5 次） | 想看多次结果 |

---

### 3.6 参数详解

#### `-s` / `--launch-skip`：跳过次数

```bash
-s 10    # 跳过前 10 次 kernel 调用
```

#### `-c` / `--launch-count`：抓取次数

```bash
-c 1     # 只抓取 1 次
-c 5     # 抓取 5 次
```

#### `--section`：选择分析模块

```bash
# 只看 SpeedOfLight（最快）
ncu -s 10 -c 1 --section SpeedOfLight ./app

# 看多个 Section
ncu -s 10 -c 1 --section SpeedOfLight,Occupancy ./app

# 不加 --section = 默认 4 个 Section
ncu -s 10 -c 1 ./app
```

#### `--kernel-name`：过滤 kernel

```bash
# 只分析名字包含 “topkGatingSoftmax” 的 kernel
ncu --kernel-name “topkGatingSoftmax” -s 10 -c 1 ./app
```

#### `-o`：生成 UI 文件

```bash
# 不加 -o：只输出到终端
ncu -s 10 -c 1 ./app

# 加 -o：生成 .ncu-rep 文件
ncu -o profile -s 10 -c 1 ./app
# → 生成 profile.ncu-rep

# 用 UI 打开
ncu-ui profile.ncu-rep
```

**注意：推荐 `-c 1`，否则文件很大，UI 打开很慢。**

---

### 3.7 输出解读（完整版）

#### SpeedOfLight Section（10 个指标）

| 指标 | 单位 | 含义 | 理想值 |
|------|------|------|--------|
| `DRAM Frequency` | GHz | 显存频率 | - |
| `SM Frequency` | GHz | SM 频率 | - |
| `Elapsed Cycles` | cycle | 总周期数 | - |
| `Duration` | us | **kernel 执行时间** | 越低越好 |
| `Memory Throughput` | % | **内存综合利用率** | > 70% |
| `DRAM Throughput` | % | 显存带宽利用率 | > 70% |
| `L1/TEX Cache Throughput` | % | L1 缓存利用率 | - |
| `L2 Cache Throughput` | % | L2 缓存利用率 | - |
| `SM Active Cycles` | cycle | SM 活跃周期 | - |
| `Compute (SM) Throughput` | % | **计算利用率** | > 70% |

#### LaunchStats Section（8 个核心指标）

| 指标 | 含义 | 关注点 |
|------|------|--------|
| `Grid Size` | block 数量 | 太小 → 利用率低 |
| `Block Size` | 每 block 线程数 | 建议 128-256 |
| `Threads` | 总线程数 | Grid × Block |
| `Registers Per Thread` | 每线程寄存器数 | 太多 → occupancy 降低 |
| `Static Shared Memory Per Block` | 静态共享内存 | 太多 → occupancy 降低 |
| `Dynamic Shared Memory Per Block` | 动态共享内存 | - |
| `# SMs` | GPU 的 SM 数量 | 硬件参数 |
| `Waves Per SM` | 每 SM 的 wave 数 | 太少 → 利用率低 |

#### Occupancy Section（8 个指标）

| 指标 | 含义 | 理想值 |
|------|------|--------|
| `Theoretical Occupancy` | 理论占用率 | > 75% |
| `Achieved Occupancy` | **实际占用率** | 接近理论值 |
| `Theoretical Active Warps per SM` | 理论活跃 warp 数 | - |
| `Achieved Active Warps Per SM` | 实际活跃 warp 数 | - |
| `Block Limit SM` | SM 最多容纳 block 数 | - |
| `Block Limit Registers` | 寄存器限制的 block 数 | - |
| `Block Limit Shared Mem` | 共享内存限制的 block 数 | - |
| `Block Limit Warps` | warp 限制的 block 数 | - |

#### WorkloadDistribution Section（10 个指标）

| 指标 | 含义 |
|------|------|
| `Average DRAM Active Cycles` | 平均 DRAM 活跃周期 |
| `Total DRAM Elapsed Cycles` | 总 DRAM 周期 |
| `Average L1 Active Cycles` | 平均 L1 活跃周期 |
| `Total L1 Elapsed Cycles` | 总 L1 周期 |
| `Average L2 Active Cycles` | 平均 L2 活跃周期 |
| `Total L2 Elapsed Cycles` | 总 L2 周期 |
| `Average SM Active Cycles` | 平均 SM 活跃周期 |
| `Total SM Active Cycles` | 总 SM 周期 |
| `Average SMSP Active Cycles` | 平均 SMSP 活跃周期 |
| `Total SMSP Active Cycles` | 总 SMSP 周期 |

#### 性能问题诊断

| 现象 | 可能原因 | 建议 |
|------|----------|------|
| Memory Throughput < 30% | 内存带宽没跑满 | 检查合并访问、增加并行度 |
| Compute Throughput < 30% | 计算单元闲置 | 增加计算密度、检查分支发散 |
| Grid Size = 1 | 只启动了 1 个 block | 增加 batch size 或数据并行 |
| Achieved << Theoretical Occupancy | 实际占用远低于理论 | 检查负载均衡、同步开销 |

---

### 3.8 命令速查表

| 场景 | 命令 |
|------|------|
| **快速看 kernel 时间** | `ncu -s 10 -c 1 ./app` |
| **只看利用率** | `ncu -s 10 -c 1 --section SpeedOfLight ./app` |
| **生成 UI 文件** | `ncu -o profile -s 10 -c 1 ./app` |
| **只看特定 kernel** | `ncu --kernel-name “topkGatingSoftmax” -s 10 -c 1 ./app` |
| **分析内存瓶颈** | `ncu -s 10 -c 1 --section SpeedOfLight,MemoryWorkloadAnalysis ./app` |
| **分析线程等待** | `ncu -s 10 -c 1 --section Occupancy,WarpStateStats ./app` |

---

### 3.9 常见问题

**Q: 报错 `ERR_NVGPUCTRPERM`？**
A: 需要 root 权限，命令前加 `sudo`。

**Q: ncu 太慢？**
A:
1. 用 `-c 1` 限制抓取次数
2. 用 `--section SpeedOfLight` 只看关键指标
3. 用 `--kernel-name` 只分析目标 kernel

**Q: 输出太多刷屏？**
A: 用 `-s` 和 `-c` 控制只抓一次。

**Q: GUI 看不到源码？**
A: 编译时加 `-lineinfo` 参数。

**Q: 如何对比优化前后？**
A:
```bash
# 生成 baseline
ncu -o baseline -s 10 -c 1 ./app_before

# 生成优化后
ncu -o optimized -s 10 -c 1 ./app_after

# 在 GUI 中对比
ncu-ui baseline.ncu-rep
# 然后导入 optimized.ncu-rep 进行对比
```

---

## 🖥 4. 界面与实战 (GUI Workflow)

### 4.1 "Source View" (源码视图) —— 杀手级功能

当你用 `.ncu-rep` 在本地打开 GUI 后：
1.  切换到 **Source** 标签页。
2.  在下拉菜单选择 **"Source Analysis"**。
3.  **看这里**：
    *   **Sampling Data (采样数据)**：某一行的指令被执行了多少次。
    *   **Stall Reasons (停滞原因)**：每一行代码导致了什么类型的停滞。
    *   **Heat Map (热力图)**：右侧滚动条会有**红色色块**。
    *   **直接定位**：点那个最红的色块，系统会高亮那行 C++ 代码，告诉你：“就是这一行 `a[i] = b[k]`，造成了 80% 的 Memory Stall”。**这才是性能优化的终极答案。**

### 4.2 编译参数的玄学

要在 GUI 里看到源码对应，必须在编译时加：
*   **`-lineinfo`**：(推荐) 生成行号与汇编的映射。对性能影响极小。
*   **`-G`**：(不推荐) 生成完整的 Debug 信息。会导致编译器禁用所有优化，性能会变得极差，分析结果不再准确（除非你在 Debug 逻辑错误）。

---

## 🔗 5. 官方资源导航

收藏这些链接，解决疑难杂症：

*   **[Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)**
    *   查阅所有 `nsys` 命令行参数解释。
*   **[Nsight Compute GUI User Guide](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)**
    *   教你如何操作那个复杂的仪表盘界面。
*   **[Kernel Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)**
    *   **必读**。官方解释每一个 Metric 的物理含义。如果你不懂 "DRAM__bytes_write" 是啥，去这里查。
*   **[CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)**
    *   NVIDIA 官方教你如何写出高性能代码。

---

## 🛡️ 6. 安全性与副作用 (Safety & Impact)

新手最担心的问题：“我跑这个会不会把服务器搞挂？会不会影响别人训练？”

### 6.1 `nsys` (Nsight Systems) —— ✅ 几乎完全安全
*   **原理**：它像一个旁观者（Observer），默默记录 API 调用和 OS 事件。它是所谓的 "Trace" 工具，而不是 "Profile" 工具（虽然名字里有 profile）。
*   **开销 (Overhead)**：非常低。通常只有 2% - 5%。
*   **共享环境**：**完全可用**。在生产环境或多人共享的机器上跑 `nsys` 是标准操作，不会干扰显卡上的其他任务，也不会因为“锁住”硬件而导致别人死机。
*   **唯一风险**：如果你生成的报告文件 (.rep) 太大（几十 GB），可能会把磁盘塞满。
    *   *避坑指南*：一定要加 `--duration=10` 或者 `--delay` 参数，不要无限期跑下去。

### 6.2 `ncu` (Nsight Compute) —— ⚠️ 需要谨慎
*   **原理**：它需要“暂停”你的 Kernel，甚至对同一个 Kernel 重放几十次 (Replay) 来读取底层硬件计数器。
*   **开销**：**巨大**。程序运行时间可能增加 10倍 到 100倍。
*   **共享环境**：
    *   它**不会**弄坏硬件。
    *   但是，它在分析的时候会**独占** GPU 的计算资源（看起来像显卡卡死了）。如果这块卡上有别人的任务在跑，可能会因为长时间占用而受到轻微影响（驱动层面的上下文切换）。
    *   **权限**：如前所述，它通常需要 Root 权限来访问 Performance Counters。
*   **建议**：最好在没人抢占的卡上用。如果是共享卡，请务必使用 `--launch-count 1` 限制次数，或者只分析极短的时间。

### 6.3 "Locking" clocks (锁定频率)
*   为了获得稳定的测量结果（比如对比优化前后的微小差异），专业人士通常会用 `nvidia-smi -lgc` 锁定显卡时钟频率，防止动态频率调整（Boost）干扰数据。
*   **注意**：**这个操作会影响整张卡**。如果在共享机器上，**千万不要由你手动去锁定频率**，除非你是管理员且知道自己在做什么。普通 `nsys` 分析不需要这一步，只有在做精细对比实验时才需要。

---

## 🛠️ 7. 脚本植入最佳实践 (Workflow Integration)

不要每次都手动敲命令行。把 Profiling 集成到你的 `run.sh` 或 `build.sh` 里，形成习惯。

### 范例脚本 (batch_profile.sh)

```bash
#!/bin/bash

# 1. 编译 (确保带上 -lineinfo 方便看源码)
# -lineinfo 是必须的，否则你在 GUI 里看不到代码行号
nvcc -O3 -lineinfo src/main.cu -o build/app

# 2. 快速宏观检查 (Nsys)
# 只抓取开始后 第5秒 到 第10秒 的数据，避免数据量爆炸
echo "========================================"
echo "Step 1: Running Nsight Systems (Macro)..."
echo "========================================"
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=report_macro \
    --force-overwrite=true \
    --delay=5 \
    --duration=5 \
    ./build/app

# 3. 深度微观检查 (Ncu)
# 假设我们已经知道瓶颈 Kernel 叫 "softmax_kernel"
# 我们只关心显存利用率 (SpeedOfLight)
echo "========================================"
echo "Step 2: Running Nsight Compute (Micro)..."
echo "========================================"
ncu \
    --kernel-name-regex "softmax_kernel" \
    --section SpeedOfLight \
    --output report_micro \
    --force-overwrite \
    ./build/app

echo "Done! Generated report_macro.nsys-rep and report_micro.ncu-rep"
```

---
*Generated by GitHub Copilot | Last Update: Feb 2026*