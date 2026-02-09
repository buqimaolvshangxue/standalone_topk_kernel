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

不要一上来就硬啃代码，先看时间线。

### 2.1 常见“病症”速查表

在 GUI 的时间轴上，你可能会看到以下现象，它们分别代表了不同的病症：

| 现象 (Visual Pattern) | 病症诊断 | 潜在原因 | 解决方案 |
| :--- | :--- | :--- | :--- |
| **大量白色空隙 (Gaps)** | **GPU 饥饿 (Starvation)** | CPU 准备数据太慢；API 调用开销太大。 | 使用多流 (Multi-stream)；使用 CUDA Graphs；优化 CPU 逻辑。 |
| **HtoD 传输条极长** | **PCIe 瓶颈** | 在循环里频繁拷贝数据；数据没在 GPU 上常驻。 | 减少传输次数；使用 Pinned Memory (`cudaMallocHost`)；批处理传输。 |
| **细碎的 Kernel** | **启动开销过大** | 每个 Kernel 跑太快 (<10us)，导致启动时间 > 运行时间。 | Kernel Fusion (内核融合)；增大 Batch Size。 |
| **OS Runtime 条很长** | **调度延迟** | 操作系统在忙着上下文切换。 | 绑核 (CPU Affinity)；检查其他进程干扰。 |

### 2.2 NVTX：给时间轴加“书签” (强烈推荐)

在茫茫时间轴中找自己的函数就像大海捞针。**NVTX** 是官方提供的“挂钩”工具。

```cpp
#include <nvtx3/nvToolsExt.h>

void train_step() {
    nvtxRangePushA("Train Step");  // <--- 在时间轴上画一个大色块
    kernel_A<<<...>>>();
    kernel_B<<<...>>>();
    nvtxRangePop();                // <--- 色块结束
}
```
*编译时需链接库：`-lnvToolsExt` (部分新版 CUDA 只需 include 头文件即可)*

### 2.4 Nsight Systems (`nsys`) 到底能看啥？(实战案例)

很多新手觉得 `nsys` 只是看个红绿条，没啥用。**错！大错特错！**

在 90% 的性能优化案例中，**瓶颈根本不在 Kernel 内部，而在 Kernel 外面。** 这时候 `ncu` 是瞎子，只有 `nsys` 能救命。

#### 案例 1：显存拷贝地狱 (The Copy Hell)
*   **现象**: 在 `nsys` 时间轴上，你看到 Compute（计算）条很短，但 `MemCpy (HtoD)` 和 `MemCpy (DtoH)` 的条极长，而且非常密集。
*   **诊断**: 你在 `for` 循环里频繁地把数据在 CPU 和 GPU 之间搬运。
*   **后果**: GPU 计算 1ms，搬数据花了 10ms。这也是为什么你优化 Kernel 没用的原因——**路都在路上耽误了**。
*   **图示**: `[HtoD] [Kernel] [DtoH] ... [HtoD] [Kernel] [DtoH]`

#### 案例 2：CPU 拖后腿 (CPU Bound)
*   **现象**: GPU 的时间轴上有巨大的**空隙 (Gaps)**。两个 Kernel 之间隔了很久。
*   **诊断**: CPU 在准备下一个 Batch 的数据太慢了（比如在做图片解码），导致 GPU 饿得发慌。
*   **这叫**: **GPU Starvation (GPU 饥饿)**。此时你去优化 GPU 代码毫无意义，得去优化刚才那个 Python DataLoader。

#### 案例 3：API 启动延迟 (Launch Latency)
*   **现象**: 你的 Kernel 运行只需要 5us (微秒)，但时间轴上显示 `cudaLaunchKernel` 这个 API 调用花了 10us。
*   **诊断**: **任务太碎了**。
*   **比喻**: 就像送快递，每次只送一颗螺丝钉，送一次就要填一次单子（API开销）。
*   **解法**: 把小任务合并成大任务 (Kernel Fusion)，或者用 CUDA Graph 一次性提交。

> **结论**: 即使 `ncu` 全线飘红，你也必须**先看 `nsys`**。如果 GPU 只有 20% 的时间在干活，那你把 Kernel 优化快 10 倍，整体程序也只能快 18%。

### 2.5 常用命令 (Linux 服务器端)
| :--- | :--- | :--- |
| **标准体检 (推荐)** | `nsys profile --trace=cuda,osrt,nvtx -o report ./app` | 追踪 CUDA API、操作系统调度、NVTX 标签。最常用。 |
| **定点抓取** | `nsys profile --delay=10 --duration=5 -o report ./app` | 运行 10秒后，只抓取 5秒 数据。避免文件过大。 |
| **Python 分析** | `nsys profile --python-backtrace=cuda python train.py` | 能够看到 PyTorch/TensorFlow 到底在哪行 Python 代码调用的 GPU。 |
| **只看统计** | `nsys profile --stats=true ./app` | 此时不生成 `rep` 文件，直接在终端打印 Top 10 耗时内核。快速排查用。 |
| **多 GPU 过滤** | `nsys profile --cuda-devices=0,1 ./app` | 只追踪 0 号和 1 号卡。 |

---

## 🔬 3. Nsight Compute (`ncu`)：深度解剖

当你确定了 `topk_kernel` 是瓶颈，用 `ncu` 把它切片研究。

### 3.1 核心指标字典 (Metric Dictionary)

`ncu` 的数据多到让人恐惧。看这个表，只关注最致命的指标：

| 领域 | 关键指标 (Metric) | 理想值 (Good) | 糟糕值 (Bad) | 含义与通俗解释 |
| :--- | :--- | :--- | :--- | :--- |
| **瓶颈总览** | **Speed of Light (SOL)** | > 60% | < 30% | **SM SOL** 高 = 算力瓶颈；**Memory SOL** 高 = 带宽瓶颈。如果双低，说明在空转（Latency Bound）。 |
| **内存** | **DRAM Throughput** | > 70% | < 10% | **显存带宽利用率**。越高越好。低说明数据没喂饱。 |
| **内存** | **L2 Cache Hit Rate** | > 50% | -- | **L2 缓存命中率**。Hit Rate 越高，也就越少去读慢吞吞的 DRAM。 |
| **内存** | **Coalesced Efficiency** | 100% | < 50% | **合并访问效率**。这是**新手最容易犯的错**。GPU 一次读32字节，你只用了4字节？那就是 12.5% 的效率。浪费带宽。 |
| **计算** | **SM Efficiency** | > 80% | < 40% | SM 真的在忙着算数吗？还是被别的什么卡住了？ |
| **调度** | **Theoretical Occupancy** | > 75% | < 50% | **理论最大载客量**。如果低，通常是因为你用了太多的 **寄存器 (Registers)** 或 **共享内存 (Shared Mem)**，导致 SM 放不下更多线程。 |
| **调度** | **Achieved Occupancy** | 接近理论值 | 远低于理论值 | **实际载客量**。如果远低于理论值，说明发生了严重的 **Tail Effect (尾部效应)** 或流水线停滞。 |
| **代码** | **Branch Divergence** | 0% | > 20% | **分支发散**。同一个 Warp 里的线程走了不同的 `if-else` 路径。除了必须的逻辑，应尽量避免。 |

### 3.2 停滞原因 (Stall Reasons) —— 为什么线程不动了？

这是 `ncu` 最有价值的图表之一 (Warp State Statistics)。

| 停滞代码 | 含义 (The Why) | 如何解决 (The Fix) |
| :--- | :--- | :--- |
| **Long Scoreboard** | **等数据**。试图从 Global Memory 读数据，但还没回来。 | 见效最快。优化合并访问；使用 L2 预取；增加计算密度以掩盖延迟。 |
| **Math Pipe Throttle** | **算不过来**。数学运算单元 (FMA, ALU) 已经塞满了。 | 减少数学指令；降低精度 (FP16/BF16)；使用 Tensor Cores。 |
| **Barrier** | **等队友**。`__syncthreads()` 卡住。 | 检查是否存在个别线程跑得特别慢 (Load Imbalance)；检查分支发散。 |
| **MIO Throttle** | **接口拥堵**。Shared Memory 或 Texture 读写太频繁。 | 减少 Shared Memory 冲突 (Bank Conflicts)。 |

### 3.3 `ncu` 命令速查手册

⚠️ **高危提醒**：不要直接运行 `ncu ./app`。它会让程序慢 100 倍。**必须**使用过滤器。

| 场景 | 命令示例 | 解释 |
| :--- | :--- | :--- |
| **精准打击 (Top 1)** | `ncu -k "softmax" --set full -o result ./app` | `-k`: 正则匹配 Kernel 名。<br>`--set full`: 收集全指标 (推荐)。 |
| **轻量分析** | `ncu -k "softmax" --section SpeedOfLight ./app` | `--section`: 只收集特定板块，速度快。适合快速验证。 |
| **狙击模式 (查指标)** | `ncu --metrics gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed ./app` | `--metrics`: 只查这一个特定的指标 (显存带宽利用率)。不用跑几分钟的全量分析。 |
| **对比优化前后** | `ncu -k "softmax" --import baseline.ncu-rep ./app` | 直接在运行时对比基准文件，输出 diff 结果。 |
| **多次重放控制** | `ncu -k "softmax" --launch-count 1 ./app` | 默认 `ncu` 会重放 Kernel 数十次。加上这个只抓一次，防卡死。 |
| **导出 CSV** | `ncu --csv --page raw -k "softmax" ./app > out.csv` | 导出为 Excel 可读的格式，方便做报表。 |
| **指定显卡** | `ncu --target-processes all --devices 0 ./app` | 只分析 0 号卡上的进程。 |

> 💡 **如何查找指标名？**
> 运行 `ncu --query-metrics` 可以列出几千个指标。
> 常用: `gpu__dram_throughput` (显存), `sm__throughput` (计算), `sm__warps_active` (占用率)。


### 3.4 怎么只查一个指标？(保姆级教程)

你可能会觉得：“我不想看那么多废话，我就想知道 **显存带宽** 到底跑了多少 GB/s，或者利用率是百分之几，怎么弄？”

这两个步骤教给你：

#### 第一步：找到指标的名字
NVIDIA 的指标名字特别长，没人记得住。你可以用 `--query-metrics` 配合 `grep` 来搜。

```bash
# 假设你想找 "DRAM" (显存) 相关的指标
ncu --query-metrics | grep dram
```
*你会看到一大堆输出，比如 `gpu__dram_throughput`...*

#### 第二步：使用 --metrics 参数
把找到的名字复制下来，填到命令里。

**例子 1：看百分比 (利用率)**
```bash
# pct_of_peak_sustained_elapsed = 也就是 "跑到了理论峰值的百分之多少"
ncu --metrics gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed ./my_app
```
*   **输出结果**: 会显示 `85.2%`。说明你把显存带宽吃到了 85%。

**例子 2：看绝对值 (GB/s)**
```bash
# bytes_per_second = "每秒传多少字节"
ncu --metrics dram__bytes.sum.per_second ./my_app
```
*   **输出结果**: 会显示 `800 Gbytes/second`。

### 3.5 常见问题 (FAQ)

**Q: 为什么 `ncu` 跑得这么慢？**
A: 因为它要“重放” (Replay) Kernel。比如你要采集 10 种指标，它可能就要把同一个 Kernel 重复运行 10-20 次，每次只采集一种数据。
**解法**：尽量锁定特定的 Kernel (`-k`)，或者使用狙击模式只查特定指标。

**Q: 我在 GUI 里看不到 C++ 源码？**
A: 99% 是因为编译时没加 `-lineinfo`。请在 `nvcc` 编译命令中加上这个标志。

**Q: `nsys` 里的 GPU 时间轴全是空白？**
A: 可能是 Kernel 跑得太快了 (< 2us)。使用 CUDA Graph 或者增加 workload。也可能是你根本没点 GPU 选项开关 (默认是开的 `--trace=cuda`)。

**Q: 遇到 `ERR_NVGPUCTRPERM` 报错怎么办？**
A: 这是最常见的权限问题。分析 GPU 计数器通常需要 Root 权限。
*   **解法 1 (临时)**: 在命令前加 `sudo` (如果你有权限)：`sudo ncu ...`
*   **解法 2 (永久)**: 让管理员加载内核模块选项 `options nvidia NVreg_RestrictProfilingToAdminUsers=0`。
*   **解法 3 (无奈)**: 如果你是租的显卡（如 Colab/Kaggle）且没有 root，那通常就**用不了 ncu**，只能用 nsys 看时间线。

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