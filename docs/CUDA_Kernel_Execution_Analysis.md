# CUDA Kernel 执行流程与性能分析报告

## 概述

本文档详细分析一个 CUDA Kernel 从调用到执行完成的完整流程，以及每个步骤的时间开销。以 `topk_softmax` kernel 为例（配置：Qwen3-30B-A3B 模型参数 - 1 token, 128 experts, topk=8, bf16），**NCU 实测 Duration 6.56 μs**（含 GPC 端启动 + kernel 执行 + GPC 端完成，不含 GPU Front End 命令处理、Event 打点）。

---

## NCU 性能测试命令与实测数据

### 测试环境

- **GPU**: NVIDIA A40
- **显存**: 46068 MiB
- **Compute Capability**: 8.6
- **Peak Bandwidth**: 696 GB/s

### 1. empty kernel 测试

**目的**：测量纯 kernel launch 的 GPU 端基础开销（无任何计算和访存）。

**NCU 命令**：
```bash
cd /LocalRun/xiaolong.zhu/standalone_topk_kernel
sudo /usr/local/cuda/bin/ncu --set full -o ./ncu_empty -f ./build/bench_factor empty 10 10
```

**cudaEvent 命令**（来自 `compare_baseline.sh`）：
```bash
./build/bench_factor empty 1000 1000    # warmup=1000, iters=1000
```

**NCU 实测结果**：
```
kernel_empty() (1,1,1) x (1,1,1)    ← 1 个 block, 1 个 thread

Duration                         us     1.99 (avg)    1.86 (min)    2.40 (max)
Elapsed Cycles                cycle  2,525.70 (avg)
SM Active Cycles              cycle      4.37 (avg)    ← 真正执行指令的周期
SM Frequency                    Ghz      1.27

Executed Instructions           inst     2.00          ← 只有 2 条指令（进入/退出）
Issued Instructions             inst     4.00
SM Busy                            %     0.27
Compute (SM) Throughput            %     0.00
Memory Throughput                  %     2.00
```

**关键指标解读**：

| 指标 | 值 | 含义 |
|------|------|------|
| Duration | ~2.0 μs | = 2526 / 1.27GHz。GPC 端完整生命周期，几乎全是启动/等待开销 |
| SM Active Cycles | 4.37 | 真正执行指令的 cycle，只有 2 条指令（进入/退出） |
| Elapsed Cycles | 2526 | 总 cycle，与 SM Active 4.37 的巨大差距 = 阶段 3b 启动开销 |
| Executed Instructions | 2 | kernel 体为空，仅有进入和退出两条指令 |

**NCU vs cudaEvent 对比**：
```
NCU Duration  ≈ 2.0  μs   (GPC 全部周期，含 dispatch/retirement)
cudaEvent     ≈ 4.25 μs   (compare_baseline.sh 实测 empty_kernel = 4.247 μs)
差值          ≈ 2.2  μs   (= GPU Front End 命令处理 + Event 打点开销)
```

**差值 ~2.2 μs 的组成**：kernel 极短（2 μs），Front End 命令处理和 E-stop 准备无法被 kernel 执行时间覆盖（详见 3.5 节流水线图解），开销几乎完全暴露。

**NSYS 实测数据**（第三种度量）：

**NSYS 命令**：
```bash
sudo /usr/local/cuda/bin/nsys profile -o ./nsys_empty --force-overwrite true ./build/bench_factor empty 10 10
```

**NSYS 结果**（`nsys stats --report cuda_gpu_kern_sum`）：
```
kernel_empty() — 20 次调用

Duration  Avg: 928 ns   Med: 912 ns   Min: 896 ns   Max: 1,120 ns
```

稳态值（去掉前 2 次 cache 冷启动）：中位数 896 ns ≈ **0.90 μs**。

**三工具对比**：
```
NSYS Duration ≈ 0.93 μs   (CUPTI activity timestamp，Compute Engine 级)
NCU Duration  ≈ 2.0  μs   (gpc__cycles_elapsed.max / SM Freq，GPC/SM 级)
cudaEvent     ≈ 4.25 μs   (T_stop − T_start，Stream 级)

排序：NSYS (0.93) < NCU (2.0) < cudaEvent (4.25)
```

NSYS 显著小于 NCU（仅 46%）——NCU 在 replay 模式下运行条件不同（L2 刷新 + 隔离调度），导致 NCU 值膨胀，详见 3.6 节分析。

### 2. topk_softmax kernel 测试

**目的**：测量真实业务 kernel 的 GPU 端执行时间。

**NCU 命令**：
```bash
sudo /usr/local/cuda/bin/ncu --set full -o ./ncu_topk -f ./build/bench_perf 128 1 bf16 8 10 10
```
- 参数含义：128 experts, 1 token, bf16, topk=8, warmup=10, iters=10

**cudaEvent 命令**（来自 `compare_baseline.sh`，warmup=1000, iters=1000）：
```bash
./build/bench_perf 128 1 bf16 8 1000 1000
```

**NCU 实测结果**：
```
topkGatingSoftmax<8,128,4,16,32,int,__nv_bfloat16>  (1,1,1) x (32,4,1)
  ← 1 个 block, 128 个 thread (32×4)

Duration                         us     6.56
Elapsed Cycles                cycle  8,497
SM Active Cycles              cycle    75.20         ← 真正执行指令
SM Frequency                    Ghz     1.30
SM Busy                            %     4.35

Executed Ipc Active       inst/cycle     0.16
Issued Ipc Active         inst/cycle     0.17
No Eligible Warps                  %    88.18         ← 88% 时间无可发射 warp

Memory Throughput              Gbyte/s    2.48
L1/TEX Hit Rate                    %    85.25
L2 Hit Rate                        %    87.20
Mem Busy                           %     1.22

Avg Active Threads Per Warp             13.85         ← 接近 16（sub-warp 模式）
Warp Cycles Per Issued Instruction       8.47
```

**关键指标解读**：

| 指标 | 值 | 含义 |
|------|-----|------|
| Duration | 6.56 μs | = 8497 / 1.30GHz。kernel 在 GPC/SM 上的完整生命周期 |
| SM Active Cycles | 75.20 | 真正执行指令的 cycle。vs Elapsed Cycles 8497，说明 ~99% 时间 warp 在等待（内存延迟） |
| No Eligible Warps | 88.18% | 88% 的时间没有 warp 可以发射指令 → **延迟受限**（latency-bound）,不是计算或带宽瓶颈 |
| L1 Hit Rate | 85.25% | 大部分数据命中 L1，256 bytes 输入很小 |
| Avg Active Threads | 13.85 | 接近 16，符合代码中的 sub-warp（width=16）工作模式 |

**NCU vs cudaEvent 对比**：
```
NCU Duration  = 6.56 μs   (GPC 全部周期，含 dispatch/retirement)
cudaEvent     = 6.773 μs  (compare_baseline.sh 实测 tokens=1)
差值          = 0.21 μs   (= GPU Front End 命令处理尾部 + Event 打点)
```

**差值只有 ~0.2 μs 的原因**：kernel 执行 6.56 μs 足够长，Front End 在 kernel 还在跑的时候已经完成了 E-start 打点、3a 命令处理和 E-stop 的拉取解析，这些开销被 kernel 执行时间**几乎完全覆盖**，只有极小的尾部暴露在 kernel 结束之后（详见 3.5 节流水线图解）。

**NSYS 实测数据**（第三种度量）：

**NSYS 命令**：
```bash
sudo /usr/local/cuda/bin/nsys profile -o ./nsys_topk --force-overwrite true ./build/bench_perf 128 1 bf16 8 10 10
```

**NSYS 结果**（`nsys stats --report cuda_gpu_kern_sum`）：
```
topkGatingSoftmax<8,...,__nv_bfloat16> — 20 次调用

Duration  Avg: 3,994 ns   Med: 3,936 ns   Min: 3,904 ns   Max: 4,736 ns
```

稳态值（去掉前 2 次 cache 冷启动）：中位数 3,936 ns ≈ **3.94 μs**。

**三工具对比**：
```
NSYS Duration ≈ 3.94 μs   (CUPTI activity timestamp，Compute Engine 级)
NCU Duration  = 6.56 μs   (gpc__cycles_elapsed.max / SM Freq，GPC/SM 级)
cudaEvent     = 6.773 μs  (T_stop − T_start，Stream 级)

排序：NSYS (3.94) < NCU (6.56) < cudaEvent (6.77)
```

NSYS 同样显著小于 NCU（60%），原因见 3.6 节。

### 3. 两组数据的对比总结

```
                        empty kernel          topk kernel
                        ────────────          ───────────
Grid × Block            (1,1,1)×(1,1,1)       (1,1,1)×(32,4,1)
Threads                 1                      128
Executed Instructions   2                      ~12 (per warp)

NSYS Duration           ~0.93 μs               ~3.94 μs

NCU Duration            ~2.0 μs                6.56 μs
  Elapsed Cycles          2,526                  8,497
  SM Active Cycles        4.37                   75.20
  SM Frequency            1.27 GHz               1.30 GHz

cudaEvent               ~4.25 μs               ~6.77 μs

三工具排序              NSYS < NCU < cudaEvent  NSYS < NCU < cudaEvent
NSYS/NCU 比值           46%                    60%
cudaEvent−NCU 差值      ~2.2 μs                ~0.2 μs

cudaEvent−NCU 原因      kernel 太短,            kernel 够长,
                        Front End 无法重叠      Front End 与执行重叠
NSYS < NCU 原因         NCU replay 模式膨胀（L2 刷新 + 隔离调度）（见 3.6 节）
```

> **关于 cudaEvent 的运行间波动**：cudaEvent 是统计平均值，受 GPU 时钟频率（动态调频）、PCIe 延迟、系统负载等影响，不同运行间典型波动 ±0.3-0.5 μs。NCU Duration 基于硬件周期计数器，相对稳定。上述 cudaEvent 数据来自 `compare_baseline.sh 1000 1000`（warmup=1000, iters=1000）。

### 3.5 流水线重叠机制详解

上面反复提到"重叠"，这一节用时间线图把它画清楚。

#### 3.5.1 硬件前提：Front End 和 SM 是独立硬件

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         GPU 内部简化结构                                  │
│                                                                          │
│  ┌───────────────┐         ┌───────────────┐        ┌────────────────┐  │
│  │  Front End    │ 分派CTA │  CTA 分配器    │ 调度   │  SM (执行单元)  │  │
│  │ (命令处理器)  │────────▶│               │───────▶│               │  │
│  │              │         └───────────────┘        │  GPC 处理范围 │  │
│  │ 职责:        │                                   │  (NCU Duration)│  │
│  │ · 从 pushbuf │  Front End 把 kernel 交给          └────────────────┘  │
│  │   拉取命令   │  CTA 分配器后，自身立刻空闲，                          │
│  │ · 解析命令   │  可以去处理 stream 中的下一条                          │
│  │ · 执行 event │  命令（比如 E-stop）。                                 │
│  │   打点       │                                                        │
│  └───────────────┘                                                       │
│                                                                          │
│  关键：Front End 和 SM 是独立硬件，可以同时工作。                        │
│  同一 stream 保证的是命令的"完成顺序"，不是"处理不能并行"。             │
└──────────────────────────────────────────────────────────────────────────┘
```

#### 3.5.2 bench_perf 每次迭代的 stream 命令队列

`bench_perf` 每轮迭代往 GPU stream 顺序提交三条命令：

```
  提交顺序 ──▶

  命令 ①  cudaEventRecord(start)    ← Front End 写起始时间戳 T_start
  命令 ②  kernel launch (topk)      ← Front End 做 3a → SM 做 3b/4/5a → 5b
  命令 ③  cudaEventRecord(stop)     ← Front End 写结束时间戳 T_stop

  cudaEvent ms = T_stop − T_start
```

Front End **按顺序**从 pushbuffer 拉取这三条命令。关键在于：
- Front End 处理完命令②（把 kernel 分派给 SM）后，**不需要等 SM 跑完**，立刻去处理命令③。
- 但命令③的时间戳（T_stop）必须等 kernel 在 SM 上完成后才写入（stream 顺序保证）。
- 所以 Front End 对命令③的"拉取 + 解析"可以与 SM 执行 kernel **同时进行**。

#### 3.5.3 场景 A：topk kernel（长 kernel，NCU 6.56 μs）— 重叠充分

```
时间(μs) 0                                                           ≈6.8
         ├────────────────────────────────────────────────────────────┤

Front    ▓E-st▓  ▓▓▓▓ 3a ▓▓▓▓  ▓ E-stop ▓                    ▓ts▓
End:     写T1    DMA+解析       拉取+解析   (空闲,等SM)         写T2
                 kernel命令     stop命令

SM:                             ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ▓5b▓
                                3b ──── 阶段 4 ────────── 5a
                                ←────── NCU Duration ──────→
                                        6.56 μs

                                 ↑                           ↑
                                 │ ← 这段时间内 Front End    │
                                 │   和 SM 同时在干活：      │
                                 │   Front End 处理 E-stop   │
                                 │   SM 执行 kernel          │
                                 └───── 重叠区间 ────────────┘

cudaEvent  ◄──────────────────────────────────────────────────►
           T_start                                         T_stop

差值 = 6.77 − 6.56 = 0.21 μs
     ≈ 5b + E-stop 打点尾部
       （E-start、3a、E-stop 拉取准备几乎全部被 kernel 的 6.56 μs 覆盖）
```

**为什么差值只有 ~0.2 μs？**
- 3a→3b 的交接是流水线式的：Front End 还在做 3a 尾部时，CTA 分配器已开始 3b，两者有一定重叠。
- Front End 做完 3a 后立刻处理 E-stop 命令（拉取+解析），这些工作在 SM 执行 kernel 期间**全部完成**。
- kernel 结束后只剩 5b（全局发布）和 E-stop 时间戳写入。
- 结果：**~2 μs 的 Front End 工作被 kernel 的 6.56 μs 吞没**，只露出 ~0.2 μs 的尾巴。

#### 3.5.4 场景 B：empty kernel（极短 kernel，NCU ~2.0 μs）— 重叠不充分

```
时间(μs) 0                                                    ≈4.2
         ├────────────────────────────────────────────────────┤

Front    ▓E-st▓  ▓▓▓▓ 3a ▓▓▓▓  ▓▓▓▓▓▓ E-stop ▓▓▓▓▓▓  ▓ts▓
End:     写T1    DMA+解析       拉取+解析                写T2
                 kernel命令     stop命令(含等待kernel完成)
                                ←── 虽然已开始准备，
                                     但必须等 SM 结束
                                     才能写 T_stop ──→

SM:                             ▓▓ 3b+4+5a ▓▓  ▓5b▓
                                ← NCU Dur. →
                                   ~2.0 μs
                                              ↑
                                              SM 很快就结束了
                                              但 E-stop 处理
                                              的开销无法被
                                              kernel 完全覆盖

cudaEvent  ◄────────────────────────────────────────────────►
           T_start                                       T_stop

差值 = 4.25 − 2.0 = 2.25 μs
     = E-start + 3a + 5b + E-stop + 各环节的 3a↔3b 交接间隙
       （kernel 太短，Front End 的工作没怎么被覆盖）
```

**为什么差值高达 ~2.2 μs？**
- kernel 只跑了 2 μs（而且 SM Active 只有 4.37 cycle，几乎是空转）。
- Front End 做 3a 和 E-stop 准备这些工作，kernel 的执行时间不够长，无法把它们完全"吞掉"。
- 结果：Front End 的命令处理 + Event 打点开销**几乎全部暴露**在 cudaEvent 差值里。

#### 3.5.5 一句话总结

```
              Front End 工作量（3a + E-stop + E-start + 5b）≈ ~2 μs 量级
              SM 执行时间（NCU Duration）= 因 kernel 而异

              ┌──────────────────────────────────────────────────────────┐
              │  NCU Duration >> Front End 工作量                        │
              │  → Front End 被 SM 执行时间覆盖 → 差值小 (topk: ~0.2)  │
              ├──────────────────────────────────────────────────────────┤
              │  NCU Duration ≈ Front End 工作量                        │
              │  → 无法充分覆盖 → 差值大 (empty: ~2.2)                 │
              └──────────────────────────────────────────────────────────┘

              所以：kernel 越长，cudaEvent 和 NCU 的差值越接近于零。
```

### 3.6 NSYS vs NCU vs cudaEvent：三种 GPU 时间度量的差异分析

> **方法论**：本节首先基于 NVIDIA 官方文档，精确定义三种工具各自"测的是什么"、"怎么测的"、"在什么条件下测的"（3.6.1）。然后再结合实测数据分析差异成因（3.6.2–3.6.5）。定义是基础，分析建立在定义之上。

#### 3.6.1 三种工具的官方定义与测量原理

> **本节内容全部来自 NVIDIA 官方文档**，有一个例外需要特别注意：
> 官方清楚地定义了每个工具「Duration 公式是什么」「运行条件是什么」，但**没有定义**每个工具从 kernel 生命周期的哪个硬件环节开始计时、到哪个环节结束。
> 换句话说，官方告诉你用什么公式算、在什么环境下测，但没有告诉你「这个值包含 CTA 分配吗？包含 Front End 吗？」。
> 后续第 4 节的阶段映射（3a/3b/3c/4/5a/5b/5c）是本文基于 GPU 架构知识的推断模型，不是官方定义。

##### A. NSYS（Nsight Systems）— CUPTI Activity Tracing

**官方文档来源**：[CUPTI Documentation](https://docs.nvidia.com/cupti/main/main.html)

**测量机制**：NSYS 底层使用 CUPTI（CUDA Profiling Tools Interface）的 Activity API 来采集 kernel 执行的时间戳。CUPTI 文档明确说明：

> *"CUPTI instruments the kernel code to collect the timing information."*
> — CUPTI Documentation, CONCURRENT_KERNEL Tracing Mode

即 CUPTI 通过**对 kernel 代码进行软件插桩**来获取时间信息。在 Blackwell 架构之前，这套机制基于"software instrumentation and semaphore-based approaches"（软件插桩与信号量方案）：

> *"[HES provides] more accurate timestamps and is expected to add minimal launch overhead compared to traditional software instrumentation and semaphore-based approaches used in CUPTI."*
> — CUPTI Documentation, Hardware Event Sampling (Blackwell+)

**Duration 定义**：
- 数据结构：`CUpti_ActivityKernel` 记录中的 `start` 和 `end` 字段
- Duration = `end − start`
- 时钟源：GPU `globaltimer`（全局纳秒计时器）
- 量化粒度：实测约 32 ns（与 globaltimer 的 ~31.25 MHz 时钟频率一致）

> **实测量化验证**：empty kernel 的 20 次 Duration 全部为 32 ns 的整数倍（896=28×32, 928=29×32, 1056=33×32, 1120=35×32），证实有效量化粒度约 32 ns。

**运行条件**：
- 应用**正常执行**，CUPTI 仅注入低开销的时间戳采集
- 不改变 kernel 的调度方式、cache 状态、时钟频率
- 多次迭代可享受 warmup 后的热 cache 和稳态 pipeline

> **官方未说明的**：`start` 和 `end` 时间戳精确对应 kernel 生命周期的哪个硬件时刻（例如：`start` 是在 CTA 被调度时记的，还是在第一条指令执行时记的？）。官方只说 *"instruments the kernel code"*，即插桩点在 kernel 代码本身内部，因此可以推断 `start`/`end` 对应 kernel 代码执行的起止，但这是推断。

##### B. NCU（Nsight Compute）— 硬件性能计数器 + Replay 模式

**官方文档来源**：[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)

**Duration 计算公式**：
```
Duration = gpc__cycles_elapsed.max / GPC_clock_frequency
```
其中：
- **GPC**（General Processing Cluster）：官方定义为 *"The General Processing Cluster contains SM, Texture and L1 in the form of TPC(s). It is replicated several times across a chip."*
- **`.max` roll-up**：官方定义为 *"maximum counter value across all unit instances"*
- 因此 `gpc__cycles_elapsed.max` = **所有 GPC 实例中，经历的最大周期数**
- GPC 时钟频率可从 `gpc__cycles_elapsed.avg.per_second` 获取（官方描述：*"GPC Clock Frequency: The average GPC clock frequency in hertz"*）
- **`cycles_elapsed` 的官方定义**：*"The number of cycles within a range. The cycles' DimUnits are specific to the unit's clock domain."*（Profiling Guide, Cycle Metrics 章节）。这里的 "range" 是指 kernel 的 profiling 范围，但**官方未进一步定义 range 的精确起止点**（即未说明是从 CTA 分配开始还是从首条指令执行开始，是到最后一个 warp 退出结束还是到 CTA 完成信号结束）

**运行条件 — 三大控制机制（官方文档明确描述）**：

1. **Kernel Serialization（串行化）**：
   > *"NVIDIA Nsight Compute serializes kernel launches within the profiled application."*
   
   所有 kernel launch 被串行化执行，不允许并发。

2. **Cache Control（缓存刷新）**：
   > *"NVIDIA Nsight Compute by default flushes all GPU caches before each replay pass."*
   
   每个 replay pass 之前，**默认刷新所有 GPU cache**（包括 L2）。这意味着被 profile 的 kernel 始终面对 cold cache。

3. **Clock Control（时钟锁定）**：
   > *"NVIDIA Nsight Compute attempts to limit GPU clock frequencies to their base value."*
   
   GPU 时钟被锁定到基频，避免 Boost 频率波动影响测量。

**Replay（重放）机制**：
> *"Due to hardware limitations, only a limited number of metrics can be collected in a single pass of the kernel execution. If more metrics are requested, the kernel launch is replayed multiple times."*

> *"Before the first replay pass, all GPU memory that can be accessed by the kernel is saved. After the first pass, the subset of memory that is written by the kernel is determined. Before each pass (except the first one), this subset is restored."*

即：NCU 会多次重放 kernel，每次收集不同的性能计数器子集。首次 pass 前保存所有可访问 GPU 内存，后续每个 pass 前恢复被写入的内存子集，确保每次 replay 的初始状态一致。

**关键影响**：由于 cache 刷新 + 串行化 + 时钟锁定，NCU 测得的 Duration **不代表应用正常运行时的 kernel 执行时间**，而是**在受控隔离环境下的执行时间**。

##### C. cudaEvent — GPU Stream 时间戳

**官方文档来源**：
- [CUDA Runtime API — Event Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html)
- [CUDA C++ Best Practices Guide — Section 9.1.2](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#using-cuda-gpu-timers)

**测量机制**：

`cudaEventRecord()` — 官方定义：
> *"Captures in event the contents of stream at the time of this call."*

`cudaEventElapsedTime()` — 官方定义：
> *"Computes the elapsed time between two events (in milliseconds with a resolution of around 0.5 microseconds)."*

Best Practices Guide 进一步解释了底层行为：
> *"Here `cudaEventRecord()` is used to place the start and stop events into the default stream, stream 0. The device will record a timestamp for the event when it reaches that event in the stream."*
> *"Note that the timings are measured on the GPU clock, so the timing resolution is operating-system-independent."*

**工作方式**：
1. `cudaEventRecord(start, stream)` 在 stream 中插入一个"打点命令"
2. GPU 按 stream 顺序执行到该命令时，记录当前 GPU 时钟时间戳
3. `cudaEventRecord(stop, stream)` 同理，在 kernel 之后记录
4. `cudaEventElapsedTime()` 计算两个 GPU 时间戳的差值

**关键注意事项**（官方文档原文）：
> *"If either event was last recorded in a non-NULL stream, the resulting time may be greater than expected (even if both used the same stream handle). This happens because the `cudaEventRecord()` operation takes place asynchronously and there is no guarantee that the measured latency is actually just between the two events. Any number of other different stream operations could execute in between the two measured events, thus altering the timing in a significant way."*

这意味着 cudaEvent 测量的是 **GPU stream 时间线上两个打点之间的全部时间**，包含 kernel 执行本身以及打点命令自身在 stream 中的处理开销。

**运行条件**：
- 应用正常执行，GPU 时钟和 cache 不受额外控制
- 时间戳精度约 0.5 μs
- 打点命令本身在 GPU stream 中有执行开销

##### D. 汇总对比表

| 特征 | NSYS (Nsight Systems) | NCU (Nsight Compute) | cudaEvent |
|------|----------------------|---------------------|-----------|
| **底层 API** | CUPTI Activity API | 硬件性能计数器 | CUDA Runtime Event API |
| **Duration 定义** | `CUpti_ActivityKernel.end − start` | `gpc__cycles_elapsed.max / GPC_freq` | `cudaEventElapsedTime(start, stop)` |
| **时钟源** | GPU globaltimer（~31.25 MHz） | GPC 周期计数器（~1.3 GHz） | GPU 时钟（~0.5 μs 精度） |
| **采集方式** | 软件插桩注入时间戳 | 硬件计数器 + 多 pass replay | Stream 中插入打点命令 |
| **Cache 状态** | 正常（warmup 后热） | **每 pass 前刷新所有 GPU cache** | 正常（warmup 后热） |
| **时钟控制** | 无（允许 Boost） | **锁定到基频** | 无（允许 Boost） |
| **Kernel 调度** | 正常并发 | **串行化** | 正常并发 |
| **时间分辨率** | ~32 ns | ~0.77 ns (@1.3GHz) | ~500 ns |
| **测量区间** | kernel 代码执行（插桩入口→出口） | GPC 全部周期（含 dispatch/retirement） | Stream 中 start→stop 全部时间 |

> **核心结论**：三者测量的"Duration"含义不同——
> - **NSYS** 测的是正常执行下 kernel **代码执行**的耗时（插桩入口→出口，范围最窄）；
> - **NCU** 测的是受控隔离环境（cold cache + 基频 + 串行化）下 **GPC 全部周期**的耗时（含 dispatch/retirement 开销，范围更宽）；
> - **cudaEvent** 测的是 GPU stream 时间线上两个打点之间的总耗时（含 Front End + kernel + 打点开销，范围最宽）。
>
> **官方边界**：以上 Duration 公式、运行条件、采集方式均来自官方文档。但**官方没有定义**这些 Duration 从 kernel 生命周期的哪个具体硬件环节开始、到哪个环节结束（见上方 A、B 节的说明）。后续第 4 节的阶段映射（3a/3b/3c/4/5a/5b/5c → 工具覆盖）是本文基于实测数据（特别是 empty kernel 控制实验）的推断模型。

#### 3.6.2 实测数据汇总

```
                    empty kernel        topk (tokens=1)         topk (tokens=65536)
                    ────────────        ───────────────         ───────────────────
NSYS Duration       ~0.93 μs            ~3.94 μs                ~132.16 μs
NCU Duration        ~2.0  μs            6.56 μs                 142.72 μs
cudaEvent           ~4.25 μs            ~6.77 μs                ~133.93 μs

三工具排序          NSYS < NCU < Event  NSYS < NCU < Event      NSYS < Event < NCU
NSYS/NCU            46%                 60%                     93%
(cudaEvent−NCU)     +2.25 μs (113%)     +0.21 μs (3.2%)         −8.79 μs (−6.2%)
(NCU−NSYS)          +1.07 μs (115%)     +2.62 μs (66%)          +10.56 μs (8.0%)
```

> **tokens=65536 NSYS 数据来源**：`nsys_topk_65536.nsys-rep`，20 次调用，Avg 132,161 ns（Med 131,729 ns, Min 130,753 ns, Max 138,112 ns, StdDev 1,579 ns）。Grid (8192,1,1)×(32,4,1)，29 reg/thread。

#### 3.6.3 为什么 NSYS < NCU？——测量范围差异 + NCU 重放（replay）膨胀

NSYS < NCU 的原因有**两层**：

**第一层：测量范围不同（测量机制差异）**
NSYS/CUPTI 和 NCU 使用完全不同的测量机制，导致它们覆盖的时间范围不同：
- **NSYS/CUPTI**：在 kernel **代码内部**插入时间戳读取指令（*"instruments the kernel code"*，见 3.6.1 A）——只捕获 kernel 代码执行期间的时间，**不包含**代码执行之前/之后的 GPC 级 dispatch/retirement 开销
- **NCU**：使用 GPC 硬件周期计数器 `gpc__cycles_elapsed`（*"The number of cycles within a range"*，见 3.6.1 B）——计数 GPC 处理 kernel 的**全部周期**（active + stalled + idle），包含 CTA dispatch 和 retirement 等 GPC 管线开销

> **Empty kernel 控制实验（决定性证据）**：
> - Empty kernel 没有任何内存访问 → L2 cache 刷新完全无关
> - NCU: 2526 GPC elapsed cycles @ 1.27 GHz = 2.0 μs；其中 SM Active Cycles = 4.37（仅 0.17%）
> - NSYS: 0.93 μs
> - 如果两者测的是同一范围，只是时钟不同（NCU 基频 1.27 GHz, NSYS boost ~1.74 GHz）：同一个 2526 cycle 范围 @ 1.74 GHz = **1.45 μs**——但 NSYS 实测 **0.93 μs**，远小于 1.45 μs
> - 0.93 μs × 1.74 GHz ≈ 1618 cycles，比 NCU 的 2526 cycles **少了 908 cycles (36%)**
> - **结论：NSYS 不是在更快的时钟下跑相同的周期数，而是在数更少的周期数 → 测量范围更窄**
>
> 那 2526 中多出来的 908 cycles 是什么？是 GPC 级 CTA dispatch/scheduling/retirement 开销——发生在 kernel 代码执行之前/之后，NSYS 的代码内插桩捕获不到它们，但 NCU 的 GPC 硬件计数器把它们全部计入了。

**第二层：运行条件不同（replay 环境差异）**
即使在各自的测量范围内，两者的运行条件也截然不同——这进一步拉大了差距。以下差异全部来自官方文档（引用见 3.6.1 节 A 和 B）：

| | NSYS（Nsight Systems） | NCU（Nsight Compute） |
|---|---|---|
| **运行方式** | 正常执行（与应用一起跑，warmup 后稳态） | Replay 模式（多 pass 重放，*"replayed multiple times"*） |
| **L2 Cache 状态** | 热（warmup 后数据留在 cache） | 冷（*"flushes all GPU caches before each replay pass"*） |
| **时钟** | 允许 Boost | 锁定基频（*"limit GPU clock frequencies to their base value"*） |
| **调度环境** | 正常 GPU pipeline，前后 kernel 流水线衔接 | 串行化（*"serializes kernel launches"*） |
| **采集方式** | CUPTI 软件插桩（*"instruments the kernel code"*） | 硬件性能计数器 + 内存状态 save/restore |

```
正常执行（NSYS 环境）                     重放模式（NCU 环境）
──────────────────                        ──────────────────

  warmup iterations...                    NCU 控制的隔离环境：
  cache 已热，pipeline 稳态               · L2 cache 每个 pass 之间被刷新
  ↓                                       · kernel 在隔离环境中执行
                                          · 时钟锁定基频
  GPC 处理范围：                          ↓
  ┌─ dispatch ─┬── kernel code ──┬─ retire ─┐
  │  (GPC      │  ●── code ──●  │  (GPC    │
  │  overhead) │  ↑ NSYS 范围 ↑  │  overhead)│
  └────────────┴─────────────────┴──────────┘
  ↑──────────── NCU 范围 ──────────────────↑

  NSYS 测得（仅 kernel 代码执行）：       NCU 测得（全部 GPC 周期 + replay 膨胀）：
    empty  0.93 μs                          empty  2.0  μs  (× 2.15)
    topk   3.94 μs                          topk   6.56 μs  (× 1.66)
```

##### 直接证据：tokens=65536 时 NCU 反超 cudaEvent

对于长 kernel（tokens=65536），实测数据给出了 NCU replay 膨胀的**决定性证据**：

```
  NSYS Duration = 132.16 μs    （正常执行，kernel 代码插桩，范围最窄）
  cudaEvent     = 133.93 μs    （正常执行，Stream 级，含 Front End + Event 开销）
  NCU Duration  = 142.72 μs    （replay 模式，GPC 级，不含 Front End + Event）

  排序：NSYS (132.16) < cudaEvent (133.93) < NCU (142.72)
  NCU − cudaEvent = +8.79 μs   ← NCU 反而更大！
  NCU − NSYS      = +10.56 μs  ← 范围差 + replay 膨胀
```

cudaEvent 的测量区间比 NCU **更宽**（包含 Front End + Event 开销），但 NCU 反而更大。唯一合理的解释是 **NCU replay 膨胀**：65536 tokens 的工作集较大，replay 时 L2 被刷新导致大量额外 cache miss（正常执行时命中 L2 的数据，在 replay 中必须从 DRAM 重取），使 NCU Duration 膨胀到超过 cudaEvent。

**这证明：NCU Duration 不是"真实执行时间"，而是"replay 环境下的执行时间"，会因 cache 状态和隔离调度而偏大。**

##### 不同 kernel 的膨胀幅度

| | empty kernel | topk (tokens=1) | topk (tokens=65536) |
|---|---|---|---|
| NSYS（正常执行） | 0.93 μs | 3.94 μs | **132.16 μs** |
| NCU（replay） | 2.0 μs | 6.56 μs | 142.72 μs |
| cudaEvent（正常+FE） | 4.25 μs | 6.77 μs | 133.93 μs |
| NCU / NSYS | 2.15× | 1.66× | **1.08×** |
| NCU − cudaEvent | −2.25 μs | −0.21 μs | **+8.79 μs** |

- **empty kernel（无访存）**：NCU/NSYS = 2.15×。无 L2 cache 效应，差异来自两个因素：(1) NCU 测量范围更宽（包含 GPC dispatch/retirement 开销，约 908 extra cycles），(2) NCU 时钟锁定基频（1.27 vs ~1.74 GHz）。
- **topk tokens=1（256 bytes 访存）**：NCU/NSYS = 1.66×。比值下降，说明 GPC 固定开销被更多的实际执行时间稀释。L2 刷新影响有限（数据量极小）。
- **topk tokens=65536（大量访存）**：NCU/NSYS = 1.08×，比值接近 1，说明 GPC 固定开销（~908 cycles）在长 kernel 中已被完全稀释；剩余差异几乎全部来自 L2 刷新导致的 cache miss 膨胀。NCU 已超过 cudaEvent（+8.79 μs），排序变为 NSYS < cudaEvent < NCU。NSYS 与 cudaEvent 仅差 ~1.8 μs（Front End 开销被充分重叠）。

> **关于时钟域差异**：NSYS 和 NCU 使用不同时钟源（globaltimer vs GPC cycle counter）和不同量化粒度（~32 ns vs ~0.77 ns）。时钟频率差异（NCU 基频 ~1.27 GHz vs NSYS boost ~1.74 GHz）可贡献约 1.37× 的时间膨胀，但 empty kernel 实测比值 = 2.15×，远超纯时钟效应。差异由两层因素构成：(1) **测量范围不同** — NCU 的 GPC 计数器包含 dispatch/retirement 开销，NSYS 代码内插桩不包含；(2) **运行条件不同** — NCU replay 模式的 L2 刷新、基频锁定和隔离调度。

#### 3.6.4 NSYS trace 的额外观察：Event 打点开销

从 NSYS GPU trace 中可以直接观察到 cudaEventRecord 的开销：

```
empty kernel — NSYS trace 中 kernel 间距：

  warmup 阶段（无 Event）：kernel 间距 ≈ 3-4 μs
  measured 阶段（有 Event）：kernel 间距 ≈ 18-19 μs

  额外开销 ≈ 14-15 μs / iteration
  = 2 × cudaEventRecord + cudaEventSynchronize 的 GPU+Host 开销
```

这从第三方视角佐证了 cudaEventRecord 本身在 GPU stream 上有明显开销，进一步解释了 cudaEvent > NCU 的差异。

#### 3.6.5 三种度量的关系

```
小 kernel 上满足：NSYS < NCU < cudaEvent
长 kernel 上可能：NSYS < cudaEvent < NCU（replay 膨胀超过 Front End 开销）

原因来自三个独立因素：

  ┌─ cudaEvent ───────────────────────────────────────────────────────────┐
  │ E-start  3a ┌─ GPC 处理范围 ─────────────────────────────┐ 5b E-stop│
  │             │ dispatch ┌── kernel 代码执行 ──┐ retirement │         │
  │             │ (GPC     │                      │ (GPC      │         │
  │             │ overhead)│  · NSYS 只测这里     │ overhead) │         │
  │             │          │  · 正常执行 → 值最小 │           │         │
  │             │          └──────────────────────┘           │         │
  │             │                                              │         │
  │             │  · NCU 测整个 GPC 范围（含 dispatch/retire） │         │
  │             │  · + replay 模式膨胀 → 值更大               │         │
  │             └──────────────────────────────────────────────┘         │
  └─────────────────────────────────────────────────────────────────────┘

  差异来源（三个独立因素）：
  ──────────────────────────────
  ① NSYS < NCU 的「范围差」：NCU 的 GPC 计数器覆盖范围 > NSYS 的代码内插桩范围
  ② NSYS < NCU 的「条件差」：NCU replay 膨胀（L2 刷新 + 基频锁定 + 隔离调度）
  ③ cudaEvent > NCU/NSYS   ：Front End + Event 开销（3.5 节，受流水线重叠压缩）

  ⚠ 对于长 kernel（如 tokens=65536）：
     NSYS (132.16 μs) < cudaEvent (133.93 μs) < NCU (142.72 μs)
     因素 ①② 的总膨胀超过因素 ③ → 排序变为 NSYS < cudaEvent < NCU
     NCU/NSYS = 1.08×（GPC 固定开销被稀释，差异主要来自 L2 cache miss 膨胀）
```

**一句话总结**：
- **NSYS Duration** — 范围最窄（kernel 代码内部插桩）、条件最自然（正常执行，热 cache，boost 频率），值最小。
- **NCU Duration** — 范围更宽（GPC 全部周期，含 dispatch/retirement 开销），且在受控环境下运行（L2 刷新 + 基频锁定 + 串行化），值更大。
- **cudaEvent** — 范围最宽（整个 stream 打点区间，含 Front End + Event 开销），但在正常执行环境下运行。
- 小 kernel 上 Front End 开销占主导 → NSYS < NCU < cudaEvent；长 kernel 上 replay 膨胀占主导 → NSYS < cudaEvent < NCU。
- 三者的差异来自**范围覆盖差异** + **运行条件差异** + **Front End 开销**，共三个独立因素。

### 4. 度量边界一览表（核心参考）

> **⚠️ 重要区分：官方定义 vs 本文推断模型**
>
> **NVIDIA 官方定义的（见 3.6.1 节引用）**：
> - NCU Duration = `gpc__cycles_elapsed.max / GPC_freq`（GPC 周期计数器）
> - NSYS Duration = CUPTI Activity `end − start`（软件插桩时间戳）
> - cudaEvent = `cudaEventElapsedTime(start, stop)`（stream 中两个 GPU 时间戳差）
> - NCU 的三大运行控制：串行化、缓存刷新、时钟锁定
>
> **NVIDIA 官方没有定义的**：
> - `gpc__cycles_elapsed` 从 kernel 执行的哪个硬件环节开始计数、到哪个环节停止（官方只说 *"The number of cycles within a range"*，未进一步定义 range 的起止点）
> - CUPTI Activity 的 `start`/`end` 到底对应 kernel 生命周期的哪个精确时刻（官方只说 *"instruments the kernel code to collect the timing information"*）
> - 不存在「阶段 3a/3b/3c/4/5a/5b/5c」这样的官方编号流水线模型
>
> **下面的阶段分解（3a, 3b, 3c, 4, 5a, 5b, 5c）和工具-阶段映射是本文基于 GPU 架构知识和实测数据的推断模型**，目的是帮助理解三种工具测量值的差异。它是合理的推断，但不是 NVIDIA 官方定义。

**阶段划分原则**：每个阶段整体属于或不属于某工具的覆盖范围。**NCU 和 NSYS 覆盖的阶段不同**——NCU 的 GPC 硬件计数器范围更宽，NSYS 的代码内插桩范围更窄（见 3.6.3 节的 empty kernel 控制实验）。

**官方定义（确定的）**：
```
NCU Duration  = gpc__cycles_elapsed.max / GPC_freq（官方公式，见 3.6.1 节 B）
NSYS Duration = CUpti_ActivityKernel.end − start （官方公式，见 3.6.1 节 A）
cudaEvent     = cudaEventElapsedTime(start, stop) （官方公式，见 3.6.1 节 C）
```

**推断模型（本文构建，非官方）**：

以下阶段编号和工具-阶段映射是基于 GPU 架构知识和实测数据（特别是 empty kernel 控制实验）的推断，用于解释三种度量值的差异：

```
GPU 时间线（推断模型）
══════════════════════════════════════════════════════════════════════════════════════

步骤          描述                        cudaEvent    NCU Duration    NSYS Duration
──────        ────                        ─────────    ────────────    ─────────────
E-start       Event Start 打点               ✓ (起点)     ✗              ✗

阶段 3a       Front End 命令 DMA 拉取        ✓            ✗              ✗
              + 命令解析

              ─────────── NCU Duration 起点（推断）─────────────────

阶段 3b       GPC 接收 CTA、资源分配         ✓            ✓              ✗
              dispatch 管线开销
              (GPC cycles_elapsed 开始计数)

              ─────────── NSYS Duration 起点（推断）────────────────

阶段 3c       取指/I-Cache 装填              ✓            ✓              ✓
              首 warp 发射
              (kernel 代码开始执行，
               CUPTI 插桩入口代码记 start)

阶段 4        数据加载 + 计算 + 写回         ✓            ✓              ✓

阶段 5a       最后 warp 退出                 ✓            ✓              ✓
              (CUPTI 插桩出口代码记 end)

              ─────────── NSYS Duration 终点（推断）────────────────

阶段 5b       CTA 完成信号 + GPC retirement  ✓            ✓              ✗
              (GPC cycles_elapsed 停止计数)

              ─────────── NCU Duration 终点（推断）─────────────────

阶段 5c       全局完成发布                   ✓            ✗              ✗
              (stream 状态 + event 时间戳)

E-stop        Event Stop 打点               ✓ (终点)     ✗              ✗

══════════════════════════════════════════════════════════════════════════════════════
注：此表中 NCU 和 NSYS 的起止点不同——这是基于 empty kernel 控制实验的推断。
    empty kernel 中 NCU = 2.0 μs 而 NSYS = 0.93 μs（无内存访问，cache 刷新无关），
    即使校正时钟差异（base 1.27 vs boost ~1.74 GHz），NCU 仍多出 ~908 GPC cycles，
    这些 cycles 对应 GPC dispatch/retirement 开销，落在 NSYS 插桩范围之外。
    NVIDIA 官方未明确定义这些工具从 kernel 生命周期的哪个环节开始/结束计数。
```

**推断依据**：
- **NCU 范围 > NSYS 范围（核心发现）**：empty kernel 控制实验证明（见 3.6.3 节）——NCU 的 `gpc__cycles_elapsed` 包含 GPC dispatch/retirement 管线开销（阶段 3b + 5b），而 NSYS 的代码内插桩只从 kernel 代码入口到出口（阶段 3c → 5a）
- NCU 计数器 `gpc__cycles_elapsed` 属于 GPC 时钟域（GPC 包含 SM），当 kernel 未被分派到 GPC 上时（如 Front End 解析阶段）GPC 不会为这个 kernel 计数，因此推断 NCU 不包含 Front End（3a）
- CUPTI 文档说 *"instruments the kernel code"*，即在 kernel 代码本身注入时间戳，因此推断 NSYS 不包含 GPC dispatch/retirement 开销
- cudaEvent 在 kernel 前后各插入一个 stream 命令，stream 是串行的，因此 cudaEvent 区间覆盖从 start 打点到 stop 打点之间的一切

**NCU vs NSYS 差异的双重来源**：
1. **范围差**（固有）：NCU 包含 GPC dispatch/retirement 开销，NSYS 不包含。这是测量机制的固有差异，即使在完全相同的运行条件下也会存在。empty kernel 上约 908 GPC cycles（~0.5-0.7 μs）。
2. **条件差**（环境）：NCU 在 replay 模式下运行（L2 刷新、基频锁定、串行化），NSYS 在正常执行下运行（热 cache、boost 频率）。对有访存的 kernel，这个因素更显著。

实测：NSYS/NCU 比值 = 46%（empty）~ 60%（topk tokens=1）。
长 kernel 时 NCU 甚至可能超过 cudaEvent（tokens=65536 实测证明）。

**一句话总结**：
```
官方确定的：
  NSYS Duration < NCU Duration（范围不同 + 运行条件不同，见 3.6 节）
  小 kernel: NSYS < NCU < cudaEvent
  大 kernel: NSYS < cudaEvent < NCU（replay 膨胀）

本文推断模型（帮助理解差异的概念工具，非官方）：
  NSYS Duration ≈ 阶段 3c + 4 + 5a（kernel 代码执行，正常环境）     ← 最窄、最短
  NCU Duration  ≈ 阶段 3b + 3c + 4 + 5a + 5b（GPC 全部周期，replay） ← 更宽、更长
  cudaEvent     ≈ E-start + 3a + 3b + 3c + 4 + 5a + 5b + 5c + E-stop − 流水线重叠  ← 最宽

  推断依据：empty kernel 控制实验 — NCU 2526 GPC cycles vs NSYS ~1618 cycles（校正时钟后），
  多出的 ~908 cycles 对应 GPC dispatch/retirement 开销（阶段 3b + 5b）
```

---

### 术语约定（避免混淆）

#### CUDA 执行模型

- **Grid / Block / Thread**：CUDA 的三级并行层次。一次 kernel launch 产生一个 **Grid**；Grid 由若干 **Block**（= CTA）组成；每个 Block 包含若干 **Thread**。例如 `dim3 block_dim(32, 4)` 表示每个 Block 有 128 个 Thread。
- **CTA（Cooperative Thread Array）**：就是一个 **thread block** 的硬件层面叫法。同一个 CTA 内的线程共享 shared memory，可以用 `__syncthreads()` 同步。GPU 按 CTA 粒度分配寄存器/shared memory/调度槽位。文中"分配 CTA 资源"可直接理解为"给 block 分配寄存器/shared memory 并准备调度"。
- **Warp**：GPU 调度和执行的最小单位，固定 32 个连续 thread。同一 Warp 内的 thread 以 SIMT（Single Instruction, Multiple Threads）方式执行同一条指令。一个 128-thread 的 Block 有 4 个 Warp。
- **SM（Streaming Multiprocessor）**：GPU 上的计算核心单元。每个 SM 有自己的寄存器文件、shared memory、L1 cache、warp scheduler。多个 CTA 可以同时驻留在同一个 SM 上（取决于资源占用），由 SM 的 warp scheduler 交替调度执行。
- **Stream**：CUDA 的命令队列。同一 stream 内的命令按提交顺序在 GPU 上依次执行；不同 stream 的命令可以并发。`kernel<<<grid, block, 0, stream>>>` 中最后一个参数就是 stream。
- **Kernel Launch（本文约定）**：Host 侧 `kernel<<<...>>>(...)` 的提交/入队（enqueue），也叫 **host launch overhead**。Host 把 launch 命令入队后立即返回，不等待 GPU 执行完成。
- **GPU 侧启动延迟**：GPU 收到 launch 命令后，到真正开始执行 kernel 前的解析、资源分配、调度准备。
- **Kernel 执行时间**：kernel 在 SM 上实际执行（含数据读写与计算）的时间。

#### 存储层次

- **寄存器（Register）**：每个 thread 私有的最快存储，延迟 ~0 cycle。数量有限（每 SM 约 64K 个 32-bit 寄存器），多个 CTA 共享寄存器文件。
- **Shared Memory**：同一 CTA 内所有 thread 共享的片上高速存储（延迟 ~20 cycle），可用于 thread 间通信。本 kernel 未使用 shared memory（纯寄存器 + warp shuffle）。
- **L1 Cache**：SM 上的片上缓存（~128KB/SM），缓存 global memory 数据，延迟约 20-30 cycle。
- **L2 Cache**：GPU 芯片级共享缓存（~50MB），所有 SM 共用，延迟约 100-200 cycle。
- **HBM（High Bandwidth Memory）**：GPU 显存（主存），容量大（~80GB）但延迟高（300-500 cycle），带宽约 3 TB/s。也叫 DRAM / Global Memory。
- **Pinned Memory（锁页内存）**：Host 侧通过 `cudaMallocHost` 分配的页锁定内存，不会被 OS 换出到磁盘，GPU 可通过 DMA 直接访问，是 Host↔GPU 数据传输的高速通道。
- **I-Cache（Instruction Cache）**：SM 上的指令缓存，缓存从显存代码区取回的 kernel 机器码，避免每次取指都访问显存。

#### 计算与通信原语

- **ALU（Arithmetic Logic Unit）**：执行加减乘除、比较等基本算术和逻辑运算的硬件单元。
- **SFU（Special Function Unit）**：GPU 上专门执行 `exp()`、`sin()`、`cos()`、`rsqrt()` 等超越函数的硬件单元。比 ALU 慢，但比软件模拟快得多。
- **Warp Shuffle**：同一 Warp 内 thread 之间直接交换寄存器数据的硬件指令（如 `__shfl_xor_sync`），不需要经过 shared memory 或 global memory，延迟极低（~1 cycle）。本 kernel 用它做 sub-warp reduce（width=16，即 16 个 thread 一组协作归约）。
- **Coalesced Access（合并访问）**：同一 Warp 的 thread 访问连续的内存地址时，硬件可以把多个 thread 的请求合并成一次内存事务，大幅提高带宽利用率。反之（散乱访问）会产生多次独立事务。
- **VPT（Values Per Thread）**：本 kernel 的模板参数，表示每个 thread 负责处理多少个值。128 experts / 16 threads = VPT=8。

#### Host↔GPU 通信

- **PCIe / NVLink**：Host CPU 与 GPU 之间的物理互连总线。PCIe 是通用接口（~64 GB/s），NVLink 是 NVIDIA 私有高带宽互连（~900 GB/s）。命令和数据都通过它们传输。
- **DMA（Direct Memory Access）**：硬件直接在内存之间传输数据，不需要 CPU 逐字节搬运。GPU Front End 通过 DMA 从 host pinned memory 拉取命令。
- **Pushbuffer**：Host driver 在 pinned memory 中维护的命令缓冲区。CPU 侧把 launch 命令写入 pushbuffer，GPU Front End 通过 DMA 主动拉取（pull 模型）。
- **Doorbell**：Host 写完 pushbuffer 后，通过写一个特殊的 MMIO 寄存器通知 GPU"有新命令了"。GPU Front End 收到 doorbell 后开始拉取命令。
- **GPU Front End**：GPU 内部的命令处理器，负责从 pushbuffer 拉取命令、解析、分发到对应硬件单元（如 CTA 分配器、SM 等）。

#### 编译与二进制

- **CUDA Context**：进程在某张 GPU 上的运行环境（地址空间、模块状态、stream/event 状态等）；通常首次创建，后续复用。
- **模块装载（Module Load）**：driver 把 kernel 对应模块准备成"可执行状态"，并把代码放到 GPU 显存中的代码区。
- **PTX**：NVIDIA 的中间表示（类似汇编），与具体 GPU 架构无关。编译器先把 CUDA C++ 编译成 PTX。
- **SASS**：GPU 的原生机器码，与具体架构绑定（如 sm_80、sm_90）。是 GPU 实际执行的指令。
- **PTX JIT**：当没有可直接执行的目标机器码（SASS）时，driver 在运行时把 PTX 编译成当前 GPU 的 SASS，再执行。
- **fatbin / cubin**：CUDA 可执行文件中嵌入的 GPU 二进制格式。fatbin 可包含多个架构的 cubin + PTX，运行时自动选择匹配的版本。

#### 性能测量

- **NSYS（Nsight Systems）**：NVIDIA 的系统级性能分析工具。底层通过 CUPTI Activity API 采集 kernel 的 GPU 端时间戳（官方：*"CUPTI instruments the kernel code to collect the timing information"*，基于软件插桩与信号量方案）。Duration = `CUpti_ActivityKernel` 记录的 `end − start`，时钟源为 GPU `globaltimer`（~31.25 MHz，量化粒度 ~32 ns）。**运行条件**：应用正常执行，不改变 cache 状态、时钟频率和调度方式。详见 3.6.1 节 A。
- **NCU（Nsight Compute）**：NVIDIA 的 GPU kernel 性能分析工具。Duration = `gpc__cycles_elapsed.max / GPC_clock_frequency`（`.max` 官方定义：*"maximum counter value across all unit instances"*）。**运行条件**（官方文档三大控制）：(1) 串行化 — *"serializes kernel launches"*；(2) 缓存刷新 — *"flushes all GPU caches before each replay pass"*；(3) 时钟锁定 — *"attempts to limit GPU clock frequencies to their base value"*。因此 NCU Duration 反映的是**受控隔离环境下的执行时间**，通常大于正常执行下的 NSYS Duration；对长 kernel 甚至可能大于 cudaEvent（见 3.6 节）。详见 3.6.1 节 B。
- **cudaEvent**：CUDA 的 GPU 时间戳机制。官方定义：*"cudaEventRecord() captures in event the contents of stream at the time of this call"*；*"cudaEventElapsedTime() computes the elapsed time between two events (in milliseconds with a resolution of around 0.5 microseconds)"*；*"The device will record a timestamp for the event when it reaches that event in the stream. The timings are measured on the GPU clock."*（CUDA Best Practices Guide 9.1.2）。详见 3.6.1 节 C。
- **cudaEventSynchronize**：Host 阻塞等待直到指定 event 在 GPU 上被执行完毕。
- **cudaStreamSynchronize**：Host 阻塞等待直到指定 stream 中所有已提交的命令在 GPU 上全部执行完毕。
- **三种 GPU 时间度量的大小关系**：小 kernel 上通常 NSYS Duration < NCU Duration ≤ cudaEvent ms；长 kernel（如 tokens=65536）上 NCU 可能反超 cudaEvent（replay 膨胀 > Front End 开销）。NSYS vs NCU 的差异来自运行条件（正常执行 vs replay），cudaEvent vs NCU/NSYS 的差异来自 Front End + Event 开销。详见 3.6 节分析。

> 后文只写 "launch" 时，默认指 Kernel Launch（Host 侧 enqueue），不等于 GPU 执行时间。

**这三个概念的关系与目的（首次调用常见）**：
1. 先有 `CUDA Context`（先把运行环境建好）。
2. 再做 `模块装载`（让代码进入可执行状态并可被取指）。
3. 必要时做 `PTX JIT`（没有现成机器码时才发生）。
4. 目的：保证 kernel 能在当前 GPU 上正确执行，并把后续调用变成稳态路径。

### 三个概念再说清楚（按“是什么/何时/哪里/对 ms 影响”）

#### A) CUDA Context
- 是什么：进程在某张 GPU 上的“运行环境”（地址空间、stream/event 状态、模块状态等）。
- 何时发生：常见在首次触发 CUDA 路径（第一次 launch 或第一次显式 CUDA 调用）。
- 在哪里：由 CUDA Driver 维护，既有 Host 侧管理状态，也有对应的 GPU 侧运行状态。
- 对 `ms` 的影响：这是 Host 侧成本，不会被 event“直接计时”；但若发生在 `start` 到 `stop` 之间，会通过延迟后续命令提交（形成 GPU 空等间隙）抬高这次 `ms`。

#### B) 模块装载（Module Load）
- 是什么：把可执行文件中的 `fatbin/cubin` 模块准备成“设备可执行状态”。
- 何时发生：常见在首次使用该模块/首次 launch 对应 kernel 时。
- 到哪里：装到 **GPU 显存中的代码区**（Driver 管理，不是业务 tensor 缓冲区）。
- 对 `ms` 的影响：通常只影响首次调用；影响方式同上（间接体现在 `start->stop` 区间里的空等/时序拉长），稳态调用一般不再重复。

#### C) PTX JIT
- 是什么：若没有匹配当前 GPU 的机器码，Driver 在运行时把 PTX 编译成 SASS。
- 何时发生：首次触发该 PTX kernel 且需要 JIT 时。
- 发生位置：编译动作在 Host 路径触发，产物再装入 GPU 显存代码区。
- 对 `ms` 的影响：JIT 本身是 Host 侧编译，不被 event直接计时；但若发生在事件区间内，会延迟 launch/stop 提交，从而显著抬高该次 `ms`。Warmup 可把它移出正式统计。

### 一句话边界（最容易混淆）
- `launch` = Host 把命令入队，不等于 kernel 已执行。
- `cudaEvent` 的 `ms` = GPU 时间线里 `start` 到 `stop` 的总间隔，不是“只算阶段 4”。
- 所以 `ms` 通常包含阶段 `3a + 3b + 3c + 4 + 5a + 5b + 5c`（外加 event 本身开销与可能的空等间隙）。

---

## 第一部分：CUDA Kernel 完整调用流程

### 1. 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              完整调用链路                                    │
│                                                                             │
│   CPU 端                              GPU 端                                │
│   ══════                              ══════                                │
│                                                                             │
│   用户代码                                                             │
│       │                                                                     │
│       ▼                                                                     │
│   CUDA Runtime API                                                          │
│       │                                                                     │
│       │  PCIe                                                               │
│       ▼                                                                     │
│                                    GPU 命令队列                              │
│                                        │                                    │
│                                        ▼                                    │
│                                    GPU 命令处理器                            │
│                                        │                                    │
│                                        ▼                                    │
│                                    SM 执行 Kernel                           │
│                                        │                                    │
│                                        ▼                                    │
│                                    写回结果                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2. 详细步骤分解

**统一粒度约定（下面每个阶段都按同一模板写）**：
- `子步骤`：用 `Sx-y` 标号，避免“阶段”和“细项”混在一起。
- `阶段输出`：该阶段结束后，系统状态发生了什么变化。
- `计时归属`：是否进入 `cudaEvent` 的 `ms` 区间。

**阶段与第三部分分项的固定映射（先定死，后文不跳表述）**：
1. `Event Start/Stop` <-> `第三部分 2.1 Event 处理`
2. `阶段 1` <-> 无独立 GPU 分项（Host 准备路径）
3. `阶段 2` <-> 无独立 GPU 分项（Host enqueue；只可能通过“间隙”间接影响 `ms`）
4. `阶段 3a` <-> `第三部分 2.2 Front End 命令处理`（**不在** NCU/NSYS 内）
5. `阶段 3b` <-> `第三部分 2.3 GPC dispatch`（**在** NCU 内 — NCU 起点，**不在** NSYS 内）
6. `阶段 3c` <-> `第三部分 2.3 取指/首 warp 发射`（**在** NCU 和 NSYS 内 — NSYS 起点）
7. `阶段 4` <-> `第三部分 2.4 数据加载` + `2.5 计算` + `2.6 结果写回`（**在** NCU 和 NSYS 内）
8. `阶段 5a` <-> `第三部分 2.7a SM 内部完成`（**在** NCU 和 NSYS 内 — NSYS 终点）
9. `阶段 5b` <-> `第三部分 2.7b GPC retirement`（**在** NCU 内 — NCU 终点，**不在** NSYS 内）
10. `阶段 5c` <-> `第三部分 2.7c 全局完成发布`（**不在** NCU/NSYS 内）

#### 阶段 1：CPU 端函数调用

```
用户代码调用
─────────────

    topk_softmax(weights_d, indices_d, nullptr, gating_d, nullptr,
                 tokens, experts, topk, dtype_enum, false, 0);
        │
        ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  void topk_softmax(...)                                             │
    │  {                                                                  │
    │      // 1. 参数校验和准备工作                                        │
    │      const bool is_pow_2 = (num_experts & (num_experts - 1)) == 0;  │
    │      const bool needs_workspace = !is_pow_2 || num_experts > 256;   │
    │                                                                     │
    │      // 2. 类型分发                                                  │
    │      if (dtype == Float16) {                                        │
    │          dispatch_topk_softmax_launch<__half>(...);                 │
    │      }                                                              │
    │      ...                                                            │
    │  }                                                                  │
    └─────────────────────────────────────────────────────────────────────┘
        │
        ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  void dispatch_topk_softmax_launch<__half>(...)                     │
    │  {                                                                  │
    │      topkGatingSoftmaxKernelLauncher<int, __half>(...);             │
    │  }                                                                  │
    └─────────────────────────────────────────────────────────────────────┘
        │
        ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  void topkGatingSoftmaxKernelLauncher(...)                          │
    │  {                                                                  │
    │      // 计算启动配置                                                 │
    │      const int num_blocks = (num_warps + 3) / 4;                    │
    │      dim3 block_dim(32, 4);  // 128 threads per block               │
    │                                                                     │
    │      // 调用 kernel                                                  │
    │      topkGatingSoftmax<<<num_blocks, block_dim, 0, stream>>>(...);  │
    │  }                                                                  │
    └─────────────────────────────────────────────────────────────────────┘
```

**子步骤（S1-x）：**
1. `[S1-1]` 解析入参与基本条件判断（`num_experts`、`topk`、dtype 路径等）。
2. `[S1-2]` 判定 workspace 需求（如 `needs_workspace`）。
3. `[S1-3]` 进行类型分发（fp32/fp16/bf16 -> 对应模板实例）。
4. `[S1-4]` 计算 launch 配置（`num_blocks`、`block_dim`）。
5. `[S1-5]` 准备进入 `<<<...>>>` 提交点（进入阶段 2）。

**阶段输出：**
- 得到确定的 kernel 入口、参数布局、grid/block 配置。

**计时归属：**
- 纯 Host 侧路径，不直接计入 `cudaEvent` 的 `ms`。

---

#### 阶段 2：Host Kernel Launch（enqueue，CPU → GPU）

```
Host Launch 流程（统一样式）
────────────────────────────

    topkGatingSoftmax<<<num_blocks, block_dim, 0, stream>>>(args...);
        │
        ▼
    [S2-1] Host Runtime 打包 launch 参数
        │
        ▼
    [S2-2] 判断：是否首次调用该 kernel？
        │
        ├─ 是（首次） ─▶ [S2-2a] 初始化 CUDA Context（第一次创建，后续复用）
        │               [S2-2b] PTX 路径先 JIT 到机器码（如需要）
        │               [S2-2c] 将 kernel 机器码装入 GPU 显存中的代码区
        │                        （Driver 管理，区别于业务数据区）
        │
        └─ 否（非首次） ─▶ 跳过一次性装载路径
        │
        ▼
    [S2-3] 构造并提交 Launch 命令（grid/block/参数指针/stream）
        │
        ▼
    [S2-4] 写入 Host Pinned Memory 中的 Pushbuffer + Doorbell 通知
           - 命令缓冲区位于 host 锁页内存，GPU Front End 通过 PCIe/NVLink DMA 拉取
        │
        ▼
    [S2-5] Host 立即返回（异步，不等待 kernel 完成）
```

**子步骤（S2-x）：**
1. `[S2-1]` Host runtime 打包 launch 元数据。
2. `[S2-2]` 首次调用分支：可能触发 Context/JIT/模块装载。
3. `[S2-3]` 构造 launch 命令并提交到 driver/runtime 队列。
4. `[S2-4]` 写入 host pinned memory 中的 pushbuffer 并 doorbell 通知。
5. `[S2-5]` Host 返回（异步语义，不等执行完成）。

**阶段输出：**
- GPU 端已经“看得见”一条待执行 launch 命令，等待命令处理器消费。

**关键澄清：**
- 这里写入的主要是**命令/参数元数据 +（仅首次）代码装载结果**。
- 其中命令写入目标是 **host pinned memory（锁页内存）中的 pushbuffer**，GPU Front End 通过 PCIe/NVLink DMA 从中拉取命令。
- `gating/weights/indices` 这类业务数据不在该阶段搬运（应在 launch 前已位于 device memory）。
- 首次调用的代码装载目标位置是 GPU 显存中的代码区（Driver 管理），不是业务 tensor 缓冲区。

**计时归属：**
- Host launch 开销（CPU 发起与提交）本身不直接进入 `cudaEvent` 的 `ms`。
- 但若发生在 `start` 与 `stop` 之间并拖慢后续提交，会以 GPU 空等间隙反映到 `ms`。

**和第三部分的一一映射：**
- 阶段 2 在第三部分没有独立 GPU 分项；仅在 cudaEvent ms 的"间隙/其他"中以间接形式出现。

---

#### 阶段 3a：Front End 命令处理（不在 NCU/NSYS 内，仅 cudaEvent）

```
Front End 命令处理流程
──────────────────────

    [S3-0] GPU 收到 Doorbell
        │
        ▼
    [S3-1] 命令获取（GPU Front End 通过 PCIe/NVLink 从 host pushbuffer 拉取 launch 命令）
        │
        ▼
    [S3-2] 命令解析（函数地址 / grid / block / 参数指针）
        │
        ▼
    ── 交接给 CTA 分配器 → 进入阶段 3b ──
```

**子步骤（S3-1~S3-2）：**
1. `[S3-1] 命令读取`  
   - 做什么：GPU Front End 通过 PCIe/NVLink DMA 从 host pinned memory 的 pushbuffer 拉取本次 launch 描述符。  
   - 输入/输出：输入是 pushbuffer 中的队列项；输出是可解析的 launch 包。  
   - 为什么耗时：需要走 PCIe/NVLink DMA 读取路径。
2. `[S3-2] 命令解析`  
   - 做什么：解析 kernel 入口、grid/block、参数地址、stream 顺序关系。  
   - 输入/输出：输入是 launch 包；输出是“可调度任务描述”。  
   - 为什么耗时：需要做字段校验与硬件可执行格式转换。

**阶段输出：**
- 命令已被 Front End 解析，准备交给 CTA 分配器进行 SM 端资源分配。

**计时归属：**
- ✓ cudaEvent：计入（在 start→stop 区间内）。
- ✗ NCU Duration：**不计入**。这是 Front End 路径，不属于 kernel 在 GPC/SM 上的生命周期。
- ✗ NSYS Duration：**不计入**。NSYS 仅测 kernel 代码执行（插桩入口→出口），Front End 路径不在其覆盖范围内。
- 这就是 empty kernel 差值 ~2.2 μs 的主要来源：Front End 命令处理 + Event 打点开销在 kernel 极短时几乎无法被覆盖。
- 当 kernel 执行时间足够长（如 topk 6.56 μs），Front End 完成 3a 后可立即处理后续命令（如 E-stop），这些工作与 SM 执行 kernel **并行进行**（详见 3.5 节流水线图解），所以 cudaEvent 差值只有 ~0.2 μs。

---

#### 阶段 3b：SM 端启动准备（在 NCU 内，NCU Duration 起点）

```
SM 端启动准备流程
────────────────

    ── NCU Duration 起点 ──
        │
        ▼
    [S3-3] 资源分配（CTA 分派到 SM / 寄存器 / shared memory）
        │
        ▼
    [S3-4] 取指与 I-Cache 装填
           - 从 GPU 显存中的代码区取指
           - 经 L2 装填到 SM 指令缓存
           - 首次/非首次调用都会执行该步骤
        │
        ▼
    [S3-5] 首批 Warp 发射 → 进入阶段 4
```

**子步骤（S3-3~S3-5）：**
3. `[S3-3] 资源可行性检查与分配`
   - 做什么：检查寄存器/shared memory/可用 SM 配额，建立首批 CTA 的资源映射。
   - 输入/输出：输入是任务描述；输出是"可发射资源上下文"。
   - 为什么耗时：硬件调度器要做占用与并发可行性判断。
4. `[S3-4] 取指与 I-Cache 装填`
   - 做什么：从代码区取首批指令，经 L2 装填到 SM 指令缓存。
   - 输入/输出：输入是 kernel PC/入口；输出是"可取指执行"的指令缓存状态。
   - 为什么耗时：首次取指或 I-Cache 命中不佳会增加延迟。
5. `[S3-5] 发射门槛判定`
   - 做什么：确认依赖与资源都就绪，发射首批 warp，切入阶段 4。
   - 输入/输出：输入是可发射上下文；输出是"执行中"状态。

**阶段输出：**
- kernel 已具备"可发射"条件，首批 warp 已进入执行入口（阶段 4）。

**补充边界：**
- 首次调用的"模块级代码装载/JIT"主要在阶段 2 的 Host 路径触发。
- 本阶段只覆盖"CTA 分派到 SM → 首批 warp 发射"这段 SM 端路径。

**计时归属：**
- ✓ cudaEvent：计入。
- ✓ NCU Duration：**计入**（这是 NCU Duration 的起始段）。
- empty kernel 的 Elapsed Cycles 2526 中，绝大部分（~2500 cycle）是本阶段开销（SM 上下文配置 + I-Cache 装填），SM Active Cycles 仅 4.37 cycle。

---

#### 阶段 4：Kernel 执行

```
Kernel 在 SM 上执行
───────────────────

    以 topkGatingSoftmax 为例（128 experts, 1 token, topk=8, bf16）
        │
        ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                                                                     │
    │  执行形状（由阶段 1 已确定）：                                      │
    │  ─────────────────────────                                          │
    │  - ELTS_PER_LDG = 16 / sizeof(bf16) = 8                            │
    │  - VPT = 8（每 thread 处理 8 个 expert 值）                         │
    │  - THREADS_PER_ROW = 128 / 8 = 16                                   │
    │  - ROWS_PER_WARP = 32 / 16 = 2                                     │
    │  - WARPS_PER_TB = 4                                                 │
    │  - ROWS_PER_CTA = 4 × 2 = 8                                        │
    │  - num_warps = ceil(1 / 2) = 1                                      │
    │  - num_blocks = ceil(1 / 4) = 1                                     │
    │                                                                     │
    │  Grid (1,1,1) × Block (32,4,1) = 1 个 Block, 128 个 thread          │
    │  实际只有 1 个 warp 中的 16 个 thread 处理唯一的 1 个 token          │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
        │
        ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                                                                     │
    │  4.1 数据加载 (Data Loading)                                        │
    │  ──────────────────────────                                         │
    │                                                                     │
    │  16 个 thread 协作加载 128 个 expert 值（每个 thread 加载 VPT=8 个）： │
    │  - 从 Global Memory 读取 gating[token_id * 128 + expert_id]         │
    │  - 通过 L2 Cache → L1 Cache → 寄存器                                │
    │  - 数据量：1 token × 128 experts × 2 bytes (bf16) = 256 bytes       │
    │                                                                     │
    │  内存访问模式：                                                     │
    │  - Coalesced access（合并访问）                                     │
    │  - 同一 warp 的 32 个 thread 访问连续内存                           │
    │  - 有效利用内存带宽                                                 │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
        │
        ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                                                                     │
    │  4.2 Softmax 计算 (Softmax Computation)                             │
    │  ────────────────────────────────────                               │
    │                                                                     │
    │  a) 找最大值（Max Reduction）：                                     │
    │     thread_max = max(row_chunk[0..7])   // 每个 thread VPT=8       │
    │     group_max = shuffle_xor_reduce(thread_max, width=16)            │
    │                                                                     │
    │  b) 计算 exp 和求和：                                               │
    │     row_sum = 0                                                     │
    │     for (i in 0..7):          // VPT=8                              │
    │         row_chunk[i] = exp(row_chunk[i] - group_max)                │
    │         row_sum += row_chunk[i]                                     │
    │     group_sum = shuffle_xor_reduce(row_sum, width=16)               │
    │                                                                     │
    │  c) 归一化：                                                        │
    │     for (i in 0..7):                                                │
    │         row_chunk[i] /= group_sum                                   │
    │                                                                     │
    │  计算特点：                                                         │
    │  - 纯寄存器操作，非常快                                             │
    │  - Warp shuffle 进行 reduce（width=16，即 sub-warp reduce）          │
    │  - exp() 使用 SFU（Special Function Unit）                          │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
        │
        ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                                                                     │
    │  4.3 TopK 选择 (TopK Selection)                                     │
    │  ──────────────────────────                                         │
    │                                                                     │
    │  迭代 k 次（k=8），每次找一个最大值：                               │
    │                                                                     │
    │  for (k_idx = 0; k_idx < 8; k_idx++) {                              │
    │      // a) 在本 thread 的数据中找最大值（VPT=8 个元素）              │
    │      max_val = row_chunk[0]                                         │
    │      expert = start_col                                             │
    │      for (i in 0..7):                                               │
    │          if (row_chunk[i] > max_val):                               │
    │              max_val = row_chunk[i]                                 │
    │              expert = i                                             │
    │                                                                     │
    │      // b) 在 thread group 范围内找全局最大值（width=16）            │
    │      shuffle_xor_reduce_argmax(max_val, expert, width=16)           │
    │                                                                     │
    │      // c) 记录结果                                                 │
    │      if (thread_group_idx == 0):                                    │
    │          output[k * token + k_idx] = max_val                        │
    │          indices[k * token + k_idx] = expert                        │
    │                                                                     │
    │      // d) 清除已选中的值，准备下一次迭代                           │
    │      row_chunk[expert] = -10000.f                                   │
    │  }                                                                  │
    │                                                                     │
    │  计算特点：                                                         │
    │  - 8 次迭代，每次每 thread 8 次比较 + 1 次 sub-warp reduce（width=16）│
    │  - 纯 ALU 操作（+ shuffle），非常快                                  │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
        │
        ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                                                                     │
    │  4.4 结果写回 (Result Writeback)                                    │
    │  ──────────────────────────                                         │
    │                                                                     │
    │  写入三个输出数组：                                                 │
    │  - weights[1 × 8] = 32 bytes (float)                               │
    │  - indices[1 × 8] = 32 bytes (int)                                 │
    │  - source_rows[1 × 8] = 32 bytes (int)                             │
    │  - 总计：96 bytes                                                     │
    │                                                                     │
    │  写入模式：                                                         │
    │  - Coalesced write（合并写入）                                      │
    │  - 从寄存器 → L1 → L2 → DRAM                                        │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
```

**子步骤（S4-x）：**
1. `[S4-1]` 数据加载：从 global memory 读入 `gating` 到寄存器路径。
2. `[S4-2]` Softmax：max-reduce + exp/sum + normalize。
3. `[S4-3]` TopK：迭代 argmax + 清零已选位置。
4. `[S4-4]` 结果写回：`weights/indices` 回写显存。

**阶段输出：**
- 目标输出张量已写回（对后续命令可见）。

**和第三部分的一一映射：**
1. `S4-1` <-> `第三部分 2.4 数据加载`
2. `S4-2/S4-3` <-> `第三部分 2.5 计算`
3. `S4-4` <-> `第三部分 2.6 结果写回`

**计时归属：**
- ✓ cudaEvent：计入。
- ✓ NCU Duration：**计入**。
- NCU Duration = 阶段 3b + 3c + **阶段 4** + 阶段 5a + 5b（GPC 全部周期，含 dispatch/retirement 开销）。
- NSYS Duration = 阶段 3c + **阶段 4** + 阶段 5a（kernel 代码执行，不含 GPC dispatch/retirement 开销）。

---

#### 阶段 5a：SM 内部完成（在 NCU 和 NSYS 内，NSYS Duration 终点）

**先回答"为什么执行结束后还需要这个阶段"**：
- "执行结束"只表示 kernel 指令跑完，不等于系统层面的"调用已完成"。
- 资源（寄存器/shared memory/调度槽位）需要回收，否则后续 kernel 不能稳定调度。

```
SM 内部完成流程
──────────────

    阶段 4 执行期间，资源回收已在持续进行：
    ┌─────────────────────────────────────────────────────────────────────┐
    │  S5-0 Per-CTA 资源回收（与阶段 4 交织发生）                        │
    │  - 每个 CTA 执行完毕后，其占用的寄存器/shared memory/调度槽位       │
    │    立即释放，可被同一 SM 上的后续 CTA 或其他 kernel 复用             │
    │  - 这是流式/增量的，不是所有 block 完成后的一次性批量操作           │
    └─────────────────────────────────────────────────────────────────────┘

    最后一个 CTA 执行完毕
        │
        ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  S5-1 完成判定 (Completion Point)                                   │
    │  - 所有已 launch 的 CTA 全部结束                                    │
    │  - 只有最慢的 CTA 结束后，kernel 才算真正完成                       │
    │  - 此时所有 per-CTA 资源已在先前逐步回收完毕                        │
    └─────────────────────────────────────────────────────────────────────┘
        │
        ▼
    ── NSYS Duration 终点（推断：CUPTI 插桩出口代码记 end）──
```

**子步骤（S5-0~S5-1）：**
1. `[S5-0]` Per-CTA 资源回收（与阶段 4 交织）：每个 CTA 结束后其占用的寄存器/shared memory/调度槽位立即释放。
2. `[S5-1]` 完成判定：最后一个 CTA 结束，warp 退出，CTA 完成信号发送。

**计时归属：**
- ✓ cudaEvent：计入。
- ✓ NCU Duration：**计入**。
- ✓ NSYS Duration：**计入**（这是 NSYS Duration 的终止段，CUPTI 插桩出口代码在此记录 end 时间戳）。

---

#### 阶段 5b：GPC Retirement（在 NCU 内，NCU Duration 终点；不在 NSYS 内）

```
GPC Retirement 流程
──────────────

    ┌─────────────────────────────────────────────────────────────────────┐
    │  S5-1b CTA 完成信号 + GPC retirement                             │
    │  - CTA 完成信号从 SM 发回 GPC 级控制器                          │
    │  - GPC 层面的资源回收和状态更新                              │
    │  - GPC cycles_elapsed 停止计数                               │
    └─────────────────────────────────────────────────────────────────────┘
        │
        ▼
    ── NCU Duration 终点（推断：GPC cycles_elapsed 停止计数）──
```

**计时归属：**
- ✓ cudaEvent：计入。
- ✓ NCU Duration：**计入**（这是 NCU Duration 的终止段，GPC cycles_elapsed 在此停止计数）。
- ✗ NSYS Duration：**不计入**。NSYS 的 CUPTI 代码内插桩已在阶段 5a 记录 end，GPC retirement 开销不在其覆盖范围内。

---

#### 阶段 5c：全局完成发布（不在 NCU/NSYS 内，仅 cudaEvent）

```
全局完成发布流程
──────────────

    ┌─────────────────────────────────────────────────────────────────────┐
    │  S5-2 完成发布 (Completion Publish)                                 │
    │  - 更新 stream 中该 kernel 的完成状态                                │
    │  - 若有 cudaEvent，写入 event 时间戳                                 │
    │  - 若 CPU 正在等待，同步原语据此解除等待                              │
    │  - 这是对 Host 可见的"完成发布点"                                    │
    └─────────────────────────────────────────────────────────────────────┘
```

**子步骤（S5-2）：**
3. `[S5-2]` 完成发布：stream/event/等待方看到"已完成"状态。

**计时归属：**
- ✓ cudaEvent：计入。
- ✗ NCU Duration：**不计入**。这属于 GPU 全局控制路径，已超出 GPC 生命周期。
- ✗ NSYS Duration：**不计入**。同上。

**阶段输出：**
- 对同一 stream 的后续命令，前序 kernel 已完成这一事实可见。

**是否任何 kernel 都需要 5a + 5b + 5c？**
- 需要。这是 GPU/runtime 的通用收尾逻辑，不是 `topk_softmax` 特有代码。

**`cudaStreamSynchronize(stream)` 到底在等什么？**
1. 等阶段 5a（S5-1）：所有已 launch 的 CTA 全部执行完，最后 warp 退出。
2. 等阶段 5b（S5-1b）：GPC retirement 完成。
3. 等阶段 5c（S5-2）：完成状态被发布到 stream（必要时 event 时间戳也写入）。
4. 上述 1~3 都满足后，`cudaStreamSynchronize(stream)` 才返回。

**源码补充（当前仓库实现）：**
- `src/topk_softmax.cu:672` 有 `cudaStreamSynchronize(stream)`，因此 `topk_softmax` 返回前会等待"5a → 5b"整条链路完成。
- 这属于封装层额外语义：底层 kernel launch 仍是异步，封装函数把返回时机改成"完成后再返回"。

---
## 第二部分：时间统计方法分析

### 1. 测试代码

```cpp
// 来自 tests/bench_perf.cpp

cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

// Warmup - 消除首次调用的额外开销
for (int i = 0; i < warmup; i++) {
    topk_softmax_async(...);              // 纯异步，无内部同步
}
cudaDeviceSynchronize();

// 正式测量
for (int i = 0; i < iters; i++) {
    cudaEventRecord(start);                    // 记录起点
    topk_softmax_async(...);                   // 调用 kernel（纯异步）
    cudaEventRecord(stop);                     // 记录终点
    cudaEventSynchronize(stop);                // 等待完成

    float ms;
    cudaEventElapsedTime(&ms, start, stop);    // 计算时间差
    total_time_ms += ms;
}
```

**计时循环细拆（M-x）：**
1. `[M-1]` 创建 `start/stop` event（一次性）。
2. `[M-2]` 执行 warmup（不计入 `total_time_ms`）。
3. `[M-3]` `cudaEventRecord(start)`：把“开始打点命令”入队。
4. `[M-4]` 调用 `topk_softmax_async(...)`：纯异步 kernel launch，无内部同步。
5. `[M-5]` `cudaEventRecord(stop)`：把“结束打点命令”入队。
6. `[M-6]` `cudaEventSynchronize(stop)`：Host 等到 stop 事件真正执行完成。
7. `[M-7]` `cudaEventElapsedTime(start, stop)`：读取 GPU 时间线间隔并累加。

### 2. cudaEvent 是什么（先说人话）

- `cudaEventRecord(start)`：往指定 stream 里插入一个“开始打点命令”。
- `cudaEventRecord(stop)`：往同一 stream 里插入一个“结束打点命令”。
- `cudaEventElapsedTime(start, stop)`：返回这两个打点在 **GPU 时间线** 上的时间差。

> 关键点：`cudaEvent` 量的是“同一 stream 上 start->stop 之间的时间区间”，不是单纯 CPU 函数耗时。

```
CPU 提交顺序（同一 stream）：
    record(start) -> topk_softmax_async(...) -> record(stop)

GPU 看到的时间线：
    [start打点] -> (可能空等) -> 命令处理/启动 -> kernel执行 -> 完成发布 -> (可能空等) -> [stop打点]
```

**细节补充（避免误解）：**
1. event 不是“CPU 立即取时间戳”，而是 GPU 执行到该 event 命令时才打点。
2. `cudaEventSynchronize(stop)` 等的是“stop 已被 GPU 执行”，不是只等 CPU 提交完成。
3. 只有当 `start/stop` 在同一 stream 的正确顺序上，`ElapsedTime` 才有可比性。

### 3. 这个 `ms` 到底是什么时间

**结论**：`ms` 不是只等于“阶段 4：Kernel 执行时间”。  
它是 `start` 到 `stop` 两个 event 在 GPU 时间线上的总间隔。

**在本工程这段 benchmark 里，`ms` 通常由这些部分构成：**
- `Event Start` 打点处理时间。
- `阶段 3a`：Front End 命令处理（不在 NCU/NSYS 内）。
- `阶段 3b`：GPC dispatch（在 NCU 内，不在 NSYS 内）。
- `阶段 3c`：取指/首 warp 发射（在 NCU 和 NSYS 内）。
- `阶段 4`：Kernel 执行（真正算子计算，在 NCU 和 NSYS 内）。
- `阶段 5a`：SM 内部完成（在 NCU 和 NSYS 内）。
- `阶段 5b`：GPC retirement（在 NCU 内，不在 NSYS 内）。
- `阶段 5c`：全局完成发布（不在 NCU/NSYS 内）。
- `Event Stop` 打点处理时间。

> 注：`bench_perf` 使用的是 `topk_softmax_async()`，内部无 `cudaStreamSynchronize`，也无 `cudaMalloc/cudaFree`，
> 因此 start 到 stop 之间不会出现由封装层导致的 GPU 空等间隙。

**所以为什么看起来包住 `topk_softmax_async(...)`，却不是“只测阶段4”？**
- 因为 event 测的是 start->stop 之间整个 GPU 时间区间，不会自动只截取 kernel 本体。

**如果想更接近“纯阶段4”**：
- 用 profiler 看 kernel `duration`（如 Nsight）。
- 或把打点尽量贴近真实 `<<<>>>` 提交点，并去掉封装内同步/内存管理对时序的影响。

**边界公式（统一口径）：**
- `ms_event = T_gpu_exec(stop_event) - T_gpu_exec(start_event)`
- `t_wall = T_cpu_return(topk_softmax) - T_cpu_call(topk_softmax)`（这是另一个量，不等于 `ms_event`）

**包含/不包含（按当前写法）：**
1. 包含：`阶段 3a + 3b + 3c + 4 + 5a + 5b + 5c + event 命令执行 + 可能空等间隙`。
2. 不直接包含：`start 前`和`stop 后`的 Host 代码执行。
3. 可能间接反映：`start~stop` 期间 Host 提交变慢导致的 GPU 空等。

### 4. Warmup 的作用

```
Warmup（不计时）：
    for i in warmup:
        topk_softmax_async(...)
    cudaDeviceSynchronize()

正式计时：
    record(start) -> topk_softmax_async(...) -> record(stop)
```

**这段要回答“Warmup 对你测到的 `ms` 到底影响什么”**：
- 你的 `ms` 定义是：`ms = T(stop_event_on_gpu) - T(start_event_on_gpu)`，也就是 start/stop 两个 event 在 GPU 时间线上的间隔。
- Warmup 会影响 `ms` 的部分：首次调用才出现的一次性成本（Context 初始化、模块装载、PTX JIT 等）。这些属于 Host 侧，不会被 event 直接计时；但如果发生在 `start` 到 `stop` 之间，会通过“延迟 launch/stop 提交 -> GPU 空等间隙”抬高前几次 `ms`。
- Warmup 不会消除的部分：每次调用都发生的开销（GPU 命令处理、kernel 执行、完成发布）。
- 本工程 `bench_perf` 使用 `topk_softmax_async`，内部无 `cudaMalloc/cudaFree`，因此稳态下无额外分配/释放开销。
- 一句话：Warmup 主要“去掉首轮偏大值”，不会“去掉稳态每次调用成本”。

**Warmup 细拆（W-x）：**
1. `[W-1]` 先跑若干次 `topk_softmax_async`，把一次性路径尽量前置消耗。
2. `[W-2]` `cudaDeviceSynchronize()`，确保 warmup 的 GPU 工作全部收尾。
3. `[W-3]` 正式开始 event 计时，只统计稳态区间。
4. `[W-4]` `bench_perf` 使用 `topk_softmax_async`，无内部分配/同步，稳态开销干净。

---

## 第三部分：时间开销详细分解（与前文阶段对齐）

### 1. 先对齐：`ms` 与“阶段 1~5”怎么对应

- 你测的 `ms` 是 `start/stop` 两个 event 在 GPU 时间线上的间隔。
- 与前文阶段的映射是：
  1. `阶段 2（Host launch）`：CPU 入队，不直接计入 `ms`。
  2. `Event Start`：GPU 打点。
  3. `阶段 3a`：Front End 命令处理。
  4. `阶段 3b`：GPC dispatch（CTA 接收、资源分配）。
  5. `阶段 3c`：取指/I-Cache 装填、首 warp 发射。
  6. `阶段 4`：Kernel 执行（数据加载 + 计算 + 写回）。
  7. `阶段 5a`：SM 内部完成（最后 warp 退出，CUPTI 插桩出口记 end，per-CTA 资源回收与阶段 4 交织）。
  8. `阶段 5b`：GPC retirement（CTA 完成信号，GPC cycles_elapsed 停止计数）。
  9. `阶段 5c`：全局完成发布（stream 状态、event 时间戳）。
  10. `Event Stop`：GPU 打点。
  11. 可能还有"间隙"：Host 后续命令提交延迟导致的 GPU 空等时间。

在本仓库里，`bench_perf` 使用 `topk_softmax_async`，内部无 `cudaStreamSynchronize` 且无 `cudaMalloc/cudaFree`，因此稳态下不存在封装层引入的空等间隙。

> **NCU 实测值 vs cudaEvent 测量值**：
> - **NCU Duration** = `gpc__cycles_elapsed.max / GPC Frequency`，测的是 kernel 在 GPC 上的全部周期（含 dispatch/retirement 开销）：阶段 3b + 3c + 阶段 4 + 阶段 5a + 5b。
> - **NSYS Duration** = CUPTI 代码内插桩 `end − start`，仅测 kernel 代码执行：阶段 3c + 阶段 4 + 阶段 5a。
> - **cudaEvent ms** = GPU 时间线上 start→stop 总间隔，额外包含 Event 打点 + Front End 命令处理 + 完成发布。

**topk kernel 实测数据（128 experts, 1 token, bf16, topk=8, A40）**：
```
NCU Duration = 6.56 μs
  内含：GPC 端启动 + 数据加载/计算/写回 + GPC 端完成
  公式：gpc__cycles_elapsed.max / GPC Frequency = 8497 / 1.30GHz

cudaEvent ms = 6.773 μs  (compare_baseline.sh, warmup=1000, iters=1000)
  内含：Event打点 + Front End命令处理 + NCU Duration全部 + 完成发布
  差值：6.773 - 6.56 = 0.21 μs
```

**为什么 topk 的差值这么小（~0.2 μs），而 empty kernel 差值很大（~2.2 μs）？**
- Front End 和 SM 是独立硬件，可以并行工作（详见前文 3.5 节流水线图解）。
- kernel 够长（6.56 μs）时：Front End 对 E-stop 的拉取/解析在 SM 执行 kernel 期间已完成，kernel 结束后只需写时间戳 → 差值小。
- kernel 极短（~2 μs）时：Front End 的工作无法被 kernel 执行时间覆盖 → 差值大。

**empty kernel 实测对比**：
```
NCU Duration ≈ 2.0 μs（SM Active Cycles 仅 4.37 cycle，其余 ~2500 cycle 是 SM 端启动/等待/完成）
cudaEvent   = 4.247 μs（compare_baseline.sh 实测）
差值        = 2.25 μs（= GPU Front End 命令处理 + Event 打点）
```

`Kernel 启动` 不是独立新阶段，它属于 `阶段 3a + 3b` 的子过程。

### 2. 每个部分的详细分析（按阶段顺序）

#### 2.1 Event 处理（~0.2 μs，对应 Event Start/Stop）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Event Start + Event Stop                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  做什么：                                                                   │
│  - GPU 读取 event 命令                                                      │
│  - 读取 GPU 内部高精度时钟寄存器                                            │
│  - 把时间戳写入 event 对应的内存位置                                        │
│                                                                             │
│  为什么需要时间：                                                           │
│  - 读取时钟寄存器：几个时钟周期                                             │
│  - 写入显存：需要经过存储层次                                               │
│                                                                             │
│  最低下限：~0.05 μs（纯硬件操作）                                           │
│  实际开销：~0.1-0.2 μs（两个 event）                                        │
│  能否优化：不可优化（硬件固有操作）                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 2.2 Front End 命令处理（~1-2 μs，对应阶段 3a）

> **计时归属：✗ NCU Duration（不计入）| ✗ NSYS Duration（不计入）| ✓ cudaEvent（计入）。本节对应阶段 3a（Front End 路径），不属于 GPC/SM 生命周期。**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  GPU 命令队列处理和解析（Front End 路径）                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  做什么：                                                                   │
│  1. GPU Front End 从 host pushbuffer 拉取命令（通过 PCIe/NVLink DMA）       │
│  2. 解析命令类型和参数                                                      │
│  3. 读取 kernel 元数据（grid/block 配置等），转发给 CTA 分配器              │
│                                                                             │
│  为什么需要这么长时间：                                                     │
│  - 命令缓冲区在 host 锁页内存中，需通过 PCIe/NVLink DMA 拉取               │
│  - 命令需要在多个硬件单元间传递（Front End → CTA 分配器）            │
│                                                                             │
│  最低下限：~0.5-1 μs                                                        │
│  实际开销：~1-2 μs                                                          │
│  能否优化：不可优化（GPU 架构决定）                                         │
│                                                                             │
│  关键点：                                                                   │
│  - 这是 GPU 工作的基础开销，任何 kernel 都有                                │
│  - 同一 stream 内按提交顺序执行；不同 stream 可能并发                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 2.3 SM 端启动准备（~0.5-1 μs，对应阶段 3b + 3c）

> **计时归属：✓ NCU Duration（计入，NCU 起点）| ✓ cudaEvent（计入）。阶段 3b 不在 NSYS 内（GPC dispatch），阶段 3c 在 NSYS 内（NSYS 起点，取指/首 warp 发射）。**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  GPU 侧启动和初始化（SM 端路径，非 Host launch）                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  做什么：                                                                   │
│  1. 为 CTA 分配寄存器与 shared memory（S3-3）                           │
│  2. 取指并装填 I-Cache（S3-4）                                          │
│  3. 建立首批 warp 的调度状态（S3-5）                                 │
│  4. 首个 warp 发射（阶段 4 入口）                                           │
│                                                                             │
│  为什么需要这么长时间：                                                     │
│  - 代码取指与缓存装填有访问延迟                                             │
│  - CTA 资源分配与占用检查需要硬件调度协调                                   │
│  - 从“命令已解析”到“首条指令发射”存在启动路径                               │
│                                                                             │
│  最低下限：~0.3-0.5 μs                                                      │
│  实际开销：~0.5-1 μs                                                        │
│  能否优化：基本不可优化（硬件自动完成）                                     │
│                                                                             │
│  特殊情况：                                                                 │
│  - 如果 kernel 代码较大或 I-Cache 命中差，启动会变长                        │
│  - 如果代码已在缓存中，启动会更短                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**阶段映射一句话：**
- 本节对应 `阶段 3b`（S3-3~S3-5），最后“首个 warp 发射”是 `阶段 4` 的入口，不是新增阶段。

#### 2.4 数据加载（~0.5-1 μs，对应阶段 4）

> **计时归属：✓ NCU Duration（计入）| ✓ cudaEvent（计入）。本节对应阶段 4。**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  从显存读取输入数据                                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  数据量（本测试配置：1 token, 128 experts）：                               │
│  - 输入：1 token × 128 experts × 2 bytes (bf16) = 256 bytes                │
│                                                                             │
│  访问模式：                                                                 │
│  - Coalesced access（合并访问）                                             │
│  - 同一 warp 的 32 个 thread 访问连续内存                                   │
│                                                                             │
│  内存层次：                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                                                                     │  │
│  │  DRAM (HBM) ──▶ L2 Cache ──▶ L1 Cache ──▶ 寄存器                    │  │
│  │                                                                     │  │
│  │  延迟：        延迟：         延迟：        延迟：                    │  │
│  │  ~200-400ns   ~50-100ns     ~10-20ns     ~0ns                       │  │
│  │                                                                     │  │
│  │  带宽：        带宽：         带宽：        带宽：                    │  │
│  │  ~3 TB/s      ~3 TB/s        ~10 TB/s     ~10 TB/s                  │  │
│  │                                                                     │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  理论最低时间（纯带宽限制）：                                               │
│  - 256 bytes / 3 TB/s ≈ 0.085 ns                                           │
│                                                                             │
│  实际开销：~0.5-1 μs（受延迟主导，非带宽限制）                              │
│                                                                             │
│  为什么比理论值高：                                                         │
│  - 首次访问的 cache miss 延迟                                               │
│  - Memory controller 调度和排队                                             │
│  - L2 Cache 竞争                                                           │
│                                                                             │
│  最低下限：~0.05 μs（假设全部命中 L2）                                      │
│  实际开销：~0.5-1 μs                                                        │
│  能否优化：可优化但空间有限（已是较优访问模式）                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 2.5 计算（~1-2 μs，对应阶段 4）

> **计时归属：✓ NCU Duration（计入）| ✓ cudaEvent（计入）。本节对应阶段 4。**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Softmax 计算 + TopK 选择                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  计算内容：                                                                 │
│                                                                             │
│  1. Softmax（每个 token，16 个 thread 协作）：                                  │
│     - 每个 thread 找 VPT=8 个值中的最大值，shuffle reduce 跨 16 threads     │
│     - 计算 128 个 exp（每 thread 8 个 SFU 操作）                              │
│     - 求和（每 thread 8 次加法 + shuffle reduce）                            │
│     - 归一化（每 thread 8 次乘法）                                        │
│                                                                             │
│  2. TopK（每个 token，迭代 8 次）：                                        │
│     - 每次每 thread 扫描 VPT=8 个值 + shuffle reduce 跨 16 threads       │
│     - 清除已选值（8 次赋值）                                                │
│                                                                             │
│  计算量：                                                                   │
│  - 每个 token：128 次 exp + 多轮 shuffle reduce + ~128 次算术操作          │
│  - 1 个 token：总计算量很小，主要受延迟主导                              │
│                                                                             │
│  计算特点：                                                                 │
│  - 大部分是 ALU 操作，非常快                                                │
│  - exp() 使用 SFU，稍慢但可接受                                             │
│  - Warp shuffle 进行 sub-warp reduce（width=16），无 shared memory 开销      │
│                                                                             │
│  最低下限：~0.5 μs（假设计算资源无限）                                      │
│  实际开销：~1-2 μs                                                          │
│  能否优化：基本不可优化（计算路径已较高效）                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 2.6 结果写回（~0.3-0.5 μs，对应阶段 4）

> **计时归属：✓ NCU Duration（计入）| ✓ cudaEvent（计入）。本节对应阶段 4。**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  写回结果到显存                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  数据量（本测试配置：1 token, topk=8）：                                    │
│  - weights: 1 × 8 × 4 bytes (float) = 32 bytes                             │
│  - indices: 1 × 8 × 4 bytes (int) = 32 bytes                               │
│  - source_rows: 1 × 8 × 4 bytes (int) = 32 bytes                           │
│  - 总计：96 bytes                                                           │
│                                                                             │
│  写入模式：                                                                 │
│  - Coalesced write（合并写入）                                              │
│  - 从寄存器 → L1 → L2 → DRAM                                                │
│                                                                             │
│  理论最低时间（纯带宽限制）：                                               │
│  - 96 bytes / 3 TB/s ≈ 0.03 ns                                             │
│                                                                             │
│  实际开销：~0.3-0.5 μs                                                      │
│                                                                             │
│  为什么比理论值高：                                                         │
│  - 写入也需要经过存储层次                                                   │
│  - Write buffer 填充和刷新                                                  │
│  - 可能的 write-back 延迟                                                   │
│                                                                             │
│  最低下限：~0.1 μs                                                          │
│  实际开销：~0.3-0.5 μs                                                      │
│  能否优化：可优化但空间有限（已是较优写入模式）                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 2.7 完成同步（~0.3-0.5 μs，对应阶段 5a + 5b + 5c）

> **计时归属：**
> - **阶段 5a（SM 内部完成）**：✓ NCU Duration（计入）| ✓ NSYS Duration（计入，NSYS 终点）| ✓ cudaEvent（计入）— 最后 warp 退出、CUPTI 插桩出口记 end
> - **阶段 5b（GPC retirement）**：✓ NCU Duration（计入，NCU 终点）| ✗ NSYS Duration（不计入）| ✓ cudaEvent（计入）— CTA 完成信号、GPC cycles_elapsed 停止
> - **阶段 5c（全局完成发布）**：✗ NCU Duration（不计入）| ✗ NSYS Duration（不计入）| ✓ cudaEvent（计入）— stream 状态更新、event 时间戳写入

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Kernel 完成和同步                                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  做什么：                                                                   │
│  1. Per-CTA 资源回收（与阶段 4 交织，每个 CTA 结束即释放）                │
│  2. 等待最后一个 CTA 完成                                                │
│  3. 更新 kernel 完成状态                                                    │
│  4. 如果有 event，写入时间戳                                                │
│                                                                             │
│  为什么需要时间：                                                           │
│  - 最后一个 CTA 可能因为调度原因稍慢                                      │
│  - 完成状态发布需要硬件协调                                               │
│                                                                             │
│  最低下限：~0.2 μs                                                          │
│  实际开销：~0.3-0.5 μs                                                      │
│  能否优化：不可优化（正确性要求）                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3. 内存访问的物理下限分析

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  问题：即使只访问 1 个字节，最快需要多久？                                   │
│                                                                             │
│  答案：取决于数据在哪个存储层级                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

GPU 存储层次延迟（以 H100 为例，示意值）
──────────────────────────────────────

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  存储层级          延迟              带宽              容量                 │
│  ──────────────────────────────────────────────────────────────────────     │
│  寄存器           ~0 cycle         ~10 TB/s           ~256KB               │
│  L1 Cache         ~20-30 cycles    ~10 TB/s           ~128KB/SM            │
│  L2 Cache         ~100-200 cycles  ~3 TB/s            ~50MB                │
│  HBM (显存)       ~300-500 cycles  ~3 TB/s            ~80GB                │
│                                                                             │
│  以 H100 (2.0 GHz) 为例：                                                   │
│  - 1 cycle ≈ 0.5 ns                                                         │
│  - L1 延迟 ≈ 10-15 ns                                                       │
│  - L2 延迟 ≈ 50-100 ns                                                      │
│  - HBM 延迟 ≈ 150-250 ns                                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

一次全局内存访问的典型延迟路径（示意）
───────────────────────────────────

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  即使业务上只需要很少数据，底层也按内存事务粒度访问：                        │
│                                                                             │
│  1. 指令发射        ~5-10 cycles    ~3-5 ns                                │
│     Warp Scheduler 发射 load 指令                                           │
│                                                                             │
│  2. 地址计算        ~5 cycles       ~2.5 ns                                │
│     计算 effective address                                                  │
│                                                                             │
│  3. L1 Cache 查找   ~20 cycles      ~10 ns                                 │
│     Cache tag 查找                                                          │
│                                                                             │
│  4. L2 Cache 查找   ~100 cycles     ~50 ns                                 │
│     （如果 L1 未命中）                                                       │
│                                                                             │
│  5. HBM 访问        ~300-500 cycles ~150-250 ns                            │
│     （如果 L2 未命中）                                                       │
│     - Memory controller 调度                                                │
│     - DRAM 行激活                                                           │
│     - DRAM 列读取                                                           │
│                                                                             │
│  6. 数据返回        ~20-50 cycles   ~10-25 ns                              │
│     通过总线返回到寄存器                                                     │
│                                                                             │
│  合计：~400-700 cycles ≈ 200-350 ns（一次 DRAM 路径访问的量级）              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

本 Kernel 的数据传输分析（1 token, 128 experts, topk=8 配置）
───────────────────────────────────────────────────────────────

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  读取输入：256 bytes (1 token × 128 experts × 2 bytes bf16)                │
│  写回输出：96 bytes (weights 32B + indices 32B + source_rows 32B)            │
│  总计：352 bytes                                                            │
│                                                                             │
│  理论纯带宽限制：352 bytes / 696 GB/s ≈ 0.51 ns（A40 峰值带宽）            │
│                                                                             │
│  实测时间（A40, 128 experts, 1 token, bf16, topk=8）：                      │
│  - NCU Duration = 6.56 μs（GPC 端启动 + 执行 + GPC 端完成）                │
│  - cudaEvent   = 6.773 μs（含 Front End + Event 打点 + 完成发布）           │
│                                                                             │
│  差距原因（延迟主导，非带宽限制）：                                         │
│  - SM 端启动开销（资源分配、取指/I-Cache 装填、首 warp 调度）               │
│  - 多次内存访问的延迟叠加（L1/L2/HBM 路径）                                │
│  - SFU 计算（exp）延迟                                                      │
│  - Warp 同步开销                                                            │
│                                                                             │
│  结论：数据量极小（352 bytes），完全受延迟主导                              │
│        6.56 μs（NCU）/ 6.77 μs（Event）主要是                               │
│        SM 启动开销 + 计算延迟 + 同步开销，不是带宽瓶颈                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
