# CUDA Kernel 执行流程与性能分析报告

## 概述

本文档详细分析一个 CUDA Kernel 从调用到执行完成的完整流程，以及每个步骤的时间开销。以 `topk_softmax` kernel 为例（配置：Qwen3-30B-A3B 模型参数 - 1 token, 128 experts, topk=8, bf16），**NCU 实测 kernel 执行时间 6.56 μs**。

---

## NCU 性能测试命令

**目标配置**: 1 token, 128 experts, topk=8, bf16

```bash
cd /LocalRun/xiaolong.zhu/standalone_topk_kernel
sudo /usr/local/cuda/bin/ncu --set full ./build/bench_perf 128 1 bf16 8 1 2>&1 | tee ncu_output.txt
```

查看 kernel 执行时间：
```bash
grep -i duration ncu_output.txt
```

**说明**：
- `./build/bench_perf 128 1 bf16 8 1` 参数含义：128 experts, 1 token, bf16, topk=8, 1 iteration
- bench_perf 默认有 10 次 warmup，所以实际会跑 11 次 kernel
- NCU 输出中的 `Duration` 字段就是 kernel 的实际 GPU 执行时间

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

- **NCU（Nsight Compute）**：NVIDIA 的 GPU kernel 性能分析工具，可以精确测量单个 kernel 的执行时间（Duration）、内存吞吐量、计算利用率等，不包含 launch 开销和 event 开销。
- **cudaEvent**：CUDA 的 GPU 时间戳机制。`cudaEventRecord` 在 stream 中插入一个打点命令，GPU 执行到时记录高精度时钟。`cudaEventElapsedTime` 计算两个 event 在 GPU 时间线上的间隔。
- **cudaEventSynchronize**：Host 阻塞等待直到指定 event 在 GPU 上被执行完毕。
- **cudaStreamSynchronize**：Host 阻塞等待直到指定 stream 中所有已提交的命令在 GPU 上全部执行完毕。

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
- 所以 `ms` 通常包含阶段 `3 + 4 + 5`（外加 event 本身开销与可能的空等间隙）。

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
4. `阶段 3` <-> `第三部分 2.2 命令处理` + `2.3 启动延迟`
5. `阶段 4` <-> `第三部分 2.4 数据加载` + `2.5 计算` + `2.6 结果写回`
6. `阶段 5` <-> `第三部分 2.7 完成同步`

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
- 阶段 2 在第三部分没有独立 GPU 分项；只在 `6.9 μs` 公式里的"间隙/其他"中以间接形式出现。

---

#### 阶段 3：GPU 端命令处理

```
GPU 端命令处理流程（统一样式）
──────────────────────────────

    [S3-0] GPU 收到 Doorbell
        │
        ▼
    [S3-1] 命令获取（GPU Front End 通过 PCIe/NVLink 从 host pushbuffer 拉取 launch 命令）
        │
        ▼
    [S3-2] 命令解析（函数地址 / grid / block / 参数指针）
        │
        ▼
    [S3-3] 资源分配（SM / 寄存器 / shared memory）
        │
        ▼
    [S3-4] 取指与 I-Cache 装填
           - 从 GPU 显存中的代码区取指
           - 经 L2 装填到 SM 指令缓存
           - 首次/非首次调用都会执行该步骤
        │
        ▼
    [S3-5] 首批 Warp 发射（阶段 4 入口）
```

**子步骤（S3-x，细粒度版）：**
1. `[S3-1] 命令读取`  
   - 做什么：GPU Front End 通过 PCIe/NVLink DMA 从 host pinned memory 的 pushbuffer 拉取本次 launch 描述符。  
   - 输入/输出：输入是 pushbuffer 中的队列项；输出是可解析的 launch 包。  
   - 为什么耗时：需要走 PCIe/NVLink DMA 读取路径。
2. `[S3-2] 命令解析`  
   - 做什么：解析 kernel 入口、grid/block、参数地址、stream 顺序关系。  
   - 输入/输出：输入是 launch 包；输出是“可调度任务描述”。  
   - 为什么耗时：需要做字段校验与硬件可执行格式转换。
3. `[S3-3] 资源可行性检查与分配`  
   - 做什么：检查寄存器/shared memory/可用 SM 配额，建立首批 CTA 的资源映射。  
   - 输入/输出：输入是任务描述；输出是“可发射资源上下文”。  
   - 为什么耗时：硬件调度器要做占用与并发可行性判断。
4. `[S3-4] 取指与 I-Cache 装填`  
   - 做什么：从代码区取首批指令，经 L2 装填到 SM 指令缓存。  
   - 输入/输出：输入是 kernel PC/入口；输出是“可取指执行”的指令缓存状态。  
   - 为什么耗时：首次取指或 I-Cache 命中不佳会增加延迟。
5. `[S3-5] 发射门槛判定`  
   - 做什么：确认依赖与资源都就绪，发射首批 warp，切入阶段 4。  
   - 输入/输出：输入是可发射上下文；输出是“执行中”状态。

**和第三部分的一一映射：**
1. `第三部分 2.2 命令处理` 对应 `S3-1 ~ S3-3`。
2. `第三部分 2.3 启动延迟` 对应 `S3-3 ~ S3-5`（`S3-3` 是共享边界）。

**阶段输出：**
- kernel 已具备“可发射”条件，首批 warp 已进入执行入口（阶段 4）。

**补充边界：**
- 首次调用的“模块级代码装载/JIT”主要在阶段 2 的 Host 路径触发。
- 本阶段只覆盖“命令消费到首批 warp 发射前后”这段 GPU 路径。

**计时归属：**
- 该阶段属于 GPU 时间线，计入 `cudaEvent` 的 `ms`（经验量级约 `~1.5-3.0 μs`，对应第三部分合并口径）。

---

#### 阶段 4：Kernel 执行

```
Kernel 在 SM 上执行
───────────────────

    以 topkGatingSoftmax 为例（64 experts, 1024 tokens, topk=4）
        │
        ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                                                                     │
    │  执行形状（由阶段 1 已确定）：                                      │
    │  ─────────────────────────                                          │
    │  - WARPS_PER_TB = 4                                                 │
    │  - ROWS_PER_WARP = 32 / (64 / 64) = 32                             │
    │  - ROWS_PER_CTA = 4 * 32 = 128                                      │
    │  - num_warps = ceil(1024 / 32) = 32                                 │
    │  - num_blocks = ceil(32 / 4) = 8                                    │
    │                                                                     │
    │  每个 Block 处理 128 个 token                                       │
    │  8 个 Block 处理 1024 个 token                                      │
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
    │  写入两个输出数组：                                                 │
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
- 该阶段是 kernel 主体执行时间，计入 `cudaEvent` 的 `ms`（经验量级约 `~1.8-3.5 μs`，对应第三部分合并口径）。

---

#### 阶段 5：Kernel 完成和同步

**先回答“为什么执行结束后还需要这个阶段”**：
- “执行结束”只表示 kernel 指令跑完，不等于系统层面的“调用已完成”。
- 同一 stream 里后续命令（下一个 kernel、event、memcpy）必须知道一个明确的“完成点”。
- 资源（寄存器/shared memory/调度槽位）需要回收，否则后续 kernel 不能稳定调度。
- CPU 侧如果在等（`cudaEventSynchronize` / `cudaStreamSynchronize`），必须有可见的完成状态。

```
Kernel 完成与同步流程（统一样式）
────────────────────────────────

    阶段 4 执行期间，资源回收已在持续进行：
    ┌─────────────────────────────────────────────────────────────────────┐
    │  5.0 Per-CTA 资源回收（与阶段 4 交织发生）                         │
    │  - 每个 CTA 执行完毕后，其占用的寄存器/shared memory/调度槽位       │
    │    立即释放，可被同一 SM 上的后续 CTA 或其他 kernel 复用             │
    │  - 这是流式/增量的，不是所有 block 完成后的一次性批量操作           │
    └─────────────────────────────────────────────────────────────────────┘

    最后一个 CTA 执行完毕
        │
        ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  5.1 完成判定 (Completion Point)                                    │
    │  - 所有已 launch 的 CTA 全部结束                                    │
    │  - 只有最慢的 CTA 结束后，kernel 才算真正完成                       │
    │  - 此时所有 per-CTA 资源已在先前逐步回收完毕                        │
    └─────────────────────────────────────────────────────────────────────┘
        │
        ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  5.2 完成发布 (Completion Publish)                                  │
    │  - 更新 stream 中该 kernel 的完成状态                                │
    │  - 若有 cudaEvent，写入 event 时间戳                                 │
    │  - 若 CPU 正在等待，同步原语据此解除等待                              │
    │  - 这是对 Host 可见的"完成发布点"                                    │
    └─────────────────────────────────────────────────────────────────────┘
```

**子步骤（S5-x）：**
1. `[S5-0]` Per-CTA 资源回收（与阶段 4 交织）：每个 CTA 结束后其占用的寄存器/shared memory/调度槽位立即释放。
2. `[S5-1]` 完成判定：最后一个 CTA 结束，kernel 完成。
3. `[S5-2]` 完成发布：stream/event/等待方看到"已完成"状态。

**阶段输出：**
- 对同一 stream 的后续命令，前序 kernel 已完成这一事实可见。

**和第三部分的一一映射：**
- `S5-0/S5-1/S5-2` <-> `第三部分 2.7 完成同步`。

**是否任何 kernel 都需要 5.0/5.1/5.2？**
- 需要。这是 GPU/runtime 的通用收尾逻辑，不是 `topk_softmax` 特有代码。

**`cudaStreamSynchronize(stream)` 到底在等什么？**
1. 等 `5.1`：所有已 launch 的 CTA 全部执行完（per-CTA 资源已在此前逐步回收）。
2. 等 `5.2`：完成状态被发布到 stream（必要时 event 时间戳也写入）。
3. 上述 1~2 都满足后，`cudaStreamSynchronize(stream)` 才返回，`topk_softmax` 才返回。

**计时归属：**
- 该阶段在 GPU 时间线内，计入 `cudaEvent` 的 `ms`（经验量级约 `~0.3-0.5 μs`）。

**源码补充（当前仓库实现）：**
- `src/topk_softmax.cu:672` 有 `cudaStreamSynchronize(stream)`，因此 `topk_softmax` 返回前会等待“5.1 -> 5.2”整条链路完成。
- 这属于封装层额外语义：底层 kernel launch 仍是异步，封装函数把返回时机改成“完成后再返回”。

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
- `阶段 3`：GPU 命令处理 + 启动准备。
- `阶段 4`：Kernel 执行（真正算子计算）。
- `阶段 5`：完成发布（含 event 可见性）。
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
1. 包含：`阶段 3 + 阶段 4 + 阶段 5 + event 命令执行 + 可能空等间隙`。
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
3. `阶段 3`：GPU 命令处理 + 启动准备 + 取指/I-Cache 装填。
4. `阶段 4`：Kernel 执行（数据加载 + 计算 + 写回）。
5. `阶段 5`：完成判定 + 完成发布（per-CTA 资源回收与阶段 4 交织）。
6. `Event Stop`：GPU 打点。
7. 可能还有“间隙”：Host 后续命令提交延迟导致的 GPU 空等时间。

在本仓库里，`bench_perf` 使用 `topk_softmax_async`，内部无 `cudaStreamSynchronize` 且无 `cudaMalloc/cudaFree`，因此稳态下不存在封装层引入的空等间隙。

> **NCU 实测值 vs cudaEvent 测量值**：
> - **NCU Duration = 6.56 μs**：纯 kernel 执行时间（阶段 4），不包含 event 开销、启动延迟等
> - **cudaEvent ms ≈ 6.6 μs**：包含 event 开销、启动延迟、完成同步等的总时间

```
NCU 实测 kernel 执行时间（阶段 4）= 6.56 μs

cudaEvent 测量的总时间（阶段 3+4+5+event）≈ 6.6 μs
  阶段 4（Kernel 执行）    6.56 μs  ← NCU 实测
  阶段 3+5+event          ~0.04 μs （几乎可忽略）
```

`Kernel 启动` 不是独立新阶段，它属于 `阶段 3` 的子过程。

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

#### 2.2 命令处理（~1-2 μs，对应阶段 3）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  GPU 命令队列处理和解析                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  做什么：                                                                   │
│  1. GPU Front End 从 host pushbuffer 拉取命令（通过 PCIe/NVLink DMA）       │
│  2. 解析命令类型和参数                                                      │
│  3. 读取 kernel 元数据（grid/block 配置等）                                 │
│  4. 确定需要多少 SM 资源                                                    │
│                                                                             │
│  为什么需要这么长时间：                                                     │
│  - 命令缓冲区在 host 锁页内存中，需通过 PCIe/NVLink DMA 拉取               │
│  - GPU 是个庞然大物，命令需要在多个硬件单元间传递                           │
│  - 资源分配需要协商（哪些 SM 空闲？寄存器够用吗？）                          │
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

#### 2.3 GPU 侧启动延迟（~0.5-1 μs，对应阶段 3 为主）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  GPU 侧启动和初始化（非 Host launch）                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  做什么：                                                                   │
│  1. 取指并装填 I-Cache（阶段 3.4）                                          │
│  2. 为 CTA 分配寄存器与 shared memory（阶段 3.3）                           │
│  3. 建立首批 warp 的调度状态（阶段 3 末尾）                                 │
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
- 这一节主要属于 `阶段 3（S3-3/S3-4）`，最后“首个 warp 发射”是 `阶段 4` 的入口，不是新增阶段。

#### 2.4 数据加载（~0.5-1 μs，对应阶段 4）

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

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  写回结果到显存                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  数据量（本测试配置：1 token, topk=8）：                                    │
│  - weights: 1 × 8 × 4 bytes (float) = 32 bytes                             │
│  - indices: 1 × 8 × 4 bytes (int) = 32 bytes                               │
│  - 总计：64 bytes                                                           │
│                                                                             │
│  写入模式：                                                                 │
│  - Coalesced write（合并写入）                                              │
│  - 从寄存器 → L1 → L2 → DRAM                                                │
│                                                                             │
│  理论最低时间（纯带宽限制）：                                               │
│  - 64 bytes / 3 TB/s ≈ 0.02 ns                                             │
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

#### 2.7 完成同步（~0.3-0.5 μs，对应阶段 5）

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
│  写回输出：64 bytes (weights 32B + indices 32B)                             │
│  总计：320 bytes                                                            │
│                                                                             │
│  理论纯带宽限制：320 bytes / 3 TB/s ≈ 0.1 ns                               │
│                                                                             │
│  NCU 实测：6.56 μs                                                          │
│                                                                             │
│  差距原因（延迟主导，非带宽限制）：                                         │
│  - GPU 启动开销（命令处理、资源分配）                                       │
│  - 多次内存访问的延迟叠加                                                   │
│  - SFU 计算（exp）延迟                                                      │
│  - Warp 同步开销                                                            │
│                                                                             │
│  结论：数据量极小（320 bytes），完全受延迟主导                              │
│        6.56 μs 主要是 GPU 基础开销 + 计算延迟，不是带宽瓶颈                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
