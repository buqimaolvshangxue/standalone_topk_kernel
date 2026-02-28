# Kernel 计时方法对比分析

## 背景

在对 TopK Softmax kernel 进行性能测量时，发现 CUDA Event 和 DLPTI 两种工具测出的时间存在系统性差异。本文档记录观测现象并给出分析。

---

## 测试环境

- **硬件**: 国产 GPU（非 NVIDIA）
- **测试命令**: `./build/bench_perf 128 1 bf16 8 2 1`（128 experts, 1 token, bf16, topk=8）
- **DLPTI 命令**: `dlpti_tools kprof --targets cu,tu --data-file ./topk.db -- ./build/bench_perf 128 1 bf16 8 2 1`

---

## 观测数据

| 测量工具 | 空 kernel | 实际 kernel (128 experts, 1 token, bf16, topk=8) |
|---------|----------|------------------------------------------------|
| CUDA Event (`cudaEventElapsedTime`) | **~20 μs** | **~40 μs** |
| DLPTI (`--targets cu,tu`) | **~1 μs** | **~20 μs** |

关键观察：
- 两者的**差值（增量）一致**：实际 kernel - 空 kernel ≈ 19~20 μs
- CUDA Event 有一个恒定的 ~20 μs 基底偏移

---

## 两种工具的测量原理

### CUDA Event（`cudaEventRecord` + `cudaEventElapsedTime`）

```cpp
cudaEventRecord(start);        // 向 stream 中插入 start 时间戳命令
kernel<<<...>>>();             // 向 stream 中插入 kernel launch 命令
cudaEventRecord(stop);         // 向 stream 中插入 stop 时间戳命令
cudaEventSynchronize(stop);    // CPU 等待 stop event 完成
cudaEventElapsedTime(&ms, start, stop);  // 计算 T_stop - T_start
```

**在 NVIDIA GPU 上**，T_start 和 T_stop 都是 GPU 硬件时钟直接记录的，不涉及 CPU 时钟。因此测的是纯 GPU 端时间。

**猜测：在本国产 GPU 上**，Event 的时间戳记录机制可能不同，或者 Event 测量的范围覆盖了 kernel 的完整生命周期（包括调度开销）。

### DLPTI（`--targets cu,tu`）

DLPTI 以 `cu`（Compute Unit）和 `tu`（Tensor Unit）为采集目标，通过硬件 performance counter 测量。

**猜测**：DLPTI 测量的是 **CU/TU 实际活跃执行指令的时间**，不包含 kernel 的调度、grid launch、block 分配等前后开销。

---

## 分析与猜测

### 猜测 1：DLPTI 只测 CU 活跃时间，CUDA Event 测完整 kernel 生命周期

```
完整 kernel 时间线 (CUDA Event 测量范围):
│←── 调度开销 ──→│←── CU 执行指令 ──→│←── 尾部清理 ──→│
│    ~20 μs       │     ~20 μs         │                 │
│                 │←─ DLPTI 测量范围 ─→│                 │
│←──────────── CUDA Event 测量范围 (~40 μs) ────────────→│
```

- 空 kernel 时，CU 几乎无指令可执行 → DLPTI ≈ 1 μs
- 空 kernel 时，调度开销仍存在 → CUDA Event ≈ 20 μs
- 实际 kernel 时，CU 计算 ~20 μs → DLPTI ≈ 20 μs
- 实际 kernel 时，调度 + 计算 → CUDA Event ≈ 20 + 20 = 40 μs

**支持证据**：两者的增量完全一致（~19-20 μs），偏移恰好等于空 kernel 的 Event 时间。

### 猜测 2：CUDA Event 在该硬件上混入了 CPU-GPU 命令提交延迟

在 NVIDIA GPU 上，CPU 调用 `cudaEventRecord` 几乎瞬间返回，GPU 异步消费命令。但该硬件的 CUDA 兼容层可能在 Event 实现中包含了隐式同步或额外的 CPU-GPU 通信开销。

### 猜测 3：该 GPU 的 Command Processor 调度开销较大

无论上述哪种解释，~20 μs 的固有开销（从空 kernel 的 Event 时间可判断）显著高于 NVIDIA GPU 的典型值（~2-5 μs），这可能反映了 Grid Scheduler / Command Processor 的架构差异。

---

## 对比 NVIDIA GPU 的参考值

| 指标 | NVIDIA GPU (典型值) | 本国产 GPU (观测值) |
|------|-------------------|-------------------|
| 空 kernel (CUDA Event) | 2~5 μs | ~20 μs |
| 空 kernel (ncu / DLPTI) | 1~3 μs | ~1 μs |
| ncu 与 Event 差异 | < 5% | 系统性偏移 ~20 μs |

在 NVIDIA GPU 上，CUDA Event 和 ncu 测出的时间非常接近（因为都是 GPU 硬件时钟）。而在本硬件上两者差异很大，说明底层实现确实不同。

---

## 实用结论

| 场景 | 推荐工具 | 原因 |
|------|---------|------|
| 测 kernel 的 **CU 实际计算时间** | DLPTI | 直接测量硬件计算单元活跃时间 |
| 测 kernel 的 **端到端 GPU 时间**（含调度） | CUDA Event | 覆盖完整 kernel 生命周期 |
| **两个 kernel 的相对性能比较** | 两者均可 | 固有开销被抵消，增量一致 |
| **绝对性能评估**（如 roofline 分析） | 需明确定义"时间" | CU 时间 vs 端到端时间，结论不同 |

### 注意事项

1. **做相对比较时两种工具都可靠**：因为固有偏移是常数，kernel A vs kernel B 的差值在两种工具下一致
2. **做绝对性能报告时需注明工具**：用 DLPTI 报 20 μs 和用 Event 报 40 μs 都是"对的"，但含义不同
3. **与 NVIDIA GPU 横向对比时需对齐口径**：NVIDIA 的 Event 时间 ≈ ncu Duration ≈ "真实 GPU 时间"，而本硬件的 Event 时间包含额外调度开销

---

## DLPTI 官方文档（dlpti_manual.pdf）中的相关信息

阅读了 `/home/xiaolong.zhu/dlpti_manual.pdf`（登临 Hamming V2 dlPTI 使用手册，更新至 2026-02-26），提取关键信息如下：

### 文档明确说明的

1. **kprof 的定位**：`dlpti_tools kprof` 用于 "Collect profiling counters for specified targets"，即采集硬件性能计数器
2. **`--targets cu,tu` 的含义**：
   - `cu` = Compute Unit 相关的 hardware counter
   - `tu` = Tensor Unit 相关的 hardware counter
   - 还支持 `graph`（内部功能）
3. **工作流程**：kprof 采集原始 hardware counter → `export --format metric` 分析 counter 生成 metrics → `dlkprof-ui` 可视化
4. **Replay 机制**：文档提到 "kernel profiling 可能不能获取完全可靠的数据，对于应用的需求是在多次执行中保证内部多个 Kernel 的执行顺序是固定的"，说明存在类似 ncu 的 replay 机制
5. **时钟问题**：文档明确警告 "Device Clock 和 Host Clock 无法完全同步"，并提到 DVFS 会导致 Device Timestamp 不稳定，建议进行锁频

### 文档没有明确说明的

- **kprof 报告的 "Duration" 的精确定义**：没有说明是 CU 活跃时间还是 kernel 端到端时间
- **cudaEventRecord 的实现细节**：没有说明在该硬件上 Event 时间戳的打点位置
- **capture（Application Profiling）中 `cmd` activity 的时间范围**：capture 模式下有 Device Command timeline，但文档没说明其与 kprof Duration 的关系

### 文档给出的一种验证思路

文档提到 `dlpti_tools capture --activity-mask cmd` 可以采集 Device Command 级别的 timeline，这个应该反映 kernel 的完整 Device 端生命周期（从 Command Processor 接收命令到执行完毕）。可以用这种方式采集一个 timeline，看 Device Command 的时长是更接近 DLPTI kprof 的 ~20μs 还是 CUDA Event 的 ~40μs，从而区分两种猜测。

```bash
# 用 capture 模式采集 Device Command timeline
dlpti_tools capture --activity-mask cmd --data-file ./topk_cmd.db -- ./build/bench_perf 128 1 bf16 8 2 1

# 导出后用 dlsys-ui 查看 Device Command 的时长
dlpti_tools export --format perfetto-json ./topk_cmd.db
```

如果 Device Command timeline 显示 kernel 时长为 ~20μs，则说明 CUDA Event 的额外 20μs 是 Event 实现带来的开销；如果显示 ~40μs，则说明 DLPTI kprof 确实只测 CU 活跃时间。

---

## 待验证事项

- [ ] 用 `dlpti_tools capture --activity-mask cmd` 采集 Device Command timeline，对比 kernel 时长与 CUDA Event / kprof 的结果
- [ ] 测试不同规模的 kernel（如增大 tokens），观察 20 μs 偏移是否恒定
- [ ] 确认空 kernel 的定义和实现（是否真正编译出了空 kernel）
- [ ] 咨询 GPU 厂商（登临），确认 `cudaEventRecord` 在 Hamming V2 上的精确语义
- [ ] 尝试锁频后重新测试，排除 DVFS 对 Device Timestamp 的影响
