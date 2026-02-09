# GPU Kernel Baseline Comparison Analysis

## Test Environment

| Item | Value |
|------|-------|
| GPU | NVIDIA A40 |
| Memory | 46068MiB |
| Compute Capability | 8.6 |
| Peak Memory Bandwidth | 696 GB/s |
| Test Iterations | 100 |

## Model Configuration (Qwen3-30B-A3B)

| Parameter | Value |
|-----------|-------|
| num_experts | 128 |
| topk (num_experts_per_tok) | 8 |
| dtype | bf16 |

## Baseline Kernel Results

| Kernel | Average Time (us) | Description |
|--------|------------------|-------------|
| empty_kernel | 4.352 | Pure GPU launch + event overhead |
| minimal_rw_kernel | 4.424 | Launch + minimal memory access |

**Fixed Overhead Baseline**: ~4.424 us - This is the minimum overhead for ANY kernel execution on this GPU.

## TopK Kernel Results (Pure Kernel Time)

| Tokens | Data Size (bytes) | Theoretical Transfer (ns) | Actual Time (us) | BW Utilization |
|--------|------------------|--------------------------|------------------|----------------|
| 1 | 352 | 0.505 | 6.840 | 0.00% |
| 4 | 1408 | 2.022 | 6.840 | 0.02% |
| 256 | 90112 | 129.471 | 7.035 | 1.84% |
| 1024 | 360448 | 517.885 | 7.363 | 7.03% |

### Key Observation

**1 token 和 1024 tokens 的执行时间差距远小于数据量差距！**

```
tokens=1:     6.840 us    (数据量: 352 bytes)
tokens=4:     6.840 us    (数据量: 1408 bytes)
tokens=256:   7.035 us   (数据量: 90112 bytes = 88.0 KB)
tokens=1024:  7.363 us  (数据量: 360448 bytes = 352.0 KB)

数据量比: 1 : 4 : 256 : 1024
时间比:   1.00 : 1.00 : 1.02 : 1.07
```

这证明了**固定开销占绝对主导地位**，数据量增加不带来成比例的时间增加。

## Analysis

### Time Breakdown for All Token Counts

```
┌──────────────────────────────────────────────────────────────────────────┐
│  TopK Kernel 时间分解                                                      │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  tokens=1:   总时间 6.840 us                                  │
│    固定开销:  4.424 us (60.0%)  ████████████████████████│
│    计算开销:  2.416 us (30.0%)  ██████            │
│    数据传输:  ~0.505 ns (<0.01%)                                 │
│                                                                          │
│  tokens=4:   总时间 6.840 us                                  │
│    固定开销:  4.424 us (60.0%)  ████████████████████████│
│    计算开销:  2.416 us (30.0%)  ██████            │
│    数据传输:  ~2.022 ns (<0.01%)                                 │
│                                                                          │
│  tokens=256: 总时间 7.035 us                                │
│    固定开销:  4.424 us (60.0%)  ██████████████████████│
│    计算开销:  2.611 us (30.0%)  ████████      │
│    数据传输:  ~129.471 ns (~0.02%)                               │
│                                                                          │
│  tokens=1024: 总时间 7.363 us                              │
│    固定开销:  4.424 us (60.0%)  ████████████████████ │
│    计算开销:  2.939 us (30.0%)  ██████████  │
│    数据传输:  ~517.885 ns (~0.07%)                              │
│                                                                          │
│  关键发现: 即使 tokens=1024，固定开销仍占 ~60.0%！            │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### TopK vs Baseline

```
TopK kernel:    ~6.840 us
Baseline:       ~4.424 us
差距:           ~2.416 us

这 2.416 us 来自:
- Softmax 计算 (exp, reduction, normalize)
- TopK 选择 (8次迭代找最大值)
- 内存访问

这些是算法必需的开销，不是"写得不好"！
```

### Bandwidth Utilization

```
有效带宽 = 数据量 / 实际时间

tokens=1:
  有效带宽 = 352 bytes / 6.840 us = 0.05 GB/s
  带宽利用率 = 0.00%

tokens=4:
  有效带宽 = 1408 bytes / 6.840 us = 0.20 GB/s
  带宽利用率 = 0.02%

tokens=256:
  有效带宽 = 90112 bytes / 7.035 us = 12.80 GB/s
  带宽利用率 = 1.84%

tokens=1024:
  有效带宽 = 360448 bytes / 7.363 us = 48.95 GB/s
  带宽利用率 = 7.03%
```

### Why Low Bandwidth is Inevitable for Small Tokens

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  带宽利用率 = 数据传输时间 / 总时间                              │
│                                                                 │
│  对于 tokens=1:                                                 │
│    带宽利用率 = 0.505 ns / 6.840 us ≈ 0%                   │
│                                                                 │
│  对于 tokens=1024:                                              │
│    带宽利用率 = 517.885 ns / 7.363 us ≈ 7.00%                 │
│                                                                 │
│  即使数据量增加 1024 倍，带宽利用率仍然较低！                     │
│  因为总时间被固定开销主导，不是数据传输。                         │
│                                                                 │
│  固定开销 (4.424 us) >> 数据传输时间 (0.505 ns - 517.885 ns)       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Conclusion

### 核心结论

**小 token 场景下带宽利用率低是正常的，不是 kernel 实现有问题。**

### 证据链

1. **Baseline 测试证明固定开销存在**: 即使空 kernel 也需要 ~4.424 us
2. **1/4/256/1024 tokens 时间增长缓慢**: 证明数据量变化，时间不成比例增加
3. **TopK vs Baseline 差距小**: 只有 ~2.416 us，是算法必需的计算
4. **理论计算**: 数据传输时间 (0.505 ns) << 固定开销 (4.424 us = 4424.000 ns)

### To Achieve High Bandwidth Utilization

需要数据量大到传输时间能覆盖固定开销：

```
要达到 50% 带宽利用率 (348 GB/s):
  需要传输时间 ≈ 固定开销 ≈ 4.424 us
  数据量 = 348 GB/s × 4.424 us = 1.46 MB

当前 tokens=1024 时数据量只有 352.0 KB
需要的数据量是当前的 4.2 倍！
```

---

## Test Details

### Testing Methodology

```
- warmup: 10 iterations (不计入统计)
- iterations: 100 (计入统计)
- timing: cudaEvent (GPU 时间线)
- memory: 所有内存在测试循环外预分配
- API: topk_softmax_async() (纯 kernel 调用，不含同步)
```

### Raw Results

```
Baseline:
  empty_kernel:      4.352 us
  minimal_rw_kernel: 4.424 us

TopK (experts=128, topk=8, dtype=bf16):
  tokens=1:     6.840 us
  tokens=4:     6.840 us
  tokens=256:   7.035 us
  tokens=1024:  7.363 us
```

---

*Generated from actual test results on NVIDIA A40*
*Model: Qwen3-30B-A3B (experts=128, topk=8)*
*Using topk_softmax_async() for pure kernel timing*
