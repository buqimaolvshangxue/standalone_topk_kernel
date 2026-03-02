# 登临 dlPTI 使用指南

> 本文档基于《登临 Hamming V2 dlPTI 使用手册》整理，用于替代 NVIDIA Nsight (ncu/nsys) 进行 GPU 性能分析。

---

## 一、dlPTI 是什么

dlPTI 是登临 GPU 的性能分析工具集，用于分析 GPU 程序的性能瓶颈。

**和 NVIDIA 工具的对应关系：**

| NVIDIA 工具 | dlPTI 工具 | 用途 |
|-------------|-----------|------|
| nsys | `dlpti_tools capture` | 看时间线、API 调用、并发情况 |
| ncu | `dlpti_tools kprof` | 看 kernel 内部硬件计数器 |
| nsys-ui | `dlsys-ui` | 查看时间线分析结果 |
| ncu-ui | `dlkprof-ui` | 查看 kernel 指标分析结果 |

---

## 二、理解 GPU 程序执行

在学习工具之前，先理解 GPU 程序是怎么执行的：

```
┌─────────────────────────────────────────────────────────────────┐
│   CPU (Host)                              GPU (Device)          │
│   ┌──────────────┐                        ┌──────────────┐     │
│   │ cudaMalloc   │ ────────────────────>  │ 分配显存      │     │
│   │ cudaMemcpy   │ ────────────────────>  │ 拷贝数据      │     │
│   │ myKernel<<<>>>│ ───(异步发射)───────>  │ 排队等待执行  │     │
│   │              │                        │ Kernel 执行   │     │
│   │ (CPU继续干活) │                        │  ├─ CU 计算   │     │
│   │              │                        │  └─ TU 计算   │     │
│   │ cudaSync     │ <────(等待完成)──────  │ 执行完毕      │     │
│   └──────────────┘                        └──────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

**关键点：**
- CPU 调用 CUDA API 是**异步**的（发射命令后立刻返回）
- GPU 真正执行是**之后**才发生的
- 所以有两条时间线：**Host Timeline** 和 **Device Timeline**

**两种 Profiling 模式：**

| 模式 | 看什么 | 解决什么问题 |
|------|--------|--------------|
| **Application Profiling** | 整体时间线、API 调用、并发 | 哪个 API 慢？kernel 什么时候执行？有没有并行？ |
| **Kernel Profiling** | kernel 内部硬件计数器 | kernel 内部算得有多满？用了多少计算单元？ |

---

## 三、dlpti_tools 命令行工具

dlpti_tools 包含三个子命令：

```bash
dlpti_tools capture   # Application Profiling（时间线分析）
dlpti_tools kprof     # Kernel Profiling（硬件计数器分析）
dlpti_tools export    # 数据格式转换
```

### 3.1 Application Profiling (capture)

**用途：** 分析程序的整体时间线，看 API 调用、kernel 执行、并发情况。

#### 基本用法

```bash
# 最简单的用法
dlpti_tools capture -- ./your_app args

# 输出文件默认为 capture-{datetime}.db
# 生成: capture-20250110103126.db
```

#### 指定输出文件

```bash
dlpti_tools capture --data-file ./my_profile.db -- ./your_app args
```

#### 指定采集的事件类型 (--activity-mask)

```bash
# 采集特定类型的 activity
dlpti_tools capture --activity-mask cmd,cu,curt,nne --data-file ./out.db -- ./your_app
```

**activity-mask 选项说明：**

| mask | 含义 | 说明 |
|------|------|------|
| `cmd` | GPU Command | kernel launch、memcpy 等命令 |
| `cu` | Compute Unit | 计算单元的活动 |
| `curt` | CUDA Runtime | CUDA runtime API 调用 |
| `nne` | NNE | 神经网络引擎活动（如果有） |
| `cublas` | cuBLAS | cuBLAS 库调用 |
| `cufft` | cuFFT | cuFFT 库调用 |
| `cudnn` | cuDNN | cuDNN 库调用 |
| `ext` | Extension | 自定义标记（Range/Marker） |
| `res` | Resource | 资源相关 |

**默认值：** `cmd,cu,curt`

**使用 `+` 添加到默认值：**
```bash
# 在默认基础上添加 ext 和 cublas
dlpti_tools capture --activity-mask +ext,cublas -- ./your_app
```

#### 限定采集范围 (--capture-range)

有时程序很长，但只想分析其中一段：

**方法1：在代码中标记范围**

```cpp
// 你的程序
void main() {
    init();        // 不想分析

    cudaProfilerStart();   // 从这里开始采集
    train_model(); // 想分析的部分
    cudaProfilerStop();    // 到这里结束采集

    cleanup();     // 不想分析
}
```

```bash
# 运行时只采集标记范围内的
dlpti_tools capture --capture-range cudaProfilerApi -- ./your_app
```

**方法2：export 时指定范围**

```bash
# 导出时只导出特定时间段
dlpti_tools export --export-range 0:2.4 --format perfetto-json ./out.db   # 绝对时间（秒）
dlpti_tools export --export-range 20%:40% --format perfetto-json ./out.db # 百分比
```

#### 导出到 Perfetto JSON（可选）

```bash
# dlsys-ui 已经可以直接打开 db 文件，这步是可选的
dlpti_tools export --format perfetto-json ./capture.db
# 生成: capture.perfetto.json
```

### 3.2 Kernel Profiling (kprof)

**用途：** 采集 kernel 执行时的硬件计数器，分析 kernel 内部的执行效率。

#### 基本用法

```bash
# 采集硬件计数器
dlpti_tools kprof --targets cu,tu --data-file ./kernel.db -- ./your_app args

# 导出为 metric 格式
dlpti_tools export --format metric ./kernel.db
# 生成: kernel.metric.db

# 用 GUI 查看
dlkprof-ui ./kernel.metric.db
```

#### 参数说明

| 参数 | 说明 |
|------|------|
| `--targets cu,tu` | 采集目标：cu (计算单元), tu (张量单元) |
| `--data-file` | 输出文件 |
| `--devices <ID>` | 指定 GPU 设备 |
| `-k, --kernel-name <REGEX>` | 过滤特定 kernel |
| `--metrics <QUERY>` | 指定要收集的指标 |
| `--list-metrics` | 列出所有可用指标 |

#### 示例

```bash
# 只采集特定 kernel
dlpti_tools kprof --targets cu --kernel-name "my_kernel" --data-file ./out.db -- ./app

# 查看可用的 metrics
dlpti_tools kprof --targets cu --list-metrics -- ./app
```

**注意：** 多进程、多线程应用中，kernel profiling 可能获取的数据不完全可靠。

### 3.3 Export 数据转换

```bash
# Application Profiling 导出为 Perfetto JSON
dlpti_tools export --format perfetto-json ./capture.db

# Kernel Profiling 导出为 metric 格式
dlpti_tools export --format metric ./kernel.db

# 导出特定时间范围
dlpti_tools export --export-range 0:2.4 --format perfetto-json ./capture.db
dlpti_tools export --export-range 20%:40% --format perfetto-json ./capture.db
```

---

## 四、dlinspect GUI 工具

dlinspect 是 GUI 分析工具集，包含：
- **dlsys-ui**: 查看 Application Profiling 结果
- **dlkprof-ui**: 查看 Kernel Profiling 结果

### 4.1 dlsys-ui (时间线分析)

**用途：** 查看 Application Profiling 的时间线数据。

#### 基本用法

```bash
# 直接打开 db 文件
dlsys-ui ./capture.db

# 启动空 GUI
dlsys-ui
```

#### 加载大文件的部分内容 (--import-range)

如果 db 文件太大，可以只加载部分：

```bash
# 按百分比加载
dlsys-ui --import-range 10%:50% ./capture.db    # 加载 10% 到 50% 时间段

# 按绝对时间加载（秒）
dlsys-ui --import-range 1.2:3.5 ./capture.db    # 加载 1.2s 到 3.5s
dlsys-ui --import-range 2.0: ./capture.db       # 从 2.0s 到结束
dlsys-ui --import-range :5.0 ./capture.db       # 从开始到 5.0s
```

**import-range 语法：**

| 格式 | 含义 |
|------|------|
| `s:e` | 从时间 s 到时间 e |
| `s:` | 从时间 s 到结束 |
| `:e` | 从开始到时间 e |
| `-s:-e` | 从 (结束-s) 到 (结束-e) |
| `s:+st` | 从时间 s 开始，持续 st |
| `s%:e%` | 从 s% 到 e% |
| `s%:` | 从 s% 到结束 |

### 4.2 dlkprof-ui (Kernel 指标分析)

**用途：** 查看 Kernel Profiling 的硬件计数器数据。

#### 基本用法

```bash
# 打开 metric.db 文件
dlkprof-ui ./kernel.metric.db

# 启动空 GUI
dlkprof-ui
```

**注意：** 必须先用 `dlpti_tools export --format metric` 导出后才能用 dlkprof-ui 查看。

---

## 五、Extension API (自定义标记)

### 5.1 为什么需要自定义标记

默认的时间线上只有 CUDA API 和 kernel，看不出哪个对应你代码的哪个业务逻辑。

通过在代码中插入标记，可以让时间线显示你的业务逻辑：

```
没有标记:
|--cudaLaunchKernel--|--cudaMemcpy--|--cudaLaunchKernel--|
     kernel_A             memcpy          kernel_B

有标记:
|--ForwardPass--------|--BackwardPass----| OptimizerStep
     kernel_A             memcpy          kernel_B
```

### 5.2 使用方法

**1. 包含头文件**

```cpp
#include <dlpti/ext.hpp>
```

**2. 注册 Domain（可选，用于分类）**

```cpp
// 通常放在头文件中
DLPTI_REGISTER_DOMAIN(network_domain, "NetworkLayer");
DLPTI_REGISTER_DOMAIN(data_domain, "DataProcessing");
```

**3. 添加标记**

```cpp
// 方式1: 使用注册的 domain
void process_request() {
    DLPTI_DOMAIN_RANGE(network_domain, "ProcessRequest");  // 范围标记
    network_domain.mark("Checkpoint");                      // 单点标记
}

// 方式2: 使用全局 domain
void my_function() {
    DLPTI_RANGE("MyFunction");           // 范围标记
    DLPTI_MARK("ImportantPoint");        // 单点标记
}
```

**4. CMake 链接**

```cmake
target_link_libraries(your_target PRIVATE dlpti_ext)
```

**5. 运行时开启 ext activity**

```bash
dlpti_tools capture --activity-mask +ext -- ./your_app
```

---

## 六、完整工作流程

### 6.1 Application Profiling 流程

```bash
# Step 1: 采集数据
dlpti_tools capture --data-file ./app.db -- ./your_app args

# Step 2: 直接用 GUI 查看
dlsys-ui ./app.db

# [可选] Step 3: 导出到 Perfetto JSON（给别人用 Perfetto UI 看）
dlpti_tools export --format perfetto-json ./app.db
```

### 6.2 Kernel Profiling 流程

```bash
# Step 1: 采集硬件计数器
dlpti_tools kprof --targets cu,tu --data-file ./kernel.db -- ./your_app args

# Step 2: 导出为 metric 格式
dlpti_tools export --format metric ./kernel.db

# Step 3: 用 GUI 查看
dlkprof-ui ./kernel.metric.db
```

### 6.3 和 NV 命令对比

**Application Profiling (时间线):**

| NV | DL |
|----|-----|
| `nsys profile -o out ./app args` | `dlpti_tools capture --data-file out.db -- ./app args` |
| `nsys-ui out.qdrep` | `dlsys-ui out.db` |

**Kernel Profiling (硬件计数器):**

| NV | DL |
|----|-----|
| `ncu --set full -o out ./app args` | `dlpti_tools kprof --targets cu,tu --data-file out.db -- ./app args` |
| `ncu-ui out.ncu-rep` | `dlpti_tools export --format metric out.db` → `dlkprof-ui out.metric.db` |

---

## 七、常见问题

### 7.1 db 文件太大怎么办

**方法1: 减少采集内容**
```bash
# 只采集 cmd，比默认减少约 70%
dlpti_tools capture --activity-mask cmd -- ./app
```

**方法2: 限定采集范围**
```bash
# 只采集 cudaProfilerStart/Stop 之间的内容
dlpti_tools capture --capture-range cudaProfilerApi -- ./app
```

**方法3: 只加载部分数据**
```bash
# 只加载 10% 到 50% 时间段
dlsys-ui --import-range 10%:50% ./app.db
```

### 7.2 Host/Device Timeline 错位

**现象：** Device Command 显示的执行时间早于 API 调用时间

**原因：**
1. V2 Device Clock 和 Host Clock 无法完全同步，长时间 profiling 会有累计误差
2. 硬件开启了 DVFS 导致 Device Timestamp 不稳定

**解决：** 用 `dlsmi` 工具锁频

### 7.3 Windows 打不开 db 文件

**错误：** `Failed to read from stream: DLPTI library not loaded`

**解决：** 安装 [Visual C++ Redistributable](https://aka.ms/vc14/vc_redist.x64.exe)

### 7.4 COMMAND 无对应的 HOST API

**原因：** 上层软件没有 trace 相应的 API

---

## 八、API 库说明

| 库 | 用途 | 头文件位置 |
|----|------|-----------|
| `libdlpti.so` | Callback 和 Activity API | `${SDK_DIR}/include/dlpti/` |
| `libdlpti_tools.so` | 数据文件读取 | `${SDK_DIR}/include/dlpti/tools` |
| `libdlpti_ext.so` | Extension API (Range/Marker) | `${SDK_DIR}/include/dlpti/ext.h, ext.hpp` |

### 动态加载注意事项

如果用 `dlopen` 动态加载 `libdlpti.so`，必须使用 `RTLD_GLOBAL`：

```cpp
void* handle = dlopen("libdlpti.so", RTLD_NOW | RTLD_GLOBAL);
```

### 自动加载

设置环境变量自动加载 dlpti library：

```bash
export DLPTI_AUTO_LOAD=1
/your-command ...
```

---

## 九、快速参考

### 常用命令速查

```bash
# === Application Profiling ===
# 基本采集
dlpti_tools capture --data-file ./app.db -- ./your_app args

# 带自定义标记采集
dlpti_tools capture --activity-mask +ext --data-file ./app.db -- ./your_app

# 限定范围采集
dlpti_tools capture --capture-range cudaProfilerApi --data-file ./app.db -- ./your_app

# 查看
dlsys-ui ./app.db

# === Kernel Profiling ===
# 采集
dlpti_tools kprof --targets cu,tu --data-file ./kernel.db -- ./your_app args

# 导出
dlpti_tools export --format metric ./kernel.db

# 查看
dlkprof-ui ./kernel.metric.db

# === 查看帮助 ===
dlpti_tools --help
dlpti_tools capture --help
dlpti_tools kprof --help
dlpti_tools export --help
dlsys-ui --help
dlkprof-ui --help
```

### 文件格式对应

| 文件类型 | 来源 | 查看工具 |
|----------|------|----------|
| `capture.db` / `*.db` | `dlpti_tools capture` | `dlsys-ui` |
| `*.metric.db` | `dlpti_tools export --format metric` | `dlkprof-ui` |
| `*.perfetto.json` | `dlpti_tools export --format perfetto-json` | Perfetto UI |
