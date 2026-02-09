/*
 * Baseline Kernels for GPU Overhead Measurement
 *
 * These kernels are designed to measure the minimum fixed overhead of GPU execution.
 * Used for comparing with actual kernels to determine if performance issues are due to:
 * - Kernel implementation problems, OR
 * - Inherent GPU overhead that dominates small workloads
 */

#ifndef BASELINE_KERNELS_H
#define BASELINE_KERNELS_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * launch_empty_kernel: Launches an empty kernel
 *
 * Purpose: Measure the minimum GPU launch + event overhead
 * Expected time: ~2-4 microseconds (GPU architecture dependent)
 *
 * @param stream: CUDA stream to launch on (0 for default)
 */
void launch_empty_kernel(cudaStream_t stream);

/*
 * launch_minimal_rw_kernel: Launches a kernel that reads and writes one value
 *
 * Purpose: Measure minimum GPU overhead with minimal memory access
 * Expected time: ~3-5 microseconds (slightly higher than empty_kernel)
 *
 * @param out: Output buffer (must be GPU memory, at least 1 float)
 * @param in: Input buffer (must be GPU memory, at least 1 float)
 * @param stream: CUDA stream to launch on (0 for default)
 */
void launch_minimal_rw_kernel(float* out, const float* in, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // BASELINE_KERNELS_H
