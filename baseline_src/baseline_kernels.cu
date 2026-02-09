/*
 * Baseline Kernels Implementation
 */

#include "baseline_kernels.h"

/*
 * empty_kernel: Does absolutely nothing
 *
 * This kernel measures the absolute minimum overhead of:
 * - GPU command processing
 * - Kernel launch
 * - Kernel completion
 * - Event timing
 *
 * Any kernel will have at least this much overhead.
 */
__global__ void empty_kernel_impl() {
    // Intentionally empty - measures pure launch overhead
}

/*
 * minimal_rw_kernel: Minimal read + write operation
 *
 * This kernel performs exactly one read and one write to global memory.
 * The volatile qualifier forces the compiler to:
 * - Actually read from memory (not use cached register value)
 * - Actually write to memory (not optimize away the store)
 *
 * The "+ 1.0f" operation ensures the read value must be used,
 * preventing the compiler from eliminating the read.
 */
__global__ void minimal_rw_kernel_impl(volatile float* out, volatile const float* in) {
    // Only thread 0 performs the operation
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Read one value from global memory
        float val = in[0];

        // Use the value (required to prevent read optimization)
        // Write one value to global memory
        out[0] = val + 1.0f;
    }
}

/*
 * Wrapper functions for calling from .cpp files
 */

void launch_empty_kernel(cudaStream_t stream) {
    empty_kernel_impl<<<1, 1, 0, stream>>>();
}

void launch_minimal_rw_kernel(float* out, const float* in, cudaStream_t stream) {
    minimal_rw_kernel_impl<<<1, 1, 0, stream>>>(out, in);
}
