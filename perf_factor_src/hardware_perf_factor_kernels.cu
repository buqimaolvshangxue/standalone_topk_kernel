/*
 * Hardware Performance Factor Kernels Implementation
 */

#include "hardware_perf_factor_kernels.h"
#include <stdio.h>

// ============================================
// Launch Overhead Kernels
// ============================================

__global__ void kernel_empty() {
    // Intentionally empty - measures pure launch overhead
}

__global__ void kernel_minimal_rw(volatile float* out, volatile const float* in) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out[0] = in[0] + 1.0f;
    }
}

void perf_launch_empty(cudaStream_t stream) {
    kernel_empty<<<1, 1, 0, stream>>>();
}

void perf_launch_minimal_rw(float* out, const float* in, cudaStream_t stream) {
    kernel_minimal_rw<<<1, 1, 0, stream>>>(out, in);
}

// ============================================
// Memory Bandwidth Kernels
// ============================================

// Pure read - each thread reads one element
// Use reduction pattern to sum results (prevents compiler optimization)
__global__ void kernel_coalesced_read(float* dst, const float* src, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Simply copy to dst - the read is what we're measuring
        dst[idx] = src[idx];
    }
}

// Pure write - each thread writes one element
__global__ void kernel_coalesced_write(float* dst, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = (float)idx * 0.001f + 1.0f;
    }
}

// Copy - read + write
__global__ void kernel_coalesced_copy(float* dst, const float* src, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// Strided read - simulates non-coalesced memory access
// Thread i reads address i*stride, creating non-coalesced access pattern
// Total elements read: min(total_threads, n/stride)
// Bandwidth should be calculated based on actual data read
__global__ void kernel_strided_read(float* dst, const float* src, size_t n, int stride) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idx = tid * stride;  // Non-coalesced: adjacent threads access addresses stride apart

    if (idx < n) {
        dst[tid] = src[idx];
    }
}

// Random read - indirect access pattern
__global__ void kernel_random_read(float* dst, const float* src, const int* indices, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < n; i += stride) {
        sum += src[indices[i]];  // Indirect access
    }

    if (idx < n) {
        dst[idx] = sum;
    }
}

void perf_mem_coalesced_read(float* dst, const float* src, size_t n_elements,
                             int grid, int block, cudaStream_t stream) {
    kernel_coalesced_read<<<grid, block, 0, stream>>>(dst, src, n_elements);
}

void perf_mem_coalesced_write(float* dst, size_t n_elements,
                              int grid, int block, cudaStream_t stream) {
    kernel_coalesced_write<<<grid, block, 0, stream>>>(dst, n_elements);
}

void perf_mem_coalesced_copy(float* dst, const float* src, size_t n_elements,
                             int grid, int block, cudaStream_t stream) {
    kernel_coalesced_copy<<<grid, block, 0, stream>>>(dst, src, n_elements);
}

void perf_mem_strided_read(float* dst, const float* src, size_t n_elements, int stride,
                           int grid, int block, cudaStream_t stream) {
    kernel_strided_read<<<grid, block, 0, stream>>>(dst, src, n_elements, stride);
}

void perf_mem_random_read(float* dst, const float* src, const int* indices, size_t n_elements,
                          int grid, int block, cudaStream_t stream) {
    kernel_random_read<<<grid, block, 0, stream>>>(dst, src, indices, n_elements);
}

// ============================================
// Warp Operations Kernels
// ============================================

// Butterfly reduction pattern (similar to TopK kernel)
// Each iteration does 5 shuffle operations (mask = 16, 8, 4, 2, 1)
__global__ void kernel_warp_shfl_xor(float* output, int n_iters) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    float val = (float)(blockIdx.x * blockDim.x + tid);

    for (int iter = 0; iter < n_iters; iter++) {
        // Butterfly reduction for max (5 steps for 32 threads)
        for (int mask = 16; mask > 0; mask /= 2) {
            float other = __shfl_xor_sync(0xffffffff, val, mask, 32);
            val = max(val, other);
        }
    }

    // Write result to prevent optimization
    if (lane_id == 0) {
        output[blockIdx.x * (blockDim.x / 32) + warp_id] = val;
    }
}

// Prefix sum - 5 shuffle operations per iteration
__global__ void kernel_warp_shfl_up(float* output, int n_iters) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    float val = (float)(blockIdx.x * blockDim.x + tid);

    for (int iter = 0; iter < n_iters; iter++) {
        // Prefix sum using shfl_up (5 steps)
        for (int offset = 1; offset <= 16; offset *= 2) {
            float other = __shfl_up_sync(0xffffffff, val, offset, 32);
            if (lane_id >= offset) {
                val += other;
            }
        }
    }

    if (lane_id == 0) {
        output[blockIdx.x * (blockDim.x / 32) + warp_id] = val;
    }
}

// Broadcast - 1 shuffle operation per iteration
__global__ void kernel_warp_broadcast(float* output, int n_iters) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    float val = (float)(blockIdx.x * blockDim.x + tid);

    for (int iter = 0; iter < n_iters; iter++) {
        // Broadcast from lane 0
        float broadcast_val = __shfl_sync(0xffffffff, val, 0, 32);
        val = broadcast_val + (float)lane_id;
    }

    if (lane_id == 0) {
        output[blockIdx.x * (blockDim.x / 32) + warp_id] = val;
    }
}

// Warp reduce sum - 5 shuffle operations per iteration
__global__ void kernel_warp_reduce_sum(float* output, int n_iters) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    float val = (float)(blockIdx.x * blockDim.x + tid);

    for (int iter = 0; iter < n_iters; iter++) {
        // Warp reduce sum
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset, 32);
        }
    }

    if (lane_id == 0) {
        output[blockIdx.x * (blockDim.x / 32) + warp_id] = val;
    }
}

// Warp reduce max - 5 shuffle operations per iteration
__global__ void kernel_warp_reduce_max(float* output, int n_iters) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    float val = (float)(blockIdx.x * blockDim.x + tid);

    for (int iter = 0; iter < n_iters; iter++) {
        // Warp reduce max
        for (int offset = 16; offset > 0; offset /= 2) {
            float other = __shfl_down_sync(0xffffffff, val, offset, 32);
            val = max(val, other);
        }
    }

    if (lane_id == 0) {
        output[blockIdx.x * (blockDim.x / 32) + warp_id] = val;
    }
}

void perf_warp_shfl_xor(float* output, int n_blocks, int n_iters, cudaStream_t stream) {
    kernel_warp_shfl_xor<<<n_blocks, 128, 0, stream>>>(output, n_iters);
}

void perf_warp_shfl_up(float* output, int n_blocks, int n_iters, cudaStream_t stream) {
    kernel_warp_shfl_up<<<n_blocks, 128, 0, stream>>>(output, n_iters);
}

void perf_warp_broadcast(float* output, int n_blocks, int n_iters, cudaStream_t stream) {
    kernel_warp_broadcast<<<n_blocks, 128, 0, stream>>>(output, n_iters);
}

void perf_warp_reduce_sum(float* output, int n_blocks, int n_iters, cudaStream_t stream) {
    kernel_warp_reduce_sum<<<n_blocks, 128, 0, stream>>>(output, n_iters);
}

void perf_warp_reduce_max(float* output, int n_blocks, int n_iters, cudaStream_t stream) {
    kernel_warp_reduce_max<<<n_blocks, 128, 0, stream>>>(output, n_iters);
}

// ============================================
// Synchronization Kernels
// ============================================

// 4 __syncthreads() per iteration
__global__ void kernel_sync_threads(float* output, int n_iters) {
    int tid = threadIdx.x;
    float val = (float)tid;

    for (int iter = 0; iter < n_iters; iter++) {
        __syncthreads();
        __syncthreads();
        __syncthreads();
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = val;
    }
}

// 4 __syncwarp() per iteration
__global__ void kernel_sync_warp(float* output, int n_iters) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    float val = (float)tid;

    for (int iter = 0; iter < n_iters; iter++) {
        __syncwarp(0xffffffff);
        __syncwarp(0xffffffff);
        __syncwarp(0xffffffff);
        __syncwarp(0xffffffff);
    }

    if (lane_id == 0) {
        output[blockIdx.x * (blockDim.x / 32) + warp_id] = val;
    }
}

void perf_sync_threads(float* output, int n_blocks, int n_threads, int n_iters, cudaStream_t stream) {
    kernel_sync_threads<<<n_blocks, n_threads, 0, stream>>>(output, n_iters);
}

void perf_sync_warp(float* output, int n_blocks, int n_iters, cudaStream_t stream) {
    kernel_sync_warp<<<n_blocks, 128, 0, stream>>>(output, n_iters);
}

// ============================================
// Compute Kernels
// ============================================

// 1 expf() per iteration
__global__ void kernel_compute_expf(float* output, int n_iters) {
    int tid = threadIdx.x;
    float val = (float)(tid + 1) * 0.001f;

    for (int iter = 0; iter < n_iters; iter++) {
        val = expf(val);
        // Prevent value from exploding/shrinking
        if (val > 10.0f) val = 0.001f;
        if (val < 0.0001f) val = 0.001f;
    }

    if (tid == 0) {
        output[blockIdx.x] = val;
    }
}

// 128 comparisons per iteration (simulating TopK over 128 experts)
__global__ void kernel_compute_compare(float* output, const float* input, int n_iters) {
    int tid = threadIdx.x;
    float max_val = -1e30f;
    int max_idx = -1;

    // Simulate TopK-like comparison over 128 elements
    for (int iter = 0; iter < n_iters; iter++) {
        max_val = -1e30f;
        max_idx = -1;
        for (int i = 0; i < 128; i++) {
            float val = input[i];
            if (val > max_val) {
                max_val = val;
                max_idx = i;
            }
        }
    }

    if (tid == 0) {
        output[blockIdx.x] = max_val + (float)max_idx;
    }
}

void perf_compute_expf(float* output, int n_blocks, int n_threads, int n_iters, cudaStream_t stream) {
    kernel_compute_expf<<<n_blocks, n_threads, 0, stream>>>(output, n_iters);
}

void perf_compute_compare(float* output, const float* input, int n_blocks, int n_threads,
                          int n_iters, cudaStream_t stream) {
    kernel_compute_compare<<<n_blocks, n_threads, 0, stream>>>(output, input, n_iters);
}

// ============================================
// Block Size Kernels
// ============================================

__global__ void kernel_block_size(float* output, int n_iters) {
    int tid = threadIdx.x;
    float val = (float)tid;

    for (int iter = 0; iter < n_iters; iter++) {
        val = val * 1.0001f + 0.001f;
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = val;
    }
}

void perf_block_size(float* output, int n_blocks, int block_size, int n_iters, cudaStream_t stream) {
    kernel_block_size<<<n_blocks, block_size, 0, stream>>>(output, n_iters);
}
