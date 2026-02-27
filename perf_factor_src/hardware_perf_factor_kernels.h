/*
 * Hardware Performance Factor Kernels
 *
 * Measure various hardware performance factors to understand
 * performance differences between GPU platforms (NV/DL).
 *
 * Test Categories:
 * 1. Kernel Launch Overhead - Empty kernel, minimal RW
 * 2. Memory Bandwidth - Coalesced, strided, random access
 * 3. Warp Operations - Shuffle, broadcast, reduce
 * 4. Synchronization - __syncthreads, __syncwarp
 * 5. Compute - expf, comparison operations
 * 6. Parallelism - Block size impact
 */

#ifndef HARDWARE_PERF_FACTOR_KERNELS_H
#define HARDWARE_PERF_FACTOR_KERNELS_H

#include <cuda_runtime.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================
// Configuration
// ============================================

#define PERF_WARMUP_ITERATIONS  100
#define PERF_TEST_ITERATIONS    100
#define PERF_DATA_SIZE_MB       10      // 10 MB for memory tests
#define PERF_COMPUTE_ITERATIONS 1000    // Iterations for compute/warp/sync tests
#define PERF_N_BLOCKS           256     // Number of blocks for compute tests
#define PERF_N_THREADS          128     // Number of threads for compute tests

// ============================================
// Test Categories
// ============================================

typedef enum {
    // Launch overhead
    TEST_LAUNCH_EMPTY = 0,
    TEST_LAUNCH_MINIMAL_RW,

    // Memory bandwidth (10 MB data)
    TEST_MEM_COALESCED_READ,
    TEST_MEM_COALESCED_WRITE,
    TEST_MEM_COALESCED_COPY,
    TEST_MEM_STRIDED_READ,
    TEST_MEM_RANDOM_READ,

    // Warp operations (per operation, 1000 iterations)
    TEST_WARP_SHFL_XOR,
    TEST_WARP_SHFL_UP,
    TEST_WARP_BROADCAST,
    TEST_WARP_REDUCE_SUM,
    TEST_WARP_REDUCE_MAX,

    // Synchronization (per operation, 1000 iterations)
    TEST_SYNC_THREADS,
    TEST_SYNC_WARP,

    // Compute (per operation, 1000 iterations)
    TEST_COMPUTE_EXPF,
    TEST_COMPUTE_COMPARE,

    // Parallelism (100 iterations each)
    TEST_BLOCK_32,
    TEST_BLOCK_64,
    TEST_BLOCK_128,
    TEST_BLOCK_256,
    TEST_BLOCK_512,

    // Total count
    TEST_COUNT
} PerfTestType;

// ============================================
// Result Structure
// ============================================

typedef struct {
    const char* name;           // Test name
    const char* category;       // Category for grouping
    float time_us;              // Average time in microseconds
    float bandwidth_gbps;       // Effective bandwidth (GB/s), 0 if N/A
    float per_op_us;            // Time per operation (us), 0 if N/A
    const char* description;    // What is being measured
} PerfTestResult;

// ============================================
// Kernel Launchers - Launch Overhead
// ============================================

void perf_launch_empty(cudaStream_t stream);
void perf_launch_minimal_rw(float* out, const float* in, cudaStream_t stream);

// ============================================
// Kernel Launchers - Memory Bandwidth
// ============================================

void perf_mem_coalesced_read(float* dst, const float* src, size_t n_elements,
                             int grid, int block, cudaStream_t stream);
void perf_mem_coalesced_write(float* dst, size_t n_elements,
                              int grid, int block, cudaStream_t stream);
void perf_mem_coalesced_copy(float* dst, const float* src, size_t n_elements,
                             int grid, int block, cudaStream_t stream);
void perf_mem_strided_read(float* dst, const float* src, size_t n_elements, int stride,
                           int grid, int block, cudaStream_t stream);
void perf_mem_random_read(float* dst, const float* src, const int* indices, size_t n_elements,
                          int grid, int block, cudaStream_t stream);

// ============================================
// Kernel Launchers - Warp Operations
// ============================================

void perf_warp_shfl_xor(float* output, int n_blocks, int n_iters, cudaStream_t stream);
void perf_warp_shfl_up(float* output, int n_blocks, int n_iters, cudaStream_t stream);
void perf_warp_broadcast(float* output, int n_blocks, int n_iters, cudaStream_t stream);
void perf_warp_reduce_sum(float* output, int n_blocks, int n_iters, cudaStream_t stream);
void perf_warp_reduce_max(float* output, int n_blocks, int n_iters, cudaStream_t stream);

// ============================================
// Kernel Launchers - Synchronization
// ============================================

void perf_sync_threads(float* output, int n_blocks, int n_threads, int n_iters, cudaStream_t stream);
void perf_sync_warp(float* output, int n_blocks, int n_iters, cudaStream_t stream);

// ============================================
// Kernel Launchers - Compute
// ============================================

void perf_compute_expf(float* output, int n_blocks, int n_threads, int n_iters, cudaStream_t stream);
void perf_compute_compare(float* output, const float* input, int n_blocks, int n_threads,
                          int n_iters, cudaStream_t stream);

// ============================================
// Kernel Launchers - Block Size
// ============================================

void perf_block_size(float* output, int n_blocks, int block_size, int n_iters, cudaStream_t stream);

// ============================================
// Benchmark Runner
// ============================================

// Run all tests and populate results array
// Returns 0 on success, non-zero on error
int run_all_perf_tests(PerfTestResult* results, int warmup_iters, int test_iters);

// Print results in formatted table
void print_perf_results(const PerfTestResult* results, int count, const char* platform_info);

#ifdef __cplusplus
}
#endif

#endif // HARDWARE_PERF_FACTOR_KERNELS_H
