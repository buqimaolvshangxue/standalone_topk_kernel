/*
 * Hardware Performance Factor Benchmark
 *
 * Measures various hardware performance factors to compare GPU platforms.
 *
 * Usage: ./bench_hardware_perf_factor [warmup_iters] [test_iters]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include "hardware_perf_factor_kernels.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "[ERROR] CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Helper: get platform info
static void get_platform_info(char* info, size_t max_len) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    snprintf(info, max_len, "%s (%d SMs, %zu MB)",
             prop.name, prop.multiProcessorCount,
             prop.totalGlobalMem / (1024 * 1024));
}

int run_all_perf_tests(PerfTestResult* results, int warmup_iters, int test_iters) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Allocate working memory
    size_t data_size = (size_t)PERF_DATA_SIZE_MB * 1024 * 1024;
    size_t n_elements = data_size / sizeof(float);

    float *d_src = NULL, *d_dst = NULL, *d_output = NULL;
    int* d_indices = NULL;

    CUDA_CHECK(cudaMalloc(&d_src, data_size));
    CUDA_CHECK(cudaMalloc(&d_dst, data_size));
    CUDA_CHECK(cudaMalloc(&d_output, 1024 * 1024));
    CUDA_CHECK(cudaMalloc(&d_indices, n_elements * sizeof(int)));

    // Initialize data
    float* h_data = (float*)malloc(data_size);
    int* h_indices = (int*)malloc(n_elements * sizeof(int));
    for (size_t i = 0; i < n_elements; i++) {
        h_data[i] = (float)i * 0.001f;
        h_indices[i] = (int)(rand() % n_elements);
    }
    CUDA_CHECK(cudaMemcpy(d_src, h_data, data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_indices, h_indices, n_elements * sizeof(int), cudaMemcpyHostToDevice));

    // Memory tests: use enough blocks to cover all elements
    int block = 256;
    int grid = (n_elements + block - 1) / block;
    // Don't limit grid size for bandwidth tests - need to cover all data

    int idx = 0;
    float time_us, total_ms;
    int i;

    // ========================================
    // Launch Overhead Tests
    // ========================================

    // Empty kernel
    results[idx].name = "Empty Kernel";
    results[idx].category = "Launch Overhead";
    results[idx].description = "Pure kernel launch overhead";
    for (i = 0; i < warmup_iters; i++) perf_launch_empty(0);
    CUDA_CHECK(cudaDeviceSynchronize());
    total_ms = 0.0f;
    for (i = 0; i < test_iters; i++) {
        cudaEventRecord(start);
        perf_launch_empty(0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    time_us = (total_ms / test_iters) * 1000.0f;
    results[idx].time_us = time_us;
    results[idx].bandwidth_gbps = 0;
    results[idx].per_op_us = time_us;
    idx++;

    // Minimal RW
    results[idx].name = "Minimal RW";
    results[idx].category = "Launch Overhead";
    results[idx].description = "Launch + 1 memory access";
    for (i = 0; i < warmup_iters; i++) perf_launch_minimal_rw(d_dst, d_src, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    total_ms = 0.0f;
    for (i = 0; i < test_iters; i++) {
        cudaEventRecord(start);
        perf_launch_minimal_rw(d_dst, d_src, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    time_us = (total_ms / test_iters) * 1000.0f;
    results[idx].time_us = time_us;
    results[idx].bandwidth_gbps = 0;
    results[idx].per_op_us = time_us;
    idx++;

    // ========================================
    // Memory Bandwidth Tests (10 MB)
    // ========================================

    // Coalesced Read
    results[idx].name = "Coalesced Read";
    results[idx].category = "Memory Bandwidth";
    results[idx].description = "Sequential memory read (10 MB)";
    for (i = 0; i < warmup_iters; i++) perf_mem_coalesced_read(d_dst, d_src, n_elements, grid, block, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    total_ms = 0.0f;
    for (i = 0; i < test_iters; i++) {
        cudaEventRecord(start);
        perf_mem_coalesced_read(d_dst, d_src, n_elements, grid, block, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    time_us = (total_ms / test_iters) * 1000.0f;
    results[idx].time_us = time_us;
    results[idx].bandwidth_gbps = data_size / (time_us * 1000.0f);
    results[idx].per_op_us = 0;
    idx++;

    // Coalesced Write
    results[idx].name = "Coalesced Write";
    results[idx].category = "Memory Bandwidth";
    results[idx].description = "Sequential memory write (10 MB)";
    for (i = 0; i < warmup_iters; i++) perf_mem_coalesced_write(d_dst, n_elements, grid, block, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    total_ms = 0.0f;
    for (i = 0; i < test_iters; i++) {
        cudaEventRecord(start);
        perf_mem_coalesced_write(d_dst, n_elements, grid, block, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    time_us = (total_ms / test_iters) * 1000.0f;
    results[idx].time_us = time_us;
    results[idx].bandwidth_gbps = data_size / (time_us * 1000.0f);
    results[idx].per_op_us = 0;
    idx++;

    // Coalesced Copy
    results[idx].name = "Coalesced Copy";
    results[idx].category = "Memory Bandwidth";
    results[idx].description = "Sequential read + write (10 MB)";
    for (i = 0; i < warmup_iters; i++) perf_mem_coalesced_copy(d_dst, d_src, n_elements, grid, block, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    total_ms = 0.0f;
    for (i = 0; i < test_iters; i++) {
        cudaEventRecord(start);
        perf_mem_coalesced_copy(d_dst, d_src, n_elements, grid, block, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    time_us = (total_ms / test_iters) * 1000.0f;
    results[idx].time_us = time_us;
    results[idx].bandwidth_gbps = (data_size * 2) / (time_us * 1000.0f);
    results[idx].per_op_us = 0;
    idx++;

    // Strided Read (NOTE: only reads n/stride elements due to access pattern)
    results[idx].name = "Strided Read";
    results[idx].category = "Memory Bandwidth";
    results[idx].description = "Non-coalesced read (10MB/128 actual)";
    for (i = 0; i < warmup_iters; i++) perf_mem_strided_read(d_dst, d_src, n_elements, 128, grid, block, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    total_ms = 0.0f;
    for (i = 0; i < test_iters; i++) {
        cudaEventRecord(start);
        perf_mem_strided_read(d_dst, d_src, n_elements, 128, grid, block, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    time_us = (total_ms / test_iters) * 1000.0f;
    results[idx].time_us = time_us;
    // Actual data read is data_size / 128 (due to strided pattern)
    results[idx].bandwidth_gbps = (data_size / 128) / (time_us * 1000.0f);
    results[idx].per_op_us = 0;
    idx++;

    // Random Read
    results[idx].name = "Random Read";
    results[idx].category = "Memory Bandwidth";
    results[idx].description = "Random access memory read (10 MB)";
    for (i = 0; i < warmup_iters; i++) perf_mem_random_read(d_dst, d_src, d_indices, n_elements, grid, block, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    total_ms = 0.0f;
    for (i = 0; i < test_iters; i++) {
        cudaEventRecord(start);
        perf_mem_random_read(d_dst, d_src, d_indices, n_elements, grid, block, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    time_us = (total_ms / test_iters) * 1000.0f;
    results[idx].time_us = time_us;
    results[idx].bandwidth_gbps = data_size / (time_us * 1000.0f);
    results[idx].per_op_us = 0;
    idx++;

    // ========================================
    // Warp Operations Tests (1000 iterations)
    // ========================================

    int n_warp_iters = PERF_COMPUTE_ITERATIONS;
    int n_blocks = PERF_N_BLOCKS;

    // Warp Shuffle XOR
    results[idx].name = "Warp Shfl XOR";
    results[idx].category = "Warp Operations";
    results[idx].description = "Butterfly reduction via shfl_xor (1000 iters)";
    for (i = 0; i < warmup_iters; i++) perf_warp_shfl_xor(d_output, n_blocks, n_warp_iters, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    total_ms = 0.0f;
    for (i = 0; i < test_iters; i++) {
        cudaEventRecord(start);
        perf_warp_shfl_xor(d_output, n_blocks, n_warp_iters, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    time_us = (total_ms / test_iters) * 1000.0f;
    results[idx].time_us = time_us;
    results[idx].bandwidth_gbps = 0;
    results[idx].per_op_us = time_us / n_warp_iters / 5;
    idx++;

    // Warp Shuffle UP
    results[idx].name = "Warp Shfl UP";
    results[idx].category = "Warp Operations";
    results[idx].description = "Prefix sum via shfl_up (1000 iters)";
    for (i = 0; i < warmup_iters; i++) perf_warp_shfl_up(d_output, n_blocks, n_warp_iters, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    total_ms = 0.0f;
    for (i = 0; i < test_iters; i++) {
        cudaEventRecord(start);
        perf_warp_shfl_up(d_output, n_blocks, n_warp_iters, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    time_us = (total_ms / test_iters) * 1000.0f;
    results[idx].time_us = time_us;
    results[idx].bandwidth_gbps = 0;
    results[idx].per_op_us = time_us / n_warp_iters / 5;
    idx++;

    // Warp Broadcast
    results[idx].name = "Warp Broadcast";
    results[idx].category = "Warp Operations";
    results[idx].description = "Broadcast from lane 0 (1000 iters)";
    for (i = 0; i < warmup_iters; i++) perf_warp_broadcast(d_output, n_blocks, n_warp_iters, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    total_ms = 0.0f;
    for (i = 0; i < test_iters; i++) {
        cudaEventRecord(start);
        perf_warp_broadcast(d_output, n_blocks, n_warp_iters, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    time_us = (total_ms / test_iters) * 1000.0f;
    results[idx].time_us = time_us;
    results[idx].bandwidth_gbps = 0;
    results[idx].per_op_us = time_us / n_warp_iters;
    idx++;

    // Warp Reduce Sum
    results[idx].name = "Warp Reduce Sum";
    results[idx].category = "Warp Operations";
    results[idx].description = "Warp sum via shfl_down (1000 iters)";
    for (i = 0; i < warmup_iters; i++) perf_warp_reduce_sum(d_output, n_blocks, n_warp_iters, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    total_ms = 0.0f;
    for (i = 0; i < test_iters; i++) {
        cudaEventRecord(start);
        perf_warp_reduce_sum(d_output, n_blocks, n_warp_iters, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    time_us = (total_ms / test_iters) * 1000.0f;
    results[idx].time_us = time_us;
    results[idx].bandwidth_gbps = 0;
    results[idx].per_op_us = time_us / n_warp_iters / 5;
    idx++;

    // Warp Reduce Max
    results[idx].name = "Warp Reduce Max";
    results[idx].category = "Warp Operations";
    results[idx].description = "Warp max via shfl_down (1000 iters)";
    for (i = 0; i < warmup_iters; i++) perf_warp_reduce_max(d_output, n_blocks, n_warp_iters, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    total_ms = 0.0f;
    for (i = 0; i < test_iters; i++) {
        cudaEventRecord(start);
        perf_warp_reduce_max(d_output, n_blocks, n_warp_iters, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    time_us = (total_ms / test_iters) * 1000.0f;
    results[idx].time_us = time_us;
    results[idx].bandwidth_gbps = 0;
    results[idx].per_op_us = time_us / n_warp_iters / 5;
    idx++;

    // ========================================
    // Synchronization Tests (1000 iterations)
    // ========================================

    int n_sync_iters = PERF_COMPUTE_ITERATIONS;
    int n_threads = PERF_N_THREADS;

    // __syncthreads
    results[idx].name = "__syncthreads()";
    results[idx].category = "Synchronization";
    results[idx].description = "Block-level barrier (1000 iters x 4)";
    for (i = 0; i < warmup_iters; i++) perf_sync_threads(d_output, n_blocks, n_threads, n_sync_iters, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    total_ms = 0.0f;
    for (i = 0; i < test_iters; i++) {
        cudaEventRecord(start);
        perf_sync_threads(d_output, n_blocks, n_threads, n_sync_iters, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    time_us = (total_ms / test_iters) * 1000.0f;
    results[idx].time_us = time_us;
    results[idx].bandwidth_gbps = 0;
    results[idx].per_op_us = time_us / n_sync_iters / 4;
    idx++;

    // __syncwarp
    results[idx].name = "__syncwarp()";
    results[idx].category = "Synchronization";
    results[idx].description = "Warp-level barrier (1000 iters x 4)";
    for (i = 0; i < warmup_iters; i++) perf_sync_warp(d_output, n_blocks, n_sync_iters, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    total_ms = 0.0f;
    for (i = 0; i < test_iters; i++) {
        cudaEventRecord(start);
        perf_sync_warp(d_output, n_blocks, n_sync_iters, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    time_us = (total_ms / test_iters) * 1000.0f;
    results[idx].time_us = time_us;
    results[idx].bandwidth_gbps = 0;
    results[idx].per_op_us = time_us / n_sync_iters / 4;
    idx++;

    // ========================================
    // Compute Tests (1000 iterations)
    // ========================================

    int n_compute_iters = PERF_COMPUTE_ITERATIONS;

    // expf
    results[idx].name = "expf()";
    results[idx].category = "Compute";
    results[idx].description = "Exponential function (1000 iters)";
    for (i = 0; i < warmup_iters; i++) perf_compute_expf(d_output, n_blocks, n_threads, n_compute_iters, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    total_ms = 0.0f;
    for (i = 0; i < test_iters; i++) {
        cudaEventRecord(start);
        perf_compute_expf(d_output, n_blocks, n_threads, n_compute_iters, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    time_us = (total_ms / test_iters) * 1000.0f;
    results[idx].time_us = time_us;
    results[idx].bandwidth_gbps = 0;
    results[idx].per_op_us = time_us / n_compute_iters;
    idx++;

    // Comparison (TopK-like)
    results[idx].name = "Compare/Select";
    results[idx].category = "Compute";
    results[idx].description = "TopK-like comparison (1000 iters, 128 elements)";
    for (i = 0; i < warmup_iters; i++) perf_compute_compare(d_output, d_src, n_blocks, n_threads, n_compute_iters, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    total_ms = 0.0f;
    for (i = 0; i < test_iters; i++) {
        cudaEventRecord(start);
        perf_compute_compare(d_output, d_src, n_blocks, n_threads, n_compute_iters, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    time_us = (total_ms / test_iters) * 1000.0f;
    results[idx].time_us = time_us;
    results[idx].bandwidth_gbps = 0;
    results[idx].per_op_us = time_us / n_compute_iters / 128;
    idx++;

    // ========================================
    // Block Size Tests (100 iterations)
    // ========================================

    int n_block_iters = 100;

    // Block 32
    results[idx].name = "Block=32";
    results[idx].category = "Block Size";
    results[idx].description = "Block size 32 (100 iters)";
    for (i = 0; i < warmup_iters; i++) perf_block_size(d_output, n_blocks, 32, n_block_iters, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    total_ms = 0.0f;
    for (i = 0; i < test_iters; i++) {
        cudaEventRecord(start);
        perf_block_size(d_output, n_blocks, 32, n_block_iters, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    time_us = (total_ms / test_iters) * 1000.0f;
    results[idx].time_us = time_us;
    results[idx].bandwidth_gbps = 0;
    results[idx].per_op_us = 0;
    idx++;

    // Block 64
    results[idx].name = "Block=64";
    results[idx].category = "Block Size";
    results[idx].description = "Block size 64 (100 iters)";
    for (i = 0; i < warmup_iters; i++) perf_block_size(d_output, n_blocks, 64, n_block_iters, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    total_ms = 0.0f;
    for (i = 0; i < test_iters; i++) {
        cudaEventRecord(start);
        perf_block_size(d_output, n_blocks, 64, n_block_iters, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    time_us = (total_ms / test_iters) * 1000.0f;
    results[idx].time_us = time_us;
    results[idx].bandwidth_gbps = 0;
    results[idx].per_op_us = 0;
    idx++;

    // Block 128
    results[idx].name = "Block=128";
    results[idx].category = "Block Size";
    results[idx].description = "Block size 128 (100 iters)";
    for (i = 0; i < warmup_iters; i++) perf_block_size(d_output, n_blocks, 128, n_block_iters, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    total_ms = 0.0f;
    for (i = 0; i < test_iters; i++) {
        cudaEventRecord(start);
        perf_block_size(d_output, n_blocks, 128, n_block_iters, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    time_us = (total_ms / test_iters) * 1000.0f;
    results[idx].time_us = time_us;
    results[idx].bandwidth_gbps = 0;
    results[idx].per_op_us = 0;
    idx++;

    // Block 256
    results[idx].name = "Block=256";
    results[idx].category = "Block Size";
    results[idx].description = "Block size 256 (100 iters)";
    for (i = 0; i < warmup_iters; i++) perf_block_size(d_output, n_blocks, 256, n_block_iters, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    total_ms = 0.0f;
    for (i = 0; i < test_iters; i++) {
        cudaEventRecord(start);
        perf_block_size(d_output, n_blocks, 256, n_block_iters, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    time_us = (total_ms / test_iters) * 1000.0f;
    results[idx].time_us = time_us;
    results[idx].bandwidth_gbps = 0;
    results[idx].per_op_us = 0;
    idx++;

    // Block 512
    results[idx].name = "Block=512";
    results[idx].category = "Block Size";
    results[idx].description = "Block size 512 (100 iters)";
    for (i = 0; i < warmup_iters; i++) perf_block_size(d_output, n_blocks, 512, n_block_iters, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    total_ms = 0.0f;
    for (i = 0; i < test_iters; i++) {
        cudaEventRecord(start);
        perf_block_size(d_output, n_blocks, 512, n_block_iters, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    time_us = (total_ms / test_iters) * 1000.0f;
    results[idx].time_us = time_us;
    results[idx].bandwidth_gbps = 0;
    results[idx].per_op_us = 0;
    idx++;

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_indices));
    free(h_data);
    free(h_indices);

    return 0;
}

void print_perf_results(const PerfTestResult* results, int count, const char* platform_info) {
    printf("\n");
    printf("==============================================\n");
    printf("  Hardware Performance Factor Analysis\n");
    printf("  Platform: %s\n", platform_info);
    printf("==============================================\n\n");

    const char* current_category = "";

    for (int i = 0; i < count; i++) {
        if (strcmp(current_category, results[i].category) != 0) {
            current_category = results[i].category;
            printf("[%s]\n", current_category);
        }

        printf("  %-20s", results[i].name);

        if (results[i].bandwidth_gbps > 0) {
            printf(" %8.2f us, %6.1f GB/s", results[i].time_us, results[i].bandwidth_gbps);
        } else if (results[i].per_op_us > 0) {
            printf(" %8.2f us, %.4f us/op", results[i].time_us, results[i].per_op_us);
        } else {
            printf(" %8.2f us", results[i].time_us);
        }

        printf("\n");
    }

    printf("\n==============================================\n");
}

int main(int argc, char** argv) {
    int warmup_iters = (argc > 1) ? atoi(argv[1]) : PERF_WARMUP_ITERATIONS;
    int test_iters = (argc > 2) ? atoi(argv[2]) : PERF_TEST_ITERATIONS;

    printf("=== Hardware Performance Factor Benchmark ===\n");
    printf("Warmup: %d iterations\n", warmup_iters);
    printf("Test: %d iterations\n", test_iters);
    printf("Data size: %d MB\n\n", PERF_DATA_SIZE_MB);

    char platform_info[256];
    get_platform_info(platform_info, sizeof(platform_info));
    printf("Platform: %s\n", platform_info);

    PerfTestResult results[TEST_COUNT];
    int ret = run_all_perf_tests(results, warmup_iters, test_iters);
    if (ret != 0) {
        fprintf(stderr, "Error running tests\n");
        return 1;
    }

    print_perf_results(results, TEST_COUNT, platform_info);

    return 0;
}
