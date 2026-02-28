/*
 * Hardware Performance Factor Benchmark (Single Factor)
 *
 * Usage: ./bench_factor <factor_name> [warmup] [iters]
 *
 * factor_name:
 *   Launch Overhead:
 *     - empty:        Empty kernel (pure launch overhead)
 *     - minimal_rw:   Minimal read-write kernel
 *
 *   Memory Bandwidth:
 *     - coalesced_read:   Sequential memory read (10 MB)
 *     - coalesced_write:  Sequential memory write (10 MB)
 *     - coalesced_copy:   Sequential read + write (10 MB)
 *     - strided_read:     Non-coalesced read (stride=128)
 *     - random_read:      Random access memory read (10 MB)
 *
 *   Warp Operations:
 *     - warp_shfl_xor:    Butterfly reduction via shfl_xor
 *     - warp_shfl_up:     Prefix sum via shfl_up
 *     - warp_broadcast:   Broadcast from lane 0
 *     - warp_reduce_sum:  Warp sum via shfl_down
 *     - warp_reduce_max:  Warp max via shfl_down
 *
 *   Synchronization:
 *     - syncthreads:      Block-level barrier
 *     - syncwarp:         Warp-level barrier
 *
 *   Compute:
 *     - expf:             Exponential function
 *     - compare_select:   TopK-like comparison
 *
 *   Block Size:
 *     - block_32, block_64, block_128, block_256, block_512
 *
 * Example: ./bench_factor empty 100 100
 *          ./bench_factor coalesced_read 100 100
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

// Factor name to test type mapping
typedef struct {
    const char* name;
    PerfTestType type;
    const char* description;
} FactorInfo;

static FactorInfo FACTOR_MAP[] = {
    // Launch Overhead
    {"empty", TEST_LAUNCH_EMPTY, "Pure kernel launch overhead"},
    {"minimal_rw", TEST_LAUNCH_MINIMAL_RW, "Launch + 1 memory access"},

    // Memory Bandwidth
    {"coalesced_read", TEST_MEM_COALESCED_READ, "Sequential memory read (10 MB)"},
    {"coalesced_write", TEST_MEM_COALESCED_WRITE, "Sequential memory write (10 MB)"},
    {"coalesced_copy", TEST_MEM_COALESCED_COPY, "Sequential read + write (10 MB)"},
    {"strided_read", TEST_MEM_STRIDED_READ, "Non-coalesced read (stride=128)"},
    {"random_read", TEST_MEM_RANDOM_READ, "Random access memory read (10 MB)"},

    // Warp Operations
    {"warp_shfl_xor", TEST_WARP_SHFL_XOR, "Butterfly reduction via shfl_xor"},
    {"warp_shfl_up", TEST_WARP_SHFL_UP, "Prefix sum via shfl_up"},
    {"warp_broadcast", TEST_WARP_BROADCAST, "Broadcast from lane 0"},
    {"warp_reduce_sum", TEST_WARP_REDUCE_SUM, "Warp sum via shfl_down"},
    {"warp_reduce_max", TEST_WARP_REDUCE_MAX, "Warp max via shfl_down"},

    // Synchronization
    {"syncthreads", TEST_SYNC_THREADS, "Block-level barrier"},
    {"syncwarp", TEST_SYNC_WARP, "Warp-level barrier"},

    // Compute
    {"expf", TEST_COMPUTE_EXPF, "Exponential function"},
    {"compare_select", TEST_COMPUTE_COMPARE, "TopK-like comparison"},

    // Block Size
    {"block_32", TEST_BLOCK_32, "Block size 32"},
    {"block_64", TEST_BLOCK_64, "Block size 64"},
    {"block_128", TEST_BLOCK_128, "Block size 128"},
    {"block_256", TEST_BLOCK_256, "Block size 256"},
    {"block_512", TEST_BLOCK_512, "Block size 512"},
};

#define FACTOR_COUNT (sizeof(FACTOR_MAP) / sizeof(FACTOR_MAP[0]))

void print_usage(const char* prog) {
    printf("Usage: %s <factor_name> [warmup] [iters]\n\n", prog);
    printf("Available factors:\n");

    const char* categories[] = {"Launch Overhead", "Memory Bandwidth",
                                "Warp Operations", "Synchronization",
                                "Compute", "Block Size"};
    int cat_start[] = {0, 2, 7, 12, 14, 16};

    for (int c = 0; c < 6; c++) {
        printf("\n  [%s]\n", categories[c]);
        int end = (c < 5) ? cat_start[c+1] : FACTOR_COUNT;
        for (int i = cat_start[c]; i < end; i++) {
            printf("    %-18s - %s\n", FACTOR_MAP[i].name, FACTOR_MAP[i].description);
        }
    }

    printf("\n  warmup: Number of warmup iterations (default: 100)\n");
    printf("  iters:  Number of test iterations (default: 100)\n");
    printf("\nExample: %s empty 100 100\n", prog);
    printf("         %s coalesced_read 100 100\n", prog);
}

int find_factor(const char* name) {
    for (size_t i = 0; i < FACTOR_COUNT; i++) {
        if (strcmp(FACTOR_MAP[i].name, name) == 0) {
            return (int)i;
        }
    }
    return -1;
}

int run_single_factor_test(PerfTestType test_type, int warmup_iters, int test_iters,
                           float* out_time_us, float* out_bandwidth_gbps, float* out_per_op_us) {
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

    // Grid/Block configuration for memory tests
    int block = 256;
    int grid = (n_elements + block - 1) / block;

    // Compute test iterations
    int n_warp_iters = PERF_COMPUTE_ITERATIONS;
    int n_blocks = PERF_N_BLOCKS;
    int n_threads = PERF_N_THREADS;
    int n_block_iters = 100;

    float time_us, total_ms;
    int i;

    // Warmup and test based on test type
    switch (test_type) {
        // Launch Overhead
        case TEST_LAUNCH_EMPTY:
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
            *out_time_us = time_us;
            *out_bandwidth_gbps = 0;
            *out_per_op_us = time_us;
            break;

        case TEST_LAUNCH_MINIMAL_RW:
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
            *out_time_us = time_us;
            *out_bandwidth_gbps = 0;
            *out_per_op_us = time_us;
            break;

        // Memory Bandwidth
        case TEST_MEM_COALESCED_READ:
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
            *out_time_us = time_us;
            *out_bandwidth_gbps = data_size / (time_us * 1000.0f);
            *out_per_op_us = 0;
            break;

        case TEST_MEM_COALESCED_WRITE:
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
            *out_time_us = time_us;
            *out_bandwidth_gbps = data_size / (time_us * 1000.0f);
            *out_per_op_us = 0;
            break;

        case TEST_MEM_COALESCED_COPY:
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
            *out_time_us = time_us;
            *out_bandwidth_gbps = (data_size * 2) / (time_us * 1000.0f);
            *out_per_op_us = 0;
            break;

        case TEST_MEM_STRIDED_READ:
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
            *out_time_us = time_us;
            *out_bandwidth_gbps = (data_size / 128) / (time_us * 1000.0f);
            *out_per_op_us = 0;
            break;

        case TEST_MEM_RANDOM_READ:
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
            *out_time_us = time_us;
            *out_bandwidth_gbps = data_size / (time_us * 1000.0f);
            *out_per_op_us = 0;
            break;

        // Warp Operations
        case TEST_WARP_SHFL_XOR:
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
            *out_time_us = time_us;
            *out_bandwidth_gbps = 0;
            *out_per_op_us = time_us / n_warp_iters / 5;
            break;

        case TEST_WARP_SHFL_UP:
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
            *out_time_us = time_us;
            *out_bandwidth_gbps = 0;
            *out_per_op_us = time_us / n_warp_iters / 5;
            break;

        case TEST_WARP_BROADCAST:
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
            *out_time_us = time_us;
            *out_bandwidth_gbps = 0;
            *out_per_op_us = time_us / n_warp_iters;
            break;

        case TEST_WARP_REDUCE_SUM:
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
            *out_time_us = time_us;
            *out_bandwidth_gbps = 0;
            *out_per_op_us = time_us / n_warp_iters / 5;
            break;

        case TEST_WARP_REDUCE_MAX:
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
            *out_time_us = time_us;
            *out_bandwidth_gbps = 0;
            *out_per_op_us = time_us / n_warp_iters / 5;
            break;

        // Synchronization
        case TEST_SYNC_THREADS:
            for (i = 0; i < warmup_iters; i++) perf_sync_threads(d_output, n_blocks, n_threads, n_warp_iters, 0);
            CUDA_CHECK(cudaDeviceSynchronize());
            total_ms = 0.0f;
            for (i = 0; i < test_iters; i++) {
                cudaEventRecord(start);
                perf_sync_threads(d_output, n_blocks, n_threads, n_warp_iters, 0);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                float ms;
                cudaEventElapsedTime(&ms, start, stop);
                total_ms += ms;
            }
            time_us = (total_ms / test_iters) * 1000.0f;
            *out_time_us = time_us;
            *out_bandwidth_gbps = 0;
            *out_per_op_us = time_us / n_warp_iters / 4;
            break;

        case TEST_SYNC_WARP:
            for (i = 0; i < warmup_iters; i++) perf_sync_warp(d_output, n_blocks, n_warp_iters, 0);
            CUDA_CHECK(cudaDeviceSynchronize());
            total_ms = 0.0f;
            for (i = 0; i < test_iters; i++) {
                cudaEventRecord(start);
                perf_sync_warp(d_output, n_blocks, n_warp_iters, 0);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                float ms;
                cudaEventElapsedTime(&ms, start, stop);
                total_ms += ms;
            }
            time_us = (total_ms / test_iters) * 1000.0f;
            *out_time_us = time_us;
            *out_bandwidth_gbps = 0;
            *out_per_op_us = time_us / n_warp_iters / 4;
            break;

        // Compute
        case TEST_COMPUTE_EXPF:
            for (i = 0; i < warmup_iters; i++) perf_compute_expf(d_output, n_blocks, n_threads, n_warp_iters, 0);
            CUDA_CHECK(cudaDeviceSynchronize());
            total_ms = 0.0f;
            for (i = 0; i < test_iters; i++) {
                cudaEventRecord(start);
                perf_compute_expf(d_output, n_blocks, n_threads, n_warp_iters, 0);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                float ms;
                cudaEventElapsedTime(&ms, start, stop);
                total_ms += ms;
            }
            time_us = (total_ms / test_iters) * 1000.0f;
            *out_time_us = time_us;
            *out_bandwidth_gbps = 0;
            *out_per_op_us = time_us / n_warp_iters;
            break;

        case TEST_COMPUTE_COMPARE:
            for (i = 0; i < warmup_iters; i++) perf_compute_compare(d_output, d_src, n_blocks, n_threads, n_warp_iters, 0);
            CUDA_CHECK(cudaDeviceSynchronize());
            total_ms = 0.0f;
            for (i = 0; i < test_iters; i++) {
                cudaEventRecord(start);
                perf_compute_compare(d_output, d_src, n_blocks, n_threads, n_warp_iters, 0);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                float ms;
                cudaEventElapsedTime(&ms, start, stop);
                total_ms += ms;
            }
            time_us = (total_ms / test_iters) * 1000.0f;
            *out_time_us = time_us;
            *out_bandwidth_gbps = 0;
            *out_per_op_us = time_us / n_warp_iters / 128;
            break;

        // Block Size
        case TEST_BLOCK_32:
        case TEST_BLOCK_64:
        case TEST_BLOCK_128:
        case TEST_BLOCK_256:
        case TEST_BLOCK_512:
            {
                int bs = 32;
                if (test_type == TEST_BLOCK_64) bs = 64;
                else if (test_type == TEST_BLOCK_128) bs = 128;
                else if (test_type == TEST_BLOCK_256) bs = 256;
                else if (test_type == TEST_BLOCK_512) bs = 512;

                for (i = 0; i < warmup_iters; i++) perf_block_size(d_output, n_blocks, bs, n_block_iters, 0);
                CUDA_CHECK(cudaDeviceSynchronize());
                total_ms = 0.0f;
                for (i = 0; i < test_iters; i++) {
                    cudaEventRecord(start);
                    perf_block_size(d_output, n_blocks, bs, n_block_iters, 0);
                    cudaEventRecord(stop);
                    cudaEventSynchronize(stop);
                    float ms;
                    cudaEventElapsedTime(&ms, start, stop);
                    total_ms += ms;
                }
                time_us = (total_ms / test_iters) * 1000.0f;
                *out_time_us = time_us;
                *out_bandwidth_gbps = 0;
                *out_per_op_us = 0;
            }
            break;

        default:
            fprintf(stderr, "Error: Unknown test type\n");
            return -1;
    }

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

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char* factor_name = argv[1];
    int warmup_iters = (argc > 2) ? atoi(argv[2]) : PERF_WARMUP_ITERATIONS;
    int test_iters = (argc > 3) ? atoi(argv[3]) : PERF_TEST_ITERATIONS;

    int factor_idx = find_factor(factor_name);
    if (factor_idx < 0) {
        fprintf(stderr, "Error: Unknown factor '%s'\n\n", factor_name);
        print_usage(argv[0]);
        return 1;
    }

    FactorInfo* info = &FACTOR_MAP[factor_idx];

    printf("=== Single Factor Benchmark ===\n");
    printf("Factor: %s\n", info->name);
    printf("Description: %s\n", info->description);
    printf("Iterations: %d (warmup: %d)\n\n", test_iters, warmup_iters);

    printf("Running benchmark...\n");

    float time_us, bandwidth_gbps, per_op_us;
    int ret = run_single_factor_test(info->type, warmup_iters, test_iters,
                                     &time_us, &bandwidth_gbps, &per_op_us);
    if (ret != 0) {
        fprintf(stderr, "Error running test\n");
        return 1;
    }

    printf("\n=== Results ===\n");
    printf("Average time: %.3f us\n", time_us);
    if (bandwidth_gbps > 0) {
        printf("Bandwidth: %.2f GB/s\n", bandwidth_gbps);
    }
    if (per_op_us > 0) {
        printf("Per-operation: %.4f us\n", per_op_us);
    }

    // Machine-parsable output
    printf("\n[PARSABLE OUTPUT]\n");
    printf("factor=%s\n", info->name);
    printf("avg_time_us=%.3f\n", time_us);
    printf("bandwidth_gbps=%.2f\n", bandwidth_gbps);
    printf("per_op_us=%.4f\n", per_op_us);

    return 0;
}
