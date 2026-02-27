/*
 * Baseline Kernel Performance Benchmark
 *
 * Usage: ./bench_baseline <kernel_type> [warmup] [iters]
 *
 * kernel_type:
 *   - empty:  Empty kernel (measures pure launch overhead)
 *   - rw:     Minimal read-write kernel (measures launch + minimal memory access)
 *
 * warmup: Number of warmup iterations (default: 100)
 * iters:  Number of test iterations (default: 100)
 *
 * This benchmark uses the SAME testing methodology as bench_perf.cpp:
 * - cudaEvent timing
 * - Multiple iterations for stable average
 *
 * Example: ./bench_baseline empty 100 100
 *          ./bench_baseline rw 100 100
 */

#include <iostream>
#include <string>
#include <cstdlib>
#include <cuda_runtime.h>
#include "baseline_kernels.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "[ERROR] CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

void print_usage(const char* prog) {
    printf("Usage: %s <kernel_type> [warmup] [iters]\n", prog);
    printf("  kernel_type: empty | rw\n");
    printf("    empty: Empty kernel (pure launch overhead)\n");
    printf("    rw:    Minimal read-write kernel\n");
    printf("  warmup: Number of warmup iterations (default: 100)\n");
    printf("  iters:  Number of test iterations (default: 100)\n");
    printf("\nExample: %s empty 100 100\n", prog);
    printf("         %s rw 100 100\n", prog);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string kernel_type = argv[1];
    int warmup = (argc > 2) ? atoi(argv[2]) : 100;
    int iters = (argc > 3) ? atoi(argv[3]) : 100;
    cudaStream_t stream = 0;  // Default stream

    // Validate kernel type
    if (kernel_type != "empty" && kernel_type != "rw") {
        fprintf(stderr, "Error: Unknown kernel type '%s'\n", kernel_type.c_str());
        print_usage(argv[0]);
        return 1;
    }

    printf("=== Baseline Kernel Benchmark ===\n");
    printf("Kernel type: %s\n", kernel_type.c_str());
    printf("Iterations: %d (warmup: %d)\n\n", iters, warmup);

    // Allocate GPU memory (for rw kernel)
    float* in_d = nullptr;
    float* out_d = nullptr;

    if (kernel_type == "rw") {
        CUDA_CHECK(cudaMalloc(&in_d, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&out_d, sizeof(float)));

        // Initialize input
        float init_val = 1.0f;
        CUDA_CHECK(cudaMemcpy(in_d, &init_val, sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(out_d, 0, sizeof(float)));
    }

    // Create CUDA events for timing (same as bench_perf.cpp)
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup (same as bench_perf.cpp)
    printf("Warming up (%d runs)...\n", warmup);
    for (int i = 0; i < warmup; i++) {
        if (kernel_type == "empty") {
            launch_empty_kernel(stream);
        } else {
            launch_minimal_rw_kernel(out_d, in_d, stream);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    printf("Warmup complete.\n\n");

    // Benchmark (same methodology as bench_perf.cpp)
    printf("Running benchmark (%d iterations)...\n", iters);
    float total_time_ms = 0.0f;

    for (int i = 0; i < iters; i++) {
        cudaEventRecord(start);

        if (kernel_type == "empty") {
            launch_empty_kernel(stream);
        } else {
            launch_minimal_rw_kernel(out_d, in_d, stream);
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_time_ms += ms;
    }

    float avg_time_ms = total_time_ms / iters;
    float avg_time_us = avg_time_ms * 1000.0f;

    printf("\n=== Results ===\n");
    printf("Average kernel time: %.3f us (%.4f ms)\n", avg_time_us, avg_time_ms);

    // Output in a format easy to parse for scripts
    printf("\n[PARSABLE OUTPUT]\n");
    printf("kernel_type=%s\n", kernel_type.c_str());
    printf("avg_time_us=%.3f\n", avg_time_us);
    printf("avg_time_ms=%.4f\n", avg_time_ms);

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    if (in_d) CUDA_CHECK(cudaFree(in_d));
    if (out_d) CUDA_CHECK(cudaFree(out_d));

    return 0;
}
