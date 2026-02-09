/*
 * Pure Performance Benchmark for TopK Kernel
 *
 * Usage: ./bench_perf <experts> <tokens> <dtype> <topk> [iters]
 *
 * This measures pure kernel execution time:
 * - Warmup runs to eliminate cold-start overhead
 * - Multiple iterations for stable average
 * - Uses CUDA events for accurate timing
 * - No malloc/free overhead in measurement
 */

#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "../src/topk_softmax.h"

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
    printf("Usage: %s <experts> <tokens> <dtype> <topk> [iters]\n", prog);
    printf("  experts: number of experts (e.g., 64, 128, 256)\n");
    printf("  tokens:  number of tokens (e.g., 16, 128, 1024)\n");
    printf("  dtype:   data type (fp32, fp16, bf16)\n");
    printf("  topk:    top-k value (e.g., 2, 4, 6, 8)\n");
    printf("  iters:   number of iterations (default: 100)\n");
    printf("\nExample: %s 64 1024 fp16 4 1000\n", prog);
}

int main(int argc, char** argv) {
    if (argc < 5) {
        print_usage(argv[0]);
        return 1;
    }

    int experts = atoi(argv[1]);
    int tokens = atoi(argv[2]);
    std::string dtype = argv[3];
    int topk = atoi(argv[4]);
    int iters = (argc > 5) ? atoi(argv[5]) : 100;
    int warmup = 10;

    printf("=== Pure Kernel Performance Benchmark ===\n");
    printf("Configuration: experts=%d, tokens=%d, dtype=%s, topk=%d\n",
           experts, tokens, dtype.c_str(), topk);
    printf("Iterations: %d (warmup: %d)\n\n", iters, warmup);

    // Map dtype string to enum
    TopkSoftmaxDtype dtype_enum;
    size_t element_size;
    if (dtype == "fp16") {
        dtype_enum = TopkSoftmaxDtype::Float16;
        element_size = 2;
    } else if (dtype == "bf16") {
        dtype_enum = TopkSoftmaxDtype::BFloat16;
        element_size = 2;
    } else {
        dtype_enum = TopkSoftmaxDtype::Float32;
        element_size = 4;
    }

    // Seed random number generator for reproducible results
    srand(42);

    // Allocate GPU memory (all allocations done before timing loop)
    void* gating_d = nullptr;
    float* weights_d = nullptr;
    int* indices_d = nullptr;
    int* token_expert_indices_d = nullptr;  // Pre-allocate to avoid malloc in timed loop

    size_t input_bytes = tokens * experts * element_size;
    CUDA_CHECK(cudaMalloc(&gating_d, input_bytes));
    CUDA_CHECK(cudaMalloc(&weights_d, tokens * topk * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&indices_d, tokens * topk * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&token_expert_indices_d, tokens * topk * sizeof(int)));

    // Initialize input with random data (on host, then copy)
    std::vector<float> input_host(tokens * experts);
    for (size_t i = 0; i < input_host.size(); i++) {
        input_host[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }

    // Convert and copy to device
    if (dtype == "fp16") {
        std::vector<__half> input_flat;
        input_flat.reserve(input_host.size());
        for (float v : input_host)
            input_flat.push_back(__float2half(v));
        CUDA_CHECK(cudaMemcpy(gating_d, input_flat.data(), input_flat.size() * sizeof(__half),
                              cudaMemcpyHostToDevice));
    } else if (dtype == "bf16") {
        std::vector<__nv_bfloat16> input_flat;
        input_flat.reserve(input_host.size());
        for (float v : input_host)
            input_flat.push_back(__float2bfloat16(v));
        CUDA_CHECK(cudaMemcpy(gating_d, input_flat.data(), input_flat.size() * sizeof(__nv_bfloat16),
                              cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(gating_d, input_host.data(), input_host.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    printf("Warming up (%d runs)...\n", warmup);
    for (int i = 0; i < warmup; i++) {
        topk_softmax_async(
            weights_d, indices_d, token_expert_indices_d, gating_d, nullptr,
            tokens, experts, topk, dtype_enum, false, 0
        );
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());  // Check for kernel launch errors
    printf("Warmup complete.\n\n");

    printf("Running benchmark (%d iterations)...\n", iters);
    float total_time_ms = 0.0f;

    for (int i = 0; i < iters; i++) {
        cudaEventRecord(start);
        topk_softmax_async(
            weights_d, indices_d, token_expert_indices_d, gating_d, nullptr,
            tokens, experts, topk, dtype_enum, false, 0
        );
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
    printf("Throughput: %.2f tokens/sec\n", tokens / (avg_time_us / 1e6f));
    printf("Per-token: %.3f us\n", avg_time_us / tokens);

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(gating_d));
    CUDA_CHECK(cudaFree(weights_d));
    CUDA_CHECK(cudaFree(indices_d));
    CUDA_CHECK(cudaFree(token_expert_indices_d));

    return 0;
}
