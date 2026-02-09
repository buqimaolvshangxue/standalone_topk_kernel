#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// =============================================================================
// Standalone TopK Softmax Kernel Interface
// Extracted from vLLM csrc/moe/topk_softmax_kernels.cu
//
// Maintains same structure as vLLM for easy profiling comparison:
//   topk_softmax() -> dispatch_topk_softmax_launch<T>() -> topkGatingSoftmaxKernelLauncher()
// =============================================================================

// Supported data types for gating_output
enum class TopkSoftmaxDtype {
    Float32 = 0,
    Float16 = 1,
    BFloat16 = 2
};

// Main entry point - mirrors vLLM's topk_softmax() structure
// 
// Parameters:
//   topk_weights:          [num_tokens, topk] output - softmax weights for top-k experts
//   topk_indices:          [num_tokens, topk] output - indices of top-k experts
//   token_expert_indices:  [num_tokens, topk] output - can be nullptr (allocated internally)
//   gating_output:         [num_tokens, num_experts] input - raw gating scores
//   softmax_workspace:     workspace buffer - can be nullptr (allocated if needed)
//   num_tokens:            number of tokens to process
//   num_experts:           number of experts in MoE layer
//   topk:                  number of experts to select per token
//   dtype:                 data type of gating_output
//   renormalize:           whether to renormalize top-k weights to sum to 1
//   stream:                CUDA stream for async execution
//
void topk_softmax(
    float* topk_weights,
    int* topk_indices,
    int* token_expert_indices,
    void* gating_output,
    float* softmax_workspace,
    int num_tokens,
    int num_experts,
    int topk,
    TopkSoftmaxDtype dtype,
    bool renormalize,
    cudaStream_t stream);
