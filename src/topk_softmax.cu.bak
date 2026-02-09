/*
 * Standalone TopK Softmax Kernel
 * Extracted from vLLM csrc/moe/topk_softmax_kernels.cu
 * 
 * Changes from original:
 * - Removed PyTorch dependencies (torch/all.h, ATen/cuda/CUDAContext.h)
 * - Replaced TORCH_CHECK with assert
 * - Created pure CUDA interface (topk_softmax_launch)
 * - Removed workspace dynamic allocation (use stack for supported expert counts)
 * 
 * Original copyright:
 * Adapted from https://github.com/NVIDIA/TensorRT-LLM/blob/v0.7.1/cpp/tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.cu
 * Copyright (c) 2024, The vLLM team.
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <type_traits>
#include <cassert>
#include <cfloat>
#include <cstdio>
#include <cuda_runtime.h>
#include "cuda_compat.h"
#include "cub_helpers.h"

#ifndef USE_ROCM
    #include <cuda_bf16.h>
    #include <cuda_fp16.h>
#else
    #include <hip/hip_bf16.h>
    #include <hip/hip_fp16.h>
    typedef __hip_bfloat16 __nv_bfloat16;
    typedef __hip_bfloat162 __nv_bfloat162;
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

namespace vllm {
namespace moe {

/// Aligned array type
template <
    typename T,
    /// Number of elements in the array
    int N,
    /// Alignment requirement in bytes
    int Alignment = sizeof(T) * N
>
struct alignas(Alignment) AlignedArray {
    T data[N];
};

template <typename T>
__device__ __forceinline__ float toFloat(T value) {
    if constexpr (std::is_same_v<T, float>) {
        return value;
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __bfloat162float(value);
    } else if constexpr (std::is_same_v<T, __half>) {
        return __half2float(value);
    }
}

// ====================== Softmax things ===============================
template <int TPB, typename InputType>
__launch_bounds__(TPB) __global__
    void moeSoftmax(const InputType* input, const bool* finished, float* output, const int num_cols)
{
    using BlockReduce = cub::BlockReduce<float, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    __shared__ float normalizing_factor;
    __shared__ float float_max;

    const int thread_row_offset = blockIdx.x * num_cols;

    float threadData(-FLT_MAX);

    // Don't touch finished rows.
    if ((finished != nullptr) && finished[blockIdx.x])
    {
        return;
    }

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
    {
        const int idx = thread_row_offset + ii;
        const float val = toFloat(input[idx]);
        threadData = max(val, threadData);
    }

    const float maxElem = BlockReduce(tmpStorage).Reduce(threadData, CubMaxOp());
    if (threadIdx.x == 0)
    {
        float_max = maxElem;
    }
    __syncthreads();

    threadData = 0;

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
    {
        const int idx = thread_row_offset + ii;
        const float val = toFloat(input[idx]);
        threadData += expf(val - float_max);
    }

    const auto Z = BlockReduce(tmpStorage).Reduce(threadData, CubAddOp());

    if (threadIdx.x == 0)
    {
        normalizing_factor = 1.f / Z;
    }
    __syncthreads();

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
    {
        const int idx = thread_row_offset + ii;
        const float val = toFloat(input[idx]);
        const float softmax_val = expf(val - float_max) * normalizing_factor;
        output[idx] = softmax_val;
    }
}

template <int TPB, typename IndType>
__launch_bounds__(TPB) __global__ void moeTopK(
    const float* inputs_after_softmax,
    const bool* finished,
    float* output,
    IndType* indices,
    int* source_rows,
    const int num_experts,
    const int k,
    const int start_expert,
    const int end_expert,
    const bool renormalize)
{

    using cub_kvp = cub::KeyValuePair<int, float>;
    using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    cub_kvp thread_kvp;
    cub::ArgMax arg_max;

    const int num_rows = gridDim.x;
    const int block_row = blockIdx.x;

    const bool row_is_active = finished ? !finished[block_row] : true;
    const int thread_read_offset = blockIdx.x * num_experts;
    float selected_sum = 0.f;
    for (int k_idx = 0; k_idx < k; ++k_idx)
    {
        thread_kvp.key = 0;
        thread_kvp.value = -1.f;

        cub_kvp inp_kvp;
        for (int expert = threadIdx.x; expert < num_experts; expert += TPB)
        {
            const int idx = thread_read_offset + expert;
            inp_kvp.key = expert;
            inp_kvp.value = inputs_after_softmax[idx];

            for (int prior_k = 0; prior_k < k_idx; ++prior_k)
            {
                const int prior_winning_expert = indices[k * block_row + prior_k];

                if (prior_winning_expert == expert)
                {
                    inp_kvp = thread_kvp;
                }
            }

            thread_kvp = arg_max(inp_kvp, thread_kvp);
        }

        const cub_kvp result_kvp = BlockReduce(tmpStorage).Reduce(thread_kvp, arg_max);
        if (threadIdx.x == 0)
        {
            const int expert = result_kvp.key;
            const bool node_uses_expert = expert >= start_expert && expert < end_expert;
            const bool should_process_row = row_is_active && node_uses_expert;

            const int idx = k * block_row + k_idx;
            output[idx] = result_kvp.value;
            indices[idx] = should_process_row ? (expert - start_expert) : num_experts;
            assert(indices[idx] >= 0);
            source_rows[idx] = k_idx * num_rows + block_row;
            if (renormalize) {
                selected_sum += result_kvp.value;
            }
        }
        __syncthreads();
    }

    // Renormalize the k weights for this row to sum to 1, if requested.
    if (renormalize) {
        if (threadIdx.x == 0) {
            const float denom = selected_sum > 0.f ? selected_sum : 1.f;
            for (int k_idx = 0; k_idx < k; ++k_idx) {
                const int idx = k * block_row + k_idx;
                output[idx] = output[idx] / denom;
            }
        }
    }
}

// ====================== TopK softmax things ===============================

template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG, int WARP_SIZE_PARAM, typename IndType, typename InputType = float>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE_PARAM) __global__
    void topkGatingSoftmax(const InputType* input, const bool* finished, float* output, const int num_rows, IndType* indices,
        int* source_rows, const int k, const int start_expert, const int end_expert, const bool renormalize)
{
    static_assert(std::is_same_v<InputType, float> || std::is_same_v<InputType, __nv_bfloat16> ||
                      std::is_same_v<InputType, __half>,
                  "InputType must be float, __nv_bfloat16, or __half");

    static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");
    static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(InputType);
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

    if constexpr (std::is_same_v<InputType, __nv_bfloat16> || std::is_same_v<InputType, __half>) {
        static_assert(ELTS_PER_LDG == 1 || ELTS_PER_LDG % 2 == 0,
            "ELTS_PER_LDG must be 1 or even for 16-bit conversion");
    }

    static_assert(VPT % ELTS_PER_LDG == 0, "The elements per thread must be a multiple of the elements per ldg");
    static_assert(WARP_SIZE_PARAM % THREADS_PER_ROW == 0, "The threads per row must cleanly divide the threads per warp");
    static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW), "THREADS_PER_ROW must be power of 2");
    static_assert(THREADS_PER_ROW <= WARP_SIZE_PARAM, "THREADS_PER_ROW can be at most warp size");

    static constexpr int ELTS_PER_WARP = WARP_SIZE_PARAM * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

    static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0, "The elts per row must cleanly divide the total elt per warp");

    const int cta_base_row = blockIdx.x * ROWS_PER_CTA;
    const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;
    const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
    const int thread_row = warp_base_row + thread_row_in_warp;

    if (thread_row >= num_rows)
    {
        return;
    }
    const bool row_is_active = finished ? !finished[thread_row] : true;

    const InputType* thread_row_ptr = input + thread_row * ELTS_PER_ROW;
    const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const InputType* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

    float row_chunk[VPT];

    // Load data from global memory and convert to float
    if constexpr (std::is_same_v<InputType, float>) {
        using VecType = AlignedArray<float, ELTS_PER_LDG>;
        VecType* row_chunk_vec_ptr = reinterpret_cast<VecType*>(&row_chunk);
        const VecType* vec_thread_read_ptr = reinterpret_cast<const VecType*>(thread_read_ptr);
#pragma unroll
        for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
            row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
        }
    } else if constexpr (std::is_same_v<InputType, __nv_bfloat16>) {
        if constexpr (ELTS_PER_LDG >= 2) {
            using VecType = AlignedArray<__nv_bfloat16, ELTS_PER_LDG>;
            float2* row_chunk_f2 = reinterpret_cast<float2*>(row_chunk);
            const VecType* vec_thread_read_ptr = reinterpret_cast<const VecType*>(thread_read_ptr);
#pragma unroll
            for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
                VecType vec = vec_thread_read_ptr[ii * THREADS_PER_ROW];
                int base_idx_f2 = ii * ELTS_PER_LDG / 2;
#pragma unroll
                for (int jj = 0; jj < ELTS_PER_LDG / 2; ++jj) {
                    row_chunk_f2[base_idx_f2 + jj] = __bfloat1622float2(
                        *reinterpret_cast<const __nv_bfloat162*>(vec.data + jj * 2)
                    );
                }
            }
        } else {
#pragma unroll
            for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
                const __nv_bfloat16* scalar_ptr = thread_read_ptr + ii * THREADS_PER_ROW;
                row_chunk[ii] = __bfloat162float(*scalar_ptr);
            }
        }
    } else if constexpr (std::is_same_v<InputType, __half>) {
        if constexpr (ELTS_PER_LDG >= 2) {
            using VecType = AlignedArray<__half, ELTS_PER_LDG>;
            float2* row_chunk_f2 = reinterpret_cast<float2*>(row_chunk);
            const VecType* vec_thread_read_ptr = reinterpret_cast<const VecType*>(thread_read_ptr);
#pragma unroll
            for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
                VecType vec = vec_thread_read_ptr[ii * THREADS_PER_ROW];
                int base_idx_f2 = ii * ELTS_PER_LDG / 2;
#pragma unroll
                for (int jj = 0; jj < ELTS_PER_LDG / 2; ++jj) {
                    row_chunk_f2[base_idx_f2 + jj] = __half22float2(
                        *reinterpret_cast<const __half2*>(vec.data + jj * 2)
                    );
                }
            }
        } else {
#pragma unroll
            for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
                const __half* scalar_ptr = thread_read_ptr + ii * THREADS_PER_ROW;
                row_chunk[ii] = __half2float(*scalar_ptr);
            }
        }
    }

    // Max reduce within thread
    float thread_max = row_chunk[0];
#pragma unroll
    for (int ii = 1; ii < VPT; ++ii)
    {
        thread_max = max(thread_max, row_chunk[ii]);
    }

    // Butterfly reduce for max
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
    {
        thread_max = max(thread_max, VLLM_SHFL_XOR_SYNC_WIDTH(thread_max, mask, THREADS_PER_ROW));
    }

    // Subtract max and compute exp + sum
    float row_sum = 0;
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii)
    {
        row_chunk[ii] = expf(row_chunk[ii] - thread_max);
        row_sum += row_chunk[ii];
    }

    // Butterfly reduce for sum
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
    {
        row_sum += VLLM_SHFL_XOR_SYNC_WIDTH(row_sum, mask, THREADS_PER_ROW);
    }

    // Normalize
    const float reciprocal_row_sum = 1.f / row_sum;

#pragma unroll
    for (int ii = 0; ii < VPT; ++ii)
    {
        row_chunk[ii] = row_chunk[ii] * reciprocal_row_sum;
    }

    // TopK selection
    int start_col = first_elt_read_by_thread;
    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

    float selected_sum = 0.f;
    for (int k_idx = 0; k_idx < k; ++k_idx)
    {
        float max_val = row_chunk[0];
        int expert = start_col;
#pragma unroll
        for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD; ++ldg, col += COLS_PER_GROUP_LDG)
        {
#pragma unroll
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii)
            {
                float val = row_chunk[ldg * ELTS_PER_LDG + ii];
                if (val > max_val)
                {
                    max_val = val;
                    expert = col + ii;
                }
            }
        }

        // Butterfly reduce for argmax
#pragma unroll
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
        {
            float other_max = VLLM_SHFL_XOR_SYNC_WIDTH(max_val, mask, THREADS_PER_ROW);
            int other_expert = VLLM_SHFL_XOR_SYNC_WIDTH(expert, mask, THREADS_PER_ROW);

            if (other_max > max_val || (other_max == max_val && other_expert < expert))
            {
                max_val = other_max;
                expert = other_expert;
            }
        }

        // Write output
        if (thread_group_idx == 0)
        {
            const bool node_uses_expert = expert >= start_expert && expert < end_expert;
            const bool should_process_row = row_is_active && node_uses_expert;

            const int idx = k * thread_row + k_idx;
            output[idx] = max_val;
            indices[idx] = should_process_row ? (expert - start_expert) : NUM_EXPERTS;
            source_rows[idx] = k_idx * num_rows + thread_row;
            if (renormalize) {
                selected_sum += max_val;
            }
        }

        // Clear winning value for next iteration
        if (k_idx + 1 < k)
        {
            const int ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
            const int thread_to_clear_in_group = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;

            if (thread_group_idx == thread_to_clear_in_group)
            {
                const int offset_for_expert = expert % ELTS_PER_LDG;
                row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] = -10000.f;
            }
        }
    }

    // Renormalize if requested
    if (renormalize) {
        if (thread_group_idx == 0)
        {
            const float denom = selected_sum > 0.f ? selected_sum : 1.f;
            for (int k_idx = 0; k_idx < k; ++k_idx)
            {
                const int idx = k * thread_row + k_idx;
                output[idx] = output[idx] / denom;
            }
        }
    }
}

namespace detail
{
template <int EXPERTS, int BYTES_PER_LDG, int WARP_SIZE_PARAM, typename InputType>
struct TopkConstants
{
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(InputType);
    static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE_PARAM) == 0 || EXPERTS % (ELTS_PER_LDG * WARP_SIZE_PARAM) == 0, "");
    static constexpr int VECs_PER_THREAD = MAX(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE_PARAM));
    static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
    static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
    static const int ROWS_PER_WARP = WARP_SIZE_PARAM / THREADS_PER_ROW;
};
} // namespace detail

template <int EXPERTS, int WARPS_PER_TB, int WARP_SIZE_PARAM, int MAX_BYTES_PER_LDG, typename IndType, typename InputType>
void topkGatingSoftmaxLauncherHelper(const InputType* input, const bool* finished, float* output, IndType* indices,
    int* source_row, const int num_rows, const int k, const int start_expert, const int end_expert, const bool renormalize,
    cudaStream_t stream)
{
    static constexpr int BYTES_PER_LDG = MIN(MAX_BYTES_PER_LDG, sizeof(InputType) * EXPERTS);
    using Constants = detail::TopkConstants<EXPERTS, BYTES_PER_LDG, WARP_SIZE_PARAM, InputType>;
    static constexpr int VPT = Constants::VPT;
    static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
    const int num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
    const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;

    dim3 block_dim(WARP_SIZE_PARAM, WARPS_PER_TB);
    topkGatingSoftmax<VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG, WARP_SIZE_PARAM, IndType, InputType><<<num_blocks, block_dim, 0, stream>>>(
        input, finished, output, num_rows, indices, source_row, k, start_expert, end_expert, renormalize);
}

#ifndef USE_ROCM
#define LAUNCH_SOFTMAX(NUM_EXPERTS, WARPS_PER_TB, MAX_BYTES)                          \
    static_assert(WARP_SIZE == 32,                                                    \
                  "Unsupported warp size. Only 32 is supported for CUDA");            \
    topkGatingSoftmaxLauncherHelper<NUM_EXPERTS, WARPS_PER_TB, WARP_SIZE, MAX_BYTES>( \
        gating_output, nullptr, topk_weights, topk_indices, token_expert_indices,     \
        num_tokens, topk, 0, num_experts, renormalize, stream);
#else
#define LAUNCH_SOFTMAX(NUM_EXPERTS, WARPS_PER_TB, MAX_BYTES)                             \
    if (WARP_SIZE == 64) {                                                               \
        topkGatingSoftmaxLauncherHelper<NUM_EXPERTS, WARPS_PER_TB, 64, MAX_BYTES>(       \
            gating_output, nullptr, topk_weights, topk_indices, token_expert_indices,    \
            num_tokens, topk, 0, num_experts, renormalize, stream);                      \
    } else if (WARP_SIZE == 32) {                                                        \
        topkGatingSoftmaxLauncherHelper<NUM_EXPERTS, WARPS_PER_TB, 32, MAX_BYTES>(       \
            gating_output, nullptr, topk_weights, topk_indices, token_expert_indices,    \
            num_tokens, topk, 0, num_experts, renormalize, stream);                      \
    } else {                                                                             \
        assert(false && "Unsupported warp size. Only 32 and 64 are supported for ROCm"); \
    }
#endif

template <typename IndType, typename InputType>
void topkGatingSoftmaxKernelLauncher(
    const InputType* gating_output,
    float* topk_weights,
    IndType* topk_indices,
    int* token_expert_indices,
    float* softmax_workspace,
    const int num_tokens,
    const int num_experts,
    const int topk,
    const bool renormalize,
    cudaStream_t stream) {
    static constexpr int WARPS_PER_TB = 4;
    static constexpr int BYTES_PER_LDG_POWER_OF_2 = 16;
#ifndef USE_ROCM
    static constexpr int BYTES_PER_LDG_MULTIPLE_64 =
    (std::is_same_v<InputType, __nv_bfloat16> || std::is_same_v<InputType, __half>) ? 4 : 8;
#endif
    switch (num_experts) {
        case 1:
            LAUNCH_SOFTMAX(1, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
            break;
        case 2:
            LAUNCH_SOFTMAX(2, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
            break;
        case 4:
            LAUNCH_SOFTMAX(4, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
            break;
        case 8:
            LAUNCH_SOFTMAX(8, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
            break;
        case 16:
            LAUNCH_SOFTMAX(16, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
            break;
        case 32:
            LAUNCH_SOFTMAX(32, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
            break;
        case 64:
            LAUNCH_SOFTMAX(64, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
            break;
        case 128:
            LAUNCH_SOFTMAX(128, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
            break;
        case 256:
            LAUNCH_SOFTMAX(256, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
            break;
        case 512:
            LAUNCH_SOFTMAX(512, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
            break;
#ifndef USE_ROCM
        case 192:
            LAUNCH_SOFTMAX(192, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64);
            break;
        case 320:
            LAUNCH_SOFTMAX(320, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64);
            break;
        case 384:
            LAUNCH_SOFTMAX(384, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64);
            break;
        case 448:
            LAUNCH_SOFTMAX(448, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64);
            break;
        case 576:
            LAUNCH_SOFTMAX(576, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64);
            break;
#endif
        default: {
            assert(softmax_workspace != nullptr &&
                "softmax_workspace must be provided for num_experts that are not a power of 2 or multiple of 64.");
            static constexpr int TPB = 256;
            moeSoftmax<TPB, InputType><<<num_tokens, TPB, 0, stream>>>(
                gating_output, nullptr, softmax_workspace, num_experts);
            moeTopK<TPB><<<num_tokens, TPB, 0, stream>>>(
                softmax_workspace, nullptr, topk_weights, topk_indices, token_expert_indices,
                num_experts, topk, 0, num_experts, renormalize);
        }
    }
}

} // namespace moe
} // namespace vllm


// =============================================================================
// Standalone Interface - Mirrors original vLLM structure for easy profiling
// =============================================================================
// Original vLLM structure:
//   topk_softmax() -> dispatch_topk_softmax_launch<T>() -> topkGatingSoftmaxKernelLauncher<IndType, T>()
// 
// This standalone version keeps the same dispatch pattern:
//   topk_softmax() -> dispatch_topk_softmax_launch<T>() -> topkGatingSoftmaxKernelLauncher<int, T>()
// =============================================================================

#include "topk_softmax.h"

// Type dispatch helper - mirrors vLLM's dispatch_topk_softmax_launch
template<typename ComputeType>
void dispatch_topk_softmax_launch(
    const void* gating_output,
    float* topk_weights,
    int* topk_indices,
    int* token_expert_indices,
    float* softmax_workspace,
    int num_tokens, 
    int num_experts, 
    int topk, 
    bool renormalize, 
    cudaStream_t stream)
{
    // In standalone version, we only support int indices (simplified from vLLM's multi-index-type support)
    vllm::moe::topkGatingSoftmaxKernelLauncher<int, ComputeType>(
        reinterpret_cast<const ComputeType*>(gating_output),
        topk_weights,
        topk_indices,
        token_expert_indices,
        softmax_workspace,
        num_tokens, num_experts, topk, renormalize, stream);
}

// Main entry point - mirrors vLLM's topk_softmax() structure
// Key differences from vLLM:
//   - Uses raw pointers instead of torch::Tensor
//   - Uses enum dtype instead of scalar_type()
//   - Caller manages memory allocation
void topk_softmax(
    float* topk_weights,                // [num_tokens, topk] output
    int* topk_indices,                  // [num_tokens, topk] output
    int* token_expert_indices,          // [num_tokens, topk] output (can be nullptr, will alloc internally)
    void* gating_output,                // [num_tokens, num_experts] input
    float* softmax_workspace,           // workspace for non-power-of-2 experts (can be nullptr, will alloc if needed)
    int num_tokens,
    int num_experts,
    int topk,
    TopkSoftmaxDtype dtype,
    bool renormalize,
    cudaStream_t stream)
{
    // Determine if workspace is needed (same logic as vLLM)
    const bool is_pow_2 = (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
    const bool needs_workspace = !is_pow_2 || num_experts > 256;
    const int64_t workspace_size = needs_workspace ? num_tokens * num_experts : 0;
    
    // Manage internal allocations if caller didn't provide
    bool own_token_expert_indices = false;
    bool own_softmax_workspace = false;
    
    if (token_expert_indices == nullptr) {
        cudaMalloc(&token_expert_indices, num_tokens * topk * sizeof(int));
        own_token_expert_indices = true;
    }
    
    if (needs_workspace && softmax_workspace == nullptr) {
        cudaMalloc(&softmax_workspace, workspace_size * sizeof(float));
        own_softmax_workspace = true;
    }
    
    // Type dispatch - same if-else structure as vLLM's topk_softmax()
    if (dtype == TopkSoftmaxDtype::Float32) {
        dispatch_topk_softmax_launch<float>(
            gating_output, topk_weights, topk_indices,
            token_expert_indices, softmax_workspace,
            num_tokens, num_experts, topk, renormalize, stream);
    } else if (dtype == TopkSoftmaxDtype::Float16) {
        dispatch_topk_softmax_launch<__half>(
            gating_output, topk_weights, topk_indices,
            token_expert_indices, softmax_workspace,
            num_tokens, num_experts, topk, renormalize, stream);
    } else if (dtype == TopkSoftmaxDtype::BFloat16) {
        dispatch_topk_softmax_launch<__nv_bfloat16>(
            gating_output, topk_weights, topk_indices,
            token_expert_indices, softmax_workspace,
            num_tokens, num_experts, topk, renormalize, stream);
    } else {
        assert(false && "Unsupported gating_output data type");
    }
    
    // Cleanup internal allocations
    cudaStreamSynchronize(stream);
    if (own_token_expert_indices) cudaFree(token_expert_indices);
    if (own_softmax_workspace) cudaFree(softmax_workspace);
}
