#!/bin/bash
# Standalone TopK Kernel Build Script
# Usage: ./build.sh [dl|nv]

set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

# Detect or specify platform
PLATFORM=${1}

if [ -z "$PLATFORM" ]; then
    # Auto-detect platform when no argument
    if command -v dlcc &> /dev/null; then
        PLATFORM=dl
    elif command -v nvcc &> /dev/null; then
        PLATFORM=nv
    else
        echo "Error: Neither dlcc nor nvcc found in PATH"
        exit 1
    fi
    echo "Auto-detected platform: $PLATFORM"
fi

mkdir -p build

if [ "$PLATFORM" = "nv" ]; then
    # NVIDIA platform
    NVCC=${NVCC:-nvcc}
    CUDA_ARCH=${CUDA_ARCH:-86}  # Default to sm_86 (A40)

    echo "Building for NVIDIA platform with NVCC: $NVCC"
    echo "CUDA architecture: sm_$CUDA_ARCH"

    ${NVCC} -o build/benchmark_topk \
        tests/benchmark_topk.cpp \
        src/topk_softmax.cu \
        -I./src \
        -std=c++17 -O2 \
        -arch=sm_${CUDA_ARCH} \
        -lineinfo

    ${NVCC} -o build/bench_perf \
        tests/bench_perf.cpp \
        src/topk_softmax.cu \
        -I./src \
        -std=c++17 -O2 \
        -arch=sm_${CUDA_ARCH} \
        -lineinfo

    ${NVCC} -o build/bench_baseline \
        tests/bench_baseline.cpp \
        baseline_src/baseline_kernels.cu \
        -I./src -I./baseline_src \
        -std=c++17 -O2 \
        -arch=sm_${CUDA_ARCH} \
        -lineinfo

elif [ "$PLATFORM" = "dl" ]; then
    # DL platform
    SDK_DIR=${SDK_DIR:-/LocalRun/xiaolong.zhu/artifactory/sdk}
    DLCC=${DLCC:-${SDK_DIR}/bin/dlcc}

    echo "Building for DL platform with DLCC: $DLCC"
    echo "SDK_DIR: $SDK_DIR"

    ${DLCC} -o build/benchmark_topk \
        tests/benchmark_topk.cpp \
        src/topk_softmax.cu \
        -I./src -I${SDK_DIR}/include \
        -L${SDK_DIR}/lib -lcurt \
        --cuda-gpu-arch=dlgput64 \
        -mdouble-32 \
        -mllvm -dlgpu-lower-ptx=true \
        -soft-spill-allocator \
        -std=c++17 -O2 -DUSE_DLIN

    ${DLCC} -o build/bench_perf \
        tests/bench_perf.cpp \
        src/topk_softmax.cu \
        -I./src -I${SDK_DIR}/include \
        -L${SDK_DIR}/lib -lcurt \
        --cuda-gpu-arch=dlgput64 \
        -mdouble-32 \
        -mllvm -dlgpu-lower-ptx=true \
        -soft-spill-allocator \
        -std=c++17 -O2 -DUSE_DLIN

    ${DLCC} -o build/bench_baseline \
        tests/bench_baseline.cpp \
        baseline_src/baseline_kernels.cu \
        -I./src -I./baseline_src -I${SDK_DIR}/include \
        -L${SDK_DIR}/lib -lcurt \
        --cuda-gpu-arch=dlgput64 \
        -mdouble-32 \
        -mllvm -dlgpu-lower-ptx=true \
        -soft-spill-allocator \
        -std=c++17 -O2 -DUSE_DLIN
else
    echo "Error: Unknown platform '$PLATFORM'. Use 'dl' or 'nv'"
    exit 1
fi

echo "Done: build/benchmark_topk, build/bench_perf, build/bench_baseline"
