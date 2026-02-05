#!/usr/bin/env python3
"""
Generate golden data using vLLM's fused_topk function.

Usage:
    cd standalone_topk_kernel
    python tools/generate_golden.py
"""
import sys
from pathlib import Path

import torch
import numpy as np

# Configuration
FIXED_SEED = 42
RENORMALIZE = False  # Must match standalone kernel test

# Full test configs (816 total)
CONFIGS = {
    'tokens': [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,22,24,26,28,30,32,36,40,44,48,52,56,60,64,72,80,96,112,128],
    'experts': [64, 128, 256],
    'dtypes': ['fp16', 'bf16'],
    'topk': [2, 4, 6, 8]
}

DTYPE_MAP = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}


def check_environment():
    """Check if vLLM is available"""
    try:
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
        assert torch.cuda.is_available(), "CUDA not available"
        print("[OK] vLLM and CUDA ready")
        print(f"     PyTorch: {torch.__version__}, Device: {torch.cuda.get_device_name(0)}")
        return True
    except ImportError as e:
        print(f"[ERROR] Cannot import vLLM: {e}")
        return False
    except AssertionError as e:
        print(f"[ERROR] {e}")
        return False


def verify_consistency(gating_output, hidden_states, topk, num_runs=3):
    """Run multiple times to ensure results are consistent"""
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
    
    results = []
    for _ in range(num_runs):
        weights, indices, _ = fused_topk(hidden_states, gating_output, topk, RENORMALIZE)
        results.append((weights.cpu().numpy(), indices.cpu().numpy()))

    first_w, first_i = results[0]
    for w, i in results[1:]:
        if not np.allclose(first_w, w, rtol=1e-5) or not np.array_equal(first_i, i):
            return False
    return True


def generate_one(tokens, experts, dtype_str, topk, output_dir):
    """Generate golden data for a single configuration"""
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
    
    torch.manual_seed(FIXED_SEED)
    dtype = DTYPE_MAP[dtype_str]

    hidden_states = torch.randn(tokens, 128, dtype=dtype, device="cuda")
    gating_output = torch.randn(tokens, experts, dtype=dtype, device="cuda")

    if not verify_consistency(gating_output, hidden_states, topk):
        return False

    weights, indices, _ = fused_topk(hidden_states, gating_output, topk, RENORMALIZE)

    # Save as binary: [tokens][experts][topk][dtype][input][weights][indices]
    dtype_code = {'fp32': 0, 'fp16': 1, 'bf16': 2}.get(dtype_str, 0)
    path = output_dir / f"tokens_{tokens}_experts_{experts}_{dtype_str}_topk{topk}.bin"
    
    with open(path, 'wb') as f:
        f.write(tokens.to_bytes(4, 'little'))
        f.write(experts.to_bytes(4, 'little'))
        f.write(topk.to_bytes(4, 'little'))
        f.write(dtype_code.to_bytes(4, 'little'))
        f.write(gating_output.cpu().float().flatten().numpy().tobytes())
        f.write(weights.cpu().float().flatten().numpy().tobytes())
        f.write(indices.cpu().int().flatten().numpy().tobytes())

    return True


def main():
    print("=" * 50)
    print(" Golden Data Generator for Standalone TopK Kernel")
    print("=" * 50)
    
    if not check_environment():
        sys.exit(1)

    output_dir = Path("golden")
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(CONFIGS['tokens']) * len(CONFIGS['experts']) * len(CONFIGS['dtypes']) * len(CONFIGS['topk'])
    count = success = 0

    print(f"\nGenerating {total} configurations...")
    print("-" * 50)
    
    for tokens in CONFIGS['tokens']:
        for experts in CONFIGS['experts']:
            for dtype in CONFIGS['dtypes']:
                for topk in CONFIGS['topk']:
                    count += 1
                    status = "OK" if generate_one(tokens, experts, dtype, topk, output_dir) else "FAIL"
                    if status == "OK":
                        success += 1
                    print(f"[{count:3d}/{total}] tokens={tokens:3d} experts={experts:3d} {dtype} topk={topk} [{status}]")

    print("-" * 50)
    print(f"Done! {success}/{total} files in golden/")
    
    if success == total:
        print("\n[SUCCESS] Run ./verify_all.sh to verify kernel correctness")
    sys.exit(0 if success == total else 1)


if __name__ == "__main__":
    main()
