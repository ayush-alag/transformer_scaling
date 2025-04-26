import torch
import triton
import pandas as pd
from typing import Tuple, List
from einops import einsum
from math import sqrt
import argparse

from cs336_systems.flash_attention2 import FlashAttention2, TritonFlashAttention2


def generate_inputs(batch_size: int, seq_len: int, dim: int, dtype: torch.dtype, device: str = "cuda") -> Tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    q = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype, requires_grad=True)
    do = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype)
    return q, k, v, do

@torch.compile()
def flash_attn_compiled(q, k, v, is_causal):
    return FlashAttention2.apply(q, k, v, is_causal)

def pytorch_flash_attn(q, k, v, is_causal=True):
    return flash_attn_compiled(q, k, v, is_causal)

def triton_flash_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
    return TritonFlashAttention2.apply(q, k, v, is_causal)

# impl is either pytorch_flash_attn or triton_flash_attn
def benchmark_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, impl) -> float:
    return triton.testing.do_bench(lambda: impl(q, k, v, True))

def benchmark_backward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, do: torch.Tensor, impl) -> float:
    def run_backward():
        out = impl(q, k, v, True)
        out.backward(do)
        torch.cuda.synchronize()

    return triton.testing.do_bench(run_backward)

def benchmark_configs():
    seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    dims = [8, 16, 32, 64, 128]
    dtypes = [torch.bfloat16, torch.float32]

    results_by_dtype = {dtype: [] for dtype in dtypes}

    for dtype in dtypes:
        for seq_len in seq_lengths:
            for dim in dims:
                # Skip configurations that would exceed GPU memory
                if seq_len * dim * (8 if dtype == torch.float32 else 4) > 2**31:
                    continue

                print(f"Benchmarking seq_len={seq_len}, dim={dim}, dtype={dtype}")

                # Generate inputs
                q, k, v, do = generate_inputs(1, seq_len, dim, dtype)

                # Adjust tile sizes based on input dimensions
                if seq_len <= 2048:
                    q_tile_size = 64
                    k_tile_size = 64
                elif seq_len <= 8192:
                    q_tile_size = 32
                    k_tile_size = 32
                else:
                    q_tile_size = 16
                    k_tile_size = 16

                # Benchmark PyTorch
                pytorch_fwd = benchmark_forward(q, k, v, pytorch_flash_attn)
                pytorch_bwd = benchmark_backward(q, k, v, do, pytorch_flash_attn)

                # Benchmark Triton
                triton_fwd = benchmark_forward(q, k, v, triton_flash_attn)
                triton_bwd = benchmark_backward(q, k, v, do, triton_flash_attn)

                results_by_dtype[dtype].append({
                    'seq_len': seq_len,
                    'dim': dim,
                    'pytorch_forward_ms': pytorch_fwd,
                    'pytorch_backward_ms': pytorch_bwd,
                    'pytorch_total_ms': pytorch_fwd + pytorch_bwd,
                    'triton_forward_ms': triton_fwd,
                    'triton_backward_ms': triton_bwd,
                    'triton_total_ms': triton_fwd + triton_bwd,
                })

                # Clear GPU memory
                torch.cuda.empty_cache()

    # Create DataFrames and convert to LaTeX tables
    latex_tables = {}
    for dtype, results in results_by_dtype.items():
        df = pd.DataFrame(results)
        latex_table = df.to_latex(index=False, float_format=lambda x: '{:.2f}'.format(x))
        latex_tables[str(dtype).split('.')[-1]] = latex_table
    return latex_tables


if __name__ == "__main__":
    assert torch.cuda.is_available(), "This benchmark requires a CUDA GPU"
    assert torch.cuda.get_device_name().startswith("NVIDIA H100"), "This benchmark should be run on an H100 GPU"

    results_df = benchmark_configs()
    print("\nBenchmark Results:")
    print(results_df.to_string(index=False))
