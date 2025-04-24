import torch
from torch import Tensor
import argparse
from cs336_basics.model import scaled_dot_product_attention, CausalMultiHeadSelfAttention, RotaryEmbedding, BasicsTransformerLM
from timeit import default_timer as timer
from statistics import mean
import pandas as pd

def benchmark_attention(args, model):
    # create random tensors for Q, K, V on CUDA
    # x = torch.randn(args.batch_size, args.sequence_length, args.d_model, device="cuda")
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.sequence_length), device="cuda")

    # warmup steps
    for _ in range(args.num_warmup_trials):
        output = model(x).mean()
        torch.cuda.synchronize()
        output.backward()
        torch.cuda.synchronize()

    # time forward passes
    forward_times = []
    forward_mem = []
    backward_times = []
    backward_mem = []

    for _ in range(args.num_trials):
        # forward pass
        if args.memory_profile:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        start = timer()
        output = model(x).mean()
        torch.cuda.synchronize()
        forward_times.append((timer() - start) * 1000)  # Convert to ms
        if args.memory_profile:
            forward_mem.append(torch.cuda.max_memory_allocated()/ (1024 ** 2))

        # backward pass
        if args.memory_profile:
            torch.cuda.reset_peak_memory_stats()
        start = timer()
        output.backward()
        torch.cuda.synchronize()
        backward_times.append((timer() - start) * 1000)  # Convert to ms
        if args.memory_profile:
            backward_mem.append(torch.cuda.max_memory_allocated()/ (1024 ** 2))

    print(f"Average forward pass time: {sum(forward_times)/len(forward_times):.2f} ms")
    print(f"Average backward pass time: {sum(backward_times)/len(backward_times):.2f} ms")
    if args.memory_profile:
        print(f"Average forward memory: {sum(forward_mem)/len(forward_mem):.2f} MB")
        print(f"Average backward memory: {sum(backward_mem)/len(backward_mem):.2f} MB")

    mean_forward_mem = sum(forward_mem)/len(forward_mem) if args.memory_profile else 0
    mean_backward_mem = sum(backward_mem)/len(backward_mem) if args.memory_profile else 0
    return mean(forward_times), mean(backward_times), mean_forward_mem, mean_backward_mem


if __name__ == "__main__":
    # get d_model and sequence length from command line using argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_trials", type=int, default=20)
    parser.add_argument("--num_warmup_trials", type=int, default=10)
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--memory_profile", action="store_true")
    args = parser.parse_args()

    # seq_lens = [256, 1024, 4096, 8192, 16384]
    seq_lens = [256, 1024, 4096, 8192]
    d_models = [16, 32, 64, 128]
    args.vocab_size = 10000

    results = []
    for seq_len in seq_lens:
        for d_model in d_models:
            print(f"\nBenchmarking {seq_len}x{d_model}")
            # round d_ff = 8/3 * d_model to nearest 64
            d_ff = round(8/3 * d_model / 64) * 64
            # multihead_attn = CausalMultiHeadSelfAttention(d_model, 1, RotaryEmbedding(seq_len, d_model)).to("cuda") # one head for simplicity
            transformer = BasicsTransformerLM(args.vocab_size, seq_len, d_model, 12, 1, d_ff, 1e-5).to("cuda")
            compiled_transformer = transformer
            if args.torch_compile:
                compiled_transformer = torch.compile(transformer)
            uncompiled_transformer = BasicsTransformerLM(args.vocab_size, seq_len, d_model, 12, 1, d_ff, 1e-5).to("cuda")

            args.sequence_length = seq_len
            args.d_model = d_model

            compiled_forward, compiled_backward, _, __ = benchmark_attention(args, compiled_transformer)
            uncompiled_forward, uncompiled_backward, _, __ = benchmark_attention(args, transformer)
            results.append({
                "seq_len": seq_len,
                "d_model": d_model,
                "Forward (compiled)": compiled_forward,
                "Forward (uncompiled)": uncompiled_forward,
                "Backward (compiled)": compiled_backward,
                "Backward (uncompiled)": uncompiled_backward
            })

    df = pd.DataFrame(results)
    latex_table = df.to_latex(index=False, float_format=lambda x: '{:.2f}'.format(x))
    print("\nLaTeX Table:")
    print(latex_table)
