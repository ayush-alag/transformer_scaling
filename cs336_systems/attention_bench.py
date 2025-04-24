import torch
from torch import Tensor
import argparse
from cs336_basics.model import scaled_dot_product_attention, CausalMultiHeadSelfAttention, RotaryEmbedding
from timeit import default_timer as timer
from statistics import mean

def benchmark_attention(args, self_attn):
    # create random tensors for Q, K, V on CUDA
    x = torch.randn(args.batch_size, args.sequence_length, args.d_model, device="cuda")

    # warmup steps
    for _ in range(args.num_warmup_trials):
        output = self_attn(x).mean()
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
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = timer()
        output = self_attn(x).mean()
        torch.cuda.synchronize()
        forward_times.append((timer() - start) * 1000)  # Convert to ms
        forward_mem.append(torch.cuda.max_memory_allocated()/ (1024 ** 2))

        # backward pass
        torch.cuda.reset_peak_memory_stats()
        start = timer()
        output.backward()
        torch.cuda.synchronize()
        backward_times.append((timer() - start) * 1000)  # Convert to ms
        backward_mem.append(torch.cuda.max_memory_allocated()/ (1024 ** 2))

    print(f"Average forward pass time: {sum(forward_times)/len(forward_times):.2f} ms")
    print(f"Average backward pass time: {sum(backward_times)/len(backward_times):.2f} ms")
    print(f"Average forward memory: {sum(forward_mem)/len(forward_mem):.2f} MB")
    print(f"Average backward memory: {sum(backward_mem)/len(backward_mem):.2f} MB")

    return mean(forward_times), mean(backward_times), mean(forward_mem), mean(backward_mem)


if __name__ == "__main__":
    # get d_model and sequence length from command line using argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_trials", type=int, default=100)
    parser.add_argument("--num_warmup_trials", type=int, default=10)
    args = parser.parse_args()

    seq_lens = [256, 1024, 4096, 8192, 16384]
    d_models = [16, 32, 64, 128]

    results = []
    for seq_len in seq_lens:
        for d_model in d_models:
            multihead_attn = CausalMultiHeadSelfAttention(d_model, 1, RotaryEmbedding(seq_len, d_model)).to("cuda") # one head for simplicity
            args.sequence_length = seq_len
            args.d_model = d_model

            forward_time, backward_time, forward_mem, backward_mem = benchmark_attention(args, multihead_attn)
            results.append({
                "seq_len": seq_len,
                "d_model": d_model,
                "forward_time": forward_time,
                "backward_time": backward_time,
                "forward_mem": forward_mem,
                "backward_mem": backward_mem
            })

    df = pd.DataFrame(results)
    latex_table = df.to_latex(index=False, float_format=lambda x: '{:.2f}'.format(x))
    print("\nLaTeX Table:")
    print(latex_table)
