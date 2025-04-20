import argparse
from enum import Enum
from cs336_basics import BasicsTransformerLM
import torch
from timeit import default_timer as timer
import torch.cuda.nvtx as nvtx

def create_model(args):
   model = BasicsTransformerLM(
      vocab_size=args.vocab_size,
      context_length=args.context_length,
      d_model=args.d_model,
      num_layers=args.num_layers,
      num_heads=args.num_heads,
      d_ff=args.d_ff,
      rope_theta=args.rope_theta
   )

   model.to(args.device)

   return model

def benchmark_model(model, args):
    torch.cuda.empty_cache()
    model.eval()
    def run_forward(model, args):
        input_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=args.device)
        output = model(input_ids).mean()

        return output

    def run_backward(output):
        output.backward()

    # some warmup steps
    for _ in range(args.warmup_steps):
        output = run_forward(model, args)
        if args.run_backward:
            run_backward(output)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # benchmark steps
    forward_times = []
    backward_times = []
    for _ in range(args.n_steps):
        start_time = timer()

        with nvtx.range("forward"):
            output = run_forward(model, args)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        forward_times.append((timer() - start_time) * 1000)

        if args.run_backward:
            model.zero_grad()
            start_time = timer()

            with nvtx.range("backward"):
                run_backward(output)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            backward_times.append((timer() - start_time) * 1000)

    print(f"Forward times: {forward_times}")
    print(f"Backward times: {backward_times}")
    mean_forward_time = sum(forward_times) / len(forward_times)
    std_forward_time = (sum((x - mean_forward_time) ** 2 for x in forward_times) / len(forward_times)) ** 0.5
    print(f"Forward pass: {mean_forward_time:.2f} ± {std_forward_time:.2f} ms")

    if args.run_backward:
        mean_backward_time = sum(backward_times) / len(backward_times)
        std_backward_time = (sum((x - mean_backward_time) ** 2 for x in backward_times) / len(backward_times)) ** 0.5
        print(f"Backward pass: {mean_backward_time:.2f} ± {std_backward_time:.2f} ms")
        return mean_forward_time, std_forward_time, mean_backward_time, std_backward_time

    return mean_forward_time, std_forward_time, None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark a specified model")

    # fixed params
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)
    # model architecture
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--rope_theta", type=float, default=1e6)
    # benchmarking args
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--n_steps", type=int, default=10)
    parser.add_argument("--run_backward", action="store_true")
    # device args
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model = create_model(args)
    print(f"Model created on {args.device}")

    benchmark_model(model, args)
