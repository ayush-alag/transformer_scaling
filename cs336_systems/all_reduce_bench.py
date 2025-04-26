import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
from timeit import default_timer as timer
import pandas as pd

def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def all_reduce_bench(rank, world_size, data_size_mb, backend, num_trials, num_warmup_trials, result_queue):
    setup(rank, world_size, backend)

    num_elements = data_size_mb * 1024 * 1024 // 4

    device = torch.device("cuda") if backend == "nccl" else torch.device("cpu")

    # Warmup trials
    for _ in range(num_warmup_trials):
        data = torch.randint(0, 10, (num_elements,), dtype=torch.float32, device=device)
        dist.all_reduce(data, async_op=False)
        if backend == "nccl":
            torch.cuda.synchronize()

    # Actual trials
    times = []
    for _ in range(num_trials):
        data = torch.randint(0, 10, (num_elements,), dtype=torch.float32, device=device)

        start = timer()
        dist.all_reduce(data, async_op=False)
        if backend == "nccl":
            torch.cuda.synchronize()
        times.append((timer() - start) * 1000)

    avg_time = sum(times) / len(times)

    # Gather times from all processes
    gathered_times = [None] * world_size
    dist.all_gather_object(gathered_times, avg_time)

    if rank == 0:
        final_avg = sum(gathered_times) / len(gathered_times)
        print(f"Average all-reduce time over {num_trials} trials across all ranks: {final_avg:.2f} ms")
        if result_queue is not None:
            result_queue.put(final_avg)

if __name__ == "__main__":
    # add some benchmark arguments via commandline: gloo vs nccl, data size, num processes, num trials, num warmup trials
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="gloo", choices=["gloo", "nccl"])
    parser.add_argument("--data_size_mb", type=int, default=1000)
    parser.add_argument("--num_processes", type=int, default=4)
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--num_warmup_trials", type=int, default=5)
    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)

    backends = ["gloo", "nccl"]
    num_procs = [2, 4, 6]
    data_sizes_mb = [1, 10, 100, 1024]
    for backend in backends:
        results = []
        for num_proc in num_procs:
            for data_size_mb in data_sizes_mb:
                print(f"Running {backend} with {num_proc} processes and {data_size_mb} MB data")
                ctx = mp.get_context('spawn')
                result_queue = ctx.Queue()

                mp.spawn(fn=all_reduce_bench,
                        args=(num_proc, data_size_mb, backend, args.num_trials, args.num_warmup_trials, result_queue),
                        nprocs=num_proc,
                        join=True)

                avg_time = result_queue.get()
                results.append({
                    "Data Size (MB)": data_size_mb,
                    "Number of Processes": num_proc,
                    "Average Time (ms)": avg_time
                })

        # Create and print table after each configuration
        df = pd.DataFrame(results)
        print("\nResults:")
        print(df.to_string(index=False))
        print("\nLaTeX Table:")
        print(df.to_latex(index=False, float_format=lambda x: '{:.2f}'.format(x)))