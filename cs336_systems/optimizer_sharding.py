import torch
import torch.distributed as dist
from collections import defaultdict
import argparse
from cs336_basics.optimizer import AdamW
from cs336_basics.model import BasicsTransformerLM
import triton
import torch.distributed as dist
import torch.multiprocessing as mp
from multiprocessing import Manager
import os
import time

class OptimizerSharding(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls, **kwargs):
        # shard params across all the ranks
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.all_params = list(params)

        my_params = [param for i, param in enumerate(self.all_params) if i % self.world_size == self.rank]
        self.my_param_groups = [{"params": my_params}]
        self.optimizer = optimizer_cls(self.my_param_groups, **kwargs)

        self.handles = []

        super().__init__(self.all_params, {})

    def step(self, closure=None, **kwargs):
        self.optimizer.step(closure, **kwargs)

        self.synchronize_params()
        self.wait_for_all_params()

    # should handle assigning the params across all the ranks
    def add_param_group(self, param_group):
        super().add_param_group(param_group)

    def synchronize_params(self):
        for i, param in enumerate(self.all_params):
            rank = i % self.world_size
            self.handles.append(dist.broadcast(param.data, src=rank, async_op=True))

    def wait_for_all_params(self):
        for handle in self.handles:
            handle.wait()

        self.handles.clear()

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# create a simple model
def run_model(rank, world_size, shard_optimizer, result_queue, num_trials, num_warmup_trials):
    setup(rank, world_size)
    device = torch.cuda.current_device()
    xl_transformer = BasicsTransformerLM(
        vocab_size=10000,
        context_length=1024,
        d_model=1600,
        num_layers=48,
        num_heads=25,
        d_ff=6400,
        rope_theta=1e6
    ).to(device)
    print("Loaded model")

    # create some random data
    x = torch.randint(0, 10000, (1, 1024)).to(device)

    if shard_optimizer:
        optimizer = OptimizerSharding(xl_transformer.parameters(), AdamW, lr=0.01)
    else:
        optimizer = AdamW(xl_transformer.parameters(), lr=0.01)

    torch.cuda.reset_peak_memory_stats()
    mem_init = torch.cuda.memory_allocated()
    peak_init = torch.cuda.max_memory_allocated()
    print(f"[INIT] alloc={mem_init/1e9:.2f} GB  peak={peak_init/1e9:.2f} GB")

    def train_loop():
        xl_transformer.zero_grad()
        output = xl_transformer(x)
        loss = output.sum()
        loss.backward()

        torch.cuda.reset_peak_memory_stats()
        mem_pre = torch.cuda.memory_allocated()
        peak_pre = torch.cuda.max_memory_allocated()
        print(f"[BEFORE STEP] alloc={mem_pre/1e9:.2f} GB  peak={peak_pre/1e9:.2f} GB")

        param_bytes = sum(p.numel()*p.element_size() for p in xl_transformer.parameters())
        print(f"  params: {param_bytes/1e9:.2f} GB")

        optimizer.step()

        mem_post = torch.cuda.memory_allocated()
        peak_post = torch.cuda.max_memory_allocated()
        print(f"[AFTER STEP]  alloc={mem_post/1e9:.2f} GB  peak={peak_post/1e9:.2f} GB")

    for _ in range(num_warmup_trials):
        train_loop()

    step_times = []
    for _ in range(num_trials):
        start_time = time.time()
        train_loop()
        end_time = time.time()
        step_times.append(end_time - start_time)

    step_t = torch.tensor(step_times, device=device)
    gathered_steps = [torch.zeros_like(step_t) for _ in range(world_size)]
    dist.all_gather(gathered_steps, step_t)

    if rank == 0:
        steps = [x for t in gathered_steps for x in t.cpu().tolist()]
        result_queue.put(steps)

# benchmark sharding vs no sharding
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    manager = Manager()

    result_queue = manager.Queue()
    mp.spawn(run_model,
            args=(2, True, result_queue, 10, 5),
            nprocs=2,
            join=True)

    shard_optimizer_time = result_queue.get()

    print(f"Shard optimizer time: {shard_optimizer_time}")

    result_queue = manager.Queue()
    mp.spawn(run_model,
            args=(2, False, result_queue, 10, 5),
            nprocs=2,
            join=True)

    no_shard_optimizer_time = result_queue.get()
    print(f"No shard optimizer time: {no_shard_optimizer_time}")
