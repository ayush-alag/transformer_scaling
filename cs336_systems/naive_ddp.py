import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
from timeit import default_timer as timer
import pandas as pd
from toy_model import ToyModel
from cs336_basics.optimizer import AdamW
from multiprocessing import Manager

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def data_parallel_main(rank, world_size, data, num_layers, num_steps, result_queue):
    setup(rank, world_size)

    torch.manual_seed(0)

    batch_size = data.size(0)
    num_dim = data.size(1)
    local_batch_size = batch_size // world_size
    start_index = rank * local_batch_size
    end_index = start_index + local_batch_size

    device = torch.cuda.current_device()
    toy_model = ToyModel(num_dim, num_dim).to(device)
    toy_model.train()

    # put the data slice on the device
    data = data[start_index:end_index].to(device)

    optimizer = AdamW(toy_model.parameters(), lr=0.001)

    for _ in range(num_steps):
        optimizer.zero_grad()
        # get the gradients for the batch
        output = toy_model(data)
        loss = output.sum()
        loss.backward()

        # all reduce the gradients
        for param in toy_model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=False)

        optimizer.step()

    if rank == 0:
        cpu_state = {k: v.detach().cpu() for k,v in toy_model.state_dict().items()}
        result_queue.put(cpu_state)

def train_single_process(data, dim, num_steps):
    torch.manual_seed(0)
    model = ToyModel(dim, dim).to("cuda")
    data = data.to("cuda")
    opt = AdamW(model.parameters(), lr=0.001)
    model.train()

    for _ in range(num_steps):
        opt.zero_grad()
        out = model(data)
        loss = out.sum()
        loss.backward()
        opt.step()

    return {k: v.detach().cpu() for k,v in model.state_dict().items()}

if __name__ == "__main__":
    # add some benchmark arguments via commandline: gloo vs nccl, data size, num processes, num trials, num warmup trials
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--num_processes", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--num_layers", type=int, default=12)
    args = parser.parse_args()

    full_data = torch.randn(args.batch_size, args.dim)
    ref_state = train_single_process(full_data, args.dim, args.num_steps)

    mp.set_start_method('spawn', force=True)
    manager = Manager()
    result_queue = manager.Queue()

    # now spawn DDP workers
    mp.spawn(data_parallel_main,
            args=(args.num_processes, full_data, args.num_layers, args.num_steps, result_queue),
            nprocs=args.num_processes,
            join=True)

    # get the state dict from the queue
    state_dict = result_queue.get()

    # compare the state dicts
    for key in ref_state.keys():
        assert torch.allclose(ref_state[key], state_dict[key])

    print("All state dicts are equal")
