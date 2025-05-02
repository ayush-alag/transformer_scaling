import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
from timeit import default_timer as timer
import pandas as pd
import torch.cuda.nvtx as nvtx

from cs336_basics.optimizer import AdamW
from cs336_basics.model import BasicsTransformerLM
from multiprocessing import Manager
from cs336_systems.ddp_bucketed_container import DDP_Bucketed
from cs336_systems.ddp_container import DDP

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def flat_ddp(model, data, optimizer, num_trials, num_warmup_trials, step_times, communication_times):
    for _ in range(num_warmup_trials):
        optimizer.zero_grad()
        # get the gradients for the batch
        output = model(data)
        loss = output.sum()
        loss.backward()

        # all reduce the gradients
        flatten_params = torch._utils._flatten_dense_tensors(tensors=[param.grad for param in model.parameters()])
        dist.all_reduce(flatten_params, op=dist.ReduceOp.SUM, async_op=False)

        unflatten_params = torch._utils._unflatten_dense_tensors(flatten_params, tensors=[param.grad for param in model.parameters()])
        for param, unflattened_param in zip(model.parameters(), unflatten_params):
            param.grad = unflattened_param.grad

        optimizer.step()

    for _ in range(num_trials):
        optimizer.zero_grad()
        torch.cuda.synchronize()

        start_time = timer()
        # get the gradients for the batch
        output = model(data)
        loss = output.sum()
        nvtx.range_pop()

        nvtx.range_push("backward")
        loss.backward()
        torch.cuda.synchronize()
        nvtx.range_pop()

        communication_start_time = timer()
        flatten_params = torch._utils._flatten_dense_tensors(tensors=[param.grad for param in model.parameters()])

        # a single all reduce for the flattened parameters
        nvtx.range_push("all_reduce")
        dist.all_reduce(flatten_params, op=dist.ReduceOp.SUM, async_op=False)
        nvtx.range_pop()

        unflatten_params = torch._utils._unflatten_dense_tensors(flatten_params, tensors=[param.grad for param in model.parameters()])
        for param, unflattened_param in zip(model.parameters(), unflatten_params):
            param.grad = unflattened_param.grad

        torch.cuda.synchronize()
        communication_time = timer() - communication_start_time
        communication_times.append(communication_time)

        optimizer.step()
        step_times.append(timer() - start_time)

def naive_ddp(model, data, optimizer, num_trials, num_warmup_trials, step_times, communication_times):
    for _ in range(num_warmup_trials):
        optimizer.zero_grad()
        output = model(data)
        loss = output.sum()
        loss.backward()

        # flat all reduce the gradients
        for param in model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=False)

        optimizer.step()

    for _ in range(num_trials):
        optimizer.zero_grad()
        torch.cuda.synchronize()

        start_time = timer()
        # get the gradients for the batch
        nvtx.range_push("forward")
        output = model(data)
        loss = output.sum()
        nvtx.range_pop()

        nvtx.range_push("backward")
        loss.backward()
        torch.cuda.synchronize()
        nvtx.range_pop()

        communication_start_time = timer()
        # flat all reduce the gradients
        nvtx.range_push("all_reduce")
        for param in model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=False)
        nvtx.range_pop()

        torch.cuda.synchronize()
        communication_time = timer() - communication_start_time
        communication_times.append(communication_time)

        optimizer.step()
        step_times.append(timer() - start_time)

def individual_ddp(model, data, optimizer, num_trials, num_warmup_trials, step_times, communication_times):
    ddp = DDP(model)
    for _ in range(num_warmup_trials):
        optimizer.zero_grad()
        output = ddp(data)
        loss = output.sum()
        loss.backward()
        ddp.finish_gradient_synchronization()
        optimizer.step()

    for _ in range(num_trials):
        optimizer.zero_grad()
        torch.cuda.synchronize()

        start_time = timer()
        nvtx.range_push("forward")
        output = ddp(data)
        loss = output.sum()
        nvtx.range_pop()

        nvtx.range_push("backward")
        loss.backward()
        nvtx.range_pop()

        nvtx.range_push("grad_sync")
        ddp.finish_gradient_synchronization()
        nvtx.range_pop()

        optimizer.step()
        step_times.append(timer() - start_time)

def bucketed_ddp(model, data, optimizer, num_trials, num_warmup_trials, step_times, communication_times, bucket_size_mb):
    ddp = DDP_Bucketed(model, bucket_size_mb)
    for _ in range(num_warmup_trials):
        optimizer.zero_grad()
        output = ddp(data)
        loss = output.sum()
        loss.backward()
        ddp.finish_gradient_synchronization()

    for _ in range(num_trials):
        optimizer.zero_grad()
        torch.cuda.synchronize()

        start_time = timer()
        output = ddp(data)
        loss = output.sum()
        loss.backward()
        ddp.finish_gradient_synchronization()

        optimizer.step()
        step_times.append(timer() - start_time)

def benchmark_driver(rank, world_size, data, num_layers, batch_size,
                     num_trials, num_warmup_trials, vocab_size,
                     context_length, d_model, num_heads, d_ff,
                     ddp_type, bucket_size_mb, result_queue):
    setup(rank, world_size)

    torch.manual_seed(0)

    device = torch.cuda.current_device()
    transformer = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=10000,
    ).to(device)
    transformer.train()

    local_batch_size = batch_size // world_size
    start_index = rank * local_batch_size
    end_index = start_index + local_batch_size

    # put the data slice on the device
    data = data[start_index:end_index].to(device)

    optimizer = AdamW(transformer.parameters(), lr=0.001)
    step_times = []
    communication_times = []

    if ddp_type == "naive":
        naive_ddp(transformer, data, optimizer, num_trials, num_warmup_trials, step_times, communication_times)
    elif ddp_type == "flat_ddp":
        flat_ddp(transformer, data, optimizer, num_trials, num_warmup_trials, step_times, communication_times)
    elif ddp_type == "individual_ddp":
        individual_ddp(transformer, data, optimizer, num_trials, num_warmup_trials, step_times, communication_times)
    elif ddp_type == "bucketed_ddp":
        bucketed_ddp(transformer, data, optimizer, num_trials, num_warmup_trials, step_times, communication_times, bucket_size_mb)

    step_t = torch.tensor(step_times, device=device)
    gathered_steps = [torch.zeros_like(step_t) for _ in range(world_size)]
    dist.all_gather(gathered_steps, step_t)

    if ddp_type != "individual_ddp" and ddp_type != "bucketed_ddp":
        comm_t = torch.tensor(communication_times, device=device)
        gathered_comms = [torch.zeros_like(comm_t) for _ in range(world_size)]
        dist.all_gather(gathered_comms, comm_t)
    else:
        gathered_comms = []

    if rank == 0:
        steps = [x for t in gathered_steps for x in t.cpu().tolist()]
        comms = [x for t in gathered_comms for x in t.cpu().tolist()]
        result_queue.put((steps, comms))

if __name__ == "__main__":
    # add some benchmark arguments via commandline: gloo vs nccl, data size, num processes, num trials, num warmup trials
    parser = argparse.ArgumentParser()
    # model configs
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=1600)
    parser.add_argument("--d_ff", type=int, default=6400)
    parser.add_argument("--num_heads", type=int, default=25)
    parser.add_argument("--num_layers", type=int, default=48)
    parser.add_argument("--vocab_size", type=int, default=10000) # fixed at 10000

    # training configs
    parser.add_argument("--num_processes", type=int, default=2)
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--num_warmup_trials", type=int, default=5)

    # type of ddp
    parser.add_argument("--ddp_type", type=str, default="naive", choices=["naive", "flat_ddp", "individual_ddp", "bucketed_ddp"])
    parser.add_argument("--bucket_size_mb", type=float, default=100)
    args = parser.parse_args()

    full_data = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length))

    mp.set_start_method('spawn', force=True)
    manager = Manager()
    result_queue = manager.Queue()

    # now spawn DDP workers
    mp.spawn(benchmark_driver,
            args=(args.num_processes, full_data, args.num_layers, args.batch_size,
                  args.num_trials, args.num_warmup_trials, args.vocab_size,
                  args.context_length, args.d_model, args.num_heads, args.d_ff,
                  args.ddp_type, args.bucket_size_mb, result_queue),
            nprocs=args.num_processes,
            join=True)

    # get the state dict from the queue
    step_times, communication_times = result_queue.get()

    print(f"Step times: {step_times}")
    print(f"Communication times: {communication_times}")
    print(f"Average step time: {sum(step_times) / len(step_times)}")
    if args.ddp_type != "individual_ddp" and args.ddp_type != "bucketed_ddp":
        print(f"Average communication time: {sum(communication_times) / len(communication_times)}")