import torch
import torch.distributed as dist
from collections import defaultdict

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
