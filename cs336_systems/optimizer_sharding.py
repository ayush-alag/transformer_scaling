import torch
import torch.distributed as dist
from collections import defaultdict
class OptimizerSharding(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls, **kwargs):
        # shard params across all the ranks
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.all_params = list(params)
        self.my_params = [param for i, param in enumerate(self.all_params) if i % self.world_size == self.rank]
        self.all_params_to_rank = {id(param): i % self.world_size for i, param in enumerate(self.all_params)}
        self.param_groups = []

        self.optimizer = None
        self.optimizer_cls = optimizer_cls
        self.kwargs = kwargs

        self.handles = []

        super().__init__(self.all_params, {})

    def step(self, closure=None, **kwargs):
        self.optimizer.step(closure, **kwargs)

        self.synchronize_params()
        self.wait_for_all_params()

    # should handle assigning the params across all the ranks
    def add_param_group(self, param_group):
        new_param_group = defaultdict(list)

        # we only want to add params that are not already in the optimizer
        for group, params in param_group.items():
            for param in params:
                if param not in self.all_params_to_rank:
                    self.all_params.append(param)
                    rank = (len(self.all_params) - 1) % self.world_size
                    self.all_params_to_rank[param] = rank
                    if rank == self.rank:
                        self.my_params.append(param)
                        new_param_group[group].append(param)

        if self.optimizer is None:
            self.optimizer = self.optimizer_cls(self.my_params, **self.kwargs)
        else:
            self.optimizer.add_param_group(new_param_group)

    def synchronize_params(self):
        for param in self.all_params:
            self.handles.append(dist.broadcast(param.data, src=self.all_params_to_rank[param], async_op=True))

    def wait_for_all_params(self):
        for handle in self.handles:
            handle.wait()

        self.handles.clear()
