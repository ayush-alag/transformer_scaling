import torch.distributed as dist
import torch

class DDP(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.handles = []

        # initialize all parameters to be the same
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self.gradient_hook)

    def gradient_hook(self, grad):
        self.handles.append(dist.all_reduce(grad, op=dist.ReduceOp.SUM, async_op=True))
        return grad

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()

        self.handles.clear()