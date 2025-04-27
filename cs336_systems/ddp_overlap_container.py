class DDP:
    def __init__(self, model):
        self.model = model

    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        pass