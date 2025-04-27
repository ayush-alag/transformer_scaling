import torch
import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        # print("Weight type", self.fc1.weight.dtype)
        x = self.fc1(x)
        # print("Activation type", x.dtype)
        x = self.relu(x)
        # print("ReLU type", x.dtype)
        x = self.ln(x)
        # print("LN type", x.dtype)
        x = self.fc2(x)
        # print("FC2 type", x.dtype)
        return x

device = torch.device("cuda")
model = ToyModel(10, 10).to(device)
with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
    x = torch.randn(10, device=device)
    y = model(x)
    # calculate loss and gradients
    loss = y.sum()
    loss.backward()
    # print("Loss type", loss.dtype)
    # print("Gradient type", model.fc1.weight.grad.dtype)