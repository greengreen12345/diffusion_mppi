import torch

# 假设 x_nerf 的形状为 (batch_size, in_features)
x_nerf = torch.randn(1000000, 30, device="cuda", requires_grad=True)
linear = torch.nn.Linear(30, 256).to("cuda")

try:
    y = linear(x_nerf)
    print("Forward pass succeeded.")
except RuntimeError as e:
    print(f"Error: {e}")
