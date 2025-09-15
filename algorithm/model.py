import torch
import torch.nn as nn

# Policy 网络：输入 obs=16 输出 action=3
class MLPPolicy(nn.Module):
    def __init__(self, input_dim=16, output_dim=3, device="cuda:0"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, output_dim)
        )
        self.device = torch.device(device)

    def forward(self, x):
        return self.net(x)

# Critic 网络：输入 obs+action = 16+3 输出 Q-value
class MLPCritic(nn.Module):
    def __init__(self, input_dim=19, device="cuda:0"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 1)
        )
        self.device = torch.device(device)

    def forward(self, x):
        return self.net(x)
