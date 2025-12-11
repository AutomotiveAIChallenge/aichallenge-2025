import torch
import torch.nn as nn

from .encoder import LidarEncoder1D

class PolicyNet(nn.Module):
    """LiDAR-only Policy（BC, SAC, IQL 共通）"""
    def __init__(self, input_dim=1080, hidden_dim=256, action_dim=2, log_std_init=-0.5):
        super().__init__()
        self.encoder = LidarEncoder1D(input_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)

    def forward(self, lidar):
        h = self.encoder(lidar)
        mu = torch.tanh(self.mu_head(h))          # [-1, 1]
        std = self.log_std.exp().expand_as(mu)
        return mu, std

    def sample_action(self, lidar):
        mu, std = self(lidar)
        eps = torch.randn_like(mu)
        a = torch.tanh(mu + eps * std)
        return a, mu, std
