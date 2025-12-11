import torch
import torch.nn as nn

from .encoder import LidarEncoder1D

class QNet(nn.Module):
    """LiDAR-only Qネットワーク: Q(s,a)"""
    def __init__(self, input_dim=1080, action_dim=2, hidden_dim=256):
        super().__init__()
        self.encoder = LidarEncoder1D(input_dim, hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, lidar, action):
        h = self.encoder(lidar)
        x = torch.cat([h, action], dim=-1)
        return self.fc(x)
