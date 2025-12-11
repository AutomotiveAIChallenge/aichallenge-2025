import torch
import torch.nn as nn

from .encoder import LidarEncoder1D

class VNet(nn.Module):
    """LiDAR-only Valueネットワーク: V(s)"""
    def __init__(self, input_dim=1080, hidden_dim=256):
        super().__init__()
        self.encoder = LidarEncoder1D(input_dim, hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, lidar):
        h = self.encoder(lidar)
        return self.fc(h)
