import torch
import torch.nn as nn

class LidarEncoder1D(nn.Module):
    """1D LiDARスキャン（1080/540/270点）を入力とする特徴抽出器"""
    def __init__(self, input_dim=1080, hidden_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=10, stride=4), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=6, stride=2), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=4, stride=2), nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim)
            feat_dim = self.conv(dummy).view(1, -1).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )

    def forward(self, x):
        # x: (B, N)
        x = x.unsqueeze(1)          # (B, 1, N)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.fc(x)           # (B, hidden_dim)
