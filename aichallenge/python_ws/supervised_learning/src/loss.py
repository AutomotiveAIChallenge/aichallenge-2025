import torch
import torch.nn as nn


class WeightedSmoothL1Loss(nn.Module):
    """
    加速度(accel)と舵角(steer)の損失にそれぞれ重みを付けて計算する損失関数。
    """
    def __init__(self, accel_weight: float = 1.0, steer_weight: float = 1.0):
        super().__init__()
        self.accel_weight = accel_weight
        self.steer_weight = steer_weight
        # reduction='none'にすることで、要素ごとの損失を計算できる
        self.criterion = nn.SmoothL1Loss(reduction='none')

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            outputs (torch.Tensor): モデルの出力 (B*T, 2)。[accel, steer]の順。
            targets (torch.Tensor): 教師データ (B*T, 2)。[accel, steer]の順。
        
        Returns:
            torch.Tensor: 重み付けされた最終的な損失。
        """
        # (B*T, 2) の形状で要素ごとの損失を計算
        loss = self.criterion(outputs, targets)

        # データセットの実装から、0番目がaccel、1番目がsteer
        loss_accel = loss[:, 0]
        loss_steer = loss[:, 1]
        
        # 各損失の平均を取り、重みを掛けて合計する
        weighted_loss = (self.accel_weight * loss_accel.mean()) + \
                        (self.steer_weight * loss_steer.mean())
        
        return weighted_loss