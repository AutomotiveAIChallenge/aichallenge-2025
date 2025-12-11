import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyLidarNet(nn.Module):
    """
    LiDARデータ用の標準的なCNNモデル
    - Conv層: 5層
    - FC層: 4層
    """
    def __init__(self, input_dim=1080, output_dim=2):
        super().__init__()

        # --- 畳み込み層 (Convolutional Layers) ---
        self.conv1 = nn.Conv1d(1, 24, kernel_size=10, stride=4)
        self.conv2 = nn.Conv1d(24, 36, kernel_size=8, stride=4)
        self.conv3 = nn.Conv1d(36, 48, kernel_size=4, stride=2)
        self.conv4 = nn.Conv1d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3)

        # --- 全結合層 (Fully Connected Layers) ---
        # Conv層の出力サイズを動的に計算
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)
            x = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(dummy_input)))))
            flatten_dim = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flatten_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, output_dim)

        # 重みの初期化処理を呼び出し
        self._initialize_weights()

    def _initialize_weights(self):
        """モデルの重みをKaiming He初期化する"""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):

        # 入力形状: (Batch, 1, Length)
        # Conv層 + ReLU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        # (B, C, L) -> (B, C*L) : 平坦化 (Flatten)
        x = x.view(x.size(0), -1)

        # FC層 + ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # 出力層 + Tanh
        x = torch.tanh(self.fc4(x))
        
        return x
    

class TinyLidarNetSmall(nn.Module):
    """
    LiDARデータ用の軽量版CNNモデル
    - Conv層: 3層
    - FC層: 3層
    """
    def __init__(self, input_dim=1080, output_dim=2):
        super().__init__()

        # --- 畳み込み層 (Convolutional Layers) ---
        self.conv1 = nn.Conv1d(1, 24, kernel_size=10, stride=4)
        self.conv2 = nn.Conv1d(24, 36, kernel_size=8, stride=4)
        self.conv3 = nn.Conv1d(36, 48, kernel_size=4, stride=2)

        # --- 全結合層 (Fully Connected Layers) ---
        # Conv層の出力サイズを動的に計算
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)
            x = self.conv3(self.conv2(self.conv1(dummy_input)))
            flatten_dim = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flatten_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, output_dim)

        # 重みの初期化処理を呼び出し
        self._initialize_weights()

    def _initialize_weights(self):
        """モデルの重みをKaiming He初期化する"""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 入力形状: (Batch, 1, Length)
        
        # Conv層 + ReLU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # (B, C, L) -> (B, C*L) : 平坦化 (Flatten)
        x = x.view(x.size(0), -1)
        
        # FC層 + ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # 出力層 + Tanh
        x = torch.tanh(self.fc3(x))
        
        return x