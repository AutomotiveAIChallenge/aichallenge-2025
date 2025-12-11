import torch
import torch.nn as nn
import torch.nn.functional as F

from . import (
    conv1d,
    linear,
    relu,
    tanh,   
    flatten,
    kaiming_normal_init,
    zeros_init,
)


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
    
class TinyLidarNetNp:
    """
    LiDARデータ用の標準的なCNNモデル (NumPy版)
    - Conv層: 5層
    - FC層: 4層
    """
    def __init__(self, input_dim=1080, output_dim=2):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # パラメータを辞書で管理
        self.params = {}
        
        # --- 畳み込み層 (Convolutional Layers) ---
        self.strides = {
            'conv1': 4,
            'conv2': 4,
            'conv3': 2,
            'conv4': 1,
            'conv5': 1,
        }
        
        # レイヤー形状の定義
        self.shapes = {
            'conv1_w': (24, 1, 10),  'conv1_b': (24,),
            'conv2_w': (36, 24, 8), 'conv2_b': (36,),
            'conv3_w': (48, 36, 4), 'conv3_b': (48,),
            'conv4_w': (64, 48, 3), 'conv4_b': (64,),
            'conv5_w': (64, 64, 3), 'conv5_b': (64,),
        }

        # --- 全結合層 (Fully Connected Layers) ---
        # Conv層の出力サイズを動的に計算
        flatten_dim = self._get_conv_output_dim()

        self.shapes.update({
            'fc1_w': (100, flatten_dim), 'fc1_b': (100,),
            'fc2_w': (50, 100),       'fc2_b': (50,),
            'fc3_w': (10, 50),        'fc3_b': (10,),
            'fc4_w': (output_dim, 10), 'fc4_b': (output_dim,),
        })

        # 重みの初期化処理を呼び出し
        self._initialize_weights()

    def _get_conv_output_dim(self):
        """Conv層の出力サイズを動的に計算"""
        # ダミー入力 (Batch=1, Channel=1, Length)
        l = self.input_dim
        
        l = (l - self.shapes['conv1_w'][2]) // self.strides['conv1'] + 1
        l = (l - self.shapes['conv2_w'][2]) // self.strides['conv2'] + 1
        l = (l - self.shapes['conv3_w'][2]) // self.strides['conv3'] + 1
        l = (l - self.shapes['conv4_w'][2]) // self.strides['conv4'] + 1
        l = (l - self.shapes['conv5_w'][2]) // self.strides['conv5'] + 1
        
        # 最終チャンネル数 * 最終長
        c = self.shapes['conv5_w'][0] 
        
        # (1, 64, 28) -> flatten -> 64 * 28 = 1792
        return c * l

    def _initialize_weights(self):
        """モデルの重みをKaiming He初期化する"""
        for name, shape in self.shapes.items():
            if name.endswith('_w'): # 重み
                if name.startswith('conv'):
                    # fan_out = out_channels * kernel_size
                    fan_out = shape[0] * shape[2]
                else: # linear
                    # fan_out = out_features
                    fan_out = shape[0]
                
                self.params[name] = kaiming_normal_init(shape, fan_out)
                
            elif name.endswith('_b'): # バイアス
                self.params[name] = zeros_init(shape)

    def __call__(self, x):
        """順伝播 (PyTorchのforwardに相当)"""
        
        # Conv層 + ReLU
        x = relu(conv1d(x, self.params['conv1_w'], self.params['conv1_b'], self.strides['conv1']))
        x = relu(conv1d(x, self.params['conv2_w'], self.params['conv2_b'], self.strides['conv2']))
        x = relu(conv1d(x, self.params['conv3_w'], self.params['conv3_b'], self.strides['conv3']))
        x = relu(conv1d(x, self.params['conv4_w'], self.params['conv4_b'], self.strides['conv4']))
        x = relu(conv1d(x, self.params['conv5_w'], self.params['conv5_b'], self.strides['conv5']))

        # 平坦化 (Flatten)
        x = flatten(x)

        # FC層 + ReLU
        x = relu(linear(x, self.params['fc1_w'], self.params['fc1_b']))
        x = relu(linear(x, self.params['fc2_w'], self.params['fc2_b']))
        x = relu(linear(x, self.params['fc3_w'], self.params['fc3_b']))

        # 出力層 + Tanh
        x = tanh(linear(x, self.params['fc4_w'], self.params['fc4_b']))
        
        return x
    
class TinyLidarNetSmallNp:
    """
    LiDARデータ用の軽量版CNNモデル (NumPy版)
    - Conv層: 3層
    - FC層: 3層
    """
    def __init__(self, input_dim=1080, output_dim=2):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # パラメータを辞書で管理
        self.params = {}
        
        # --- 畳み込み層 (Convolutional Layers) ---
        self.strides = {
            'conv1': 4,
            'conv2': 4,
            'conv3': 2,
        }
        
        # レイヤー形状の定義
        self.shapes = {
            'conv1_w': (24, 1, 10),  'conv1_b': (24,),
            'conv2_w': (36, 24, 8), 'conv2_b': (36,),
            'conv3_w': (48, 36, 4), 'conv3_b': (48,),
        }

        # --- 全結合層 (Fully Connected Layers) ---
        # Conv層の出力サイズを動的に計算
        flatten_dim = self._get_conv_output_dim()

        self.shapes.update({
            'fc1_w': (100, flatten_dim), 'fc1_b': (100,),
            'fc2_w': (50, 100),       'fc2_b': (50,),
            'fc3_w': (output_dim, 50), 'fc3_b': (output_dim,),
        })

        # 重みの初期化処理を呼び出し
        self._initialize_weights()

    def _get_conv_output_dim(self):
        """Conv層の出力サイズを動的に計算"""
        l = self.input_dim
        
        l = (l - self.shapes['conv1_w'][2]) // self.strides['conv1'] + 1
        l = (l - self.shapes['conv2_w'][2]) // self.strides['conv2'] + 1
        l = (l - self.shapes['conv3_w'][2]) // self.strides['conv3'] + 1
        
        # 最終チャンネル数 * 最終長
        c = self.shapes['conv3_w'][0] 
        
        # (1, 48, 32) -> flatten -> 48 * 32 = 1536
        return c * l

    def _initialize_weights(self):
        """モデルの重みをKaiming He初期化する"""
        for name, shape in self.shapes.items():
            if name.endswith('_w'): # 重み
                if name.startswith('conv'):
                    fan_out = shape[0] * shape[2]
                else: # linear
                    fan_out = shape[0]
                
                self.params[name] = kaiming_normal_init(shape, fan_out)
                
            elif name.endswith('_b'): # バイアス
                self.params[name] = zeros_init(shape)

    def __call__(self, x):
        """順伝播 (PyTorchのforwardに相当)"""
        
        # Conv層 + ReLU
        x = relu(conv1d(x, self.params['conv1_w'], self.params['conv1_b'], self.strides['conv1']))
        x = relu(conv1d(x, self.params['conv2_w'], self.params['conv2_b'], self.strides['conv2']))
        x = relu(conv1d(x, self.params['conv3_w'], self.params['conv3_b'], self.strides['conv3']))

        # 平坦化 (Flatten)
        x = flatten(x)
        
        # FC層 + ReLU
        x = relu(linear(x, self.params['fc1_w'], self.params['fc1_b']))
        x = relu(linear(x, self.params['fc2_w'], self.params['fc2_b']))
        
        # 出力層 + Tanh
        x = tanh(linear(x, self.params['fc3_w'], self.params['fc3_b']))
        
        return x