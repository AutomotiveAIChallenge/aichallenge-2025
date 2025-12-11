import torch
import torchvision.transforms.v2 as T
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

class ImageTransform:
    """
    画像データ('image_raw')に一連の前処理を適用するTransformクラス。
    Numpy (H,W,C) -> Tensor (C,H,W)
    
    __init__で処理を柔軟に切り替え可能です。
    """
    def __init__(
        self,
        resize: Optional[Tuple[int, int]] = (224, 224),
        horizontal_flip: bool = False,
        flip_p: float = 0.5,
        color_jitter: bool = False,
        normalize: bool = True,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225]
    ):
        
        transforms_list = []

        # 1. Numpy (H,W,C) [uint8] -> Tensor (C,H,W) [uint8]
        transforms_list.append(T.ToImage()) 

        # 2. リサイズ
        if resize:
            transforms_list.append(T.Resize(resize, antialias=True))

        # 3. 水平反転 (データ拡張)
        if horizontal_flip:
            transforms_list.append(T.RandomHorizontalFlip(p=flip_p))
            
        # 4. 色ジッター (データ拡張)
        if color_jitter:
            transforms_list.append(T.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ))

        # 5. [uint8] -> [float32] に変換 (0-255 -> 0.0-1.0)
        transforms_list.append(T.ToDtype(torch.float32, scale=True))

        # 6. 正規化
        if normalize:
            transforms_list.append(T.Normalize(mean=mean, std=std))

        # 7. 最終的なTransformパイプラインを構築
        self.transform = T.Compose(transforms_list)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        入力辞書を受け取り、'image_raw'キーに処理を適用し、
        'image_tensor'キーとして結果を辞書に追加して返します。
        """
        if 'image_raw' in data:
            image_np = data['image_raw']
            data['image_tensor'] = self.transform(image_np)
            
        return data

class ScanTransform:
    """
    LiDARスキャンデータ('scan')に前処理を適用するTransformクラス。
    Numpy (N,) -> Tensor (N,)
    """
    def __init__(
        self,
        max_range: float = 30.0,
        normalize: bool = True,
        add_noise: bool = False,
        noise_std: float = 0.01
    ):
        self.max_range = max_range
        self.normalize = normalize
        self.add_noise = add_noise
        self.noise_std = noise_std

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        入力辞書を受け取り、'scan'キーに処理を適用し、
        'scan_tensor'キーとして結果を辞書に追加して返します。
        """
        if 'scan' not in data:
            return data
        
        scan_data = data['scan']
        
        # HDF5SequenceDatasetの実装が 'scan' (np.void) を返す場合:
        if isinstance(scan_data, np.void) and 'ranges' in scan_data.dtype.names:
            scan_np = scan_data['ranges']
        # 'scan' が既に ranges の配列の場合:
        else:
            scan_np = scan_data
        
        # vlen (object) arrayの場合、明示的にfloatに変換
        if scan_np.dtype == 'object':
             scan_np = np.array(scan_np, dtype=np.float32)
        
        # 無限大の値をmax_rangeに置換 (一般的な前処理)
        scan_np[np.isinf(scan_np)] = self.max_range
        # NaNを0に置換
        scan_np[np.isnan(scan_np)] = 0.0
            
        scan_tensor = torch.from_numpy(scan_np.copy()).float()

        if self.normalize:
            # 0.0 ～ max_range の範囲にクリップ
            scan_tensor = torch.clamp(scan_tensor, 0.0, self.max_range)
            # 0.0 ～ 1.0 に正規化
            scan_tensor = scan_tensor / self.max_range

        if self.add_noise:
            noise = torch.randn_like(scan_tensor) * self.noise_std
            scan_tensor = scan_tensor + noise

            if self.normalize:
                scan_tensor = torch.clamp(scan_tensor, 0.0, 1.0)
                
        data['scan_tensor'] = scan_tensor
        return data