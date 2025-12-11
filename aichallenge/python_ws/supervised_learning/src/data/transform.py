import torch
import torchvision.transforms.v2 as T
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image


class ImageTransform:
    """
    画像データ ('image_raw') に一連の前処理を適用。
    - Numpy(H,W,C) / PIL -> Tensor(C,H,W)
    - Resize, Flip, ColorJitter, Normalize 等を柔軟に設定可能。
    """
    def __init__(
        self,
        resize: Optional[Tuple[int, int]] = (224, 224),
        horizontal_flip: bool = False,
        flip_p: float = 0.5,
        color_jitter: bool = False,
        normalize: bool = True,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
    ):
        transforms_list = [T.ToImage()]

        if resize:
            transforms_list.append(T.Resize(resize, antialias=True))
        if horizontal_flip:
            transforms_list.append(T.RandomHorizontalFlip(p=flip_p))
        if color_jitter:
            transforms_list.append(T.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ))

        transforms_list.append(T.ToDtype(torch.float32, scale=True))

        if normalize:
            transforms_list.append(T.Normalize(mean=mean, std=std))

        self.transform = T.Compose(transforms_list)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if 'image_raw' not in data:
            return data

        img = data['image_raw']

        # numpy, PIL どちらでもOK
        if isinstance(img, np.ndarray):
            # 万一 grayscale の場合 → RGB に変換
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
        elif isinstance(img, Image.Image):
            img = np.asarray(img)

        # 型変換安全化
        img = np.ascontiguousarray(img, dtype=np.uint8)

        try:
            data['image_tensor'] = self.transform(img)
        except Exception as e:
            print(f"[WARN] Image transform failed: {e}")
            data['image_tensor'] = torch.zeros(3, 224, 224, dtype=torch.float32)

        return data


class ScanTransform:
    """
    LiDARスキャン ('scan') の前処理。
    - 辞書形式 {'ranges': np.ndarray} に対応。
    - NaN/inf 除去・正規化・ノイズ付加に対応。
    """
    def __init__(
        self,
        max_range: float = 30.0,
        normalize: bool = True,
        add_noise: bool = False,
        noise_std: float = 0.01,
    ):
        self.max_range = max_range
        self.normalize = normalize
        self.add_noise = add_noise
        self.noise_std = noise_std

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if 'scan' not in data:
            return data

        scan_data = data['scan']

        # --- 形式ごとの分岐 ---
        if isinstance(scan_data, dict):
            scan_np = scan_data.get('ranges', None)
        elif isinstance(scan_data, np.void) and hasattr(scan_data, 'dtype') and 'ranges' in scan_data.dtype.names:
            scan_np = scan_data['ranges']
        else:
            scan_np = scan_data

        if scan_np is None:
            return data
        if isinstance(scan_np, torch.Tensor):
            scan_np = scan_np.cpu().numpy()
        if not isinstance(scan_np, np.ndarray):
            return data

        # --- NaN/Inf 除去 ---
        scan_np = np.nan_to_num(scan_np, nan=0.0, posinf=self.max_range, neginf=0.0)

        # --- Tensor化 ---
        scan_tensor = torch.from_numpy(scan_np.copy()).float()

        # --- 正規化 ---
        if self.normalize:
            scan_tensor = torch.clamp(scan_tensor, 0.0, self.max_range)
            scan_tensor = scan_tensor / self.max_range

        # --- ノイズ付加 ---
        if self.add_noise:
            noise = torch.randn_like(scan_tensor) * self.noise_std
            scan_tensor = torch.clamp(scan_tensor + noise, 0.0, 1.0)

        data['scan_tensor'] = scan_tensor
        return data
