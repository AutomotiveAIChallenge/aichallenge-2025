import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, Any, Optional, Callable, List
from PIL import Image
import torch


def _npstruct_to_dict(x: np.void) -> Dict[str, Any]:
    names = x.dtype.names or []
    return {k: x[k].item() if isinstance(x[k], np.ndarray) and x[k].shape == () else x[k] for k in names}


def _to_tensor_if_numeric(x):
    if isinstance(x, np.ndarray) and x.dtype != object:
        return torch.from_numpy(x)
    if isinstance(x, (float, int)):
        return torch.tensor(x)
    return x


class SequenceDataset(Dataset):
    def __init__(
        self,
        seq_dir: Path,
        keys_to_load: List[str],
        transform: Optional[Callable] = None,
        image_transform: Optional[Callable] = None,
        prefer_len_key_order: Optional[List[str]] = None,
    ):
        self.seq_dir = Path(seq_dir)
        self.keys_to_load = list(keys_to_load)
        self.transform = transform
        self.image_transform = image_transform

        seq_data_dir = self.seq_dir / "sequence_data"
        cam_dir = self.seq_dir / "camera_front"

        self.arrays: Dict[str, np.ndarray] = {}
        self.key_lengths: Dict[str, int] = {}

        for key in self.keys_to_load:
            npy_path = seq_data_dir / f"{key}.npy"
            if npy_path.exists():
                arr = np.load(npy_path, allow_pickle=True)
                self.arrays[key] = arr
                self.key_lengths[key] = len(arr)

        self.image_paths: List[Path] = []
        if "image" in self.keys_to_load:
            if cam_dir.exists():
                self.image_paths = sorted(cam_dir.glob("*.png"))
            elif (self.seq_dir / "images").exists():
                self.image_paths = sorted((self.seq_dir / "images").glob("*.png"))

        prefer_len_key_order = prefer_len_key_order or ["timestamps", "control_cmd", "scan"]
        length = 0
        for k in prefer_len_key_order:
            if k in self.key_lengths:
                length = self.key_lengths[k]
                break
        if length == 0:
            length = max(self.key_lengths.values(), default=0)
        length = max(length, len(self.image_paths))
        self.dataset_len = int(length)

        print(f"[LOAD] {self.seq_dir.name} → frames={self.dataset_len} (images={len(self.image_paths)}, npy={len(self.arrays)})")

    def __len__(self):
        return self.dataset_len

    def _get_item_from_key(self, key: str, idx: int):
        arr = self.arrays.get(key)
        if arr is None or idx >= len(arr):
            return None

        item = arr[idx]
        if isinstance(item, np.void):
            item = _npstruct_to_dict(item)

        if key == "scan" and isinstance(item, dict):
            ranges = item.get("ranges", None)
            if isinstance(ranges, np.ndarray) and ranges.size > 0:
                ranges = np.nan_to_num(ranges, nan=0.0, posinf=30.0, neginf=0.0)
                return {"ranges": ranges.astype(np.float32)}
            else:
                return None

        if isinstance(item, dict):
            ret = {}
            for k, v in item.items():
                if isinstance(v, np.ndarray) and v.dtype != object:
                    ret[k] = v
                elif isinstance(v, (float, int)):
                    ret[k] = v
                else:
                    ret[k] = v
            return ret

        return item

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0:
            idx = self.dataset_len + idx
        if not (0 <= idx < self.dataset_len):
            raise IndexError(f"Index {idx} out of range")

        data: Dict[str, Any] = {}

        if "image" in self.keys_to_load and self.image_paths:
            if idx < len(self.image_paths):
                img = Image.open(self.image_paths[idx]).convert("RGB")
                data["image_raw"] = np.asarray(img, dtype=np.uint8)

        for key in self.keys_to_load:
            if key == "image":
                continue
            val = self._get_item_from_key(key, idx)
            if val is not None:
                data[key] = val

        if self.image_transform:
            data = self.image_transform(data)
        if self.transform:
            data = self.transform(data)

        return data
