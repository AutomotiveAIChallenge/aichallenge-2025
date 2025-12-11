import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import h5py


# ============================================================
# LiDAR Utility
# ============================================================

def lidar_preprocess(x: np.ndarray, max_range=30.0, normalize=True):
    """LiDAR距離データを正規化"""
    x = np.nan_to_num(x, nan=max_range, posinf=max_range, neginf=0.0)
    x = np.clip(x, 0, max_range)
    if normalize:
        x = np.log1p(x) / np.log1p(max_range)
    return x.astype(np.float32)


def lidar_downsample(x: np.ndarray, target_dim: int):
    """1080点を540,270へ等間隔ダウンサンプリング"""
    if x.shape[-1] == target_dim:
        return x
    N = x.shape[-1]
    idx = np.linspace(0, N - 1, target_dim, dtype=np.int32)
    return x[..., idx]


# ============================================================
# Sequence Dataset (1 sequence = 1 h5)
# ============================================================

class LidarSequenceDataset(Dataset):
    """
    単一のHDF5 (1 bag = 1 sequence) を扱う Dataset
    """
    def __init__(self, 
                 h5_path: Path, 
                 downsample_dim=270, 
                 normalize=True,
                 require_next=True):
        super().__init__()
        self.h5_path = Path(h5_path)
        assert self.h5_path.exists(), f"Missing file: {self.h5_path}"

        self.h5 = h5py.File(self.h5_path, "r")
        self.scans = self.h5["scan"]
        self.controls = self.h5["control"]
        self.idx = self.h5["transitions/index"][:]
        self.done = self.h5["transitions/done"][:].astype(bool)

        self.downsample_dim = downsample_dim
        self.normalize = normalize
        self.require_next = require_next
        self.max_range = 30.0

        self.length = len(self.idx)

    def __len__(self):
        return self.length

    def _proc_scan(self, arr):
        arr = lidar_downsample(arr, self.downsample_dim)
        arr = lidar_preprocess(arr, self.max_range, self.normalize)
        return torch.from_numpy(arr).float()

    def __getitem__(self, i):
        t = int(self.idx[i])
        done = bool(self.done[i])
        s = self._proc_scan(self.scans[t])
        s2 = self._proc_scan(self.scans[t+1])
        a = torch.tensor(self.controls[t], dtype=torch.float32)

        if self.require_next:
            return {
                "obs": s,
                "action": a,
                "next_obs": s2,
                "done": torch.tensor(done, dtype=torch.bool),
                "seq_name": self.h5.attrs.get("bag_name", "unknown"),
                "index": i,
            }
        else:
            return {
                "obs": s,
                "action": a,
                "seq_name": self.h5.attrs.get("bag_name", "unknown"),
                "index": i,
            }

    def close(self):
        if hasattr(self, "h5"):
            self.h5.close()
