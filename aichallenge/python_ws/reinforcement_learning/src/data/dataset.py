import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import hdf5plugin

# ============================================================
# Utility
# ============================================================

def lidar_preprocess(x: np.ndarray, max_range: float = 30.0, normalize=True):
    """LiDAR距離をlogスケール＋正規化する"""
    x = np.nan_to_num(x, nan=max_range, posinf=max_range, neginf=0.0)
    x = np.clip(x, 0, max_range)
    if normalize:
        x = np.log1p(x) / np.log1p(max_range)
    return x.astype(np.float32)

def lidar_downsample(x: np.ndarray, target_dim: int):
    """等間隔にダウンサンプリング（1080→540→270など）"""
    if x.shape[-1] == target_dim:
        return x
    N = x.shape[-1]
    idx = np.linspace(0, N - 1, target_dim, dtype=np.int32)
    return x[..., idx]

# ============================================================
# Dataset Class
# ============================================================

class LidarHDF5Dataset(Dataset):
    """
    LiDAR-only dataset for BC / Offline RL (IQL/CQL)
    - scan, control, transitions/index を使用
    - normalize=True でlog正規化処理
    """
    def __init__(self,
                 h5_path,
                 normalize=True,
                 downsample_dim=1080,
                 device="cpu",
                 require_next=True):
        """
        Args:
            h5_path: Path to merged_sequence.h5
            normalize: Trueならlog正規化を適用
            device: 'cpu' or 'cuda'
            require_next: Trueなら (s, a, s', done) を返す（RL用）
                          Falseなら (s, a) のみ（BC用）
        """
        self.path = str(h5_path)
        self.normalize = normalize
        self.downsample_dim = downsample_dim
        self.device = device
        self.require_next = require_next

        # --- open file ---
        self.h5 = h5py.File(self.path, "r")
        self.scans = self.h5["scan"]
        self.actions = self.h5["control"]
        self.idx = self.h5["transitions/index"][:]  # 有効インデックス
        self.done = self.h5["transitions/done"][:].astype(np.bool_)

        # --- precompute stats for normalization ---
        self.max_range = 30.0
        self.num_beams = self.scans.shape[1]

        print(f"✅ Loaded {self.path}: {len(self.idx)} transitions")

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        t = int(self.idx[i])
        done = bool(self.done[i])

        obs = self.scans[t]
        next_obs = self.scans[t + 1]
        action = self.actions[t]

        obs = lidar_preprocess(obs, max_range=self.max_range, normalize=self.normalize)
        next_obs = lidar_preprocess(next_obs, max_range=self.max_range, normalize=self.normalize)

        obs = torch.from_numpy(obs).to(torch.float32)
        next_obs = torch.from_numpy(next_obs).to(torch.float32)
        action = torch.from_numpy(action).to(torch.float32)
        done = torch.tensor(done, dtype=torch.bool)

        if self.require_next:
            return {
                "obs": obs,           # LiDAR 1080
                "action": action,     # 行動 [steer, accel]
                "next_obs": next_obs, # 次状態
                "done": done,
            }
        else:
            return {
                "obs": obs,
                "action": action,
            }

    def close(self):
        if hasattr(self, "h5"):
            self.h5.close()
