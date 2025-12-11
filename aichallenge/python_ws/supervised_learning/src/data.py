import os
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset


class ScanControlSequenceDataset(Dataset):
    """
    単一シーケンスを扱うDataset。
    (scan, [steer, accel]) のペアを返す。
    scanはmax_range(例:30.0m)で割って正規化。
    """
    def __init__(self, seq_dir: str, max_range: float = 30.0):
        self.seq_dir = Path(seq_dir)
        self.max_range = max_range

        self.scans = np.load(self.seq_dir / "scans.npy")        # (N, num_points)
        self.steers = np.load(self.seq_dir / "steers.npy")      # (N,)
        self.accels = np.load(self.seq_dir / "accelerations.npy")  # (N,)

        n = len(self.scans)
        if not (len(self.steers) == len(self.accels) == n):
            raise ValueError(f"Data length mismatch in {seq_dir}")

        # クリッピング＋正規化
        self.scans = np.clip(self.scans, 0.0, self.max_range) / self.max_range

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, idx):
        scan = self.scans[idx].astype(np.float32)
        accel = np.float32(self.accels[idx])
        steer = np.float32(self.steers[idx])
        target = np.array([accel, steer], dtype=np.float32) 
        return scan, target


class MultiSeqConcatDataset(ConcatDataset):
    """
    複数シーケンスをまとめるConcatDataset。
    dataset_root直下のフォルダを自動探索。
    """
    def __init__(self, dataset_root: str, max_range: float = 30.0, include=None, exclude=None):
        dataset_root = Path(dataset_root)
        seq_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])

        if include:
            seq_dirs = [p for p in seq_dirs if any(i in p.name for i in include)]
        if exclude:
            seq_dirs = [p for p in seq_dirs if all(e not in p.name for e in exclude)]

        datasets = []
        for seq_dir in seq_dirs:
            scans = seq_dir / "scans.npy"
            steers = seq_dir / "steers.npy"
            accels = seq_dir / "accelerations.npy"
            if all(f.exists() for f in [scans, steers, accels]):
                ds = ScanControlSequenceDataset(seq_dir, max_range=max_range)
                datasets.append(ds)
            else:
                print(f"[WARN] Skipping {seq_dir} (missing npy files)")

        if len(datasets) == 0:
            raise RuntimeError(f"No valid sequences found in {dataset_root}")

        super().__init__(datasets)
        print(f"[INFO] Loaded {len(datasets)} sequences from {dataset_root}")

