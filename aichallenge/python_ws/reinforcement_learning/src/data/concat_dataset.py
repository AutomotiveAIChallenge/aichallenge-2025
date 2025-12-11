from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from typing import List

from .seqence_dataset import LidarSequenceDataset

class MultiSequenceDataset(Dataset):
    """
    複数のHDF5 sequenceを統合して1つのDatasetとして扱う
    """
    def __init__(self, root_dir: Path, 
                 downsample_dim=270, 
                 normalize=True, 
                 require_next=True,
                 max_seq=None):
        super().__init__()
        self.root_dir = Path(root_dir)
        assert self.root_dir.exists(), f"Invalid path: {self.root_dir}"

        self.seq_paths = sorted(list(self.root_dir.glob("*/merged_sequence.h5")))
        if max_seq is not None:
            self.seq_paths = self.seq_paths[:max_seq]

        print(f"✅ Found {len(self.seq_paths)} sequences under {root_dir}")

        # 各sequenceの長さを記録
        self.datasets: List[LidarSequenceDataset] = []
        self.seq_lengths = []
        total = 0
        for p in self.seq_paths:
            d = LidarSequenceDataset(p, downsample_dim, normalize, require_next)
            self.datasets.append(d)
            total += len(d)
            self.seq_lengths.append(len(d))

        self.total_len = total
        self.cum_lengths = np.cumsum([0] + self.seq_lengths)

        print(f"📊 Total transitions: {self.total_len}")

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        # indexから該当シーケンスを特定
        seq_idx = np.searchsorted(self.cum_lengths, index, side="right") - 1
        local_idx = index - self.cum_lengths[seq_idx]
        return self.datasets[seq_idx][local_idx]

    def close(self):
        for d in self.datasets:
            d.close()
