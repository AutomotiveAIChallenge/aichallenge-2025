from torch.utils.data import Dataset
import bisect
from typing import List, Optional, Callable, Dict, Any
import numpy as np
from pathlib import Path

from .sequence_dataset import HDF5SequenceDataset


class MultiSequenceConcatDataset(Dataset):
    def __init__(self, h5_paths: List[Path], keys_to_load: List[str], len_key: str, transform: Optional[Callable] = None):
        self.datasets: List[HDF5SequenceDataset] = []
        for p in h5_paths:
            if p.exists():
                self.datasets.append(HDF5SequenceDataset(p, keys_to_load, len_key, transform))

        if not self.datasets:
            print("Warning: No datasets loaded.")
            self.cumulative_lengths = []
            self.total_length = 0
        else:
            self.cumulative_lengths = np.cumsum([len(d) for d in self.datasets]).tolist()
            self.total_length = self.cumulative_lengths[-1]

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0:
            idx = self.total_length + idx
        if not (0 <= idx < self.total_length):
            raise IndexError(f"Index {idx} out of range for dataset length {self.total_length}")

        dataset_idx = bisect.bisect_right(self.cumulative_lengths, idx)
        sample_idx = idx if dataset_idx == 0 else idx - self.cumulative_lengths[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    def close_all_files(self):
        """全てのh5を安全にクローズ"""
        for d in self.datasets:
            try:
                d.close()
            except Exception:
                pass

    def __del__(self):
        """GC破棄時も安全にクローズ"""
        try:
            self.close_all_files()
        except Exception:
            pass
