import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import bisect
from typing import List, Optional, Callable, Dict, Any, Iterator

from .sequence_dataset import HDF5SequenceDataset

class MultiSequenceConcatDataset(Dataset):
    
    @staticmethod
    def _compute_cumulative_lengths(datasets: List[Dataset]) -> List[int]:
        lengths = [len(d) for d in datasets]
        return np.cumsum(lengths).tolist()

    def __init__(
        self,
        h5_paths: List[Path],
        keys_to_load: List[str],
        len_key: str,
        transform: Optional[Callable] = None
    ):
        self.h5_paths = h5_paths
        
        self.datasets: List[HDF5SequenceDataset] = []
        for h5_path in h5_paths:
            dataset = HDF5SequenceDataset(
                h5_path=h5_path,
                keys_to_load=keys_to_load,
                len_key=len_key,
                transform=transform
            )
            self.datasets.append(dataset)

        if not self.datasets:
            print("Warning: No datasets were loaded.")
            self.cumulative_lengths = []
            self.total_length = 0
        else:
            self.cumulative_lengths = self._compute_cumulative_lengths(self.datasets)
            self.total_length = self.cumulative_lengths[-1]

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0:
            idx = self.total_length + idx

        if not (0 <= idx < self.total_length):
            raise IndexError(f"Index {idx} out of range for concatenated dataset of length {self.total_length}")

        dataset_idx = bisect.bisect_right(self.cumulative_lengths, idx)
        
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_lengths[dataset_idx - 1]
            
        return self.datasets[dataset_idx][sample_idx]

    def iterate_sequences(self) -> Iterator[HDF5SequenceDataset]:
        for dataset in self.datasets:
            yield dataset

    def close_all_files(self):
        for dataset in self.datasets:
            dataset.close()
            
    def __del__(self):
        self.close_all_files()