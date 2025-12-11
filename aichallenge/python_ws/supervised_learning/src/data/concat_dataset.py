from torch.utils.data import Dataset
import bisect
from typing import List, Optional, Callable, Dict, Any
import numpy as np
from pathlib import Path

from .sequence_dataset import SequenceDataset


class MultiSequenceConcatDataset(Dataset):
    """
    複数の SequenceDataset を結合して1つの大規模Datasetとして扱う。
    各シーケンスは sequence_data/ 以下に .npy や images/ を持つ構造を想定。
    """

    def __init__(
        self,
        seq_dirs: List[Path],
        keys_to_load: List[str],
        transform: Optional[Callable] = None,         # ScanTransformなど
        image_transform: Optional[Callable] = None,   # ImageTransformなど
    ):
        self.datasets: List[SequenceDataset] = []

        for p in seq_dirs:
            if not p.exists():
                continue
            try:
                dataset = SequenceDataset(
                    seq_dir=p,
                    keys_to_load=keys_to_load,
                    transform=transform,              # 🔹 ScanTransformを渡す
                    image_transform=image_transform,  # 🔹 ImageTransformを渡す
                )
                if len(dataset) > 0:
                    self.datasets.append(dataset)
            except Exception as e:
                print(f"[WARN] Failed to load dataset at {p}: {e}")

        if not self.datasets:
            print("⚠️ Warning: No valid sequence datasets loaded.")
            self.cumulative_lengths = []
            self.total_length = 0
        else:
            self.cumulative_lengths = np.cumsum([len(d) for d in self.datasets]).tolist()
            self.total_length = self.cumulative_lengths[-1]
            print(f"✅ Loaded {len(self.datasets)} sequences. Total frames: {self.total_length}")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0:
            idx = self.total_length + idx
        if not (0 <= idx < self.total_length):
            raise IndexError(f"Index {idx} out of range for dataset length {self.total_length}")

        dataset_idx = bisect.bisect_right(self.cumulative_lengths, idx)
        sample_idx = idx if dataset_idx == 0 else idx - self.cumulative_lengths[dataset_idx - 1]

        # 🔹 各SequenceDatasetで ImageTransform / ScanTransform が自動適用される
        return self.datasets[dataset_idx][sample_idx]
