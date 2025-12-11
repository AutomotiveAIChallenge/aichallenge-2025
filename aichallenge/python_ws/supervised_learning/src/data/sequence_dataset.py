import h5py
import hdf5plugin
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any


class HDF5SequenceDataset(Dataset):
    def __init__(self, h5_path: Path, keys_to_load: List[str], len_key: str, transform: Optional[Callable] = None):
        self.h5_path = Path(h5_path)
        self.keys_to_load = keys_to_load
        self.len_key = len_key
        self.transform = transform
        self._file_handle: Optional[h5py.File] = None
        self._is_open = False  # ✅ 状態フラグ

        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found at: {self.h5_path}")

        try:
            with h5py.File(self.h5_path, "r") as f:
                if self.len_key not in f:
                    raise KeyError(f"Length key '{self.len_key}' not found in HDF5 file.")
                self.dataset_len = len(f[self.len_key])
                self.valid_keys = [k for k in self.keys_to_load if k in f]
        except Exception as e:
            print(f"Error opening {self.h5_path}: {e}")
            self.dataset_len = 0
            self.valid_keys = []

    def _open_file(self):
        if not self._is_open:
            self._file_handle = h5py.File(self.h5_path, "r")
            self._is_open = True

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if not self._is_open:
            self._open_file()

        if idx < 0:
            idx = self.dataset_len + idx
        if not (0 <= idx < self.dataset_len):
            raise IndexError(f"Index {idx} out of range")

        f = self._file_handle
        data = {}

        try:
            for key in self.valid_keys:
                item = f[key][idx]
                data[key] = item
            if "scan" in self.valid_keys:
                data["scan_attrs"] = dict(f["scan"].attrs)
        except Exception as e:
            print(f"Error reading index {idx} from {self.h5_path}: {e}")
            return {}

        if self.transform:
            data = self.transform(data)
        return data

    def close(self):
        """h5ファイルを安全にクローズ"""
        if self._file_handle is not None and self._is_open:
            try:
                self._file_handle.close()
            except Exception:
                pass
        self._file_handle = None
        self._is_open = False  # ✅ 状態をリセット

    def __del__(self):
        # ✅ try/exceptのみ（外部モジュール非依存）
        try:
            self.close()
        except Exception:
            pass
