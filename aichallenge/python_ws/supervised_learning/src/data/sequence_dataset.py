import h5py
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any

class HDF5SequenceDataset(Dataset):
    def __init__(
        self,
        h5_path: Path,
        keys_to_load: List[str],
        len_key: str,
        transform: Optional[Callable] = None
    ):
        self.h5_path = Path(h5_path)
        self.keys_to_load = keys_to_load
        self.len_key = len_key
        self.transform = transform

        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found at: {self.h5_path}")

        try:
            with h5py.File(self.h5_path, 'r') as f:
                if self.len_key not in f:
                    raise KeyError(f"Length key '{self.len_key}' not found in HDF5 file.")
                self.dataset_len = len(f[self.len_key])
                
                for key in self.keys_to_load:
                    if key not in f:
                        print(f"Warning: Key '{key}' not found in {self.h5_path}. It will be skipped.")
                        
        except Exception as e:
            print(f"Error opening {self.h5_path} to get length: {e}")
            self.dataset_len = 0
            
        self._file_handle: Optional[h5py.File] = None
        self.valid_keys = [] 

    def _open_file(self):
        self._file_handle = h5py.File(self.h5_path, 'r')
        self.valid_keys = [k for k in self.keys_to_load if k in self._file_handle]


    def __len__(self) -> int:
        return self.dataset_len

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        
        if self._file_handle is None:
            self._open_file()

        if idx < 0:
            idx = self.dataset_len + idx

        if not (0 <= idx < self.dataset_len):
            raise IndexError(f"Index {idx} out of range for dataset of length {self.dataset_len}")

        data = {}
        try:
            for key in self.valid_keys:
                item = self._file_handle[key][idx]
                
                if isinstance(item, np.void):
                    data[key] = item
                elif isinstance(item, (bytes, str)):
                    data[key] = item
                elif isinstance(item, h5py.vlen_dtype):
                    data[key] = np.array(item)
                else:
                    data[key] = item

            if 'scan' in self.valid_keys:
                data['scan_attrs'] = dict(self._file_handle['scan'].attrs)

        except Exception as e:
            print(f"Error reading index {idx} from {self.h5_path}: {e}")
            return {}

        if self.transform:
            data = self.transform(data)

        return data
    
    def close(self):
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
            
    def __del__(self):
        self.close()