import h5py
import hdf5plugin
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from pathlib import Path
from typing import List, Dict, Optional

# -----------------------------------------------------------------------------
# 担当：単一HDF5ファイルのシーケンス処理 (SequenceH5Dataset)
# -----------------------------------------------------------------------------

class SequenceH5Dataset(Dataset):
    """
    単一のHDF5ファイルから'scan'と'control_cmd'のシーケンスデータを読み込むクラス。

    ファイル内の2つのデータセットを同期したシーケンスとして抽出し、
    それぞれに対応した前処理を適用します。
    オプションで、ファイル全体をメモリにキャッシュして高速なアクセスを実現します。
    """
    def __init__(self,
                 h5_file_path: Path,
                 sequence_length: int,
                 slide_step: int = 1,
                 load_into_memory: bool = False,
                 scan_num_points: Optional[int] = 1080,
                 scan_normalization_factor: float = 30.0): 
        """
        Args:
            h5_file_path (Path): HDF5ファイルのパス。
            sequence_length (int): 1サンプルあたりのシーケンス長。
            slide_step (int): シーケンスをスライドさせる幅（ストライド）。
            load_into_memory (bool): Trueの場合、全データを初期化時にRAMに読み込む。
            scan_num_points (Optional[int]): 'scan'データの点群数を固定長にする場合の点群数。
            scan_normalization_factor (float): 'scan'データの正規化に使用する除数。
        """
        super().__init__()
        self.h5_file_path = h5_file_path
        self.sequence_length = sequence_length
        self.slide_step = slide_step
        self.load_into_memory = load_into_memory
        self.scan_num_points = scan_num_points
        self.scan_normalization_factor = scan_normalization_factor 
        self.data_keys = ['scan', 'control_cmd']

        if not self.h5_file_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_file_path}")

        self.index_map: List[int] = []
        self.data_cache: Optional[Dict[str, np.ndarray]] = None

        self._initialize_dataset()

    def _initialize_dataset(self):
        """HDF5ファイルをスキャンしてインデックスを構築し、必要に応じてデータをメモリにロードする。"""
        with h5py.File(self.h5_file_path, 'r') as f:
            for key in self.data_keys:
                if key not in f:
                    raise ValueError(f"Required dataset '{key}' not found in {self.h5_file_path}")
            
            num_samples_in_file = f[self.data_keys[0]].shape[0]

            last_start = num_samples_in_file - self.sequence_length
            if last_start >= 0:
                self.index_map = list(range(0, last_start + 1, self.slide_step))

            if self.load_into_memory:
                self.data_cache = {key: f[key][:] for key in self.data_keys}

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not 0 <= idx < len(self.index_map):
            raise IndexError("Index out of range")

        start_index = self.index_map[idx]
        end_index = start_index + self.sequence_length
        
        raw_sample = self._get_raw_sample(start_index, end_index)
        tensor_sample = self._transform_sample(raw_sample)
        
        return tensor_sample

    def _get_raw_sample(self, start: int, end: int) -> Dict[str, np.ndarray]:
        """指定された範囲のデータをファイルまたはキャッシュから読み込む。"""
        if self.load_into_memory and self.data_cache:
            return {key: self.data_cache[key][start:end] for key in self.data_keys}
        
        sample_dict = {}
        with h5py.File(self.h5_file_path, 'r') as f:
            for key in self.data_keys:
                sample_dict[key] = f[key][start:end]
        return sample_dict

    def _transform_sample(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Numpy配列の辞書を、Tensorの辞書に変換する。"""
        return {
            'scan': self._process_scan(sample['scan']),
            'control_cmd': torch.from_numpy(sample['control_cmd']).float()[:, [1, 2]] 
        }

    def _process_scan(self, scan_data: np.ndarray) -> torch.Tensor:
        """LiDARスキャンデータを処理: 可変長を固定長に, 正規化"""
        ranges_data = scan_data['ranges']
        
        if self.scan_num_points is not None:
            fixed_len = self.scan_num_points
            processed = np.zeros((self.sequence_length, fixed_len), dtype=np.float32)
            for i, r_array in enumerate(ranges_data):
                current_len = len(r_array)
                if current_len > fixed_len:
                    indices = np.linspace(0, current_len - 1, fixed_len, dtype=int)
                    processed[i] = r_array[indices]
                else:
                    processed[i, :current_len] = r_array
        else:
            max_len = max(len(r) for r in ranges_data)
            processed = np.zeros((self.sequence_length, max_len), dtype=np.float32)
            for i, r_array in enumerate(ranges_data):
                processed[i, :len(r_array)] = r_array

        # 保存しておいた正規化係数を使用
        return torch.from_numpy(processed).float() / self.scan_normalization_factor

# -----------------------------------------------------------------------------
# 担当：複数HDF5ファイルの結合 (MultiFileConcatDataset)
# -----------------------------------------------------------------------------

class MultiFileConcatDataset(ConcatDataset):
    """
    指定されたディレクトリ内の全HDF5ファイルを自動で探索し、
    `SequenceH5Dataset`として結合するラッパークラス。
    """
    def __init__(self, 
                 data_dir: str, 
                 sequence_length: int, 
                 slide_step: int = 1, 
                 load_into_memory: bool = False, 
                 scan_num_points: int = 1080,
                 scan_normalization_factor: float = 30.0): 
        """
        Args:
            data_dir (str): HDF5ファイルが格納されているディレクトリのパス。
            (その他、SequenceH5Datasetのコンストラクタと同じ引数)
        """
        self.data_dir = Path(data_dir)
        h5_files = sorted(list(self.data_dir.rglob('*.h5')))
        
        if not h5_files:
            raise FileNotFoundError(f"No HDF5 files found in the directory: {data_dir}")
        
        print(f"Found {len(h5_files)} HDF5 files in '{data_dir}'. Initializing datasets...")

        datasets = [
            SequenceH5Dataset(
                h5_file_path=h5_path,
                sequence_length=sequence_length,
                slide_step=slide_step,
                load_into_memory=load_into_memory,
                scan_num_points=scan_num_points,
                scan_normalization_factor=scan_normalization_factor 
            ) for h5_path in h5_files
        ]

        super().__init__(datasets)
        print(f"Successfully created a concatenated dataset with a total of {len(self)} sequences.")