import sys
sys.path.append('../') 

from pathlib import Path
from src.data.concat_dataset import MultiSequenceConcatDataset

root = Path("./../datasets/")
h5_files = sorted(root.glob("*_synced.h5"))

dataset = MultiSequenceConcatDataset(
    h5_paths=h5_files,
    keys_to_load=["control_cmd", "velocity_status", "pose_with_covariance"],
    len_key="timestamp"
)

print(len(dataset))  # 全シーケンス合計長
sample = dataset[0]
print(sample.keys())  # dict_keys(['control_cmd', 'velocity_status', 'pose_with_covariance'])
