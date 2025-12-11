import sys
sys.path.append('../') 

from pathlib import Path
from torch.utils.data import DataLoader
from src.data.concat_dataset import MultiSequenceConcatDataset

root = Path("./../datasets")
seq_dirs = sorted([p for p in root.iterdir() if p.is_dir()])

dataset = MultiSequenceConcatDataset(
    seq_dirs=seq_dirs,
    keys_to_load=["control_cmd", "imu_raw", "velocity_status"],
)

loader = DataLoader(dataset, batch_size=8, shuffle=True)

for batch in loader:
    print(batch.keys())
    print(batch["image"].shape)
    print(batch["control_cmd"])
    break
