import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

from src.data.concat_dataset import MultiSequenceConcatDataset
from src.data.transform import ImageTransform, ScanTransform
from src.model.tinylidarnet import TinyLidarNet, TinyLidarNetSmall


def save_checkpoint(model: nn.Module, ckpt_dir: Path, is_best: bool = False):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path = ckpt_dir / "last.pth"
    best_path = ckpt_dir / "best.pth"
    torch.save(model.state_dict(), last_path)
    if is_best:
        torch.save(model.state_dict(), best_path)


@hydra.main(version_base="1.2", config_path="config", config_name="train")
def main(cfg: DictConfig):
    print("=== Training Configuration ===")
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    root_dir = Path(cfg.dataset.root)
    seq_dirs = [p for p in root_dir.iterdir() if (p / "sequence_data").exists()]
    if not seq_dirs:
        raise FileNotFoundError(f"No sequence_data directories found under {root_dir}")
    print(f"Found {len(seq_dirs)} sequences")

    dataset = MultiSequenceConcatDataset(
        seq_dirs=seq_dirs,
        keys_to_load=["scan", "control_cmd", "image"],
        transform=ScanTransform(max_range=30.0, normalize=True, add_noise=True),
        image_transform=ImageTransform(resize=(224, 224), horizontal_flip=True, normalize=True),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=cfg.dataset.shuffle,
        num_workers=cfg.dataset.num_workers,
        pin_memory=True,
    )

    if cfg.model.name == "tinylidarnet":
        model = TinyLidarNet(cfg.model.input_dim, cfg.model.output_dim)
    elif cfg.model.name == "tinylidarnet_small":
        model = TinyLidarNetSmall(cfg.model.input_dim, cfg.model.output_dim)
    else:
        raise ValueError(f"Unknown model name: {cfg.model.name}")

    model = model.to(device)
    print(f"Model initialized: {cfg.model.name}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    criterion = nn.MSELoss()

    ckpt_dir = Path(cfg.save.ckpt_dir)
    log_dir = Path(cfg.save.log_dir)
    writer = SummaryWriter(log_dir)

    best_loss = float("inf")

    print("=== Start Training ===")
    for epoch in range(cfg.train.epochs):
        model.train()
        running_loss = 0.0

        for i, batch in enumerate(tqdm(dataloader, desc=f"[Epoch {epoch+1}/{cfg.train.epochs}]")):
            scan = batch.get("scan", None)
            ctrl = batch.get("control_cmd", None)
            if scan is None or ctrl is None:
                continue

            if isinstance(scan, dict) and "ranges" in scan:
                x = scan["ranges"].to(device).unsqueeze(1).float()
            else:
                print("Invalid scan structure, skipping...")
                continue

            if isinstance(ctrl, dict) and "steer" in ctrl and "accel" in ctrl:
                y = torch.stack([ctrl["steer"], ctrl["accel"]], dim=1).to(device).float()
            else:
                print("No valid control commands found in batch, skipping...")
                continue

            num_inf = torch.isinf(x).sum().item()
            num_nan = torch.isnan(x).sum().item()
            if num_inf > 0 or num_nan > 0:
                print(f"[WARN] Detected {num_inf} inf and {num_nan} nan values in scan batch.")
                bad_idx = torch.where(torch.isinf(x[0]) | torch.isnan(x[0]))[1]
                print(f"  Example bad indices: {bad_idx[:10].tolist()}")

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % cfg.train.log_interval == 0:
                avg_loss = running_loss / cfg.train.log_interval
                writer.add_scalar("train/loss", avg_loss, epoch * len(dataloader) + i)
                running_loss = 0.0

        avg_epoch_loss = running_loss / max(1, len(dataloader))
        is_best = avg_epoch_loss < best_loss
        if is_best:
            best_loss = avg_epoch_loss

        save_checkpoint(model, ckpt_dir, is_best)
        writer.add_scalar("epoch/loss", avg_epoch_loss, epoch)
        print(f"[Epoch {epoch+1}] Average Loss: {avg_epoch_loss:.6f} | Best: {best_loss:.6f}")

    writer.close()
    print("Training finished ✅")


if __name__ == "__main__":
    main()
