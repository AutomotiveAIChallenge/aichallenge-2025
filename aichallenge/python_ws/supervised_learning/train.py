import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

from src.data.concat_dataset import MultiSequenceConcatDataset
from src.model.tinylidarnet import TinyLidarNet, TinyLidarNetSmall


# ============================================================
# Utility
# ============================================================
def save_checkpoint(model: nn.Module, ckpt_dir: Path, is_best: bool = False):
    """
    Save model weights only (no optimizer, no loss)
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path = ckpt_dir / "last.pth"
    best_path = ckpt_dir / "best.pth"
    torch.save(model.state_dict(), last_path)
    if is_best:
        torch.save(model.state_dict(), best_path)


# ============================================================
# Training Entry
# ============================================================
@hydra.main(version_base="1.2", config_path="config", config_name="train")
def main(cfg: DictConfig):

    print("=== Training Configuration ===")
    print(OmegaConf.to_yaml(cfg))

    # ------------------------------------------------------------
    # 1. デバイス設定
    # ------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------
    # 2. Dataset 構築
    # ------------------------------------------------------------
    h5_root = Path(cfg.dataset.h5_root)
    h5_files = sorted(h5_root.rglob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found under {h5_root}")

    print(f"Found {len(h5_files)} HDF5 files")
    dataset = MultiSequenceConcatDataset(
        h5_paths=h5_files,
        keys_to_load=cfg.dataset.keys_to_load,
        len_key=cfg.dataset.len_key,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=cfg.dataset.shuffle,
        num_workers=cfg.dataset.num_workers,
        pin_memory=True,
    )

    # ------------------------------------------------------------
    # 3. モデル構築
    # ------------------------------------------------------------
    if cfg.model.name == "tinylidarnet":
        model = TinyLidarNet(cfg.model.input_dim, cfg.model.output_dim)
    elif cfg.model.name == "tinylidarnet_small":
        model = TinyLidarNetSmall(cfg.model.input_dim, cfg.model.output_dim)
    else:
        raise ValueError(f"Unknown model name: {cfg.model.name}")

    model = model.to(device)
    print(f"Model initialized: {cfg.model.name}")

    # ------------------------------------------------------------
    # 4. 最適化設定
    # ------------------------------------------------------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    criterion = nn.MSELoss()

    # ------------------------------------------------------------
    # 5. ログ設定
    # ------------------------------------------------------------
    ckpt_dir = Path(cfg.save.ckpt_dir)
    log_dir = Path(cfg.save.log_dir)
    writer = SummaryWriter(log_dir)

    best_loss = float("inf")

    # ============================================================
    # 6. Training Loop
    # ============================================================
    print("=== Start Training ===")
    for epoch in range(cfg.train.epochs):
        model.train()
        running_loss = 0.0

        for i, batch in enumerate(tqdm(dataloader, desc=f"[Epoch {epoch+1}/{cfg.train.epochs}]")):
            scan = batch.get("scan", None)
            cmd = batch.get("control_cmd", None)
            if scan is None or cmd is None:
                continue

            scan_input = []
            for s in scan:
                if isinstance(s, np.void):
                    continue
                ranges = np.array(s["ranges"], dtype=np.float32)
                scan_input.append(ranges)
            if not scan_input:
                continue

            x = torch.tensor(scan_input, dtype=torch.float32, device=device).unsqueeze(1)
            y = torch.tensor([[c["steering_tire_angle"], c["speed"]] for c in cmd],
                             dtype=torch.float32, device=device)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % cfg.train.log_interval == 0:
                avg_loss = running_loss / cfg.train.log_interval
                writer.add_scalar("train/loss", avg_loss, epoch * len(dataloader) + i)
                print(f"Epoch [{epoch+1}/{cfg.train.epochs}] Step [{i+1}/{len(dataloader)}] "
                      f"Loss: {avg_loss:.6f}")
                running_loss = 0.0

        # --- エポック単位での保存 ---
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
