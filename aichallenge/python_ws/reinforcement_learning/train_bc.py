import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import argparse
import yaml
import numpy as np

from src.model.policy import PolicyNet
from src.data.concat_dataset import MultiSequenceDataset


def load_config(cfg_path):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def train_bc(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # --- Dataset ---
    dataset = MultiSequenceDataset(
        root_dir=Path(cfg["dataset"]["path"]),
        downsample_dim=cfg["dataset"]["downsample_dim"],
        normalize=True,
        require_next=False,
        max_seq=cfg["dataset"].get("max_seq", None)
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True
    )

    # --- Model ---
    model = PolicyNet(
        input_dim=cfg["dataset"]["downsample_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        action_dim=cfg["model"]["action_dim"]
    ).to(device)

    criterion = nn.SmoothL1Loss() 
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"]
    )

    # --- Save path ---
    save_dir = Path(cfg["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = cfg["train"]["epochs"]
    best_loss = np.inf
    last_ckpt = save_dir / "bc_last.pth"
    best_ckpt = save_dir / "bc_best.pth"

    print(f"📦 Training for {epochs} epochs on {len(dataset)} samples")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in tqdm(loader, desc=f"[Epoch {epoch}/{epochs}]"):
            obs = batch["obs"].to(device)
            act_gt = batch["action"].to(device)

            mu, _ = model(obs)
            loss = criterion(mu, act_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        print(f"Epoch {epoch}: Train Loss = {avg_loss:.6f}")

        # --- checkpoint save ---
        torch.save(model.state_dict(), last_ckpt)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_ckpt)
            print(f"⭐ New best model saved (Loss={best_loss:.6f})")

        if epoch % cfg["train"]["save_interval"] == 0:
            torch.save(model.state_dict(), save_dir / f"bc_epoch{epoch:03d}.pth")

    # --- final checkpoint ---
    torch.save(model.state_dict(), save_dir / "bc_final.pth")
    print(f"✅ Training completed. Final/Best models saved in {save_dir}")

    dataset.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LiDAR-only BC Policy with best checkpoint saving")
    parser.add_argument("--config", type=str, default="./config/train_bc.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_bc(cfg)
