from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from src.model import TinyLidarNet, TinyLidarNetSmall
from src.data import MultiSeqConcatDataset
from src.loss import WeightedSmoothL1Loss


def select_device(preferred: str = 'auto') -> torch.device:
    """Select computation device with graceful CUDA capability fallback.

    preferred: 'auto' | 'cpu' | 'cuda'. If 'auto', choose CUDA only if available AND supported.
    Returns torch.device instance.
    """
    if preferred not in ('auto', 'cpu', 'cuda'):
        print(f"[WARN] Unknown device preference '{preferred}', using auto.")
        preferred = 'auto'

    if preferred == 'cpu':
        return torch.device('cpu')
    if preferred == 'cuda':
        if torch.cuda.is_available():
            return torch.device('cuda')
        print('[WARN] CUDA requested but not available; falling back to CPU.')
        return torch.device('cpu')

    # auto mode
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()  # (major, minor)
        sm_tag = f"sm_{cap[0]}{cap[1]}"
        supported_sm = {('5', '0'), ('6', '0'), ('7', '0'), ('7', '5'), ('8', '0'), ('8', '6'), ('9', '0')}  # coarse markers
        # Build currently compiled arch list from torch version string heuristics is non-trivial; we attempt a runtime check instead.
        try:
            _probe = torch.empty(1, device='cuda')  # allocate to test kernel availability
            print(f"[INFO] Using CUDA device (capability {cap[0]}.{cap[1]} -> {sm_tag}).")
            return torch.device('cuda')
        except RuntimeError as e:
            # Typical message: 'no kernel image is available for execution on the device'
            if 'no kernel image' in str(e):
                print(f"[WARN] CUDA capability {cap[0]}.{cap[1]} ({sm_tag}) unsupported by this PyTorch build; falling back to CPU.")
                print('[HINT] Install a newer PyTorch/nightly or build from source with TORCH_CUDA_ARCH_LIST set, e.g. TORCH_CUDA_ARCH_LIST=12.0')
                return torch.device('cpu')
            raise  # re-raise unexpected CUDA errors
    return torch.device('cpu')



def clean_numerical_tensor(x: torch.Tensor) -> torch.Tensor:
    """NaN, infを安全に除去 (graceful fallback if CUDA kernel unsupported)."""
    try:
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x
    except RuntimeError as e:
        if 'no kernel image' in str(e):
            # Move to CPU and retry
            print('[WARN] CUDA kernel unsupported for nan/inf ops; moving tensor to CPU and retrying.')
            x_cpu = x.to('cpu')
            if torch.isnan(x_cpu).any() or torch.isinf(x_cpu).any():
                x_cpu = torch.nan_to_num(x_cpu, nan=0.0, posinf=0.0, neginf=0.0)
            return x_cpu
        raise


@hydra.main(config_path="./config", config_name="train", version_base="1.2")
def main(cfg: DictConfig):
    print("------ Configuration ------")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------------")

    # Device selection with capability-aware fallback. Allow user override via cfg.train.device (optional).
    preferred = "cpu"
    device = select_device(preferred)
    print(f"Using device: {device}")

    # === Dataset ===
    train_dataset = MultiSeqConcatDataset(cfg.data.train_dir)
    val_dataset = MultiSeqConcatDataset(cfg.data.val_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # === Model ===
    if cfg.model.name == "TinyLidarNetSmall":
        model = TinyLidarNetSmall(
            input_dim=cfg.model.input_dim,
            output_dim=cfg.model.output_dim
        ).to(device)
    else:
        model = TinyLidarNet(
            input_dim=cfg.model.input_dim,
            output_dim=cfg.model.output_dim
        ).to(device)

    if cfg.train.pretrained_path:
        model.load_state_dict(torch.load(cfg.train.pretrained_path))
        print(f"[INFO] Loaded pretrained model from {cfg.train.pretrained_path}")

    # === Loss & Optimizer ===
    criterion = WeightedSmoothL1Loss(
        steer_weight=cfg.train.loss.steer_weight,
        accel_weight=cfg.train.loss.accel_weight
    )
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)

    # === Logging & Save dirs ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(cfg.train.save_dir).expanduser().resolve()
    log_dir = Path(cfg.train.log_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir / timestamp)
    best_val_loss = float("inf")
    patience_counter = 0
    max_patience = cfg.train.get("early_stop_patience", 10)

    best_path = save_dir / "best_model.pth"
    last_path = save_dir / "last_model.pth"

    # === Training Loop ===
    for epoch in range(cfg.train.epochs):
        model.train()
        train_loss = 0.0

        for scans, targets in tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{cfg.train.epochs}"):
            scans = scans.unsqueeze(1).to(device)  # [B, 1, 1080]
            targets = targets.to(device)           # [B, 2]

            scans = clean_numerical_tensor(scans)
            targets = clean_numerical_tensor(targets)
            scans = scans.to(device)

            outputs = model(scans)  # -> [B, 2] = [accel, steer]
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = validate(model, val_loader, device, criterion)

        print(f"Epoch {epoch+1:03d}: Train={avg_train_loss:.4f} | Val={avg_val_loss:.4f}")
        writer.add_scalar("Loss/train", avg_train_loss, epoch + 1)
        writer.add_scalar("Loss/val", avg_val_loss, epoch + 1)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_path)
            print(f"[SAVE] Best model updated: {best_path} (val_loss={best_val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        torch.save(model.state_dict(), last_path)
        if patience_counter >= max_patience:
            print(f"[EarlyStop] No improvement for {max_patience} epochs.")
            break

    writer.close()
    print("Training finished.")


def validate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for scans, targets in tqdm(loader, desc="[Val]", leave=False):
            scans = scans.unsqueeze(1).to(device)
            targets = targets.to(device)
            scans = clean_numerical_tensor(scans)
            targets = clean_numerical_tensor(targets)
            outputs = model(scans)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(loader)


if __name__ == "__main__":
    main()
