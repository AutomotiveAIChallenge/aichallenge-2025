import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from src.model import TinyLidarNet, TinyLidarNetSmall
from src.data import MultiFileConcatDataset
from src.loss import WeightedSmoothL1Loss

@hydra.main(config_path="./config", config_name="train", version_base='1.2')
def main(cfg: DictConfig):
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    print("------ Configuration ------")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------------")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    
    train_dataset = MultiFileConcatDataset(
        data_dir=cfg.data.train_dir,
        sequence_length=cfg.data.sequence_length,
        slide_step=cfg.data.slide_step,
        load_into_memory=cfg.data.load_into_memory,
        scan_num_points=cfg.data.scan_num_points
    )
    val_dataset = MultiFileConcatDataset(
        data_dir=cfg.data.val_dir,
        sequence_length=cfg.data.sequence_length,
        slide_step=cfg.data.slide_step,
        load_into_memory=cfg.data.load_into_memory,
        scan_num_points=cfg.data.scan_num_points
    )
    
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.train.batch_size,
                              shuffle=True,
                              num_workers=cfg.train.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=cfg.train.batch_size,
                            shuffle=False,
                            num_workers=cfg.train.num_workers,
                            pin_memory=True)
    
    if cfg.model.name == "TinyLidarNetSmall":
        model = TinyLidarNetSmall(input_dim=cfg.model.input_dim,
                                  output_dim=cfg.model.output_dim).to(device)
    elif cfg.model.name == "TinyLidarNet":
        model = TinyLidarNet(input_dim=cfg.model.input_dim,
                                    output_dim=cfg.model.output_dim).to(device)

    if cfg.train.pretrained_path is not None:
        model.load_state_dict(torch.load(cfg.train.pretrained_path))
        
    criterion = WeightedSmoothL1Loss(
        accel_weight=cfg.train.loss.accel_weight,
        steer_weight=cfg.train.loss.steer_weight
    )

    criterion_val = WeightedSmoothL1Loss(accel_weight=0.0, steer_weight=1.0)

    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)

    save_dir = cfg.train.save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    log_dir = cfg.train.log_dir
    writer = SummaryWriter(log_dir=log_dir)

    best_model_save_path = os.path.join(save_dir, 'best.pth')
    last_model_save_path = os.path.join(save_dir, 'last.pth')
    best_val_loss = float('inf')

    # ===== 学習ループ =====
    for epoch in range(cfg.train.epochs):
        model.train()
        train_loss = 0

        for data_dict in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.train.epochs} [Train]"):
            
            scans = data_dict['scan'].to(device) # -> [64, 1, 1080]
        
            targets = data_dict['control_cmd'].to(device) 

            if torch.isnan(scans).any():
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("!!  異常値 (NaN) を入力データで検出  !!")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            if torch.isinf(scans).any():
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("!!  異常値 (Inf) を入力データで検出  !!")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            targets = targets[:, -1, :]
            outputs = model(scans) # 2次元のままモデルに入力)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        print(f'====> Epoch {epoch+1} Average Train Loss: {avg_train_loss:.4f}')

        avg_val_loss = validate_step(model, val_loader, device, criterion_val, cfg)

        writer.add_scalar('Loss/train', avg_train_loss, epoch + 1)
        writer.add_scalar('Loss/val', avg_val_loss, epoch + 1)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_save_path)
            print(f"モデルを保存しました: {best_model_save_path} (Validation Loss: {best_val_loss:.4f})")

    torch.save(model.state_dict(), last_model_save_path)
    print(f"最終エポックのモデルを保存しました: {last_model_save_path}")
    writer.close()

def validate_step(model: TinyLidarNet, val_loader: DataLoader, device: torch.device, criterion: WeightedSmoothL1Loss, cfg: DictConfig) -> float:
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data_dict in tqdm(val_loader, desc="Validation"):
            scans = data_dict['scan'].to(device) # -> [64, 1, 1080]

            targets = data_dict['control_cmd'].to(device)
            targets = targets[:, -1, :]

            outputs = model(scans) # 2次元のままモデルに入力
            
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            
    avg_val_loss = val_loss / len(val_loader)
    print(f'====> Validation set: Average loss: {avg_val_loss:.4f}')
    return avg_val_loss

if __name__ == '__main__':
    main()