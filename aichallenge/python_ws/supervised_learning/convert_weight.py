import torch
import numpy as np
from pathlib import Path
import argparse
from src.model import TinyLidarNet, TinyLidarNetSmall


# ============================================================
# ユーティリティ関数
# ============================================================
def save_params_to_npy(model: torch.nn.Module, output_path: Path):
    """
    PyTorchモデルの重みをNumPy形式に変換して保存する
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    params = {}

    for name, param in model.state_dict().items():
        np_name = name.replace('.', '_')  # conv1.weight → conv1_weight 形式に統一
        params[np_name] = param.detach().cpu().numpy()

    np.save(output_path, params)
    print(f"✅ Saved NumPy weights to: {output_path}")


# ============================================================
# メイン処理
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch TinyLidarNet weights to NumPy format (.npy)"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["tinylidarnet", "tinylidarnet_small"],
        default="tinylidarnet",
        help="Model type to convert"
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="Path to PyTorch checkpoint (.pth)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./weights/converted_weights.npy"),
        help="Output path for NumPy weight file (.npy)"
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=1080,
        help="Input dimension for the model"
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=2,
        help="Output dimension for the model"
    )
    args = parser.parse_args()

    # --- モデル読み込み ---
    if args.model == "tinylidarnet":
        model = TinyLidarNet(input_dim=args.input_dim, output_dim=args.output_dim)
    else:
        model = TinyLidarNetSmall(input_dim=args.input_dim, output_dim=args.output_dim)

    # --- 重みロード ---
    if args.ckpt.exists():
        state_dict = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"✅ Loaded checkpoint: {args.ckpt}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    # --- NumPy変換 ---
    save_params_to_npy(model, args.output)


if __name__ == "__main__":
    main()
