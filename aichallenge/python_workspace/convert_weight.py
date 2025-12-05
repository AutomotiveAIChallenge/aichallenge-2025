import argparse
from pathlib import Path
import sys

import numpy as np
import torch

try:
    from src.model import TinyLidarNet, TinyLidarNetSmall
except ImportError:
    print("Error: Could not import 'src.model'. Please ensure you are in the project root.")
    sys.exit(1)


def save_params_to_npy(model: torch.nn.Module, output_path: Path) -> None:
    """
    Extracts model parameters and saves them as a NumPy dictionary file.

    Layer names are sanitized by replacing dots ('.') with underscores ('_')
    to facilitate easier variable mapping in non-PyTorch environments.

    Args:
        model: The PyTorch model instance containing the weights to save.
        output_path: The filesystem path where the .npy file will be written.
    """
    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    params = {}
    for name, param in model.state_dict().items():
        # Sanitize parameter names (e.g., "conv1.weight" -> "conv1_weight")
        np_name = name.replace('.', '_')
        params[np_name] = param.detach().cpu().numpy()

    np.save(output_path, params)
    print(f"✅ Saved NumPy weights to: {output_path}")


def main() -> None:
    """
    Main execution entry point.
    Parses arguments, loads the model architecture and checkpoint, and executes conversion.
    """
    parser = argparse.ArgumentParser(
        description="Convert PyTorch TinyLidarNet weights to NumPy format (.npy).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model Configuration
    parser.add_argument(
        "--model",
        type=str,
        choices=["tinylidarnet", "tinylidarnet_small"],
        default="tinylidarnet",
        help="The model architecture to load."
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=1080,
        help="Input dimension size (LiDAR rays)."
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=2,
        help="Output dimension size (Control commands)."
    )

    # I/O Configuration
    parser.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="Path to the source PyTorch checkpoint file (.pth)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./weights/converted_weights.npy"),
        help="Destination path for the NumPy weight file (.npy)."
    )

    args = parser.parse_args()

    # --- 1. Initialize Model Architecture ---
    if args.model == "tinylidarnet":
        model = TinyLidarNet(input_dim=args.input_dim, output_dim=args.output_dim)
    else:
        model = TinyLidarNetSmall(input_dim=args.input_dim, output_dim=args.output_dim)

    # --- 2. Load Checkpoint ---
    if not args.ckpt.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {args.ckpt}")

    try:
        # Load weights onto CPU to avoid CUDA dependency during conversion
        state_dict = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"✅ Loaded checkpoint: {args.ckpt}")
    except RuntimeError as e:
        print(f"❌ Failed to load state dict: {e}")
        sys.exit(1)

    # --- 3. Convert and Save ---
    save_params_to_npy(model, args.output)


if __name__ == "__main__":
    main()