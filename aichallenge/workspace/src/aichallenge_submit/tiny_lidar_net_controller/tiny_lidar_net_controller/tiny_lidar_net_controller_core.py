import logging
import numpy as np
from typing import Tuple

from .model.tinylidarnet import TinyLidarNetNp, TinyLidarNetSmallNp


class TinyLidarNetCore:
    """Core logic for the TinyLidarNet autonomous driving controller.

    This class manages the neural network model lifecycle, including initialization,
    weight loading, input preprocessing, and inference execution.

    Attributes:
        input_dim (int): Dimension of the input vector for the model.
        output_dim (int): Dimension of the output vector (acceleration, steering).
        architecture (str): Model architecture type ('large' or 'small').
        acceleration (float): Fixed acceleration value used in 'fixed' control mode.
        control_mode (str): Control strategy ('ai' or 'fixed').
        model (object): The instantiated neural network model.
    """

    def __init__(
        self,
        input_dim: int = 1080,
        output_dim: int = 2,
        architecture: str = 'large',
        ckpt_path: str = '',
        acceleration: float = 0.1,
        control_mode: str = 'ai'
    ):
        """Initializes the TinyLidarNetCore with specified parameters.

        Args:
            input_dim (int, optional): The number of LiDAR points expected by the model.
                Defaults to 1080.
            output_dim (int, optional): The number of output control values.
                Defaults to 2.
            architecture (str, optional): The model architecture to use ('large' or 'small').
                Defaults to 'large'.
            ckpt_path (str, optional): Path to the numpy weight file (.npy or .npz).
                Defaults to ''.
            acceleration (float, optional): The constant acceleration value to apply
                when control_mode is set to 'fixed'. Defaults to 0.1.
            control_mode (str, optional): The control mode to determine output behavior.
                'ai' uses model output for both acceleration and steering.
                'fixed' uses the fixed acceleration value and model output for steering.
                Defaults to 'ai'.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.architecture = architecture
        self.acceleration = acceleration
        self.control_mode = control_mode.lower()
        self.logger = logging.getLogger(__name__)

        if self.architecture == 'small':
            self.model = TinyLidarNetSmallNp(input_dim=self.input_dim, output_dim=self.output_dim)
        else:
            self.model = TinyLidarNetNp(input_dim=self.input_dim, output_dim=self.output_dim)

        if ckpt_path:
            self._load_weights(ckpt_path)
        else:
            self.logger.warning("No weight file provided. Using randomly initialized weights.")

    def process(self, ranges: np.ndarray) -> Tuple[float, float]:
        """Runs the inference pipeline on LiDAR data to generate control commands.

        Args:
            ranges (np.ndarray): A 1D numpy array containing LiDAR range data.

        Returns:
            Tuple[float, float]: A tuple containing (acceleration, steering_angle).
                Values are clipped between -1.0 and 1.0.
        """
        # 1. Preprocess
        processed_ranges = self._preprocess_ranges(ranges)

        # Prepare input tensor: (1, 1, input_dim)
        x = np.expand_dims(np.expand_dims(processed_ranges, axis=0), axis=1)

        # 2. Inference
        outputs = self.model(x)[0]

        # 3. Post-process
        if self.control_mode == "ai":
            accel = float(np.clip(outputs[0], -1.0, 1.0))
        else:
            accel = self.acceleration

        steer = float(np.clip(outputs[1], -1.0, 1.0))

        return accel, steer

    def _load_weights(self, path: str) -> None:
        """Loads model weights from a file into the model parameters.

        Args:
            path (str): Path to the .npy or .npz weight file.

        Raises:
            ValueError: If the weight file format is unsupported.
            IOError: If the file cannot be read.
        """
        try:
            weights = np.load(path, allow_pickle=True)

            if isinstance(weights, np.lib.npyio.NpzFile):
                weight_dict = dict(weights.items())
            elif isinstance(weights, np.ndarray) and weights.dtype == object:
                weight_dict = weights.item()
            elif isinstance(weights, dict):
                weight_dict = weights
            else:
                raise ValueError(f"Unsupported weight format type: {type(weights)}")

            loaded_count = 0
            for key, value in weight_dict.items():
                # Normalize key to match model parameter naming convention
                key_norm = key.replace('.', '_')

                if key_norm in self.model.params:
                    self.model.params[key_norm] = value
                    loaded_count += 1

            self.logger.info(f"Successfully loaded {loaded_count} parameters from {path}")

        except Exception as e:
            self.logger.error(f"Failed to load weights from {path}: {e}")
            raise e

    def _preprocess_ranges(self, ranges: np.ndarray) -> np.ndarray:
        """Resizes and normalizes LiDAR ranges to match model input dimensions.

        Args:
            ranges (np.ndarray): Source LiDAR range data.

        Returns:
            np.ndarray: Processed data array of shape (self.input_dim,).
                Data is normalized by a factor of 30.0.
        """
        current_len = len(ranges)

        if current_len > self.input_dim:
            idx = np.linspace(0, current_len - 1, self.input_dim, dtype=int)
            ranges = ranges[idx]
        elif current_len < self.input_dim:
            ranges = np.pad(ranges, (0, self.input_dim - current_len), 'constant')

        return ranges / 30.0