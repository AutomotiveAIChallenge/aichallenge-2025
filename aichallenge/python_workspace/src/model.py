import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyLidarNet(nn.Module):
    """
    Standard CNN architecture for 1D LiDAR data processing.

    Architecture:
        - 5 Convolutional Layers (1D)
        - 4 Fully Connected Layers
        - Activation: ReLU (hidden layers), Tanh (output layer)

    Attributes:
        conv1-5 (nn.Conv1d): 1D Convolutional layers.
        fc1-4 (nn.Linear): Fully connected layers.
    """

    def __init__(self, input_dim: int = 1080, output_dim: int = 2):
        """
        Initializes the TinyLidarNet model.

        Args:
            input_dim: Number of points in the LiDAR scan (input channels=1, length=input_dim).
            output_dim: Dimension of the output vector (e.g., 2 for [steer, accel]).
        """
        super().__init__()

        # --- Convolutional Layers ---
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=24, kernel_size=10, stride=4)
        self.conv2 = nn.Conv1d(in_channels=24, out_channels=36, kernel_size=8, stride=4)
        self.conv3 = nn.Conv1d(in_channels=36, out_channels=48, kernel_size=4, stride=2)
        self.conv4 = nn.Conv1d(in_channels=48, out_channels=64, kernel_size=3)
        self.conv5 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)

        # --- Fully Connected Layers ---
        # Dynamically calculate the flattened dimension size based on input_dim
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)
            x = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(dummy_input)))))
            flatten_dim = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flatten_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, output_dim)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Initializes model weights using Kaiming He initialization.
        This helps in stabilizing the training process for ReLU networks.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x: Input tensor of shape (Batch, 1, input_dim).

        Returns:
            Output tensor of shape (Batch, output_dim) with values in range [-1, 1].
        """
        # Feature Extraction (Conv + ReLU)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        # Flatten: (Batch, Channels, Length) -> (Batch, Features)
        x = torch.flatten(x, start_dim=1)

        # Regression Head (FC + ReLU)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # Output Layer (Tanh for normalized output range [-1, 1])
        x = torch.tanh(self.fc4(x))
        
        return x


class TinyLidarNetSmall(nn.Module):
    """
    Lightweight CNN architecture for 1D LiDAR data processing.
    Suitable for resource-constrained environments (e.g., embedded systems).

    Architecture:
        - 3 Convolutional Layers (1D)
        - 3 Fully Connected Layers
        - Activation: ReLU (hidden layers), Tanh (output layer)

    Attributes:
        conv1-3 (nn.Conv1d): 1D Convolutional layers.
        fc1-3 (nn.Linear): Fully connected layers.
    """

    def __init__(self, input_dim: int = 1080, output_dim: int = 2):
        """
        Initializes the TinyLidarNetSmall model.

        Args:
            input_dim: Number of points in the LiDAR scan.
            output_dim: Dimension of the output vector.
        """
        super().__init__()

        # --- Convolutional Layers ---
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=24, kernel_size=10, stride=4)
        self.conv2 = nn.Conv1d(in_channels=24, out_channels=36, kernel_size=8, stride=4)
        self.conv3 = nn.Conv1d(in_channels=36, out_channels=48, kernel_size=4, stride=2)

        # --- Fully Connected Layers ---
        # Dynamically calculate the flattened dimension size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)
            x = self.conv3(self.conv2(self.conv1(dummy_input)))
            flatten_dim = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flatten_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, output_dim)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initializes weights using Kaiming He initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x: Input tensor of shape (Batch, 1, input_dim).

        Returns:
            Output tensor of shape (Batch, output_dim) with values in range [-1, 1].
        """
        # Feature Extraction (Conv + ReLU)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = torch.flatten(x, start_dim=1)
        
        # Regression Head (FC + ReLU)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Output Layer
        x = torch.tanh(self.fc3(x))
        
        return x
