import torch
from torch import nn


class SimpleConvNet(nn.Module):
    """A simple convolutional neural network with BatchNorm and LeakyReLU."""

    def __init__(self, num_classes: int = 120) -> None:
        """
        Initialize the SimpleConvNet model.

        Args:
            num_classes (int): Number of output classes.
        """
        super().__init__()
        # Convolutional layers with BatchNorm
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Dropout layers for regularization
        self.dropout1 = nn.Dropout(0.5)  # After conv1
        self.dropout2 = nn.Dropout(0.5)  # After conv2
        self.dropout3 = nn.Dropout(0.5)  # After conv3
        self.dropout_fc = nn.Dropout(0.5)  # For fully connected layers

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)

        # Max Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # LeakyReLU activation
        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SimpleConvNet.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.pool(x)  # Downsample
        x = self.dropout1(x)

        x = self.activation(self.bn2(self.conv2(x)))
        x = self.pool(x)  # Downsample
        x = self.dropout2(x)

        x = self.activation(self.bn3(self.conv3(x)))
        x = self.pool(x)  # Downsample
        x = self.dropout3(x)

        x = torch.flatten(x, 1)  # Flatten for fully connected layer
        x = self.activation(self.fc1(x))
        x = self.dropout_fc(x)

        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # Initialize the model
    model = SimpleConvNet(num_classes=120)

    # Print model architecture and number of parameters
    print(f"Model architecture:\n{model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Test the model with dummy input
    dummy_input = torch.randn(1, 3, 224, 224)  # 3-channel RGB input with 224x224 resolution
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
