import torch
from torch import nn
import timm  # Library for pretrained models

# Parameters
PARAMS = {
    "num_classes": 120,  # Number of output classes
    "resnet_model": "resnet50",  # ResNet model name
}

class SimpleResNetClassifier(nn.Module):
    """A minimal model using the ResNet backbone for classification."""

    def __init__(self, params: dict) -> None:
        """
        Initialize the model.

        Args:
            params (dict): Dictionary of parameters to configure the model.
        """
        super().__init__()

        # Load the pretrained ResNet model
        self.resnet_backbone = timm.create_model(params["resnet_model"], pretrained=True)
        resnet_features = self.resnet_backbone.num_features  # Feature size of ResNet backbone

        # Replace the classifier head
        self.resnet_backbone.fc = nn.Linear(resnet_features, params["num_classes"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W).
        
        Returns:
            torch.Tensor: Class logits of shape (batch_size, num_classes).
        """
        # Pass input through ResNet backbone
        logits = self.resnet_backbone(x)  # Shape: (batch_size, num_classes)
        return logits


if __name__ == "__main__":
    # Initialize the model
    model = SimpleResNetClassifier(params=PARAMS)
    
    # Print the model architecture and number of parameters
    print(f"Model architecture:\n{model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test the model with dummy input
    dummy_input = torch.randn(1, 3, 224, 224)  # ResNet expects 3-channel input (RGB)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
