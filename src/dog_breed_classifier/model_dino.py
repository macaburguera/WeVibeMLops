import torch
from torch import nn
import timm  # Library for pretrained models

# Parameters
PARAMS = {
    "num_classes": 120,  # Number of output classes
    "dino_model": "vit_small_patch16_224_dino",  # DINO model name
}

class SimpleDinoClassifier(nn.Module):
    """A minimal model using the DINO backbone for classification."""

    def __init__(self, params: dict) -> None:
        """
        Initialize the model.

        Args:
            params (dict): Dictionary of parameters to configure the model.
        """
        super().__init__()

        # Load the pretrained DINO model
        self.dino_backbone = timm.create_model(params["dino_model"], pretrained=True)
        dino_features = self.dino_backbone.num_features  # Feature size of DINO backbone

        # Add a classifier head
        self.classifier = nn.Linear(dino_features, params["num_classes"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W).
        
        Returns:
            torch.Tensor: Class logits of shape (batch_size, num_classes).
        """
        # Pass input through DINO backbone
        features = self.dino_backbone(x)  # Shape: (batch_size, dino_features)
        
        # Pass features through the classifier
        logits = self.classifier(features)  # Shape: (batch_size, num_classes)
        return logits


if __name__ == "__main__":
    # Initialize the model
    model = SimpleDinoClassifier(params=PARAMS)
    
    # Print the model architecture and number of parameters
    print(f"Model architecture:\n{model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test the model with dummy input
    dummy_input = torch.randn(1, 3, 224, 224)  # DINO expects 3-channel input (RGB)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
