import pytest
import torch
from src.dog_breed_classifier.model import SimpleResNetClassifier, PARAMS

# Test if the model initializes correctly
def test_model_initialization():
    """
    Test that the SimpleResNetClassifier initializes correctly with valid parameters.
    """
    model = SimpleResNetClassifier(params=PARAMS)
    assert model is not None, "Model should initialize without errors."
    assert isinstance(model, SimpleResNetClassifier), "Model should be an instance of SimpleResNetClassifier."

# Test the model forward pass with dummy input
def test_model_forward_pass():
    """
    Test that the model's forward pass works with dummy input.
    """
    model = SimpleResNetClassifier(params=PARAMS)
    dummy_input = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3-channel input, 224x224 image
    output = model(dummy_input)

    # Check output shape
    assert output.shape == (1, PARAMS["num_classes"]), \
        f"Expected output shape (1, {PARAMS['num_classes']}), but got {output.shape}."

# Test that the model fails with invalid input
def test_model_invalid_input():
    """Test that the model raises an error with invalid input."""
    model = SimpleResNetClassifier(params=PARAMS)

    # Expect the model to raise any RuntimeError for invalid input
    with pytest.raises(RuntimeError):
        invalid_input = torch.randn(1, 1, 224, 224)  # Invalid channel dimension
        model(invalid_input)


# Test that the model's parameters are trainable
def test_model_parameters_trainable():
    """
    Test that the model's parameters are set to be trainable.
    """
    model = SimpleResNetClassifier(params=PARAMS)
    for param in model.parameters():
        assert param.requires_grad, "All parameters should be trainable by default."

# Test that the model uses the correct ResNet backbone
def test_model_resnet_backbone():
    """
    Test that the model uses the specified ResNet backbone.
    """
    model = SimpleResNetClassifier(params=PARAMS)
    assert PARAMS["resnet_model"] in model.resnet_backbone.default_cfg["architecture"], \
        f"Expected ResNet backbone {PARAMS['resnet_model']} but got {model.resnet_backbone.default_cfg['architecture']}."
