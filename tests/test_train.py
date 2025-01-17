import os
import pytest
from unittest.mock import patch, MagicMock
import torch
from omegaconf import OmegaConf
from train import train
from model import SimpleResNetClassifier
from hydra.utils import to_absolute_path
from torch.utils.data import DataLoader, TensorDataset

# Mock the device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mock configuration for testing
@pytest.fixture
def mock_config():
    return OmegaConf.create({
        "hyperparameters": {
            "lr": 0.001,
            "batch_size": 16,
            "epochs": 2,
            "model_save_path": "tests/mock_models/resnet_model.pth",
            "figure_save_path": "tests/mock_figures/training_statistics.png",
            "model_params": {
                "num_classes": 10,
                "resnet_model": "resnet18"
            }
        }
    })

# Mock datasets
@pytest.fixture
def mock_datasets():
    torch.manual_seed(42)
    num_samples = 100
    num_features = (3, 224, 224)
    num_classes = 10

    train_data = torch.randn(num_samples, *num_features)
    train_targets = torch.randint(0, num_classes, (num_samples,))

    val_data = torch.randn(num_samples, *num_features)
    val_targets = torch.randint(0, num_classes, (num_samples,))

    train_dataset = TensorDataset(train_data, train_targets)
    val_dataset = TensorDataset(val_data, val_targets)

    return train_dataset, val_dataset

# Mock data loader
@pytest.fixture
def mock_data_loaders(mock_datasets, mock_config):
    train_dataset, val_dataset = mock_datasets
    batch_size = mock_config.hyperparameters.batch_size

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# Test model initialization
def test_model_initialization(mock_config):
    model = SimpleResNetClassifier(mock_config.hyperparameters.model_params)
    assert isinstance(model, SimpleResNetClassifier)
    assert model.resnet_backbone.fc.out_features == mock_config.hyperparameters.model_params.num_classes

# Test dataset loading
def test_dataset_loading(mock_data_loaders):
    train_loader, val_loader = mock_data_loaders

    assert len(train_loader) > 0
    assert len(val_loader) > 0

    for images, targets in train_loader:
        assert images.shape[1:] == (3, 224, 224)
        assert len(targets) == mock_config().hyperparameters.batch_size
        break

# Mock wandb initialization
@patch("wandb.init")
@patch("wandb.log")
@patch("wandb.finish")
def test_training_loop(mock_finish, mock_log, mock_init, mock_config, mock_data_loaders):
    train_loader, val_loader = mock_data_loaders

    # Ensure the directories exist
    os.makedirs(os.path.dirname(mock_config.hyperparameters.model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(mock_config.hyperparameters.figure_save_path), exist_ok=True)

    # Call the training function
    train(mock_config, use_wandb=True)

    # Verify wandb was called
    mock_init.assert_called_once()
    mock_log.assert_called()
    mock_finish.assert_called_once()

    # Verify the model file was saved
    assert os.path.exists(mock_config.hyperparameters.model_save_path)

    # Verify figure paths
    loss_figure_path = mock_config.hyperparameters.figure_save_path.replace("statistics", "loss")
    accuracy_figure_path = mock_config.hyperparameters.figure_save_path.replace("statistics", "accuracy")

    assert os.path.exists(loss_figure_path)
    assert os.path.exists(accuracy_figure_path)

# Clean up mock outputs
@pytest.fixture(scope="function", autouse=True)
def cleanup():
    yield
    mock_paths = [
        "tests/mock_models",
        "tests/mock_figures",
    ]
    for path in mock_paths:
        if os.path.exists(path):
            for root, dirs, files in os.walk(path, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(path)
