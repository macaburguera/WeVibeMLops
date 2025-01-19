import os
import pytest
from unittest.mock import patch
import torch
import pandas as pd
from omegaconf import OmegaConf
from src.dog_breed_classifier.train import train
from src.dog_breed_classifier.model import SimpleResNetClassifier
from src.dog_breed_classifier.data import albumentations_transformations
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image

# Mock the device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(scope="session")
def mock_config():
    return {
        "hyperparameters": {
            "lr": 0.001,
            "batch_size": 16,
            "epochs": 2,
            "model_save_path": "tests/mock_models/resnet_model.pt",
            "figure_save_path": "tests/mock_figures",
            "processed_data_dir": "tests/mock_data",
            "model_params": {"num_classes": 4, "resnet_model": "resnet50"},
        }
    }

@pytest.fixture(scope="session", autouse=True)
def create_mock_files(mock_config):
    """
    Create mock .pt files for training and validation datasets.
    """
    base_dir = mock_config["hyperparameters"]["processed_data_dir"]
    os.makedirs(os.path.join(base_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "validation"), exist_ok=True)

    # Generate train data
    train_images = torch.randn(300, 3, 224, 224)  # Example: 300 images
    train_targets = torch.randint(0, mock_config["hyperparameters"]["model_params"]["num_classes"], (300,))
    torch.save(train_images, os.path.join(base_dir, "train", "train_images.pt"))
    torch.save(train_targets, os.path.join(base_dir, "train", "train_targets.pt"))

    # Generate validation data
    val_images = torch.randn(100, 3, 224, 224)  # Example: 100 images
    val_targets = torch.randint(0, mock_config["hyperparameters"]["model_params"]["num_classes"], (100,))
    torch.save(val_images, os.path.join(base_dir, "validation", "validation_images.pt"))
    torch.save(val_targets, os.path.join(base_dir, "validation", "validation_targets.pt"))

    yield  # Let tests run

    # Cleanup mock files after all tests
    if os.path.exists(base_dir):
        for root, _, files in os.walk(base_dir, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            os.rmdir(root)

    # Cleanup other mock directories
    for dir_path in ["tests/mock_models", "tests/mock_figures"]:
        if os.path.exists(dir_path):
            for root, _, files in os.walk(dir_path, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                os.rmdir(root)

# Mock datasets
@pytest.fixture
def mock_datasets(mock_config):
    torch.manual_seed(42)
    num_samples = 300
    num_features = (3, 224, 224)
    num_classes = mock_config["hyperparameters"]["model_params"]["num_classes"]

    train_data = torch.randn(num_samples, *num_features)
    train_targets = torch.randint(0, num_classes, (num_samples,), dtype=torch.long)

    val_data = torch.randn(num_samples, *num_features)
    val_targets = torch.randint(0, num_classes, (num_samples,), dtype=torch.long)

    train_dataset = TensorDataset(train_data, train_targets)
    val_dataset = TensorDataset(val_data, val_targets)

    return train_dataset, val_dataset

# Mock data loader
@pytest.fixture
def mock_data_loaders(mock_datasets, mock_config):
    train_dataset, val_dataset = mock_datasets
    batch_size = mock_config["hyperparameters"]["batch_size"]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# Test model initialization
def test_model_initialization(mock_config):
    model = SimpleResNetClassifier(mock_config["hyperparameters"]["model_params"])
    assert isinstance(model, SimpleResNetClassifier)
    assert model.resnet_backbone.fc.out_features == mock_config["hyperparameters"]["model_params"]["num_classes"]

# Test dataset loading
def test_dataset_loading(mock_config):
    from src.dog_breed_classifier.data import preprocess_images_in_batches

    batch_size = mock_config["hyperparameters"]["batch_size"]
    image_size = (224, 224)
    num_classes = mock_config["hyperparameters"]["model_params"]["num_classes"]

    labels_data = {
        "id": [f"img{i}" for i in range(1, 11)],
        "breed": [i % num_classes for i in range(1, 11)]
    }
    labels_df = pd.DataFrame(labels_data)

    images_dir = "tests/mock_images"
    os.makedirs(images_dir, exist_ok=True)
    for img_id in labels_data["id"]:
        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        img = Image.new("RGB", (image_size[0], image_size[1]), color="red")
        img.save(img_path)

    transform = albumentations_transformations(image_size=image_size)

    batches = list(preprocess_images_in_batches(images_dir, labels_df, transform, batch_size=batch_size))
    total_images = sum(batch[0].shape[0] for batch in batches)

    assert total_images == len(labels_df), "Mismatch in the number of images loaded"
    assert all(batch[1].max() < num_classes for batch in batches), "Labels exceed expected num_classes"
    assert all(batch[1].min() >= 0 for batch in batches), "Labels contain negative values"

    for file in os.listdir(images_dir):
        os.remove(os.path.join(images_dir, file))
    os.rmdir(images_dir)


@patch("hydra.compose")
@patch("hydra.initialize")
def test_training_loop_local(mock_initialize, mock_compose, mock_config):
    """
    Test the training loop in local mode (without WandB).
    """
    # Mock Hydra behavior to use mock_config
    mock_initialize.return_value = None
    mock_compose.return_value = OmegaConf.create(mock_config)

    cfg = OmegaConf.create(mock_config)

    # Run the training function in local mode
    train(cfg, use_wandb=False)

    # Verify the model save path exists
    assert os.path.exists(mock_config["hyperparameters"]["model_save_path"]), "Model save path not found"

    # Verify the figure directory and files exist
    figure_dir = mock_config["hyperparameters"]["figure_save_path"]
    assert os.path.isdir(figure_dir), f"Figure directory {figure_dir} does not exist"
    assert os.path.exists(os.path.join(figure_dir, "training_loss.png")), "Loss plot not found"
    assert os.path.exists(os.path.join(figure_dir, "training_accuracy.png")), "Accuracy plot not found"
