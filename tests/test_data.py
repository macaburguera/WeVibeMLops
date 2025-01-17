import os
import torch
import pytest
from data import albumentations_transformations, preprocess_images_in_batches, split_data
from torchvision import transforms
from PIL import Image
import pandas as pd

@pytest.fixture
def mock_labels(tmp_path):
    """Creates a mock labels.csv file."""
    data = {
        "id": ["img1", "img2", "img3"],
        "breed": ["breed1", "breed2", "breed1"]
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "labels.csv"
    df.to_csv(csv_path, index=False)
    return csv_path

@pytest.fixture
def mock_images(tmp_path):
    """Creates mock image files."""
    images_dir = tmp_path / "images"
    os.makedirs(images_dir)
    for img_id in ["img1", "img2", "img3"]:
        img = Image.new("RGB", (224, 224), color="red")
        img.save(images_dir / f"{img_id}.jpg")
    return images_dir

def test_albumentations_transformations():
    """Test that albumentations transformations return a valid transform pipeline."""
    transform = albumentations_transformations()
    assert callable(transform), "Transformations should be callable."

def test_preprocess_images_in_batches(mock_labels, mock_images):
    """Test image preprocessing and batching."""
    transform = albumentations_transformations()
    labels = pd.read_csv(mock_labels)
    batches = list(preprocess_images_in_batches(str(mock_images), labels, transform, batch_size=2))

    assert len(batches) == 2, "Should create two batches with batch_size=2."
    assert batches[0][0].shape == (2, 3, 224, 224), "Each batch should have correct dimensions."
    assert batches[0][1].shape == (2,), "Batch targets should match batch size."

def test_split_data(mock_labels, mock_images, tmp_path):
    """Test splitting data into train, validation, and test sets."""
    raw_data_dir = tmp_path / "raw"
    os.makedirs(raw_data_dir)
    os.rename(mock_labels, raw_data_dir / "labels.csv")
    os.rename(mock_images, raw_data_dir / "images")

    processed_data_dir = tmp_path / "processed"
    split_data(str(raw_data_dir), str(processed_data_dir))

    for subset in ["train", "validation", "test"]:
        subset_dir = processed_data_dir / subset
        assert subset_dir.exists(), f"{subset} directory should exist."
        assert (subset_dir / f"{subset}_images.pt").exists(), f"{subset}_images.pt should be created."
        assert (subset_dir / f"{subset}_targets.pt").exists(), f"{subset}_targets.pt should be created."
