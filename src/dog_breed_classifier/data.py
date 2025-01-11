import os
import pandas as pd
import shutil
import torch
import zipfile
from torchvision import transforms
from albumentations import (
    HorizontalFlip, Rotate, Normalize, Resize, CLAHE,
    RandomBrightnessContrast, GaussNoise, ShiftScaleRotate, OneOf, Compose
)
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import typer


def download_data(gdrive_link: str, raw_data_dir: str):
    """Download the dataset from Google Drive and save it to the specified directory."""
    if not os.path.exists(raw_data_dir):
        os.makedirs(raw_data_dir)
    
    os.system(f"gdown --folder {gdrive_link} -O {raw_data_dir}")
    print(f"Downloaded data to {raw_data_dir}")

    # Check if images.zip exists and decompress it
    zip_path = os.path.join(raw_data_dir, "images.zip")
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(raw_data_dir, "images"))
        print(f"Decompressed images.zip to {os.path.join(raw_data_dir, 'images')}")
        os.remove(zip_path)  # Optionally remove the zip file after extraction


def albumentations_transformations(image_size=(224, 224)):
    """Define varied Albumentations transformations for more diversity."""
    return Compose([
        Resize(height=image_size[0], width=image_size[1]),
        OneOf([
            HorizontalFlip(p=1.0),
            Rotate(limit=30, p=1.0),
            ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=1.0),
        ], p=0.8),
        OneOf([
            CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            RandomBrightnessContrast(p=1.0),
            GaussNoise(var_limit=(10, 50), p=1.0),
        ], p=0.8),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def preprocess_images_with_augmentation(images_dir: str, labels: pd.DataFrame, transform=None, num_augmentations=0):
    """Preprocess images by applying varied transformations, resizing, and generating augmented versions."""
    image_tensors = []
    targets = []
    for _, row in labels.iterrows():
        img_id = row['id']
        breed = row['breed']
        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        
        try:
            image = np.array(Image.open(img_path).convert("RGB"))
        except FileNotFoundError:
            print(f"Image {img_path} not found. Skipping...")
            continue
        
        if transform:
            # Original image
            image_tensors.append(transform(image=image)['image'])
            targets.append(breed)
            
            # Generate augmented versions
            for _ in range(num_augmentations):
                augmented_image = transform(image=image)['image']
                image_tensors.append(augmented_image)
                targets.append(breed)
        else:
            # Default normalization for validation/test without augmentation
            image = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])(Image.fromarray(image))
            image_tensors.append(image)
            targets.append(breed)
    
    return torch.stack(image_tensors), torch.tensor(targets)


def split_data(raw_data_dir: str, processed_data_dir: str, image_size=(224, 224)):
    """Split the data into train, validation, and test sets and save them."""
    # Paths
    labels_path = os.path.join(raw_data_dir, "labels.csv")
    images_dir = os.path.join(raw_data_dir, "images")
    
    # Load labels
    labels = pd.read_csv(labels_path)
    
    # Encode breeds as integers
    labels['breed'] = labels['breed'].astype('category').cat.codes
    
    # Split into train, test, and validation sets
    train_labels, temp_labels = train_test_split(labels, test_size=0.3, stratify=labels['breed'], random_state=42)
    val_labels, test_labels = train_test_split(temp_labels, test_size=0.5, stratify=temp_labels['breed'], random_state=42)
    
    # Define transformations
    train_transform = albumentations_transformations(image_size=image_size)
    val_transform = albumentations_transformations(image_size=image_size)  # Resize and normalize
    
    # Preprocess images
    train_images, train_targets = preprocess_images_with_augmentation(images_dir, train_labels, transform=train_transform)
    val_images, val_targets = preprocess_images_with_augmentation(images_dir, val_labels, transform=val_transform)
    test_images, test_targets = preprocess_images_with_augmentation(images_dir, test_labels, transform=train_transform, num_augmentations=2)
    
    # Save data to .pt files
    subsets = {
        "train": (train_images, train_targets),
        "validation": (val_images, val_targets),
        "test": (test_images, test_targets)
    }
    for subset_name, (images, targets) in subsets.items():
        subset_dir = os.path.join(processed_data_dir, subset_name)
        os.makedirs(subset_dir, exist_ok=True)
        torch.save(images, os.path.join(subset_dir, f"{subset_name}_images.pt"))
        torch.save(targets, os.path.join(subset_dir, f"{subset_name}_targets.pt"))
        print(f"Saved {subset_name} set to {subset_dir}")


def main(
    gdrive_link: str = "https://drive.google.com/drive/folders/1kCEyO3UFiZuUH93SIJLK0Zt8mh7mBig0?usp=drive_link",  # Default Google Drive link
    raw_data_dir: str = "data/raw",                   # Default raw data folder
    processed_data_dir: str = "data/processed",       # Default processed data folder
    image_size: str = "224,224"                       # Default image size as a string
):
    """
    Download data from Google Drive, split it, and save processed subsets.
    """
    # Parse image_size string into a tuple of integers
    image_size = tuple(map(int, image_size.split(',')))
    
    # Step 1: Download data
    download_data(gdrive_link, raw_data_dir)

    # Step 2: Split, process, and save data
    split_data(raw_data_dir, processed_data_dir, image_size=image_size)


if __name__ == "__main__":
    typer.run(main)
