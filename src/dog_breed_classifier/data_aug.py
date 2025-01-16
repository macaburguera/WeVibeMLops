import os
import pandas as pd
import torch
import zipfile
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
from albumentations import (
    HorizontalFlip, Rotate, Normalize, Resize, CLAHE,
    RandomBrightnessContrast, GaussNoise, ShiftScaleRotate, OneOf, Compose
)
from albumentations.pytorch import ToTensorV2
import typer
import numpy as np

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
    """Define transformations with augmentations."""
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

def preprocess_images_in_batches(images_dir: str, labels: pd.DataFrame, transform, batch_size: int, output_dir: str):
    """Preprocess images in smaller batches to avoid memory overload."""
    os.makedirs(output_dir, exist_ok=True)

    def image_generator():
        for _, row in labels.iterrows():
            img_id = row['id']
            breed = row['breed']
            img_path = os.path.join(images_dir, f"{img_id}.jpg")
            try:
                image = Image.open(img_path).convert("RGB")
                yield np.array(image), breed
            except FileNotFoundError:
                print(f"Image {img_path} not found. Skipping...")
                continue

    batch_count = 0
    buffer_images, buffer_targets = [], []

    for image, target in image_generator():
        augmented_image = transform(image=image)['image']
        buffer_images.append(augmented_image)
        buffer_targets.append(target)

        if len(buffer_images) == batch_size:
            save_batch(buffer_images, buffer_targets, output_dir, batch_count)
            buffer_images, buffer_targets = [], []
            batch_count += 1

    if buffer_images:
        save_batch(buffer_images, buffer_targets, output_dir, batch_count)

def save_batch(images, targets, output_dir, batch_count):
    """Save a batch to a file."""
    batch_path = os.path.join(output_dir, f"batch_{batch_count}.pt")
    torch.save((torch.stack(images), torch.tensor(targets, dtype=torch.long)), batch_path)
    print(f"Saved batch {batch_count} to {batch_path}")

def combine_batches_and_save(batch_dir: str, output_images_path: str, output_targets_path: str):
    """Combine all batches without overloading memory."""
    all_images, all_targets = [], []

    for batch_file in sorted(os.listdir(batch_dir)):
        if batch_file.endswith(".pt"):
            batch_path = os.path.join(batch_dir, batch_file)
            images, targets = torch.load(batch_path)
            all_images.append(images)
            all_targets.append(targets)

    combined_images = torch.cat(all_images, dim=0)
    combined_targets = torch.cat(all_targets, dim=0)

    torch.save(combined_images, output_images_path)
    torch.save(combined_targets, output_targets_path)

    print(f"Saved combined images to {output_images_path}")
    print(f"Saved combined targets to {output_targets_path}")

def split_data(raw_data_dir: str, processed_data_dir: str, image_size=(224, 224), batch_size=100):
    """Split data into train/validation/test and process it."""
    labels_path = os.path.join(raw_data_dir, "labels.csv")
    images_dir = os.path.join(raw_data_dir, "images")

    # Load labels
    labels = pd.read_csv(labels_path)
    labels['breed'] = labels['breed'].astype('category').cat.codes

    # Split into train, validation, and test sets
    train_labels, temp_labels = train_test_split(labels, test_size=0.3, stratify=labels['breed'], random_state=42)
    val_labels, test_labels = train_test_split(temp_labels, test_size=0.5, stratify=temp_labels['breed'], random_state=42)

    transform = albumentations_transformations(image_size=image_size)

    preprocess_images_in_batches(images_dir, train_labels, transform, batch_size, os.path.join(processed_data_dir, "train"))
    preprocess_images_in_batches(images_dir, val_labels, transform, batch_size, os.path.join(processed_data_dir, "validation"))
    preprocess_images_in_batches(images_dir, test_labels, transform, batch_size, os.path.join(processed_data_dir, "test"))

    combine_batches_and_save(os.path.join(processed_data_dir, "train"),
                             os.path.join(processed_data_dir, "train_images.pt"),
                             os.path.join(processed_data_dir, "train_targets.pt"))
    combine_batches_and_save(os.path.join(processed_data_dir, "validation"),
                             os.path.join(processed_data_dir, "validation_images.pt"),
                             os.path.join(processed_data_dir, "validation_targets.pt"))
    combine_batches_and_save(os.path.join(processed_data_dir, "test"),
                             os.path.join(processed_data_dir, "test_images.pt"),
                             os.path.join(processed_data_dir, "test_targets.pt"))

def main(gdrive_link: str = "", raw_data_dir: str = "data/raw", processed_data_dir: str = "data/processed", batch_size: int = 100):
    """Complete process: download, split, process, and save datasets."""
    gdrive_link = False
    if gdrive_link:
        download_data(gdrive_link, raw_data_dir)

    split_data(raw_data_dir, processed_data_dir, batch_size=batch_size)

if __name__ == "__main__":
    typer.run(main)
