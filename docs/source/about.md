# About this project

## Overall Goal
The goal of this project is to build a **dog breed classifier** capable of distinguishing between 120 different breeds as accurately as possible. The focus will be on achieving fine-grained classification by leveraging state-of-the-art deep learning techniques and pretrained models.

## Framework
The project will be based on PyTorch and will leverage its ecosystem for efficient model development and experimentation:

- **Albumentations**: To perform advanced image augmentations and effectively extend the variability of the dataset, improving the model's generalization capabilities.
- **TIMM**: To access a variety of pretrained models and well-known architectures, enabling faster development and more accurate predictions.

## Data
The dataset used will be the Stanford Dogs Dataset, which contains:

- **120** dog breeds.
- **20,580** images, with approximately 150 images per breed. The images are of varying resolutions and backgrounds, representing real-world diversity.

**Challenge**: The dataset requires distinguishing subtle differences between breeds, often with high inter-class similarity and low intra-class variability.

**Preprocessing** will include resizing images to a uniform resolution, applying Albumentations-based augmentations (e.g., cropping, flipping, color jittering), and splitting the data into training, validation, and test sets.

## Models
The project will primarily explore pretrained Resnet variants, which will be fine-tuned to perform an accurate classification of the different existing dog breeds.
