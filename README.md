# Dog Breed Classifier (Group 45)

## Overall Goal
The goal of this project is to build a **dog breed classifier** capable of distinguishing between 120 different breeds as accurately as possible. The focus will be on achieving fine-grained classification by leveraging state-of-the-art deep learning techniques and pretrained models.

## Framework
The project will be based on PyTorch and will leverage its ecosystem for efficient model development and experimentation:
- **Albumentations**: To perform advanced image augmentations and effectively extend the variability of the dataset, improving the model's generalization capabilities.
- **Optuna**: For hyperparameter optimization to fine-tune model architecture, learning rates, and training configurations.
- **TIMM**: To access a variety of pretrained models and well-known architectures, enabling faster development and more accurate predictions.

## Data
The dataset used will be the Stanford Dogs Dataset, which contains:
- **120** dog breeds.
- **20,580** images, with approximately 150 images per breed.
The images are of varying resolutions and backgrounds, representing real-world diversity.
- **Challenge**: The dataset requires distinguishing subtle differences between breeds, often with high inter-class similarity and low intra-class variability.
**Preprocessing** will include resizing images to a uniform resolution, applying Albumentations-based augmentations (e.g., cropping, flipping, color jittering), and splitting the data into training, validation, and test sets.

## Models
The project will primarily explore pretrained models to achieve a balance between accurate feature extraction and effective classification:
- **DINO**: Transformer-based model used for its ability to capture global and fine-grained features from the images.
- **ViT**: Vision Transformer model. Potentially used as an alternative or in combination with DINO.
- **ResNet**: Widely used model, more focused on robust low and mid-level feature extraction.
- **EfficientNet**:Tested as a lightweight and efficient model, may be an alternative to ResNet.

The final architecture will combine an optimal selection of these models using a feature concatenation approach, with Optuna guiding the optimization of hyperparameters for the combined pipeline.
