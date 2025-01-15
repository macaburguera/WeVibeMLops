# Dog Breed Classifier (Group 45)

## Installation

### Setting Up the Environment

1. **Create a Conda Environment**  
   ```bash
   conda create --name dogs python=3.11
   conda activate dogs
   ```
   This project has been tested with PyTorch 2.2.0 and CUDA 11.8.

2. **Install the Required Packages**

   First, install PyTorch manually:  
   ```bash
   pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
   ```
   Then go on with the rest
   ```bash
   pip install -r requirements.txt
   ```

---

### Setting Up the Data

After setting up the environment, run `data.py` or `data_aug.py` (for data augmentation) to process the data from the source:

```bash
python src/dog_breed_classifier/data_aug.py
```

---

### Running Training Sessions

#### Train All Models at Once
In Linux, Git Bash, or WSL, you can train all models by running the following commands:
```bash
chmod +x train_all.sh
./train_all.sh
```

#### Train Each Model Individually
To train a specific model, run:
```bash
python src/dog_breed_classifier/train_xxx.py
```
Replace `xxx` with one of the following:  
- `cnn`  
- `dino`  
- `resnet`

---

### Updating `requirements.txt`

To update the `requirements.txt` file based on your current environment:
1. Install `pipreqs`:
   ```bash
   pip install pipreqs
   ```
2. Generate an updated `requirements.txt`:
   ```bash
   pipreqs . --force
   ```


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
# dog_breed_classifier

Classifying Dog Breeds from pictures, Group 45

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
