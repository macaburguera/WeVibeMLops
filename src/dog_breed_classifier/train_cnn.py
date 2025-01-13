import os
import torch
import typer
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from model_cnn import SimpleConvNet
import optuna
import json


# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training parameters
PARAMS = {
    "processed_data_dir": "data/processed",  # Directory containing processed data
    "model_save_path": "models/cnn_model.pth",  # Path to save the model
    "figure_save_path": "reports/figures/cnn_training_statistics.png",  # Path to save training stats
    "num_classes": 120,  # Number of output classes
    "patience": 5,  # Early stopping patience
}


def load_dataset(subset: str, data_dir: str) -> TensorDataset:
    """
    Load the specified dataset (train/validation/test) from .pt files.

    Args:
        subset (str): The subset to load ('train', 'validation', or 'test').
        data_dir (str): Directory containing the .pt files.

    Returns:
        TensorDataset: The loaded dataset.
    """
    images_path = os.path.join(data_dir, subset, f"{subset}_images.pt")
    targets_path = os.path.join(data_dir, subset, f"{subset}_targets.pt")

    images = torch.load(images_path)
    targets = torch.load(targets_path)

    return TensorDataset(images, targets)


def train_model(
    trial,
    train_loader,
    val_loader,
    num_classes: int,
    patience: int,
) -> float:
    """
    Train the SimpleConvNet model and return the best validation loss.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        num_classes (int): Number of output classes.
        patience (int): Early stopping patience.

    Returns:
        float: Best validation loss achieved.
    """
    # Define hyperparameters to tune
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    momentum = trial.suggest_uniform("momentum", 0.8, 0.99)
    epochs = trial.suggest_int("epochs", 10, 25)

    model = SimpleConvNet(num_classes=num_classes).to(DEVICE)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=1e-4
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    # Track early stopping
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
                outputs = model(imgs)
                loss = loss_fn(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return best_val_loss


def objective(trial):
    """
    Objective function for Optuna.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.

    Returns:
        float: Best validation loss achieved in the trial.
    """
    # Load datasets
    train_dataset = load_dataset("train", PARAMS["processed_data_dir"])
    val_dataset = load_dataset("validation", PARAMS["processed_data_dir"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
        shuffle=False,
    )

    return train_model(
        trial,
        train_loader,
        val_loader,
        num_classes=PARAMS["num_classes"],
        patience=PARAMS["patience"],
    )


def train_and_tune():
    """
    Run Optuna optimization and save the best model and training statistics.
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    print("Best hyperparameters:", study.best_params)


    # Save the best trial's parameters to a JSON file
    best_params = study.best_trial.params
    best_results = {
        "value": study.best_trial.value,
        "params": best_params,
    }
    os.makedirs("config", exist_ok=True)
    with open("config/cnn_best_params.json", "w") as f:
        json.dump(best_results, f, indent=4)

    print("Best parameters and results saved to 'best_params.json'")


if __name__ == "__main__":
    train_and_tune()
