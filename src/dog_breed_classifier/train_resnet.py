import os
import json
import torch
import typer
import optuna
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from model_resnet import SimpleResNetClassifier

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Training parameters
PARAMS = {
    "processed_data_dir": "data/processed",  # Directory containing processed data
    "model_save_path": "models/resnet_model.pth",  # Path to save the model
    "figure_save_path": "reports/figures/resnet_training_statistics.png",  # Path to save training stats
    "num_classes": 120,  # Number of output classes
    "resnet_model": "resnet50",  # ResNet model name
    "epochs": 50,  # Maximum number of training epochs
    "early_stopping_patience": 5,  # Patience for early stopping
}


def load_dataset(subset: str, data_dir: str) -> TensorDataset:
    """
    Load the specified dataset (train/validation/test) from .pt files.
    """
    images_path = os.path.join(data_dir, subset, f"{subset}_images.pt")
    targets_path = os.path.join(data_dir, subset, f"{subset}_targets.pt")
    images = torch.load(images_path)
    targets = torch.load(targets_path)
    return TensorDataset(images, targets)


def objective(trial):
    """
    Optuna objective function for hyperparameter tuning.
    """
    # Suggest hyperparameters for SGD optimizer
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    momentum = trial.suggest_float("momentum", 0.5, 0.99)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    dropout_rate = trial.suggest_uniform("dropout_rate", 0.2, 0.5)

    # Load datasets
    train_dataset = load_dataset("train", PARAMS["processed_data_dir"])
    val_dataset = load_dataset("validation", PARAMS["processed_data_dir"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model_params = {
        "num_classes": PARAMS["num_classes"],
        "resnet_model": PARAMS["resnet_model"],
        "dropout_rate": dropout_rate,
    }
    model = SimpleResNetClassifier(params=model_params).to(DEVICE)

    # Define loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    # Early stopping variables
    best_val_loss = float("inf")
    patience = 0
    early_stopping_patience = PARAMS["early_stopping_patience"]

    for epoch in range(PARAMS["epochs"]):
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

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
        else:
            patience += 1
            if patience >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    return best_val_loss


def train_with_optuna(n_trials: int = 20):
    """
    Run Optuna optimization and save the best parameters and results.
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # Save the best parameters and results to a JSON file
    best_params = study.best_params
    best_val_loss = study.best_value
    results = {
        "best_params": best_params,
        "best_val_loss": best_val_loss,
    }
    os.makedirs("config", exist_ok=True)
    with open("config/resnet_best_params.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Best parameters and results saved to 'results/best_resnet_params.json'")
    print(f"Best parameters: {best_params}")
    print(f"Best validation loss: {best_val_loss}")


if __name__ == "__main__":
    typer.run(train_with_optuna)
