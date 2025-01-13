import os
import torch
import typer
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from model_dino import SimpleDinoClassifier
import optuna
from optuna.integration import PyTorchLightningPruningCallback

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Training parameters (can later be moved to a config file)
DEFAULT_PARAMS = {
    "epochs": 15,                # Default number of epochs
    "processed_data_dir": "data/processed",  # Directory containing processed data
    "model_save_path": "models/dino_model.pth",  # Path to save the model
    "figure_save_path": "reports/figures/dino_training_statistics.png",  # Path to save training stats
    "model_params": {  # Parameters for the model
        "num_classes": 120,  # Number of output classes
        "dino_model": "vit_small_patch16_224_dino",  # Pretrained DINO model
        "hidden_units": 512,  # Hidden layer units in classifier
        "dropout_rate": 0.5,  # Dropout rate
        "freeze_dino": False,  # Whether to freeze DINO backbone
    },
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


def objective(trial: optuna.Trial) -> float:
    """
    Objective function for Optuna to optimize hyperparameters.
    """
    # Suggest hyperparameters to optimize
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    dropout_rate = trial.suggest_uniform("dropout_rate", 0.2, 0.5)
    freeze_dino = trial.suggest_categorical("freeze_dino", [True, False])

    # Update model parameters based on the trial
    model_params = DEFAULT_PARAMS["model_params"].copy()
    model_params["dropout_rate"] = dropout_rate
    model_params["freeze_dino"] = freeze_dino

    # Load datasets
    processed_data_dir = DEFAULT_PARAMS["processed_data_dir"]
    train_dataset = load_dataset("train", processed_data_dir)
    val_dataset = load_dataset("validation", processed_data_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = SimpleDinoClassifier(params=model_params).to(DEVICE)

    # Define loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Early stopping
    best_val_loss = float("inf")
    patience = 3
    patience_counter = 0

    # Training loop
    for epoch in range(DEFAULT_PARAMS["epochs"]):
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

        print(f"Epoch {epoch+1}/{DEFAULT_PARAMS['epochs']} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

        # Report intermediate results to Optuna
        trial.report(val_loss, epoch)

        # Prune the trial if necessary
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_loss


def optimize_hyperparameters() -> None:
    """
    Optimize hyperparameters using Optuna.
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    import json

    # Save the best trial's parameters to a JSON file
    best_params = study.best_trial.params
    best_results = {
        "value": study.best_trial.value,
        "params": best_params,
    }
    os.makedirs("config", exist_ok=True)
    with open("config/dino._best_paramsjson", "w") as f:
        json.dump(best_results, f, indent=4)

    print("Best parameters and results saved to 'best_params.json'")


if __name__ == "__main__":
    typer.run(optimize_hyperparameters)
