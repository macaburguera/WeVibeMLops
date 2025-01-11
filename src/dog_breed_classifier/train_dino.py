import os
import torch
import typer
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from model_dino import SimpleDinoClassifier

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Training parameters (can later be moved to a config file)
PARAMS = {
    "lr": 1e-3,             # Learning rate
    "batch_size": 32,       # Batch size
    "epochs": 5,            # Number of training epochs
    "processed_data_dir": "data/processed",  # Directory containing processed data
    "model_save_path": "models/dino_model.pth",  # Path to save the model
    "figure_save_path": "reports/figures/dino_training_statistics.png",  # Path to save training statistics
    "model_params": {  # Parameters for the model
        "num_classes": 120,  # Number of output classes
        "dino_model": "vit_small_patch16_224_dino",  # Pretrained DINO model
        "hidden_units": 512,  # Hidden layer units in classifier
        "dropout_rate": 0.25,  # Dropout rate
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


def train(
    lr: float = PARAMS["lr"],
    batch_size: int = PARAMS["batch_size"],
    epochs: int = PARAMS["epochs"],
    processed_data_dir: str = PARAMS["processed_data_dir"],
    model_save_path: str = PARAMS["model_save_path"],
    figure_save_path: str = PARAMS["figure_save_path"],
) -> None:
    """
    Train the SimpleDinoClassifier model using the specified parameters.

    Args:
        lr (float): Learning rate.
        batch_size (int): Batch size.
        epochs (int): Number of training epochs.
        processed_data_dir (str): Path to the directory with processed datasets.
        model_save_path (str): Path to save the trained model.
        figure_save_path (str): Path to save training statistics.
    """
    print("Starting training with SimpleDinoClassifier...")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    # Load datasets
    train_dataset = load_dataset("train", processed_data_dir)
    val_dataset = load_dataset("validation", processed_data_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = SimpleDinoClassifier(params=PARAMS["model_params"]).to(DEVICE)

    # Define loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Track training statistics
    statistics = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Training loop
        model.train()
        train_loss = 0.0
        train_correct = 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            train_loss += loss.item()

            # Backward pass and optimizer step
            loss.backward()
            optimizer.step()

            # Accuracy
            preds = outputs.argmax(dim=1)
            train_correct += (preds == targets).sum().item()

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)

                outputs = model(imgs)
                loss = loss_fn(outputs, targets)
                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()

        # Calculate epoch statistics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_accuracy = train_correct / len(train_loader.dataset)
        val_accuracy = val_correct / len(val_loader.dataset)

        statistics["train_loss"].append(train_loss)
        statistics["val_loss"].append(val_loss)
        statistics["train_accuracy"].append(train_accuracy)
        statistics["val_accuracy"].append(val_accuracy)

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Save the model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Plot training statistics
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"], label="Train Loss")
    axs[0].plot(statistics["val_loss"], label="Validation Loss")
    axs[0].set_title("Loss")
    axs[0].legend()

    axs[1].plot(statistics["train_accuracy"], label="Train Accuracy")
    axs[1].plot(statistics["val_accuracy"], label="Validation Accuracy")
    axs[1].set_title("Accuracy")
    axs[1].legend()

    os.makedirs(os.path.dirname(figure_save_path), exist_ok=True)
    fig.savefig(figure_save_path)
    print(f"Training statistics saved to {figure_save_path}")


if __name__ == "__main__":
    typer.run(train)
