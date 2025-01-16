import os
import torch
import typer
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from model_dino import SimpleDinoClassifier

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training parameters
PARAMS = {
    "lr": 1e-4,
    "batch_size": 32,
    "epochs": 5,
    "processed_data_dir": "data/processed",
    "model_save_path": "models/dino_model.pth",
    "figure_save_path": "reports/figures/dino_training_statistics.png",
    "model_params": {
        "num_classes": 120,
        "dino_model": "vit_small_patch16_224_dino",
        "hidden_units": 512,
        "dropout_rate": 0.5,
        "freeze_dino": False
    },
}

def train():
    print("Starting training with SimpleDinoClassifier...")
    print(f"{PARAMS['lr']=}, {PARAMS['batch_size']=}, {PARAMS['epochs']=}")

    # Load datasets
    def load_dataset(subset, data_dir):
        images = torch.load(os.path.join(data_dir, subset, f"{subset}_images.pt"))
        targets = torch.load(os.path.join(data_dir, subset, f"{subset}_targets.pt"))
        return TensorDataset(images, targets)

    train_dataset = load_dataset("train", PARAMS["processed_data_dir"])
    val_dataset = load_dataset("validation", PARAMS["processed_data_dir"])

    train_loader = DataLoader(train_dataset, batch_size=PARAMS["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=PARAMS["batch_size"], shuffle=False)

    # Initialize model
    model = SimpleDinoClassifier(params=PARAMS["model_params"]).to(DEVICE)

    # Define loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=PARAMS["lr"])

    # Training loop
    statistics = {"train_loss": [], "val_loss": []}
    for epoch in range(PARAMS["epochs"]):
        model.train()
        train_loss = 0
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
        val_loss = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
                outputs = model(imgs)
                loss = loss_fn(outputs, targets)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        statistics["train_loss"].append(train_loss)
        statistics["val_loss"].append(val_loss)

        print(f"Epoch {epoch+1}/{PARAMS['epochs']}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

    # Save model
    os.makedirs(os.path.dirname(PARAMS["model_save_path"]), exist_ok=True)
    torch.save(model.state_dict(), PARAMS["model_save_path"])

    # Save training statistics
    plt.plot(statistics["train_loss"], label="Train Loss")
    plt.plot(statistics["val_loss"], label="Validation Loss")
    plt.legend()
    plt.savefig(PARAMS["figure_save_path"])
    print(f"Training statistics saved to {PARAMS['figure_save_path']}")

if __name__ == "__main__":
    typer.run(train)
