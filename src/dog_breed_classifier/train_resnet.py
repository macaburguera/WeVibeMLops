import os
import torch
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from model_resnet import SimpleResNetClassifier

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processed_data_dir = to_absolute_path("data/processed")

def train(cfg: DictConfig):
    print("Starting training with SimpleResNetClassifier...")
    print(f"{cfg.hyperparameters.lr=}, {cfg.hyperparameters.batch_size=}, {cfg.hyperparameters.epochs=}")

    # Load datasets
    def load_dataset(subset, data_dir):
        subset_dir = os.path.join(data_dir, subset)
        images_path = os.path.join(subset_dir, f"{subset}_images.pt")
        targets_path = os.path.join(subset_dir, f"{subset}_targets.pt")

        if not os.path.exists(images_path) or not os.path.exists(targets_path):
            raise FileNotFoundError(f"Expected files not found: {images_path} or {targets_path}")

        images = torch.load(images_path)
        targets = torch.load(targets_path)
        return TensorDataset(images, targets)

    train_dataset = load_dataset("train", processed_data_dir)
    val_dataset = load_dataset("validation", processed_data_dir)

    train_loader = DataLoader(train_dataset, batch_size=cfg.hyperparameters.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.hyperparameters.batch_size, shuffle=False)

    # Initialize model
    model = SimpleResNetClassifier(params=cfg.hyperparameters.model_params).to(DEVICE)

    # Define loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.hyperparameters.lr, momentum=0.9)

    # Training loop
    statistics = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
    for epoch in range(cfg.hyperparameters.epochs):
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == targets).sum().item()
            total_train += targets.size(0)

        train_accuracy = correct_train / total_train

        # Validation loop
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
                outputs = model(imgs)
                loss = loss_fn(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == targets).sum().item()
                total_val += targets.size(0)

        val_accuracy = correct_val / total_val

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        statistics["train_loss"].append(train_loss)
        statistics["val_loss"].append(val_loss)
        statistics["train_accuracy"].append(train_accuracy)
        statistics["val_accuracy"].append(val_accuracy)

        print(f"Epoch {epoch+1}/{cfg.hyperparameters.epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Train Acc={train_accuracy:.4f}, Val Acc={val_accuracy:.4f}")

    # Save model
    os.makedirs(os.path.dirname(cfg.hyperparameters.model_save_path), exist_ok=True)
    torch.save(model.state_dict(), cfg.hyperparameters.model_save_path)

    # Save training statistics
    plt.figure()
    plt.plot(statistics["train_loss"], label="Train Loss")
    plt.plot(statistics["val_loss"], label="Validation Loss")
    plt.legend()
    plt.savefig(cfg.hyperparameters.figure_save_path.replace("statistics", "loss"))

    plt.figure()
    plt.plot(statistics["train_accuracy"], label="Train Accuracy")
    plt.plot(statistics["val_accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(cfg.hyperparameters.figure_save_path.replace("statistics", "accuracy"))

    print(f"Training statistics saved to {cfg.hyperparameters.figure_save_path}")

@hydra.main(config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    train(cfg)

if __name__ == "__main__":
    main()
