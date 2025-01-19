import os
import torch
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from torch.utils.data import DataLoader, TensorDataset
from src.dog_breed_classifier.model import SimpleResNetClassifier
import typer
import wandb
import yaml
import matplotlib.pyplot as plt

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Typer app
app = typer.Typer()

def train(cfg: DictConfig, use_wandb: bool, override_hyperparams=None):
    print("Starting training with SimpleResNetClassifier...")

    if override_hyperparams:
        cfg.hyperparameters.lr = override_hyperparams["learning_rate"]
        cfg.hyperparameters.batch_size = override_hyperparams["batch_size"]
        cfg.hyperparameters.epochs = override_hyperparams["epochs"]

    print(f"{cfg.hyperparameters.lr=}, {cfg.hyperparameters.batch_size=}, {cfg.hyperparameters.epochs=}")

    if use_wandb:
        wandb.init(
            project="dog-breed-classifier",
            config={
                "learning_rate": cfg.hyperparameters.lr,
                "batch_size": cfg.hyperparameters.batch_size,
                "epochs": cfg.hyperparameters.epochs,
                "model": cfg.hyperparameters.model_params.resnet_model,
            },
        )

    def load_dataset(subset, data_dir):
        subset_dir = os.path.join(data_dir, subset)
        images_path = os.path.join(subset_dir, f"{subset}_images.pt")
        targets_path = os.path.join(subset_dir, f"{subset}_targets.pt")

        if not os.path.exists(images_path) or not os.path.exists(targets_path):
            raise FileNotFoundError(f"Expected files not found: {images_path} or {targets_path}")

        images = torch.load(images_path)
        targets = torch.load(targets_path)

        num_classes = cfg.hyperparameters.model_params.num_classes
        if not ((targets >= 0).all() and (targets < num_classes).all()):
            raise ValueError(f"Targets contain out-of-bound values: min={targets.min()}, max={targets.max()}")

        return TensorDataset(images, targets)

    train_dataset = load_dataset("train", cfg.hyperparameters.processed_data_dir)
    val_dataset = load_dataset("validation", cfg.hyperparameters.processed_data_dir)

    train_loader = DataLoader(train_dataset, batch_size=cfg.hyperparameters.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.hyperparameters.batch_size, shuffle=False)

    model = SimpleResNetClassifier(params=cfg.hyperparameters.model_params).to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.hyperparameters.lr, momentum=0.9)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(cfg.hyperparameters.epochs):
        model.train()
        train_loss, correct_train, total_train = 0, 0, 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            correct_train += (outputs.argmax(1) == targets).sum().item()
            total_train += targets.size(0)

        train_accuracy = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        val_loss, correct_val, total_val = 0, 0, 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
                outputs = model(imgs)
                val_loss += loss_fn(outputs, targets).item()
                correct_val += (outputs.argmax(1) == targets).sum().item()
                total_val += targets.size(0)

        val_accuracy = correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{cfg.hyperparameters.epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Train Acc={train_accuracy:.4f}, Val Acc={val_accuracy:.4f}")

        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
            })

    model_save_path = cfg.hyperparameters.model_save_path
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    if use_wandb:
        # Create and log artifact
        artifact = wandb.Artifact("dog-breed-classifier-model", type="model")
        artifact.add_file(model_save_path)
        logged_artifact = wandb.log_artifact(artifact)
        artifact.wait()

        # Link artifact to registry collection
        logged_artifact.link(
            target_path="macarenaburguera-danmarks-tekniske-universitet-dtu-org/wandb-registry-model/dbc"
        )

        wandb.finish()

@app.command()
def normal():
    """Run training locally without WandB."""
    with hydra.initialize(config_path="../../configs"):
        cfg = hydra.compose(config_name="config")
        train(cfg, use_wandb=False)

@app.command()
def wandb_run():
    """Run training with WandB logging."""
    with hydra.initialize(config_path="../../configs"):
        cfg = hydra.compose(config_name="config")
        train(cfg, use_wandb=True)

@app.command()
def sweep():
    """Run a hyperparameter sweep."""
    sweep_config_path = to_absolute_path("configs/sweep.yaml")

    if not os.path.exists(sweep_config_path):
        raise FileNotFoundError(f"Could not find sweep configuration file at {sweep_config_path}")

    with open(sweep_config_path, "r") as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep_config, project="dog-breed-classifier")

    def sweep_train():
        with hydra.initialize(config_path="../../configs"):
            cfg = hydra.compose(config_name="config")
            wandb.init()
            train(cfg, use_wandb=True)

    wandb.agent(sweep_id, function=sweep_train)

if __name__ == "__main__":
    app()
