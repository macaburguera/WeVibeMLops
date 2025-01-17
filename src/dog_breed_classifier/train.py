import os
import torch
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from model import SimpleResNetClassifier
import typer
import wandb
import yaml

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processed_data_dir = to_absolute_path("data/processed")

# Typer app
app = typer.Typer()

def train(cfg: DictConfig, use_wandb: bool, override_hyperparams=None):
    print("Starting training with SimpleResNetClassifier...")

    # Override hyperparameters from wandb sweep
    if override_hyperparams:
        cfg.hyperparameters.lr = override_hyperparams["learning_rate"]
        cfg.hyperparameters.batch_size = override_hyperparams["batch_size"]
        cfg.hyperparameters.epochs = override_hyperparams["epochs"]

    print(f"{cfg.hyperparameters.lr=}, {cfg.hyperparameters.batch_size=}, {cfg.hyperparameters.epochs=}")

    # Initialize wandb if activated
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

        print(f"Epoch {epoch+1}/{cfg.hyperparameters.epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Train Acc={train_accuracy:.4f}, Val Acc={val_accuracy:.4f}")

        # Log metrics to wandb
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
            })

    # Finish wandb session
    if use_wandb:
        wandb.finish()

@app.callback(invoke_without_command=True)
def default(ctx: typer.Context, wandb_flag: bool = typer.Option(False, help="Enable Weights & Biases for logging")):
    """
    Default behavior: Run standard training if no command is provided.
    """
    if ctx.invoked_subcommand is None:
        main(wandb_flag=wandb_flag)

@app.command()
def main(wandb_flag: bool = typer.Option(False, help="Enable Weights & Biases for logging")):
    with hydra.initialize(config_path="../../configs"):
        cfg = hydra.compose(config_name="config")
        train(cfg, use_wandb=wandb_flag)

@app.command()
def sweep():
    """
    Run a Wandb hyperparameter sweep.
    """
    sweep_config_path = to_absolute_path("configs/sweep.yaml")

    # Ensure the sweep file exists
    if not os.path.exists(sweep_config_path):
        raise FileNotFoundError(f"Could not find sweep configuration file at {sweep_config_path}")

    with open(sweep_config_path, "r") as f:
        sweep_config = yaml.safe_load(f)

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="dog-breed-classifier")

    def sweep_train():
        # Use a relative path for Hydra's config path
        with hydra.initialize(config_path="../../configs"):  # Use relative path
            cfg = hydra.compose(config_name="config")
            wandb.init()
            train(cfg, use_wandb=True)



    wandb.agent(sweep_id, function=sweep_train)




if __name__ == "__main__":
    app()
