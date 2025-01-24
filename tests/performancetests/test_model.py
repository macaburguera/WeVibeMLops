import os
import time
import torch
import wandb
from src.dog_breed_classifier.model import SimpleResNetClassifier


def load_model_from_artifact(model_name: str, api_key: str, device: torch.device):
    """
    Load the model from a W&B artifact.

    Args:
        model_name (str): The name of the W&B artifact (e.g., 'entity/project/artifact-name:version').
        api_key (str): Your W&B API key.
        device (torch.device): The device to load the model onto.

    Returns:
        torch.nn.Module: The loaded model.
    """
    if not model_name or not api_key:
        raise ValueError("Model name or API key is not provided.")

    print(f"Loading model from artifact: {model_name}")

    # Initialize the W&B API with the provided API key
    api = wandb.Api(api_key=api_key)
    artifact = api.artifact(model_name, type="model")
    artifact_dir = artifact.download()

    # Load model from the artifact directory
    model_path = os.path.join(artifact_dir, "resnet_model.pth")  # Adjust if the filename differs
    params = {
        "num_classes": 120,  # Replace with the actual number of classes
        "resnet_model": "resnet50",  # Replace with the model name used during training
    }
    model = SimpleResNetClassifier(params=params)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device)


def test_model_speed():
    """
    Test that the model can make 100 predictions within a specified time limit.
    """
    # Retrieve W&B API key and model name from environment variables
    wandb_api_key = os.getenv("WANDB_API_KEY")
    model_name = os.getenv("WANDB_MODEL_NAME")

    if not wandb_api_key or not model_name:
        raise ValueError("WANDB_API_KEY or MODEL_NAME environment variable is not set.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = load_model_from_artifact(model_name, wandb_api_key, device)
    model.eval()

    # Test input
    batch_size = 1
    input_size = (3, 224, 224)  # Expected input size for ResNet models
    num_iterations = 100
    time_limit = 10  # Time limit for 100 predictions (in seconds)

    # Measure prediction speed
    inputs = torch.randn(batch_size, *input_size).to(device)
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(inputs)
    end_time = time.time()

    total_time = end_time - start_time
    print(f"Total time for {num_iterations} predictions: {total_time:.4f} seconds.")
    assert total_time < time_limit, f"Model predictions took too long: {total_time:.4f} seconds."


if __name__ == "__main__":
    test_model_speed()
