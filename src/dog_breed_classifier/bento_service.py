import os
import io
import torch
from PIL import Image
from torchvision import transforms
from bentoml import Service, Runner, Runnable
from bentoml.io import JSON, Image as BentoImage
from src.dog_breed_classifier.model import SimpleResNetClassifier

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model_path = "models/resnet_model.pth"
try:
    model = SimpleResNetClassifier(params={"num_classes": 120, "resnet_model": "resnet50"}).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    raise RuntimeError(f"Error loading the model: {e}")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define breed mapping (replace this with your actual mapping logic)
breed_mapping = {i: f"Breed {i}" for i in range(120)}

# Create a BentoML Runnable
class DogBreedClassifierRunnable(Runnable):
    def __init__(self):
        self.model = model
        self.transform = transform
        self.breed_mapping = breed_mapping

    @Runnable.method(batchable=False)
    def predict(self, image: Image.Image) -> dict:
        """
        Predict the breed of the dog from the uploaded image.
        """
        try:
            # Preprocess the image
            image_tensor = self.transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = self.model(image_tensor)
                _, predicted = torch.max(outputs, 1)

            predicted_breed = self.breed_mapping.get(predicted.item(), "Unknown breed")
            return {"predicted_breed": predicted_breed}

        except Exception as e:
            raise RuntimeError(f"Error during prediction: {str(e)}")

# Create a BentoML runner
dog_breed_runner = Runner(DogBreedClassifierRunnable)

# Define the BentoML service
svc = Service(
    name="dog_breed_classifier_service",
    runners=[dog_breed_runner]
)

@svc.api(input=BentoImage(), output=JSON())
async def predict(file: Image.Image):
    """
    API endpoint to classify the breed of the dog.
    """
    return await dog_breed_runner.async_run(file)
