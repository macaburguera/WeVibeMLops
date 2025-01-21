import os
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from torchvision import transforms
import pandas as pd
from src.dog_breed_classifier.model import SimpleResNetClassifier

# Initialize the FastAPI app
app = FastAPI()

# Load the trained model
MODEL_PATH = os.path.join(os.getcwd(), "models", "resnet_model.pth")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

print(f"Loading model from {MODEL_PATH}...")
model_params = {"num_classes": 120, "resnet_model": "resnet50"}  # Update with your model's parameters
model = SimpleResNetClassifier(params=model_params)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()
print("Model loaded successfully.")

# Load the breed mapping
BREED_MAPPING_PATH = os.path.join(os.getcwd(), "data", "breed_mapping.csv")
if not os.path.exists(BREED_MAPPING_PATH):
    raise FileNotFoundError(f"Breed mapping file not found at {BREED_MAPPING_PATH}")

print(f"Loading breed mapping from {BREED_MAPPING_PATH}...")
breed_mapping = pd.read_csv(BREED_MAPPING_PATH)
breed_id_to_name = dict(zip(breed_mapping["ID"], breed_mapping["Breed"]))
print("Breed mapping loaded successfully.")

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Predict the breed of a dog from an uploaded image.
    """
    if not file.filename.endswith((".png", ".jpg", ".jpeg")):
        return {"filename": file.filename, "predicted_breed": "Invalid file type. Please upload a .png, .jpg, or .jpeg image."}

    try:
        # Read the image file
        image = Image.open(file.file).convert("RGB")
        
        # Preprocess the image
        image_tensor = transform(image).unsqueeze(0)
        
        # Perform inference
        with torch.no_grad():
            outputs = model(image_tensor)
            predicted_class = torch.argmax(outputs, dim=1).item()
        
        # Get the breed name
        predicted_breed = breed_id_to_name.get(predicted_class, "Unknown breed")
        
        # Return the predicted breed name
        return {"filename": file.filename, "predicted_breed": predicted_breed}
    
    except Exception as e:
        # Log the error and return a graceful response
        print(f"Error processing the image: {e}")
        return {"filename": file.filename, "predicted_breed": "Unknown breed"}
