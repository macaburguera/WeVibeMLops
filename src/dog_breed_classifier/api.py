import os
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from torchvision import transforms
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
        raise HTTPException(status_code=400, detail="File must be an image (png, jpg, jpeg).")

    try:
        # Read the image file
        image = Image.open(file.file).convert("RGB")
        
        # Preprocess the image
        image_tensor = transform(image).unsqueeze(0)
        
        # Perform inference
        with torch.no_grad():
            outputs = model(image_tensor)
            predicted_class = torch.argmax(outputs, dim=1).item()
        
        # Return the predicted class
        return {"filename": file.filename, "predicted_breed": f"Breed ID: {predicted_class}"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the image: {e}")
