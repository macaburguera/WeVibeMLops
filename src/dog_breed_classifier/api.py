import os
import io
import time
import torch
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from torchvision import transforms
import pandas as pd
import numpy as np
from datetime import datetime
from google.cloud import storage
from skimage.filters import sobel
from skimage.measure import shannon_entropy
from prometheus_client import Counter, Histogram, Summary, make_asgi_app
from src.dog_breed_classifier.model import SimpleResNetClassifier

# Initialize the FastAPI app
app = FastAPI()

# Prometheus metrics setup
error_counter = Counter("api_errors", "Total number of errors encountered")
request_counter = Counter("api_requests", "Total number of requests received")
classification_time_histogram = Histogram("classification_time_seconds", "Time taken to classify an image")
file_size_summary = Summary("file_size_bytes", "Size of uploaded files in bytes")

# Mount Prometheus metrics endpoint
app.mount("/metrics", make_asgi_app())

# Google Cloud Storage setup
BUCKET_NAME = "doge_bucket45"
CSV_FILENAME = "predictions_log.csv"

# Initialize GCP Storage client
storage_client = storage.Client()

# In-memory DataFrame to store log data
log_df = pd.DataFrame(columns=[
    "timestamp", "id", "filename", "breed", "mean", "std_dev", "median", "min_pixel", "max_pixel", "entropy", "edge_density"
])

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

# Function to download the log CSV from GCP bucket at startup or create if not found
def load_or_create_log():
    global log_df
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(CSV_FILENAME)
        if blob.exists():
            blob.download_to_filename(CSV_FILENAME)
            log_df = pd.read_csv(CSV_FILENAME)
            print(f"Loaded existing log from {CSV_FILENAME}")
        else:
            print("No existing log found. Creating a new log.")
            log_df.to_csv(CSV_FILENAME, index=False)
            upload_to_gcs(BUCKET_NAME, CSV_FILENAME, CSV_FILENAME)
    except Exception as e:
        print(f"Error loading or creating the log: {e}")

# Function to upload a file to GCP bucket
def upload_to_gcs(bucket_name: str, source_file_name: str, destination_blob_name: str):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"Uploaded {source_file_name} to gs://{bucket_name}/{destination_blob_name}")

# Function to compute image metrics
def compute_image_metrics(image: np.ndarray):
    grayscale_image = np.array(image.convert("L"))  # Convert to grayscale
    mean = float(np.mean(grayscale_image))
    std_dev = float(np.std(grayscale_image))
    median = float(np.median(grayscale_image))
    min_pixel = float(np.min(grayscale_image))
    max_pixel = float(np.max(grayscale_image))
    entropy = float(shannon_entropy(grayscale_image))
    edge_density = float(np.mean(sobel(grayscale_image)))  # Edge detection using Sobel filter

    return {
        "mean": mean,
        "std_dev": std_dev,
        "median": median,
        "min_pixel": min_pixel,
        "max_pixel": max_pixel,
        "entropy": entropy,
        "edge_density": edge_density,
    }

# Function to log prediction details and sync with the bucket
def log_prediction(image_id, filename, predicted_breed, metrics):
    global log_df

    # Prepare data
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "id": image_id,
        "filename": filename,
        "breed": predicted_breed,
        **metrics,
    }

    # Append to the in-memory DataFrame
    log_df = pd.concat([log_df, pd.DataFrame([log_entry])], ignore_index=True)

    # Save updated log locally and upload
    log_df.to_csv(CSV_FILENAME, index=False)
    upload_to_gcs(BUCKET_NAME, CSV_FILENAME, CSV_FILENAME)

@app.on_event("startup")
def startup_event():
    load_or_create_log()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Predict the breed of a dog from an uploaded image.
    """
    request_counter.inc()  # Increment request counter

    if not file.filename.endswith((".png", ".jpg", ".jpeg")):
        return {"filename": file.filename, "predicted_breed": "Invalid file type. Please upload a .png, .jpg, or .jpeg image."}

    try:
        # Read the image file
        image_bytes = await file.read()
        file_size_summary.observe(len(image_bytes))  # Observe file size

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocess the image
        image_tensor = transform(image).unsqueeze(0)

        # Perform inference
        start_time = time.time()
        with torch.no_grad():
            outputs = model(image_tensor)
            predicted_class = torch.argmax(outputs, dim=1).item()
        classification_time = time.time() - start_time
        classification_time_histogram.observe(classification_time)  # Observe classification time

        # Get the breed name
        predicted_breed = breed_id_to_name.get(predicted_class, "Unknown breed")

        # Compute image metrics
        metrics = compute_image_metrics(image)

        # Log details
        image_id = f"img_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        image_filename = f"{image_id}_{file.filename}"
        image_path = os.path.join("/tmp", image_filename)
        image.save(image_path)

        # Upload image to GCP bucket
        upload_to_gcs(BUCKET_NAME, image_path, f"images/{image_filename}")

        # Log prediction details
        log_prediction(image_id, file.filename, predicted_breed, metrics)

        return {"filename": file.filename, "predicted_breed": predicted_breed, **metrics}

    except Exception as e:
        error_counter.inc()  # Increment error counter
        # Log the error and return a graceful response
        print(f"Error processing the image: {e}")
        return {"filename": file.filename, "predicted_breed": "Unknown breed"}
