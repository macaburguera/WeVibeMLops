import streamlit as st
import requests
from PIL import Image

# Define API endpoint
#API_URL = "http://localhost:8000/predict/"

API_URL = "https://doge-api-414169417184.europe-west1.run.app/predict/"

st.title("Dog Breed Classifier")
st.write("Upload an image of a dog, and the model will predict its breed.")

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Send the image to the FastAPI backend
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            prediction = response.json()
            st.success(f"Predicted Breed: {prediction['predicted_breed']}")
        else:
            st.error(f"Error: {response.json().get('detail', 'Unknown error occurred.')}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
