# Use a lightweight Python image as the base
FROM python:3.10-slim

# Install minimal dependencies required for the frontend
RUN pip install --no-cache-dir streamlit requests Pillow google-cloud-run

# Copy the frontend script into the container
COPY src/dog_breed_classifier/frontend.py .

# Expose the port Streamlit runs on
EXPOSE 8501

# Default Streamlit command to run the app
CMD ["streamlit", "run", "frontend.py", "--server.port=8501", "--server.address=0.0.0.0"]
