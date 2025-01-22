# Use an official Python image as the base
FROM python:3.10-slim

# Copy the script into the container
COPY src/dog_breed_classifier/data_drift.py .

# Install Python dependencies directly (without a requirements.txt)
RUN pip install --no-cache-dir pandas scikit-learn evidently google-cloud-storage

# Command to run the script
CMD ["python", "data_drift.py"]
