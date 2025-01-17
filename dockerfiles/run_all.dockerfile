# Base image with CUDA and PyTorch
FROM nvcr.io/nvidia/pytorch:24.10-py3 AS base

# Install necessary tools and libraries
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    build-essential \
    gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy project files (adjust paths relative to dockerfiles)
COPY ../src src/
COPY ../data data/
COPY ../configs configs/
COPY ../requirements.txt requirements.txt
COPY ../requirements_dev.txt requirements_dev.txt
COPY ../README.md README.md
COPY ../run_all.sh run_all.sh

# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt 
#RUN pip install . --no-deps --no-cache-dir --verbose

# Ensure the train_all.sh script is executable
RUN chmod +x run_all.sh

# Set the entry point to execute the train_all.sh script
ENTRYPOINT ["./run_all.sh"]
