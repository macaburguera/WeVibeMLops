# Base image with CUDA and PyTorch
FROM nvcr.io/nvidia/pytorch:24.10-py3 AS base


# Set environment variables
#ENV WANDB_MODEL_NAME="macarenaburguera-danmarks-tekniske-universitet-dtu-org/wandb-registry-model/dbc:latest"
# Allow optional passing of WANDB_API_KEY during runtime
ARG WANDB_API_KEY
ENV WANDB_API_KEY=${WANDB_API_KEY}


# Install necessary tools and libraries
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    build-essential \
    gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy project files (adjust paths relative to dockerfiles)
COPY /src src/
COPY /scripts scripts/
#COPY ../data data/
COPY /configs configs/
COPY /tests tests/
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY run_all.sh run_all.sh
COPY pyproject.toml pyproject.toml

# Install Python dependencies
#RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e . --no-cache-dir
RUN pip install wandb
RUN pip install dvc
RUN pip install dvc[gs] 

# Setup DVC
#RUN dvc init --no-scm
COPY /.dvc/config .dvc/config
COPY models.dvc models.dvc
COPY data.dvc data.dvc
#RUN dvc init --no-scm -f
#RUN dvc remote add -d myremote gs://doge_bucket45 --force

RUN dvc config core.no_scm true
RUN dvc pull data/breed_mapping.csv --no-run-cache -v
RUN dvc pull models/resnet_model.pth --no-run-cache -v
RUN dvc pull data/processed/test --no-run-cache -v
RUN dvc pull data/processed/validation --no-run-cache -v
RUN dvc pull data/processed/train --no-run-cache -v

# Set entrypoint script to allow running commands easily
RUN chmod +x ./scripts/entrypoint.sh
ENTRYPOINT ["./scripts/entrypoint.sh"]
