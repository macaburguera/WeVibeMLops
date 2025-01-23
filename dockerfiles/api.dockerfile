# Change from latest to a specific version if your requirements.txt
FROM nvcr.io/nvidia/pytorch:24.10-py3 AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml

RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install -e . --no-deps --no-cache-dir --verbose

RUN pip install dvc
RUN pip install dvc[gs]

RUN pip install fastapi
RUN pip install pydantic
RUN pip install uvicorn
RUN pip install python-multipart


# Setup DVC
#RUN dvc init --no-scm
COPY /.dvc/config .dvc/config
COPY models.dvc models.dvc
COPY data.dvc data.dvc
RUN dvc config core.no_scm true
RUN dvc pull data/breed_mapping.csv --no-run-cache
RUN dvc pull models/resnet_model.pth --no-run-cache

# Expose the API port
EXPOSE 8000

CMD exec uvicorn src.dog_breed_classifier.api:app --host 0.0.0.0 --port 8000 --reload
