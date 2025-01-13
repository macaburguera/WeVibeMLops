#!/bin/bash

# Activate the virtual environment (if required)
# source /home/maca/miniconda3/envs/dogs/bin/python

eval "$(conda shell.bash hook)"  # Ensure the conda command works in non-interactive shells
conda activate dogs

# Run train_cnn.py
echo "Starting train_cnn.py..."
python src/dog_breed_classifier/train_cnn.py > logs/train_cnn.log 2>&1
echo "Finished train_cnn.py. Logs saved to logs/train_cnn.log"

# Run train_dino.py
echo "Starting train_dino.py..."
python src/dog_breed_classifier/train_dino.py > logs/train_dino.log 2>&1
echo "Finished train_dino.py. Logs saved to logs/train_dino.log"

# Run train_resnet.py
echo "Starting train_resnet.py..."
python src/dog_breed_classifier/train_resnet.py > logs/train_resnet.log 2>&1
echo "Finished train_resnet.py. Logs saved to logs/train_resnet.log"

# Completion message
echo "All training scripts have finished."
