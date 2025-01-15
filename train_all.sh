#!/bin/bash

# Activate the virtual environment
eval "$(conda shell.bash hook)"  # Ensure the conda command works in non-interactive shells
conda activate dogs

# Create logs directory if it doesn't exist
mkdir -p logs

# Single log file for all outputs
LOG_FILE="logs/training_sesion.log"

# Clear the log file before running
> $LOG_FILE

# Run model_cnn.py and train_cnn.py
echo "Starting model_cnn.py..." | tee -a $LOG_FILE
python src/dog_breed_classifier/model_cnn.py >> $LOG_FILE 2>&1
echo "Finished model_cnn.py." | tee -a $LOG_FILE

echo "Starting train_cnn.py..." | tee -a $LOG_FILE
python src/dog_breed_classifier/train_cnn.py >> $LOG_FILE 2>&1
echo "Finished train_cnn.py." | tee -a $LOG_FILE

# Run model_dino.py and train_dino.py
echo "Starting model_dino.py..." | tee -a $LOG_FILE
python src/dog_breed_classifier/model_dino.py >> $LOG_FILE 2>&1
echo "Finished model_dino.py." | tee -a $LOG_FILE

echo "Starting train_dino.py..." | tee -a $LOG_FILE
python src/dog_breed_classifier/train_dino.py >> $LOG_FILE 2>&1
echo "Finished train_dino.py." | tee -a $LOG_FILE

# Run model_resnet.py and train_resnet.py
echo "Starting model_resnet.py..." | tee -a $LOG_FILE
python src/dog_breed_classifier/model_resnet.py >> $LOG_FILE 2>&1
echo "Finished model_resnet.py." | tee -a $LOG_FILE

echo "Starting train_resnet.py..." | tee -a $LOG_FILE
python src/dog_breed_classifier/train_resnet.py >> $LOG_FILE 2>&1
echo "Finished train_resnet.py." | tee -a $LOG_FILE

# Completion message
echo "All scripts have finished running." | tee -a $LOG_FILE
