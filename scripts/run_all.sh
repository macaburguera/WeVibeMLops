#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Single log file for all outputs
LOG_FILE="logs/training_session.log"

# Clear the log file before running
> $LOG_FILE

# Function to run a command and exit on failure
run_and_check() {
    echo "Running: $1" | tee -a $LOG_FILE
    eval $1 >> $LOG_FILE 2>&1
    if [ $? -ne 0 ]; then
        echo "Error: Command failed - $1" | tee -a $LOG_FILE
        exit 1
    fi
    echo "Finished: $1" | tee -a $LOG_FILE
}

# Run scripts in sequence
run_and_check "python src/dog_breed_classifier/data.py"
run_and_check "python -u src/dog_breed_classifier/model.py"
run_and_check "python -u src/dog_breed_classifier/train.py local"

# Completion message
echo "All scripts have finished running successfully." | tee -a $LOG_FILE
