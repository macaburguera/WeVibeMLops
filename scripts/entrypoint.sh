#!/bin/bash

# Ensure WANDB_API_KEY is set if needed
if [ -z "$WANDB_API_KEY" ]; then
    echo "Warning: WANDB_API_KEY is not set. W&B logging may fail."
fi

# Pull the latest data from the DVC remote
echo "Pulling data from DVC remote..."
dvc pull --force

# Run the desired command
exec "$@"
