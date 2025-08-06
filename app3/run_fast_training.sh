#!/bin/bash

# Fast training script optimized for M4 Pro
# Reduces computational load while maintaining learning quality

echo "Starting optimized training for M4 Pro..."
echo "Using reduced timesteps and batch sizes for faster iteration"

uv run python src/training/curriculum_trainer.py \
    --device mps \
    --batch-size 64 \
    --n-steps 512 \
    --lr 1e-3

echo "Training complete!"