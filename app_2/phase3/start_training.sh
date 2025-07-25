#!/bin/bash

# Phase 3 Training Launcher Script

echo "=================================="
echo "Phase 3: Curriculum Learning"
echo "=================================="

# Check if in correct directory
if [ ! -f "phase3/train_curriculum.py" ]; then
    echo "Error: Please run from app_2 directory"
    exit 1
fi

# Parse arguments
STAGE=""
SINGLE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --start-stage)
            STAGE="$2"
            shift 2
            ;;
        --single-stage)
            SINGLE="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Activate virtual environment if exists
if [ -f ".venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Start training
if [ ! -z "$SINGLE" ]; then
    echo "Training single stage: $SINGLE"
    uv run python phase3/train_curriculum.py --single-stage "$SINGLE"
elif [ ! -z "$STAGE" ]; then
    echo "Starting from stage: $STAGE"
    uv run python phase3/train_curriculum.py --start-stage "$STAGE"
else
    echo "Starting full curriculum training..."
    echo "This will train through 16 stages progressively."
    echo ""
    echo "Stages:"
    echo "1. Foundation (4 stages): 5-15 jobs"
    echo "2. Strategy (4 stages): 30-50 jobs"
    echo "3. Scale (4 stages): 150-400 jobs"
    echo "4. Production (4 stages): 295-500 jobs"
    echo ""
    echo "Total training time estimate: 4-6 hours on M4 Max"
    echo ""
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        uv run python phase3/train_curriculum.py
    else
        echo "Training cancelled"
        exit 0
    fi
fi

echo ""
echo "=================================="
echo "Training completed!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. View training progress: tensorboard --logdir phase3/tensorboard"
echo "2. Generate visualizations: uv run python phase3/visualize_training.py"
echo "3. Test model: uv run python phase3/test_model.py"