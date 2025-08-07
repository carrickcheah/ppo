#!/bin/bash

# Run FastAPI server for PPO scheduling

echo "Starting PPO Scheduling API..."
echo "================================"

# Navigate to app3 directory
cd /Users/carrickcheah/Project/ppo/app3

# Activate virtual environment if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "Virtual environment activated"
fi

# Install required packages if not installed
echo "Checking dependencies..."
uv add fastapi uvicorn pydantic

# Start the API server
echo "Starting FastAPI server on http://localhost:8000"
echo "API documentation available at http://localhost:8000/docs"
echo "================================"

uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000