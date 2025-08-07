#!/bin/bash

# Run React frontend for PPO scheduling visualization

echo "Starting PPO Scheduling Frontend..."
echo "================================"

# Navigate to frontend3 directory
cd /Users/carrickcheah/Project/ppo/frontend3

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Start the development server
echo "Starting React development server on http://localhost:5173"
echo "================================"
echo ""
echo "Make sure the backend API is running at http://localhost:8000"
echo "Run: cd app3 && ./run_api.sh"
echo ""

npm run dev