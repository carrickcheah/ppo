# PPO Production Scheduler - MAKEFILE
.PHONY: help run backend frontend install clean test

# Default target
help:
	@echo "PPO Production Scheduler Commands:"
	@echo "  make run       - Start both backend and frontend"
	@echo "  make backend   - Start backend API only"
	@echo "  make frontend  - Start frontend only"
	@echo "  make install   - Install all dependencies"
	@echo "  make test      - Run tests"
	@echo "  make clean     - Clean up processes"

# Run both backend and frontend
run:
	@echo "Starting PPO Production Scheduler..."
	@echo "Backend API: http://localhost:8000"
	@echo "Frontend: http://localhost:3000"
	@echo "API Docs: http://localhost:8000/docs"
	@echo "----------------------------------------"
	@echo "Checking if backend is already running..."
	@if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then \
		echo "Backend already running on port 8000"; \
	else \
		echo "Starting backend..."; \
		cd app && uv run python scripts/run_api_server.py & \
	fi
	@sleep 2
	@echo "Starting frontend..."
	@cd frontend && npm start

# Start backend only
backend:
	@echo "Starting Backend API..."
	@cd app && uv run python scripts/run_api_server.py

# Start frontend only  
frontend:
	@echo "Starting Frontend..."
	@cd frontend && npm start

# Install dependencies
install:
	@echo "Installing backend dependencies..."
	@cd app && uv sync
	@echo "Installing frontend dependencies..."
	@cd frontend && npm install

# Run tests
test:
	@echo "Running backend tests..."
	@cd app && uv run pytest tests/
	@echo "Running API integration tests..."
	@cd app && uv run python scripts/test_api_integration.py

# Clean up running processes
clean:
	@echo "Stopping all services..."
	@pkill -f "run_api_server.py" || true
	@pkill -f "vite" || true
	@pkill -f "npm start" || true
	@echo "All services stopped"