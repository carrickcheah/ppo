# Makefile for PPO Scheduling Visualization System

# Colors for output
RED=\033[0;31m
GREEN=\033[0;32m
YELLOW=\033[1;33m
NC=\033[0m # No Color

# Default target
.PHONY: help
help:
	@echo "$(GREEN)PPO Scheduling Visualization System$(NC)"
	@echo "Available commands:"
	@echo "  $(YELLOW)make run$(NC)    - Start both backend (port 8000) and frontend (port 5173)"
	@echo "  $(YELLOW)make down$(NC)   - Stop both backend and frontend services"
	@echo "  $(YELLOW)make backend$(NC) - Start only the backend server"
	@echo "  $(YELLOW)make frontend$(NC) - Start only the frontend server"
	@echo "  $(YELLOW)make status$(NC) - Check status of both services"
	@echo "  $(YELLOW)make logs$(NC)   - Show backend logs"
	@echo "  $(YELLOW)make clean$(NC)  - Clean up log files"

# Start both services
.PHONY: run
run:
	@echo "$(GREEN)Starting PPO Scheduling System...$(NC)"
	@echo "$(YELLOW)Starting backend on port 8000...$(NC)"
	@cd /Users/carrickcheah/Project/ppo/app3 && nohup uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload > api.log 2>&1 &
	@sleep 2
	@echo "$(YELLOW)Starting frontend on port 5173...$(NC)"
	@cd /Users/carrickcheah/Project/ppo/frontend3 && npm run dev > /dev/null 2>&1 &
	@sleep 3
	@echo "$(GREEN)Both services started successfully!$(NC)"
	@echo "Backend: http://localhost:8000"
	@echo "Frontend: http://localhost:5173"
	@echo "API Docs: http://localhost:8000/docs"

# Stop both services
.PHONY: down
down:
	@echo "$(RED)Stopping services...$(NC)"
	@echo "Killing processes on port 8000 (backend)..."
	@lsof -ti:8000 | xargs kill -9 2>/dev/null || echo "No process running on port 8000"
	@echo "Killing processes on port 5173 (frontend)..."
	@lsof -ti:5173 | xargs kill -9 2>/dev/null || echo "No process running on port 5173"
	@echo "$(GREEN)All services stopped.$(NC)"

# Start backend only
.PHONY: backend
backend:
	@echo "$(YELLOW)Starting backend server on port 8000...$(NC)"
	cd /Users/carrickcheah/Project/ppo/app3 && uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Start frontend only
.PHONY: frontend
frontend:
	@echo "$(YELLOW)Starting frontend server on port 5173...$(NC)"
	cd /Users/carrickcheah/Project/ppo/frontend3 && npm run dev

# Check status of services
.PHONY: status
status:
	@echo "$(GREEN)Service Status:$(NC)"
	@echo -n "Backend (port 8000): "
	@if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then \
		echo "$(GREEN)Running$(NC)"; \
	else \
		echo "$(RED)Not running$(NC)"; \
	fi
	@echo -n "Frontend (port 5173): "
	@if lsof -Pi :5173 -sTCP:LISTEN -t >/dev/null ; then \
		echo "$(GREEN)Running$(NC)"; \
	else \
		echo "$(RED)Not running$(NC)"; \
	fi

# Show backend logs
.PHONY: logs
logs:
	@echo "$(YELLOW)Backend logs:$(NC)"
	@tail -f api.log

# Clean up log files
.PHONY: clean
clean:
	@echo "$(YELLOW)Cleaning up log files...$(NC)"
	@rm -f api.log
	@echo "$(GREEN)Cleanup complete.$(NC)"

# Quick restart
.PHONY: restart
restart: down run
	@echo "$(GREEN)Services restarted successfully!$(NC)"