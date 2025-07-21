#!/usr/bin/env python3
"""
API Server Startup Script

This script starts the FastAPI server for the PPO Production Scheduler API.
It handles proper configuration loading and uvicorn server setup.
"""

import sys
import os

# Add the app directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from src.deployment.settings import get_settings


def main():
    """
    Main entry point for starting the API server.
    """
    # Load settings
    settings = get_settings()
    
    # Configure uvicorn
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["default"]["fmt"] = settings.log_format
    log_config["formatters"]["access"]["fmt"] = settings.log_format
    
    # Determine reload setting based on environment
    reload = settings.environment == "development"
    
    print(f"Starting PPO Scheduler API Server")
    print(f"Environment: {settings.environment}")
    print(f"Host: {settings.api_host}")
    print(f"Port: {settings.api_port}")
    print(f"Reload: {reload}")
    print(f"API Docs: http://{settings.api_host}:{settings.api_port}/docs")
    print("-" * 50)
    
    # Run the server
    uvicorn.run(
        "src.deployment.api_server:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=reload,
        log_level=settings.log_level.lower(),
        log_config=log_config,
        access_log=True
    )


if __name__ == "__main__":
    main()