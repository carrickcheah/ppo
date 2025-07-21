#!/usr/bin/env python3
"""
Production API server for PPO Scheduler
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

import uvicorn
from src.deployment.api_server import app
from src.utils.logger import setup_logger

# Setup logging
logger = setup_logger("api_server", "logs/api_server.log")

def parse_args():
    parser = argparse.ArgumentParser(description="Run PPO Scheduler API Server")
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("API_HOST", "0.0.0.0"),
        help="Host to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("API_PORT", "8000")),
        help="Port to bind to"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("API_WORKERS", "1")),
        help="Number of worker processes"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development only)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.getenv("LOG_LEVEL", "info"),
        choices=["debug", "info", "warning", "error"],
        help="Logging level"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    logger.info("Starting PPO Scheduler API Server")
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Log Level: {args.log_level}")
    
    # Configure uvicorn
    config = {
        "app": "src.deployment.api_server:app",
        "host": args.host,
        "port": args.port,
        "log_level": args.log_level.lower(),
        "access_log": True,
        "use_colors": True
    }
    
    # Add development options
    if args.reload:
        config["reload"] = True
        config["workers"] = 1  # Reload doesn't work with multiple workers
        logger.info("Auto-reload enabled (development mode)")
    else:
        config["workers"] = args.workers
    
    # Start server
    try:
        uvicorn.run(**config)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise

if __name__ == "__main__":
    main()