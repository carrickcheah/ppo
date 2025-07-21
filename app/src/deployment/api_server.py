"""
FastAPI Application for PPO Production Scheduler

This module implements the main API server that exposes the trained PPO model
for production scheduling via REST endpoints.
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import stable_baselines3 as sb3
import numpy as np

from .settings import get_settings, APISettings
from .models import (
    ScheduleRequest, ScheduleResponse, ScheduledJob,
    HealthResponse, ErrorResponse, ScheduleMetrics, Machine
)
from .scheduler import PPOScheduler
from .database import get_database

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and app state
ppo_model = None
ppo_scheduler = None
app_start_time = None
last_schedule_time = None
db_connection = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle - load model at startup, cleanup at shutdown.
    """
    global ppo_model, ppo_scheduler, app_start_time
    
    settings = get_settings()
    logger.info(f"Starting PPO Scheduler API in {settings.environment} mode")
    
    try:
        # Load the PPO model
        logger.info(f"Loading PPO model from {settings.model_path}")
        ppo_model = sb3.PPO.load(settings.model_path)
        logger.info("PPO model loaded successfully")
        
        # Initialize PPO scheduler - no fallback
        ppo_scheduler = PPOScheduler(ppo_model)
        logger.info("PPO scheduler initialized successfully")
        
        # Initialize database connection
        db_connection = get_database()
        if not db_connection.test_connection():
            logger.error("Failed to connect to database")
            raise RuntimeError("Database connection failed")
        logger.info("Database connection established")
        
        app_start_time = time.time()
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise
    finally:
        # Cleanup on shutdown
        logger.info("Shutting down PPO Scheduler API")
        # Database connections are handled by context manager


# Create FastAPI application
settings = get_settings()
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    lifespan=lifespan,
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url="/redoc" if settings.environment != "production" else None
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
    allow_credentials=True
)


def verify_api_key(x_api_key: str = Header(...)) -> str:
    """
    Verify API key for authentication.
    
    Args:
        x_api_key: API key from request header
        
    Returns:
        str: The verified API key
        
    Raises:
        HTTPException: If API key is invalid
    """
    if x_api_key != settings.api_key:
        logger.warning(f"Invalid API key attempt: {x_api_key[:8]}...")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return x_api_key


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify system status.
    
    Returns system health information including model status,
    database connectivity, and uptime.
    """
    global app_start_time, last_schedule_time
    
    current_time = time.time()
    uptime = current_time - app_start_time if app_start_time else 0
    
    # Check database connection
    db_connected = False
    try:
        db_connected = db_connection.test_connection() if db_connection else False
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
    
    return HealthResponse(
        status="healthy" if ppo_model and db_connected else "degraded",
        version=settings.api_version,
        model_loaded=ppo_model is not None,
        database_connected=db_connected,
        uptime=uptime,
        last_schedule_time=last_schedule_time,
        environment=settings.environment
    )


@app.post("/schedule", response_model=ScheduleResponse)
async def create_schedule(
    request: ScheduleRequest,
    api_key: str = Depends(verify_api_key)
) -> ScheduleResponse:
    """
    Create a production schedule using the PPO model.
    
    This endpoint takes a list of jobs and returns an optimized schedule
    with machine assignments and timing.
    
    Args:
        request: Schedule request containing jobs to be scheduled
        api_key: API key for authentication (injected by dependency)
        
    Returns:
        ScheduleResponse: The generated schedule with metrics
        
    Raises:
        HTTPException: If scheduling fails or times out
    """
    global last_schedule_time
    
    start_time = time.time()
    schedule_id = f"sched_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    
    logger.info(f"Creating schedule {schedule_id} with {len(request.jobs)} jobs")
    
    try:
        # Use PPO scheduler only - no fallback
        if not ppo_scheduler:
            raise HTTPException(
                status_code=503,
                detail="PPO scheduler not initialized - service unavailable"
            )
        
        # Get machines from request or load from database
        if request.machines:
            machines = request.machines
        else:
            # Load real machines from database
            try:
                machines = db_connection.get_machines()
                logger.info(f"Loaded {len(machines)} machines from database")
            except Exception as e:
                logger.error(f"Failed to load machines from database: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to load machines: {str(e)}"
                )
        
        # Use PPO scheduler to generate schedule
        scheduled_jobs, metrics_dict = ppo_scheduler.schedule(
            jobs=request.jobs,
            machines=machines,
            schedule_start=request.schedule_start
        )
        
        # Create metrics object
        metrics = ScheduleMetrics(
            makespan=metrics_dict['makespan'],
            total_jobs=metrics_dict['total_jobs'],
            scheduled_jobs=metrics_dict['scheduled_jobs'],
            completion_rate=metrics_dict['completion_rate'],
            average_utilization=metrics_dict['average_utilization'],
            total_setup_time=metrics_dict['total_setup_time'],
            important_jobs_on_time=metrics_dict['important_jobs_on_time']
        )
        
        # Record successful schedule time
        last_schedule_time = datetime.now()
        generation_time = time.time() - start_time
        
        logger.info(f"Schedule {schedule_id} created successfully in {generation_time:.3f}s using PPO scheduler")
        
        # Save schedule to database if requested
        if request.save_to_database:
            try:
                success = db_connection.save_schedule(
                    schedule_id=schedule_id,
                    scheduled_jobs=[job.dict() for job in scheduled_jobs],
                    metrics=metrics_dict
                )
                if success:
                    logger.info(f"Schedule {schedule_id} saved to database")
                else:
                    logger.warning(f"Failed to save schedule {schedule_id} to database")
            except Exception as e:
                logger.error(f"Database save error: {str(e)}")
        
        # Prepare warnings
        warnings = []
        if not request.machines:
            warnings.append("Using real machines from database")
        if metrics.completion_rate < 100:
            warnings.append(f"Only {metrics.scheduled_jobs}/{metrics.total_jobs} jobs could be scheduled")
        
        return ScheduleResponse(
            schedule_id=schedule_id,
            request_id=request.request_id,
            scheduled_jobs=scheduled_jobs,
            metrics=metrics,
            generation_time=generation_time,
            algorithm_used="ppo_full_production",
            warnings=warnings if warnings else None
        )
        
    except Exception as e:
        logger.error(f"Failed to create schedule {schedule_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Scheduling failed: {str(e)}"
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Custom exception handler for HTTP exceptions.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP_{exc.status_code}",
            message=exc.detail,
            timestamp=datetime.now().isoformat()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    General exception handler for unexpected errors.
    """
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    
    # In production, don't expose internal errors
    if settings.environment == "production":
        message = "An internal error occurred"
    else:
        message = str(exc)
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="INTERNAL_ERROR",
            message=message,
            timestamp=datetime.now().isoformat()
        ).dict()
    )


@app.get("/metrics")
async def get_metrics(api_key: str = Depends(verify_api_key)):
    """
    Prometheus-compatible metrics endpoint.
    
    Returns system metrics in Prometheus text format.
    """
    # TODO: Implement Prometheus metrics
    return {"message": "Metrics endpoint not yet implemented"}


if __name__ == "__main__":
    # This should not be run directly - use run_api_server.py instead
    logger.error("Please run the API using run_api_server.py, not directly")
    raise RuntimeError("Use run_api_server.py to start the API server")