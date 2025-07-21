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
    HealthResponse, ErrorResponse, ScheduleMetrics
)

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and app state
ppo_model = None
app_start_time = None
last_schedule_time = None
db_connection = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle - load model at startup, cleanup at shutdown.
    """
    global ppo_model, app_start_time
    
    settings = get_settings()
    logger.info(f"Starting PPO Scheduler API in {settings.environment} mode")
    
    try:
        # Load the PPO model
        logger.info(f"Loading PPO model from {settings.model_path}")
        ppo_model = sb3.PPO.load(settings.model_path)
        logger.info("PPO model loaded successfully")
        
        # Initialize database connection
        # TODO: Implement database connection
        
        app_start_time = time.time()
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise
    finally:
        # Cleanup on shutdown
        logger.info("Shutting down PPO Scheduler API")
        if db_connection:
            # TODO: Close database connection
            pass


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
        # TODO: Implement actual database health check
        db_connected = True
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
        # Validate model is loaded
        if not ppo_model:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded - service unavailable"
            )
        
        # TODO: Implement actual scheduling logic
        # For now, create a mock response
        
        # Mock scheduled jobs
        scheduled_jobs = []
        current_time = 0.0
        
        for job in request.jobs[:5]:  # Mock scheduling first 5 jobs
            scheduled_job = ScheduledJob(
                job_id=job.job_id,
                machine_id=1,  # Mock machine assignment
                machine_name="CM03",  # Mock machine name
                start_time=current_time,
                end_time=current_time + job.processing_time,
                start_datetime=request.schedule_start,
                end_datetime=request.schedule_start
            )
            scheduled_jobs.append(scheduled_job)
            current_time += job.processing_time + 0.5  # Add setup time
        
        # Calculate metrics
        makespan = current_time if scheduled_jobs else 0.0
        scheduled_count = len(scheduled_jobs)
        total_count = len(request.jobs)
        completion_rate = (scheduled_count / total_count * 100) if total_count > 0 else 0.0
        
        metrics = ScheduleMetrics(
            makespan=makespan,
            total_jobs=total_count,
            scheduled_jobs=scheduled_count,
            completion_rate=completion_rate,
            average_utilization=75.5,  # Mock value
            total_setup_time=0.5 * scheduled_count,  # Mock value
            important_jobs_on_time=95.0  # Mock value
        )
        
        # Record successful schedule time
        last_schedule_time = datetime.now()
        generation_time = time.time() - start_time
        
        logger.info(f"Schedule {schedule_id} created successfully in {generation_time:.3f}s")
        
        return ScheduleResponse(
            schedule_id=schedule_id,
            request_id=request.request_id,
            scheduled_jobs=scheduled_jobs,
            metrics=metrics,
            generation_time=generation_time,
            algorithm_used="ppo_full_production",
            warnings=["This is a mock implementation - actual PPO scheduling not yet connected"]
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
            timestamp=datetime.now()
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
            timestamp=datetime.now()
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