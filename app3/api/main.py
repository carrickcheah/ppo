"""
FastAPI application for PPO scheduling visualization.
"""

import os
import sys
from pathlib import Path
from typing import List
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.models import (
    ScheduleRequest, ScheduleResponse,
    DatasetsResponse, ModelsResponse,
    DatasetInfo, ModelInfo, ErrorResponse,
    DatasetType, ModelType
)
from api.scheduler import scheduler_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="PPO Scheduling API",
    description="API for scheduling jobs using trained PPO models",
    version="1.0.0"
)

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "PPO Scheduling API",
        "endpoints": {
            "schedule": "/api/schedule",
            "datasets": "/api/datasets",
            "models": "/api/models"
        }
    }


@app.post("/api/schedule", response_model=ScheduleResponse)
async def schedule_jobs(request: ScheduleRequest):
    """
    Schedule jobs using a trained PPO model.
    
    Args:
        request: Scheduling request with dataset and model selection
        
    Returns:
        ScheduleResponse with job allocations, machine allocations, and statistics
    """
    try:
        logger.info(f"Scheduling {request.dataset} with {request.model}")
        
        # Run scheduling
        result = scheduler_service.schedule_jobs(
            dataset_type=request.dataset,
            model_type=request.model,
            deterministic=request.deterministic,
            max_steps=request.max_steps
        )
        
        # Create response
        response = ScheduleResponse(
            success=True,
            message=f"Successfully scheduled {len(result['jobs'])} tasks",
            jobs=result['jobs'],
            machines=result['machines'],
            statistics=result['statistics'],
            dataset_used=request.dataset.value,
            model_used=request.model.value
        )
        
        logger.info(f"Scheduling complete: {result['statistics'].completion_rate:.1f}% completion")
        return response
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except NotImplementedError as e:
        logger.error(f"Not implemented: {e}")
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        logger.error(f"Scheduling error: {e}")
        raise HTTPException(status_code=500, detail=f"Scheduling failed: {str(e)}")


@app.get("/api/datasets", response_model=DatasetsResponse)
async def get_datasets():
    """
    Get list of available datasets.
    
    Returns:
        DatasetsResponse with dataset information
    """
    base_path = Path("/Users/carrickcheah/Project/ppo/app3/data")
    
    datasets = []
    dataset_info = {
        DatasetType.JOBS_10: {"tasks": 34, "families": 10},
        DatasetType.JOBS_20: {"tasks": 65, "families": 20},
        DatasetType.JOBS_40: {"tasks": 130, "families": 40},
        DatasetType.JOBS_60: {"tasks": 195, "families": 60},
        DatasetType.JOBS_100: {"tasks": 327, "families": 100}
    }
    
    for dataset_type, info in dataset_info.items():
        file_path = base_path / f"{dataset_type.value}.json"
        
        if file_path.exists():
            datasets.append(DatasetInfo(
                name=dataset_type.value,
                total_tasks=info["tasks"],
                total_families=info["families"],
                file_path=str(file_path)
            ))
    
    return DatasetsResponse(datasets=datasets)


@app.get("/api/models", response_model=ModelsResponse)
async def get_models():
    """
    Get list of available PPO models.
    
    Returns:
        ModelsResponse with model information
    """
    base_path = Path("/Users/carrickcheah/Project/ppo/app3/checkpoints")
    
    models = []
    model_info = {
        ModelType.SB3_1M: {
            "type": "SB3",
            "path": "sb3_1million/best_model.zip",
            "steps": 1000000,
            "performance": {"efficiency": "TBD", "completion": "TBD"}
        },
        ModelType.SB3_500K: {
            "type": "SB3",
            "path": "sb3_500k/best_model.zip",
            "steps": 500000,
            "performance": {"efficiency": "TBD", "completion": "TBD"}
        },
        ModelType.SB3_100X: {
            "type": "SB3",
            "path": "sb3_100x/best_model.zip",
            "steps": 100000,
            "performance": {"efficiency": "TBD", "completion": "TBD"}
        },
        ModelType.SB3_OPTIMIZED: {
            "type": "SB3",
            "path": "sb3_optimized/best_model.zip",
            "steps": 25000,
            "performance": {"efficiency": "8.9%", "completion": "100%"}
        }
    }
    
    for model_type, info in model_info.items():
        checkpoint_path = base_path / info["path"]
        
        if checkpoint_path.exists():
            models.append(ModelInfo(
                name=model_type.value,
                type=info["type"],
                checkpoint_path=str(checkpoint_path),
                training_steps=info["steps"],
                performance=info["performance"]
            ))
    
    return ModelsResponse(models=models)


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc),
            status_code=exc.status_code
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            status_code=500
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)