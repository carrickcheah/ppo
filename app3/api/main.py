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
            model_name=request.model,
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
            model_used=request.model
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
        DatasetType.JOBS_80: {"tasks": 260, "families": 80},
        DatasetType.JOBS_100: {"tasks": 327, "families": 100},
        DatasetType.JOBS_150: {"tasks": 490, "families": 150},
        DatasetType.JOBS_180: {"tasks": 588, "families": 180},
        DatasetType.JOBS_200: {"tasks": 653, "families": 200},
        DatasetType.JOBS_250: {"tasks": 816, "families": 250},
        DatasetType.JOBS_300: {"tasks": 980, "families": 300},
        DatasetType.JOBS_330: {"tasks": 1078, "families": 330},
        DatasetType.JOBS_380: {"tasks": 1241, "families": 380},
        DatasetType.JOBS_400: {"tasks": 1306, "families": 400},
        DatasetType.JOBS_430: {"tasks": 1404, "families": 430},
        DatasetType.JOBS_450: {"tasks": 1469, "families": 450},
        DatasetType.JOBS_500: {"tasks": 1633, "families": 500}
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
    Get list of available PPO models by auto-detecting from checkpoints directory.
    
    Returns:
        ModelsResponse with model information
    """
    base_path = Path("/Users/carrickcheah/Project/ppo/app3/checkpoints")
    
    models = []
    
    # Auto-detect all models with best_model.zip
    if base_path.exists():
        for model_dir in base_path.iterdir():
            if model_dir.is_dir():
                best_model_path = model_dir / "best_model.zip"
                
                # Check for best_model.zip in root of directory
                if best_model_path.exists():
                    # Generate display name from directory name
                    dir_name = model_dir.name
                    display_name = dir_name.replace("_", " ").title()
                    
                    # Determine model type
                    model_type = "SB3" if "sb3" in dir_name.lower() else "Custom"
                    
                    # Estimate training steps from name if available
                    steps = None
                    if "1million" in dir_name or "1m" in dir_name.lower():
                        steps = 1000000
                    elif "500k" in dir_name:
                        steps = 500000
                    elif "100x" in dir_name:
                        steps = 100000
                    elif "10x" in dir_name:
                        steps = 10000
                    elif "optimized" in dir_name:
                        steps = 25000
                    
                    models.append(ModelInfo(
                        name=dir_name,
                        type=model_type,
                        checkpoint_path=str(best_model_path),
                        training_steps=steps,
                        performance={"efficiency": "Auto-detected", "completion": "Auto-detected"}
                    ))
                
                # Also check for nested best_model.zip (e.g., in stage_1 subdirectory)
                for subdir in model_dir.iterdir():
                    if subdir.is_dir():
                        nested_best_model = subdir / "best_model.zip"
                        if nested_best_model.exists():
                            # Create name with subdirectory
                            nested_name = f"{model_dir.name}/{subdir.name}"
                            display_name = nested_name.replace("_", " ").title()
                            
                            models.append(ModelInfo(
                                name=nested_name,
                                type="SB3" if "sb3" in model_dir.name.lower() else "Custom",
                                checkpoint_path=str(nested_best_model),
                                training_steps=None,
                                performance={"efficiency": "Auto-detected", "completion": "Auto-detected"}
                            ))
    
    # Sort models by name for consistent ordering
    models.sort(key=lambda x: x.name)
    
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