"""
FastAPI application for PPO scheduling service.
Provides REST API endpoints for scheduling production jobs.
"""

import os
import sys
from typing import Dict, List, Optional
from datetime import datetime
import json

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.scheduling_env import SchedulingEnv
from models.ppo_scheduler import PPOScheduler
from data.snapshot_loader import SnapshotLoader

# Initialize FastAPI app
app = FastAPI(
    title="PPO Scheduler API",
    description="Production scheduling using trained PPO model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
ppo_scheduler = None
model_path = "checkpoints/fast/model_40jobs.pth"


class ScheduleRequest(BaseModel):
    """Request model for scheduling endpoint."""
    families: Dict = Field(..., description="Job families with tasks")
    machines: List[str] = Field(..., description="Available machines")
    horizon_days: int = Field(default=30, description="Planning horizon in days")
    
    class Config:
        json_schema_extra = {
            "example": {
                "families": {
                    "JOB001": {
                        "lcd_date": "2025-08-15",
                        "tasks": [
                            {
                                "sequence": 1,
                                "process_name": "Process A",
                                "processing_time": 4.5,
                                "assigned_machine": "Machine1"
                            }
                        ]
                    }
                },
                "machines": ["Machine1", "Machine2"]
            }
        }


class ScheduleResponse(BaseModel):
    """Response model for scheduling endpoint."""
    schedule: List[Dict] = Field(..., description="Scheduled tasks")
    metrics: Dict = Field(..., description="Performance metrics")
    success: bool = Field(..., description="Whether scheduling succeeded")
    message: str = Field(default="", description="Status message")


class ModelInfo(BaseModel):
    """Model information response."""
    model_loaded: bool
    model_path: str
    capacity: str
    performance: Dict


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global ppo_scheduler
    
    if os.path.exists(model_path):
        # Create dummy env to get dimensions
        dummy_env = SchedulingEnv("data/40_jobs.json")
        
        # Load model
        ppo_scheduler = PPOScheduler(
            obs_dim=dummy_env.observation_space.shape[0],
            action_dim=dummy_env.action_space.n,
            device="mps"
        )
        ppo_scheduler.load(model_path)
        
        print(f"Model loaded from {model_path}")
    else:
        print(f"Warning: Model not found at {model_path}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "PPO Scheduler API",
        "status": "running",
        "model_loaded": ppo_scheduler is not None
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": ppo_scheduler is not None
    }


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about loaded model."""
    return ModelInfo(
        model_loaded=ppo_scheduler is not None,
        model_path=model_path,
        capacity="Up to 40 job families (127 tasks)",
        performance={
            "completion_rate": "92.9%",
            "average_reward": 27150,
            "training_time": "54 seconds",
            "inference_time": "<1 second"
        }
    )


@app.post("/schedule", response_model=ScheduleResponse)
async def schedule_jobs(request: ScheduleRequest):
    """
    Schedule production jobs using trained PPO model.
    
    Args:
        request: ScheduleRequest with families and machines
        
    Returns:
        ScheduleResponse with scheduled tasks and metrics
    """
    if ppo_scheduler is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Save request to temporary file
        temp_file = "/tmp/schedule_request.json"
        with open(temp_file, 'w') as f:
            json.dump(request.dict(), f)
        
        # Create environment
        env = SchedulingEnv(temp_file, max_steps=1500)
        
        # Check capacity
        if env.n_tasks > 127:
            raise HTTPException(
                status_code=400,
                detail=f"Too many tasks ({env.n_tasks}). Model supports up to 127 tasks."
            )
        
        # Run scheduling
        obs, info = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 1500:
            action_mask = info['action_mask']
            action, _ = ppo_scheduler.predict(obs, action_mask, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        
        # Extract schedule
        schedule = []
        for machine_id, tasks in env.machine_schedules.items():
            for task in tasks:
                schedule.append({
                    "task_id": task['task_idx'],
                    "family_id": task['family_id'],
                    "sequence": task['sequence'],
                    "machine": env.loader.machines[machine_id],
                    "start_time": task['start_time'],
                    "end_time": task['end_time'],
                    "processing_time": task['processing_time']
                })
        
        # Sort by start time
        schedule.sort(key=lambda x: x['start_time'])
        
        # Calculate metrics
        completion_rate = info['tasks_scheduled'] / info['total_tasks']
        
        metrics = {
            "total_tasks": info['total_tasks'],
            "tasks_scheduled": info['tasks_scheduled'],
            "completion_rate": f"{completion_rate * 100:.1f}%",
            "makespan": max([s['end_time'] for s in schedule]) if schedule else 0,
            "steps_taken": steps
        }
        
        return ScheduleResponse(
            schedule=schedule,
            metrics=metrics,
            success=True,
            message=f"Successfully scheduled {info['tasks_scheduled']} out of {info['total_tasks']} tasks"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Scheduling failed: {str(e)}"
        )


@app.post("/schedule/file")
async def schedule_from_file(file: UploadFile = File(...)):
    """
    Schedule jobs from uploaded JSON file.
    
    Args:
        file: JSON file with job data
        
    Returns:
        ScheduleResponse with scheduled tasks
    """
    if ppo_scheduler is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    try:
        # Read file
        contents = await file.read()
        data = json.loads(contents)
        
        # Create request
        request = ScheduleRequest(
            families=data.get('families', {}),
            machines=data.get('machines', [])
        )
        
        # Use main scheduling endpoint
        return await schedule_jobs(request)
        
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON file"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )


@app.post("/model/reload")
async def reload_model():
    """Reload the model from disk."""
    global ppo_scheduler
    
    try:
        if os.path.exists(model_path):
            dummy_env = SchedulingEnv("data/40_jobs.json")
            ppo_scheduler = PPOScheduler(
                obs_dim=dummy_env.observation_space.shape[0],
                action_dim=dummy_env.action_space.n,
                device="mps"
            )
            ppo_scheduler.load(model_path)
            
            return {
                "success": True,
                "message": f"Model reloaded from {model_path}"
            }
        else:
            return {
                "success": False,
                "message": f"Model file not found at {model_path}"
            }
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model reload failed: {str(e)}"
        )


if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )