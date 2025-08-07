"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from enum import Enum


class DatasetType(str, Enum):
    """Available dataset sizes."""
    JOBS_10 = "10_jobs"
    JOBS_20 = "20_jobs"
    JOBS_40 = "40_jobs"
    JOBS_60 = "60_jobs"
    JOBS_80 = "80_jobs"
    JOBS_100 = "100_jobs"
    JOBS_150 = "150_jobs"
    JOBS_180 = "180_jobs"
    JOBS_200 = "200_jobs"
    JOBS_250 = "250_jobs"
    JOBS_300 = "300_jobs"
    JOBS_330 = "330_jobs"
    JOBS_380 = "380_jobs"
    JOBS_400 = "400_jobs"
    JOBS_430 = "430_jobs"
    JOBS_450 = "450_jobs"
    JOBS_500 = "500_jobs"


class ModelType(str, Enum):
    """Available PPO models - dynamically detected from checkpoints."""
    # This enum is deprecated - models are now auto-detected
    # Keeping for backward compatibility
    SB3_1M = "sb3_1million"
    SB3_500K = "sb3_500k"
    SB3_100X = "sb3_100x"
    SB3_OPTIMIZED = "sb3_optimized"
    CUSTOM_10X = "10x"
    CUSTOM_FAST = "fast"


class ScheduleRequest(BaseModel):
    """Request model for scheduling endpoint."""
    dataset: DatasetType = Field(default=DatasetType.JOBS_40, description="Dataset to schedule")
    model: str = Field(default="sb3_1million", description="PPO model to use (auto-detected from checkpoints)")
    deterministic: bool = Field(default=True, description="Use deterministic policy")
    max_steps: int = Field(default=10000, description="Maximum scheduling steps")


class JobTask(BaseModel):
    """Individual job task in the schedule."""
    job_id: str = Field(..., description="Job family ID")
    task_label: str = Field(..., description="Task label (FAMILY_PROCESS_SEQ/TOTAL)")
    sequence: int = Field(..., description="Sequence number")
    start: float = Field(..., description="Start time in hours")
    end: float = Field(..., description="End time in hours")
    duration: float = Field(..., description="Duration in hours")
    machine: str = Field(..., description="Assigned machine")
    color: str = Field(..., description="Color based on deadline status")
    lcd_hours: float = Field(..., description="LCD deadline in hours")
    days_to_deadline: float = Field(..., description="Days until deadline")
    process_name: str = Field(..., description="Process name")


class MachineTask(BaseModel):
    """Task allocated to a machine."""
    machine_id: str = Field(..., description="Machine ID")
    machine_name: str = Field(..., description="Machine name")
    tasks: List[JobTask] = Field(default_factory=list, description="Tasks on this machine")
    utilization: float = Field(..., description="Machine utilization percentage")
    total_busy_time: float = Field(..., description="Total busy time in hours")


class ScheduleStatistics(BaseModel):
    """Scheduling performance statistics."""
    total_tasks: int = Field(..., description="Total number of tasks")
    scheduled_tasks: int = Field(..., description="Number of scheduled tasks")
    completion_rate: float = Field(..., description="Completion percentage")
    on_time_tasks: int = Field(..., description="Tasks completed on time")
    late_tasks: int = Field(..., description="Tasks completed late")
    on_time_rate: float = Field(..., description="On-time delivery rate")
    average_tardiness: float = Field(..., description="Average tardiness in hours")
    makespan: float = Field(..., description="Total schedule duration in hours")
    machine_utilization: float = Field(..., description="Average machine utilization")
    total_reward: float = Field(..., description="Total episode reward")
    inference_time: float = Field(..., description="Time taken to schedule in seconds")


class ScheduleResponse(BaseModel):
    """Response model for scheduling endpoint."""
    success: bool = Field(..., description="Whether scheduling was successful")
    message: str = Field(..., description="Status message")
    jobs: List[JobTask] = Field(default_factory=list, description="Job allocation data")
    machines: List[MachineTask] = Field(default_factory=list, description="Machine allocation data")
    statistics: Optional[ScheduleStatistics] = Field(None, description="Performance statistics")
    dataset_used: str = Field(..., description="Dataset that was scheduled")
    model_used: str = Field(..., description="Model that was used")


class DatasetInfo(BaseModel):
    """Information about available datasets."""
    name: str = Field(..., description="Dataset name")
    total_tasks: int = Field(..., description="Number of tasks")
    total_families: int = Field(..., description="Number of job families")
    file_path: str = Field(..., description="Path to dataset file")


class ModelInfo(BaseModel):
    """Information about available models."""
    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Model type (SB3 or Custom)")
    checkpoint_path: str = Field(..., description="Path to model checkpoint")
    training_steps: Optional[int] = Field(None, description="Number of training steps")
    performance: Optional[Dict] = Field(None, description="Performance metrics")


class DatasetsResponse(BaseModel):
    """Response for datasets endpoint."""
    datasets: List[DatasetInfo] = Field(..., description="Available datasets")


class ModelsResponse(BaseModel):
    """Response for models endpoint."""
    models: List[ModelInfo] = Field(..., description="Available models")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    status_code: int = Field(..., description="HTTP status code")