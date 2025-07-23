"""
Pydantic Models for API Request/Response Validation

This module defines all data models used in the PPO scheduler API
for request validation, response formatting, and internal data structures.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class JobStatus(str, Enum):
    """Enumeration of possible job statuses"""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class MachineStatus(str, Enum):
    """Enumeration of possible machine statuses"""
    AVAILABLE = "available"
    BUSY = "busy"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


class Job(BaseModel):
    """
    Individual job to be scheduled.
    
    This model represents a single job/task that needs to be scheduled
    on one of the available machines.
    """
    job_id: str = Field(
        ...,
        description="Unique identifier for the job (e.g., JOST25060240_CM17-002-1)",
        example="JOST25060240_CM17-002-1"
    )
    family_id: str = Field(
        ...,
        description="Family/workorder ID this job belongs to",
        example="JOST25060240"
    )
    sequence: int = Field(
        ...,
        ge=1,
        description="Sequence number within the family (1-based)",
        example=1
    )
    processing_time: float = Field(
        ...,
        gt=0,
        description="Processing time in hours",
        example=2.5
    )
    machine_types: List[int] = Field(
        ...,
        min_items=1,
        description="List of compatible machine type IDs",
        example=[1, 2, 5]
    )
    is_important: bool = Field(
        default=False,
        description="Whether this is a high-priority job",
        example=True
    )
    lcd_date: datetime = Field(
        ...,
        description="Latest completion date (deadline)",
        example="2025-07-25T17:00:00"
    )
    setup_time: Optional[float] = Field(
        default=0.3,
        ge=0,
        description="Setup time required in hours",
        example=0.3
    )
    
    @validator("processing_time", "setup_time")
    def validate_positive_time(cls, v):
        if v is not None and v < 0:
            raise ValueError("Time values must be non-negative")
        return v


class Machine(BaseModel):
    """
    Machine information for scheduling context.
    """
    machine_id: int = Field(
        ...,
        description="Unique machine ID from database",
        example=42
    )
    machine_name: str = Field(
        ...,
        description="Machine name/code",
        example="CM03"
    )
    machine_type: int = Field(
        ...,
        description="Machine type ID",
        example=2
    )
    status: MachineStatus = Field(
        default=MachineStatus.AVAILABLE,
        description="Current machine status"
    )
    current_load: Optional[float] = Field(
        default=0.0,
        ge=0,
        description="Current workload in hours",
        example=5.5
    )


class ScheduleRequest(BaseModel):
    """
    Request model for creating a new schedule.
    
    This is the main input model for the /schedule endpoint.
    """
    jobs: List[Job] = Field(
        default=[],
        min_items=0,
        description="List of jobs to be scheduled (empty list loads from database)"
    )
    machines: Optional[List[Machine]] = Field(
        default=None,
        description="Optional list of available machines (if not provided, fetched from DB)"
    )
    schedule_start: datetime = Field(
        default_factory=datetime.now,
        description="Start time for the schedule",
        example="2025-07-21T06:30:00"
    )
    respect_break_times: bool = Field(
        default=True,
        description="Whether to respect break time constraints"
    )
    respect_holidays: bool = Field(
        default=True,
        description="Whether to respect holiday constraints"
    )
    optimization_objective: str = Field(
        default="makespan",
        description="Primary optimization objective",
        example="makespan"
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Optional request ID for tracking"
    )
    save_to_database: bool = Field(
        default=False,
        description="Whether to save the generated schedule to database"
    )
    
    class Config:
        extra = "ignore"
        schema_extra = {
            "example": {
                "jobs": [
                    {
                        "job_id": "JOST25060240_CM17-002-1",
                        "family_id": "JOST25060240",
                        "sequence": 1,
                        "processing_time": 2.5,
                        "machine_types": [1, 2],
                        "is_important": True,
                        "lcd_date": "2025-07-25T17:00:00"
                    }
                ],
                "schedule_start": "2025-07-21T06:30:00",
                "respect_break_times": True,
                "save_to_database": True
            }
        }


class ScheduledJob(BaseModel):
    """
    A job that has been scheduled to a specific machine and time.
    """
    job_id: str = Field(
        ...,
        description="Job ID that was scheduled"
    )
    machine_id: int = Field(
        ...,
        description="Machine ID where job is scheduled"
    )
    machine_name: str = Field(
        ...,
        description="Machine name for display"
    )
    start_time: float = Field(
        ...,
        description="Start time (hours from schedule start)",
        example=2.5
    )
    end_time: float = Field(
        ...,
        description="End time (hours from schedule start)",
        example=5.0
    )
    start_datetime: datetime = Field(
        ...,
        description="Absolute start datetime"
    )
    end_datetime: datetime = Field(
        ...,
        description="Absolute end datetime"
    )
    setup_time_included: float = Field(
        default=0.0,
        description="Setup time included in this schedule slot"
    )


class ScheduleMetrics(BaseModel):
    """
    Performance metrics for the generated schedule.
    """
    makespan: float = Field(
        ...,
        description="Total time to complete all jobs (hours)",
        example=49.2
    )
    total_jobs: int = Field(
        ...,
        description="Total number of jobs scheduled",
        example=172
    )
    scheduled_jobs: int = Field(
        ...,
        description="Number of successfully scheduled jobs",
        example=172
    )
    completion_rate: float = Field(
        ...,
        description="Percentage of jobs successfully scheduled",
        example=100.0
    )
    average_utilization: float = Field(
        ...,
        description="Average machine utilization percentage",
        example=75.5
    )
    total_setup_time: float = Field(
        ...,
        description="Total setup time across all jobs",
        example=15.3
    )
    important_jobs_on_time: float = Field(
        ...,
        description="Percentage of important jobs meeting deadlines",
        example=98.5
    )


class ScheduleResponse(BaseModel):
    """
    Response model for a successful scheduling request.
    """
    schedule_id: str = Field(
        ...,
        description="Unique identifier for this schedule",
        example="sched_20250721_093045_abc123"
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Echo of the request ID if provided"
    )
    scheduled_jobs: List[ScheduledJob] = Field(
        ...,
        description="List of all scheduled jobs with assignments"
    )
    metrics: ScheduleMetrics = Field(
        ...,
        description="Performance metrics for the schedule"
    )
    generation_time: float = Field(
        ...,
        description="Time taken to generate schedule (seconds)",
        example=0.845
    )
    algorithm_used: str = Field(
        ...,
        description="Algorithm used for scheduling",
        example="ppo_full_production"
    )
    warnings: Optional[List[str]] = Field(
        default=None,
        description="Any warnings or issues encountered"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "schedule_id": "sched_20250721_093045_abc123",
                "scheduled_jobs": [
                    {
                        "job_id": "JOST25060240_CM17-002-1",
                        "machine_id": 42,
                        "machine_name": "CM03",
                        "start_time": 0.0,
                        "end_time": 2.5,
                        "start_datetime": "2025-07-21T06:30:00",
                        "end_datetime": "2025-07-21T09:00:00"
                    }
                ],
                "metrics": {
                    "makespan": 49.2,
                    "total_jobs": 172,
                    "scheduled_jobs": 172,
                    "completion_rate": 100.0,
                    "average_utilization": 75.5,
                    "total_setup_time": 15.3,
                    "important_jobs_on_time": 98.5
                },
                "generation_time": 0.845,
                "algorithm_used": "ppo_full_production"
            }
        }


class ErrorResponse(BaseModel):
    """
    Standard error response model.
    """
    error: str = Field(
        ...,
        description="Error type/code",
        example="INVALID_REQUEST"
    )
    message: str = Field(
        ...,
        description="Human-readable error message",
        example="Invalid job data: processing_time must be positive"
    )
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Request ID for tracking"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Error timestamp"
    )


class HealthResponse(BaseModel):
    """
    Health check response model.
    """
    status: str = Field(
        ...,
        description="Overall system status",
        example="healthy"
    )
    version: str = Field(
        ...,
        description="API version",
        example="1.0.0"
    )
    model_loaded: bool = Field(
        ...,
        description="Whether PPO model is loaded",
        example=True
    )
    database_connected: bool = Field(
        ...,
        description="Whether database connection is active",
        example=True
    )
    uptime: float = Field(
        ...,
        description="API uptime in seconds",
        example=3600.5
    )
    last_schedule_time: Optional[datetime] = Field(
        default=None,
        description="Timestamp of last successful schedule"
    )
    environment: str = Field(
        ...,
        description="Deployment environment",
        example="production"
    )