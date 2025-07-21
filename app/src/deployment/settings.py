"""
API Settings Configuration using pydantic-settings

This module defines all configuration settings for the PPO scheduler API
using pydantic-settings for type-safe environment variable handling.
"""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class APISettings(BaseSettings):
    """
    API configuration settings with environment variable support.
    
    All settings can be overridden via environment variables or .env file.
    """
    
    # Model configuration
    model_path: str = Field(
        default="models/full_production/final_model.zip",
        description="Path to the trained PPO model file"
    )
    
    # API configuration
    api_host: str = Field(
        default="0.0.0.0",
        env="API_HOST",
        description="API server host address"
    )
    api_port: int = Field(
        default=8000,
        env="API_PORT",
        description="API server port"
    )
    api_key: str = Field(
        default="dev-api-key-change-in-production",
        env="API_KEY",
        description="API authentication key"
    )
    api_title: str = Field(
        default="PPO Production Scheduler API",
        description="API documentation title"
    )
    api_version: str = Field(
        default="1.0.0",
        description="API version"
    )
    
    # Database configuration (reusing existing env vars)
    db_host: str = Field(
        env="DB_HOST",
        description="MariaDB host address"
    )
    db_user: str = Field(
        env="DB_USER",
        description="MariaDB username"
    )
    db_password: str = Field(
        env="DB_PASSWORD",
        description="MariaDB password"
    )
    db_name: str = Field(
        default="nex_valiant",
        env="DB_NAME",
        description="MariaDB database name"
    )
    db_port: int = Field(
        default=3306,
        env="DB_PORT",
        description="MariaDB port"
    )
    
    # Performance settings
    max_concurrent_schedules: int = Field(
        default=10,
        env="MAX_CONCURRENT_SCHEDULES",
        description="Maximum number of concurrent scheduling requests"
    )
    schedule_timeout: float = Field(
        default=2.0,
        env="SCHEDULE_TIMEOUT",
        description="Maximum time (seconds) for generating a schedule"
    )
    inference_batch_size: int = Field(
        default=1,
        env="INFERENCE_BATCH_SIZE",
        description="Batch size for model inference"
    )
    
    # CORS settings
    cors_allow_origins: list[str] = Field(
        default=["http://localhost:3000"],
        description="Allowed CORS origins (comma-separated in env)"
    )
    cors_allow_methods: list[str] = Field(
        default=["GET", "POST"],
        description="Allowed HTTP methods"
    )
    cors_allow_headers: list[str] = Field(
        default=["*"],
        description="Allowed HTTP headers"
    )
    
    # Logging configuration
    log_level: str = Field(
        default="INFO",
        env="LOG_LEVEL",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    
    # Safety and fallback settings
    enable_fallback: bool = Field(
        default=True,
        env="ENABLE_FALLBACK",
        description="Enable fallback to baseline scheduler on errors"
    )
    fallback_policy: str = Field(
        default="first_fit",
        env="FALLBACK_POLICY",
        description="Fallback scheduling policy (first_fit, random, priority)"
    )
    max_retry_attempts: int = Field(
        default=3,
        env="MAX_RETRY_ATTEMPTS",
        description="Maximum retry attempts on failure"
    )
    
    # Monitoring and metrics
    enable_metrics: bool = Field(
        default=True,
        env="ENABLE_METRICS",
        description="Enable Prometheus metrics endpoint"
    )
    metrics_port: int = Field(
        default=9090,
        env="METRICS_PORT",
        description="Port for metrics endpoint"
    )
    
    # Environment and deployment
    environment: str = Field(
        default="development",
        env="ENVIRONMENT",
        description="Deployment environment (development, staging, production)"
    )
    debug_mode: bool = Field(
        default=False,
        env="DEBUG_MODE",
        description="Enable debug mode with detailed error messages"
    )
    
    class Config:
        """Pydantic configuration"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from .env
    
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        """
        Customize how settings are loaded to handle CORS_ALLOW_ORIGINS parsing.
        """
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )
    
    def __init__(self, **values):
        """Override init to handle CORS origins parsing."""
        # Handle CORS_ALLOW_ORIGINS if it's a string
        import os
        cors_origins = os.getenv("CORS_ALLOW_ORIGINS")
        if cors_origins and isinstance(cors_origins, str):
            values["cors_allow_origins"] = [x.strip() for x in cors_origins.split(",")]
        super().__init__(**values)


# Singleton instance
_settings: Optional[APISettings] = None


def get_settings() -> APISettings:
    """
    Get or create the settings singleton instance.
    
    Returns:
        APISettings: The API settings instance
    """
    global _settings
    if _settings is None:
        _settings = APISettings()
    return _settings