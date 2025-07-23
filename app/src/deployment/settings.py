"""
API Settings Configuration using pydantic-settings

This module defines all configuration settings for the PPO scheduler API
using pydantic-settings for type-safe environment variable handling.
"""

from typing import Optional
import os
import logging
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Setup logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load .env file before settings initialization
env_path = Path(__file__).parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    logger.debug(f"Loaded .env from: {env_path}")
else:
    logger.warning(f".env file not found at: {env_path}")


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
        validation_alias="API_HOST",
        description="API server host address"
    )
    api_port: int = Field(
        default=8000,
        validation_alias="API_PORT",
        description="API server port"
    )
    api_key: str = Field(
        default="dev-api-key-change-in-production",
        validation_alias="API_KEY",
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
    
    # Database configuration (using MARIADB_* env vars)
    db_host: str = Field(
        default="localhost",
        validation_alias="MARIADB_HOST",
        description="MariaDB host address"
    )
    db_user: str = Field(
        default="myuser",
        validation_alias="MARIADB_USERNAME",
        description="MariaDB username"
    )
    db_password: str = Field(
        default="mypassword",
        validation_alias="MARIADB_PASSWORD",
        description="MariaDB password"
    )
    db_name: str = Field(
        default="nex_valiant",
        validation_alias="MARIADB_DATABASE",
        description="MariaDB database name"
    )
    db_port: int = Field(
        default=3306,
        validation_alias="MARIADB_PORT",
        description="MariaDB port"
    )
    
    # Performance settings
    max_concurrent_schedules: int = Field(
        default=10,
        validation_alias="MAX_CONCURRENT_SCHEDULES",
        description="Maximum number of concurrent scheduling requests"
    )
    schedule_timeout: float = Field(
        default=2.0,
        validation_alias="SCHEDULE_TIMEOUT",
        description="Maximum time (seconds) for generating a schedule"
    )
    inference_batch_size: int = Field(
        default=1,
        validation_alias="INFERENCE_BATCH_SIZE",
        description="Batch size for model inference"
    )
    
    # CORS settings
    cors_allow_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:3001", "http://localhost:8080"],
        description="Allowed CORS origins (comma-separated in env)"
    )
    cors_allow_methods: list[str] = Field(
        default=["GET", "POST", "OPTIONS"],
        description="Allowed HTTP methods"
    )
    cors_allow_headers: list[str] = Field(
        default=["*"],
        description="Allowed HTTP headers"
    )
    
    # Logging configuration
    log_level: str = Field(
        default="INFO",
        validation_alias="LOG_LEVEL",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    
    # Safety and fallback settings
    enable_fallback: bool = Field(
        default=True,
        validation_alias="ENABLE_FALLBACK",
        description="Enable fallback to baseline scheduler on errors"
    )
    fallback_policy: str = Field(
        default="first_fit",
        validation_alias="FALLBACK_POLICY",
        description="Fallback scheduling policy (first_fit, random, priority)"
    )
    max_retry_attempts: int = Field(
        default=3,
        validation_alias="MAX_RETRY_ATTEMPTS",
        description="Maximum retry attempts on failure"
    )
    
    # Monitoring and metrics
    enable_metrics: bool = Field(
        default=True,
        validation_alias="ENABLE_METRICS",
        description="Enable Prometheus metrics endpoint"
    )
    metrics_port: int = Field(
        default=9090,
        validation_alias="METRICS_PORT",
        description="Port for metrics endpoint"
    )
    
    # Environment and deployment
    environment: str = Field(
        default="development",
        validation_alias="ENVIRONMENT",
        description="Deployment environment (development, staging, production)"
    )
    debug_mode: bool = Field(
        default=False,
        validation_alias="DEBUG_MODE",
        description="Enable debug mode with detailed error messages"
    )
    
    model_config = {
        "env_file": str(Path(__file__).parent.parent.parent / '.env'),
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",  # Ignore extra fields from .env
    }
    
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
        # Debug: Log environment variables
        logger.debug("Loading settings from environment...")
        logger.debug(f"MARIADB_HOST from env: {os.getenv('MARIADB_HOST')}")
        logger.debug(f"MARIADB_PORT from env: {os.getenv('MARIADB_PORT')}")
        logger.debug(f"MARIADB_USERNAME from env: {os.getenv('MARIADB_USERNAME')}")
        logger.debug(f"MARIADB_DATABASE from env: {os.getenv('MARIADB_DATABASE')}")
        logger.debug(f"MARIADB_PASSWORD exists: {bool(os.getenv('MARIADB_PASSWORD'))}")
        
        try:
            _settings = APISettings()
            logger.debug("Settings loaded successfully")
            logger.debug(f"Loaded db_host: {_settings.db_host}")
            logger.debug(f"Loaded db_port: {_settings.db_port}")
            logger.debug(f"Loaded db_user: {_settings.db_user}")
            logger.debug(f"Loaded db_name: {_settings.db_name}")
        except Exception as e:
            logger.error(f"Failed to load settings: {type(e).__name__}: {str(e)}")
            logger.error("Current working directory: " + os.getcwd())
            logger.error("Looking for .env file at: " + os.path.join(os.getcwd(), '.env'))
            raise
    
    return _settings