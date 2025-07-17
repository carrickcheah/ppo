"""
Database module for PPO Production Scheduler
Contains database connection and data access functions
"""

from .connect_db import get_db_connection

__all__ = ['get_db_connection']