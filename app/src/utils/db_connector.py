"""
Database connector for production data.
Uses MariaDB configuration from .env file.
"""

import pandas as pd
import logging
from typing import Optional
import os
from pathlib import Path
import pymysql
from pymysql.cursors import DictCursor
from dotenv import load_dotenv

# Load environment variables from app/.env
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger(__name__)


class DatabaseConnector:
    """Database connector for production data access."""
    
    def __init__(self):
        """Initialize database connector with MariaDB settings."""
        self.host = os.getenv("MARIADB_HOST", "localhost")
        self.user = os.getenv("MARIADB_USERNAME", "myuser")
        self.password = os.getenv("MARIADB_PASSWORD", "mypassword")
        self.database = os.getenv("MARIADB_DATABASE", "nex_valiant")
        self.port = int(os.getenv("MARIADB_PORT", "3306"))
        self.connection = None
        
    def connect(self):
        """Create database connection to MariaDB."""
        try:
            self.connection = pymysql.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                port=self.port,
                cursorclass=DictCursor,
                charset='utf8mb4'
            )
            logger.info(f"Successfully connected to MariaDB database: {self.database}@{self.host}")
            return self.connection
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            return None
            
    def execute_query(self, query: str) -> Optional[pd.DataFrame]:
        """Execute a query and return results as DataFrame."""
        try:
            if not self.connection:
                self.connect()
                
            if not self.connection:
                logger.warning("No database connection available")
                return None
                
            with self.connection.cursor() as cursor:
                cursor.execute(query)
                results = cursor.fetchall()
                
            if results:
                df = pd.DataFrame(results)
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return None
            
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Database connection closed")