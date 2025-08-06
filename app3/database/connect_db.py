import mysql.connector
import os
from dotenv import load_dotenv
from mysql.connector import Error

# Load environment variables from .env file at backend root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../.env'))

# Get database configuration from environment variables
DB_HOST = os.getenv("MARIADB_HOST")
DB_USER = os.getenv("MARIADB_USERNAME")
DB_PASSWORD = os.getenv("MARIADB_PASSWORD")
DB_NAME = os.getenv("MARIADB_DATABASE")
DB_PORT = os.getenv("MARIADB_PORT", "3306")


def get_db_connection():
    """Establishes a connection to the MariaDB database."""
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT
        )
        if conn.is_connected():
            return conn
    except Error as e:
        print(f"Error connecting to MariaDB Platform: {e}")
        return None