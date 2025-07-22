#!/usr/bin/env python3
"""
Interactive setup script for production configuration
"""

import os
import secrets
import shutil
from pathlib import Path
from getpass import getpass

print("PPO Production Scheduler - Setup Assistant")
print("=" * 50)
print("\nThis script will help you create a production-ready .env file")
print("Your current .env will be backed up as .env.backup\n")

# Backup existing .env
if Path(".env").exists():
    shutil.copy(".env", ".env.backup")
    print("✓ Backed up existing .env to .env.backup")

# Collect configuration
print("\n1. Database Configuration")
print("-" * 30)
db_host = input("Database host [localhost]: ").strip() or "localhost"
db_port = input("Database port [3306]: ").strip() or "3306"
db_name = input("Database name [nex_valiant]: ").strip() or "nex_valiant"
db_user = input("Database username: ").strip()
db_password = getpass("Database password: ").strip()

print("\n2. API Configuration")
print("-" * 30)
api_host = input("API host [0.0.0.0]: ").strip() or "0.0.0.0"
api_port = input("API port [8000]: ").strip() or "8000"

# Generate secure API key
generate_key = input("\nGenerate secure API key? [Y/n]: ").strip().lower()
if generate_key != 'n':
    api_key = secrets.token_urlsafe(32)
    print(f"Generated API key: {api_key}")
else:
    api_key = getpass("Enter your API key: ").strip()

print("\n3. CORS Configuration")
print("-" * 30)
print("Enter allowed origins (comma-separated)")
print("Example: https://app.company.com,https://scheduler.company.com")
cors_input = input("CORS origins: ").strip()
cors_origins = [f'"{origin.strip()}"' for origin in cors_input.split(",") if origin.strip()]
cors_string = f"[{','.join(cors_origins)}]" if cors_origins else '["*"]'

print("\n4. Environment")
print("-" * 30)
environment = input("Environment [production]: ").strip() or "production"

# Create new .env file
env_content = f"""# =============================================================================
# DATABASE CONNECTION - Used by: mariadb_parser.py
# =============================================================================
MARIADB_HOST={db_host}
MARIADB_USERNAME={db_user}
MARIADB_PASSWORD={db_password}
MARIADB_DATABASE={db_name}
MARIADB_PORT={db_port}

# =============================================================================
# DATABASE CONNECTION - Used by: ingest_data.py (PPO data ingestion)
# =============================================================================
DB_HOST={db_host}
DB_USER={db_user}
DB_PASSWORD={db_password}
DB_NAME={db_name}
DB_PORT={db_port}

# =============================================================================
# API CONFIGURATION - Used by: api_server.py (FastAPI deployment)
# =============================================================================
API_HOST={api_host}
API_PORT={api_port}
API_KEY={api_key}
CORS_ALLOW_ORIGINS={cors_string}
ENVIRONMENT={environment}
LOG_LEVEL=INFO
ENABLE_METRICS=true
METRICS_PORT=9090
"""

# Write to file
with open(".env", "w") as f:
    f.write(env_content)

# Set secure permissions
os.chmod(".env", 0o600)

print("\n" + "=" * 50)
print("✓ Configuration saved to .env")
print("✓ File permissions set to 600 (owner read/write only)")
print("\nNext steps:")
print("1. Run: uv run python test_production_connection.py")
print("2. If test passes, start server: uv run python run_api_server.py")
print("3. Access API docs at: http://{api_host}:{api_port}/docs")

if environment == "production":
    print("\n⚠️  IMPORTANT: You're configured for PRODUCTION")
    print("   - Ensure database credentials are correct")
    print("   - Keep API key secure and don't share it")
    print("   - Monitor logs closely during initial deployment")