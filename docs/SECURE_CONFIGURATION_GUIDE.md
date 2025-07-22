# Secure Configuration Guide for Production Deployment

## Overview

This guide helps you securely configure the PPO Production Scheduler for production deployment. Follow these steps to update your `.env` file with production values.

## Step 1: Generate Secure API Key

Generate a secure API key (minimum 32 characters):

```python
# Python method
import secrets
api_key = secrets.token_urlsafe(32)
print(f"API_KEY={api_key}")
```

```bash
# Linux/Mac command line
openssl rand -base64 32
```

## Step 2: Update Database Credentials

Replace the test credentials with your production MariaDB details:

```bash
# BEFORE (Test values)
DB_HOST=localhost
DB_USER=myuser
DB_PASSWORD=mypassword

# AFTER (Your production values)
DB_HOST=your-production-server.company.com
DB_USER=your_prod_username
DB_PASSWORD=your_secure_password
```

**Important**: Both `DB_*` and `MARIADB_*` variables should have the same values.

## Step 3: Configure CORS for Your Frontend

Update CORS origins to match your production frontend URLs:

```bash
# BEFORE
CORS_ALLOW_ORIGINS=["http://localhost:3000","http://localhost:8080"]

# AFTER (Example)
CORS_ALLOW_ORIGINS=["https://scheduler.yourcompany.com","https://app.yourcompany.com"]
```

## Step 4: Set Production Environment

Change environment to production:

```bash
# BEFORE
ENVIRONMENT=development

# AFTER
ENVIRONMENT=production
```

## Step 5: Test Your Configuration

Run the test script to verify your configuration:

```bash
uv run python test_production_connection.py
```

You should see:
- ✓ Database connection successful
- ✓ Found X active machines
- ✓ Found Y pending jobs
- ✓ No security warnings

## Security Checklist

Before going to production, ensure:

- [ ] API_KEY is unique and at least 32 characters
- [ ] Database password is strong and unique
- [ ] CORS origins only include your production domains
- [ ] Environment is set to "production"
- [ ] .env file has restricted permissions (chmod 600)
- [ ] .env is NOT committed to git (.gitignore includes it)
- [ ] Backup of configuration exists in secure location

## Example Production .env

```bash
# Database (both sections use same credentials)
MARIADB_HOST=prod-db.company.internal
MARIADB_USERNAME=ppo_scheduler_prod
MARIADB_PASSWORD=xK9#mP2$vL5@nQ8&
MARIADB_DATABASE=nex_valiant
MARIADB_PORT=3306

DB_HOST=prod-db.company.internal
DB_USER=ppo_scheduler_prod
DB_PASSWORD=xK9#mP2$vL5@nQ8&
DB_NAME=nex_valiant
DB_PORT=3306

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_KEY=your-32-character-secure-key-here
CORS_ALLOW_ORIGINS=["https://scheduler.company.com"]
ENVIRONMENT=production
LOG_LEVEL=INFO
ENABLE_METRICS=true
METRICS_PORT=9090
```

## File Permissions

Secure your .env file:

```bash
# Set restrictive permissions
chmod 600 .env

# Verify permissions
ls -la .env
# Should show: -rw-------
```

## Starting the Production Server

Once configured, start the server:

```bash
# Start in production mode
uv run python run_api_server.py

# Or with multiple workers
uv run python run_api_server.py --workers 4
```

## Monitoring

After deployment, monitor:
1. API health: http://your-server:8000/health
2. Logs: Check for any errors or warnings
3. Database connections: Ensure stable connectivity
4. Response times: Should be <2s for scheduling requests

## Troubleshooting

If the API shows "degraded" status:
1. Check database connectivity
2. Verify credentials are correct
3. Ensure database user has SELECT permissions
4. Check firewall rules allow connection

If schedule endpoint returns 500:
1. Verify machines exist in tbl_machine
2. Check job data in job_operations_time
3. Review API logs for specific errors

## Next Steps

1. Test with production data
2. Start shadow mode (parallel with manual)
3. Monitor performance metrics
4. Gradually increase traffic
5. Full production deployment

Remember: Always test thoroughly in a staging environment before production deployment!