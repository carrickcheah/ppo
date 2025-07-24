# PPO Scheduling System - Data Usage Report

## Overview
This report details all data sources and structures used in the PPO scheduling system.

## 1. Production Data Source

### Database Connection
- **Database:** MariaDB - `nex_valiant`
- **Tables Used:**
  - `tbl_jo_txn` - Job transactions (family-level data)
  - `tbl_jo_process` - Job processes (task-level data)
  - `tbl_machine` - Machine information

### Data Snapshot
- **Location:** `/app_2/data/real_production_snapshot.json`
- **Created:** 2025-07-24 15:45:54
- **Contents:**
  - Total Families: 88
  - Total Tasks: 295
  - Total Machines: 145
  - Planning Horizon: 30 days

## 2. Data Structure

### Production Snapshot Format
```json
{
  "metadata": {
    "created_at": "2025-07-24T15:45:54.927188",
    "database": "nex_valiant",
    "planning_horizon_days": 30,
    "total_families": 88,
    "total_tasks": 295,
    "total_machines": 145
  },
  "families": {
    "JOTP25070237": {
      "transaction_id": 8263,
      "job_reference": "JOTP25070237",
      "product": "CT10",
      "is_important": false,
      "lcd_date": "2025-08-01",
      "lcd_days_remaining": 8,
      "total_sequences": 5,
      "tasks": [
        {
          "sequence": 1,
          "process_name": "CT10-013A-1/5",
          "processing_time": 15.1,
          "capable_machines": [80],
          "status": "pending",
          "balance_quantity": 72.0,
          "original_quantity": 72.0
        }
      ]
    }
  },
  "machines": [
    {
      "machine_id": 57,
      "machine_name": "AD02-50HP",
      "machine_type_id": 2
    }
  ]
}
```

### Converted Job Format (Used by Environment)
```json
{
  "job_id": "JOTP25070237_1/5",
  "family_id": "JOTP25070237",
  "sequence": 1,
  "required_machines": [80],
  "processing_time": 15.1,
  "lcd_days_remaining": 8,
  "is_important": false,
  "product_code": "CT10",
  "status": "pending",
  "quantity": 72.0
}
```

## 3. Key Data Transformations

### Processing Time Calculation
When `CapMin_d = 1` and `CapQty_d != 0`:
```python
hourly_capacity = CapQty_d * 60
processing_time = JoQty_d / hourly_capacity + (SetupTime_d / 60)
```

### Multi-Machine Jobs
- **Input:** `Machine_v = "57,64,65,66,74"`
- **Parsed:** `required_machines = [57, 64, 65, 66, 74]`
- **Meaning:** Job requires ALL 5 machines simultaneously

## 4. Test Data Examples

### Basic Test Job
```python
{
    "job_id": "TEST001",
    "family_id": "FAM1",
    "sequence": 1,
    "required_machines": [1],
    "processing_time": 2.0,
    "lcd_days_remaining": 5,
    "is_important": True
}
```

### Multi-Machine Test Job
```python
{
    "job_id": "MULTI001",
    "family_id": "FAM2",
    "sequence": 1,
    "required_machines": [1, 2, 3],  # Needs 3 machines
    "processing_time": 4.0,
    "lcd_days_remaining": 3,
    "is_important": False
}
```

## 5. Data Loading Process

### From Database
1. Connect to MariaDB using credentials
2. Query pending jobs with balance > 0
3. Join with machine capability data
4. Calculate processing times
5. Save to JSON snapshot

### From Snapshot
1. Load JSON file
2. Convert families structure to jobs list
3. Map machine IDs to indices
4. Apply any configured limits (max_jobs, max_machines)

## 6. Data Statistics

### Production Data Summary
- **Job Families:** 88 unique product families
- **Total Tasks:** 295 individual operations
- **Sequence Lengths:** 1-10 steps per family
- **Machine Requirements:**
  - Single machine: 290 tasks (98.3%)
  - Multi-machine: 5 tasks (1.7%)
    - 2 machines: 2 tasks
    - 3 machines: 2 tasks
    - 5 machines: 1 task
- **Processing Times:** 0.5 - 100+ hours
- **LCD Days:** 0 - 30 days remaining

### Machine Fleet
- **Total Machines:** 145
- **Machine Types:** ~20 different types
- **Common Machines:**
  - CNC machines (CM series)
  - Lathes (CL series)
  - Assembly stations (AD series)

## 7. Data Quality Checks

### Validation Rules
1. All jobs must have valid machine assignments
2. Processing times must be > 0
3. Sequence numbers must be consecutive within families
4. LCD days remaining must be >= 0
5. Machine IDs must exist in machine list

### Known Issues
- Some jobs may have no capable machines (handled by environment)
- Working hours constraints removed for training
- All times in hours (converted from minutes in DB)

## 8. Configuration Files

### Data Loading Config
```yaml
# configs/data_config.yaml
data:
  source: "snapshot"  # or "database"
  snapshot_path: "data/real_production_snapshot.json"
  max_jobs: null  # No limit
  max_machines: null  # No limit
```

### Environment Config
```yaml
# configs/env_config.yaml
environment:
  time_horizon: 720  # 30 days in hours
  enforce_sequence: true
  enforce_compatibility: true
  enforce_no_overlap: true
  enforce_working_hours: false  # Disabled for training
```

## 9. Data Flow

```
MariaDB Database
    ↓
ingest_data.py
    ↓
real_production_snapshot.json
    ↓
DataLoader (converts families → jobs)
    ↓
SchedulingGameEnv (creates state/action spaces)
    ↓
PPO Model (learns scheduling policies)
```

## 10. Future Data Considerations

### Phase 4 Enhancements
- Real-time data updates from database
- Historical performance data for better estimates
- Machine maintenance schedules
- Operator skill matrices
- Material availability constraints

### Deployment Data
- Working hours: M-F 8:00-17:00, Sat 8:00-12:00
- Public holidays calendar
- Machine downtime predictions
- Rush order priorities