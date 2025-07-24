# What is a Snapshot? (Beginner's Guide)

## Simple Explanation

A **snapshot** is like taking a photograph of your data at a specific moment in time.

### Real-Life Example
Imagine you have a busy restaurant:
- Tables keep getting occupied and freed
- Orders keep coming in and getting completed  
- Staff keep starting and ending shifts

If you take a **photograph** of the restaurant at 2:00 PM, you capture:
- Which tables are occupied RIGHT NOW
- Which orders are pending RIGHT NOW
- Which staff are working RIGHT NOW

That photograph is a **snapshot** - it shows the exact state at that moment.

## In Our Scheduling System

### Without Snapshot (Direct Database)
```
Every time we need data:
1. Connect to database ← Takes time
2. Run complex queries ← Takes time
3. Process the results ← Takes time
4. Disconnect from database

If we train for 1000 iterations, we connect 1000 times!
```

### With Snapshot
```
One time only:
1. Connect to database
2. Get ALL the data we need
3. Save it to a file (snapshot)
4. Disconnect

During training:
- Just read the file ← Super fast!
- No database connection needed
- Same data every time (consistent)
```

## The Snapshot File

Our snapshot (`real_production_snapshot.json`) contains:

```json
{
  "metadata": {
    "created_at": "2025-07-24T15:45:54",  ← When photo was taken
    "total_tasks": 295,                     ← What we captured
    "total_machines": 145
  },
  "families": {
    // All the job data at that moment
  },
  "machines": [
    // All the machine data at that moment
  ]
}
```

## Why Use Snapshots?

### 1. **Speed**
- Reading a file: ~0.01 seconds
- Database query: ~1-5 seconds
- 500x faster!

### 2. **Consistency**
- Training needs the SAME data every iteration
- Real database keeps changing (new jobs, completed jobs)
- Snapshot never changes

### 3. **Offline Work**
- Can train without database connection
- Can work from home
- Can share data file with team

### 4. **Reproducibility**
- Can repeat exact same training later
- Can debug with exact same data
- Can compare results fairly

## The Config Explained

```yaml
data:
  source: "snapshot"  # ← Use the saved file
  # source: "database"  ← Would connect to live database
  
  snapshot_path: "data/real_production_snapshot.json"  # ← Which file to read
  
  max_jobs: null  # ← Use ALL jobs in snapshot (null = no limit)
  # max_jobs: 50  ← Would only use first 50 jobs
  
  max_machines: null  # ← Use ALL machines
  # max_machines: 10  ← Would only use first 10 machines
```

## How to Create a New Snapshot

```bash
# This command takes a new "photograph" of the database
cd /Users/carrickcheah/Project/ppo/app_2
uv run python src/data_ingestion/ingest_data.py

# Creates: data/real_production_snapshot.json
```

## Think of it Like...

1. **Video Game Save File**
   - Snapshot = Saving your game
   - You can load the exact same state later

2. **Phone Backup**
   - Snapshot = Backing up your phone
   - All your data at that moment is saved

3. **Time Machine**
   - Snapshot = Going back to a specific moment
   - Everything is exactly as it was then

## Summary

- **Snapshot** = Data frozen in time
- **Why?** = Fast, consistent, reliable
- **When?** = Before training, testing, or sharing
- **What?** = All jobs and machines at that moment
- **Where?** = Saved as a JSON file