# Setup Instructions for Pure DRL Scheduler

## Phase 1 Completed: Environment Foundation

The scheduling game environment has been successfully set up with the following components:

### 1. Environment Structure
```
app_2/
├── src/
│   ├── environment/
│   │   ├── scheduling_game_env.py    # Main game environment
│   │   ├── rules_engine.py           # Hard constraints (physics)
│   │   └── reward_function.py        # Soft preferences (learning signals)
│   ├── data/
│   │   ├── db_connector.py           # MariaDB connection
│   │   └── data_loader.py            # Data loading from various sources
├── configs/
│   ├── environment.yaml              # Game configuration
│   ├── training.yaml                 # PPO hyperparameters
│   └── deployment.yaml               # Production API settings
└── tests/
    └── test_environment.py           # Environment validation tests
```

### 2. Key Design Principles

**Pure Deep Reinforcement Learning**
- No hardcoded scheduling strategies
- Environment enforces only physics (hard rules)
- AI learns all strategies through experience

**Game Rules (Physics)**
1. Sequence constraints - jobs must follow order within family
2. Machine compatibility - jobs can only run on capable machines
3. No overlap - one job per machine at a time
4. Working hours - respect configured time windows

**Learning Signals (Rewards)**
- Completion reward for any scheduled job
- Importance bonus for high-priority jobs
- Urgency multiplier based on deadlines
- Efficiency bonuses for good utilization
- All configurable via YAML

### 3. Quick Test

Run the test script to verify setup:
```bash
cd /Users/carrickcheah/Project/ppo/app_2
source .venv/bin/activate
python run_test.py
```

This will:
- Test environment creation
- Validate action execution
- Check sequence constraints
- Run a complete episode with toy data

### 4. Configuration

All settings are in YAML files (no hardcoding):

**environment.yaml**
- Game rules and physics
- Reward configuration
- Data source settings

**training.yaml**
- PPO hyperparameters
- Model architecture (transformer)
- Curriculum learning stages

**deployment.yaml**
- API server settings
- Production configuration
- Monitoring and logging

### 5. Data Sources

The system supports three data sources:
1. **database** - Direct from MariaDB (production)
2. **snapshot** - JSON files for reproducibility
3. **test** - Small toy dataset for development

Currently set to "test" for initial development.

### 6. Next Steps (Phase 2)

With the environment ready, the next phase will implement:
1. Transformer-based policy network
2. PPO training algorithm
3. Action masking for valid moves only
4. Curriculum learning pipeline

The environment is designed to handle variable-sized inputs (10 to 1000+ jobs) using attention mechanisms instead of fixed-size representations.

### 7. Environment Interface

```python
# Initialize environment
env = SchedulingGameEnv(jobs, machines, working_hours, config)

# Reset to start episode
obs, info = env.reset()

# Take action (job_idx, machine_idx)
action = np.array([job_idx, machine_idx])
next_obs, reward, terminated, truncated, info = env.step(action)

# Get valid action mask
mask = env.get_action_mask()
```

The environment provides everything needed for the PPO model to learn optimal scheduling strategies through pure experience.