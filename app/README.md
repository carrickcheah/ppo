# Deep Reinforcement Learning Scheduling System Implementation Plan

## Project Location
This project is located at: `/Users/carrickcheah/Project/ppo/app`

## Table of Contents
1. [Overview & Architecture](#overview--architecture)
2. [Development Workflow](#development-workflow)
3. [Phase 1: Foundation & Learning](#phase-1-foundation--learning-weeks-1-2)
4. [Phase 2: Toy Problem](#phase-2-toy-problem-weeks-3-4)
5. [Phase 3: Realistic Constraints](#phase-3-realistic-constraints-weeks-5-8)
6. [Phase 4: Full Environment](#phase-4-full-environment-weeks-9-12)
7. [Phase 5: Validation & Safety](#phase-5-validation--safety-weeks-13-14)
8. [Phase 6: Production Deployment](#phase-6-production-deployment-weeks-15-16)
9. [Workflows](#workflows)
10. [Risk Management](#risk-management)
11. [Success Metrics](#success-metrics)

## Overview & Architecture

### System Architecture Flow
```
┌─────────────────────────────────────────────────────────────────┐
│                        Current System                            │
├─────────────────────────────────────────────────────────────────┤
│   MariaDB → Data Ingestion → Greedy Solver → Schedule → API    │
│      ↓            ↓              ↓              ↓         ↓      │
│   [Jobs]    [Constraints]   [Priorities]   [Output]   [Report]  │
└─────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Target RL System                            │
├─────────────────────────────────────────────────────────────────┤
│   MariaDB → Environment → PPO Agent → Schedule → Validator → API│
│      ↓          ↓            ↓           ↓          ↓        ↓   │
│   [Jobs]    [State]      [Actions]   [Output]  [Safety]  [Report]│
│                ↓            ↓                                    │
│            [Reward]    [Neural Network]                          │
└─────────────────────────────────────────────────────────────────┘
```

### Project Structure with UV
```
app/
├── src/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── networks.py          # Neural network architectures
│   │   ├── policies.py          # Custom policies
│   │   └── ppo_agent.py         # PPO implementation
│   ├── deployment/
│   │   ├── __init__.py
│   │   ├── api_integration.py   # FastAPI integration
│   │   ├── monitoring.py        # Real-time monitoring
│   │   └── safe_scheduler.py    # Production wrapper
│   ├── environments/
│   │   ├── __init__.py
│   │   ├── base_env.py          # Abstract base class
│   │   ├── medium_env.py        # 10-machine with constraints
│   │   ├── production_env.py    # Full 74-machine environment
│   │   └── toy_env.py           # Simple 2-machine environment
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── benchmark.py         # Compare with greedy
│   │   ├── historical_test.py   # Test on past data
│   │   └── stress_test.py       # Edge case testing
│   ├── training/
│   │   ├── __init__.py
│   │   ├── hyperparameter_tuning.py
│   │   ├── train_medium.py
│   │   ├── train_production.py
│   │   └── train_toy.py
│   └── utils/
│       ├── __init__.py
│       ├── data_loader.py       # Database connections
│       ├── metrics.py           # Performance metrics
│       ├── validators.py        # Constraint checking
│       └── visualizers.py       # Schedule visualization
├── configs/
│   ├── medium_config.yaml
│   ├── production_config.yaml
│   └── toy_config.yaml
├── notebooks/
│   └── exploratory_analysis.ipynb
├── tests/
│   └── test_*.py
├── .gitignore
├── .python-version             # Python version for UV
├── pyproject.toml              # UV project configuration
├── README.md
└── uv.lock                     # Lock file (auto-generated)
```

### Development Commands with UV
```bash
# Add new dependency
uv add torch torchvision

# Add dev dependency
uv add --dev pytest-benchmark

# Update dependencies
uv sync

# Run scripts
uv run python src/training/train_toy.py

# Run tests
uv run pytest

# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Type check
uv run mypy src/
```

## Development Workflow

### Daily Development Cycle
```
Morning (2-3 hours):
├── Review previous day's training logs
├── Analyze failure cases
├── Adjust hyperparameters/rewards
└── Start new training runs

Afternoon (3-4 hours):
├── Implement new features
├── Write tests
├── Code review (if team)
└── Documentation

Evening (1 hour):
├── Monitor training progress
├── Log findings in journal
└── Plan next day
```

### Weekly Sprint Cycle
```
Monday:
├── Sprint planning
├── Define week's objectives
└── Setup experiments

Tuesday-Thursday:
├── Core development
├── Training runs
└── Iterative improvements

Friday:
├── Evaluation & benchmarking
├── Document results
├── Demo to stakeholders
└── Plan next sprint
```

## Phase 1: Foundation & Learning (Weeks 1-2)

### Week 1: Understanding RL Basics

#### Day 1-3: Learn Fundamentals
- [ ] Complete OpenAI Spinning Up RL intro
- [ ] Study PPO algorithm specifically
- [ ] Understand concepts:
  - Markov Decision Process (MDP)
  - Policy vs Value functions
  - Actor-Critic architecture
  - Advantage estimation
- [ ] Run and modify CartPole example
- [ ] Create learning notes document

#### Day 4-7: Environment Setup with UV

First, install `uv` if you haven't:
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or on macOS with Homebrew
brew install uv
```

Create project structure:
```bash
# Create project
cd /Users/carrickcheah/Project/ppo/app
uv init

# Create pyproject.toml
```

Create `pyproject.toml`:
```toml
[project]
name = "scheduling-rl"
version = "0.1.0"
description = "Deep RL scheduling system using PPO"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "stable-baselines3[extra]>=2.1.0",
    "gymnasium>=0.29.1",
    "tensorboard>=2.15.0",
    "pandas>=2.1.4",
    "numpy>=1.26.2",
    "mysql-connector-python>=8.2.0",
    "pyyaml>=6.0.1",
    "matplotlib>=3.8.2",
    "seaborn>=0.13.0",
    "plotly>=5.18.0",
    "flask>=3.0.0",
    "optuna>=3.5.0",
    "pytest>=7.4.3",
    "black>=23.12.1",
    "ruff>=0.1.9",
]

[project.optional-dependencies]
dev = [
    "ipykernel>=6.28.0",
    "jupyter>=1.0.0",
    "pytest-cov>=4.1.0",
    "mypy>=1.8.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = [
    "ipykernel>=6.29.5",
    "jupyter>=1.1.1",
    "pytest-cov>=5.0.0",
    "mypy>=1.11.2",
]

[tool.ruff]
# Same as Black
line-length = 88
indent-width = 4

# Python 3.11
target-version = "py311"

[tool.ruff.lint]
select = [
    "E",  # pycodestyle errors
    "W",  # pycodestyle warnings
    "F",  # pyflakes
    "I",  # isort
    "B",  # flake8-bugbear
    "C4", # flake8-comprehensions
    "UP", # pyupgrade
]
ignore = [
    "E501", # line too long (handled by formatter)
]

[tool.ruff.format]
# Like Black
quote-style = "double"
indent-style = "space"

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
addopts = "-v --cov=src --cov-report=html"
```

Setup environment:
```bash
# Create virtual environment with uv
uv venv

# Activate it
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv sync

# Install development dependencies
uv sync --dev

# Test installation
python -c "import stable_baselines3; print('SB3 version:', stable_baselines3.__version__)"
```

### Week 2: Analyze Current System

#### Data Collection Script
```python
# extract_system_data.py
import mysql.connector
import pandas as pd
import json

def extract_scheduling_data():
    """Extract key data from current system"""
    
    # Connect to database
    conn = mysql.connector.connect(**db_config)
    
    # Extract job patterns
    jobs_query = """
    SELECT family_name, process_order, 
           AVG(processing_time) as avg_time,
           COUNT(*) as frequency
    FROM production_jobs
    WHERE created_at > DATE_SUB(NOW(), INTERVAL 90 DAY)
    GROUP BY family_name, process_order
    """
    
    # Extract machine utilization
    machine_query = """
    SELECT machine_id, 
           COUNT(*) as job_count,
           SUM(processing_time) as total_hours
    FROM job_schedules
    WHERE scheduled_date > DATE_SUB(NOW(), INTERVAL 30 DAY)
    GROUP BY machine_id
    """
    
    # Save results
    jobs_df = pd.read_sql(jobs_query, conn)
    machines_df = pd.read_sql(machine_query, conn)
    
    # Document findings
    analysis = {
        'total_job_families': jobs_df['family_name'].nunique(),
        'avg_jobs_per_day': len(jobs_df) / 90,
        'machine_imbalance': machines_df['job_count'].std(),
        'top_bottleneck': machines_df.nlargest(1, 'job_count')['machine_id'].iloc[0]
    }
    
    return jobs_df, machines_df, analysis
```

#### Key Metrics to Document
- [ ] Current average lateness: **25.9 days**
- [ ] Scheduling time: **4.09 seconds for 443 jobs**
- [ ] Machine utilization variance
- [ ] Job family sequences mapping
- [ ] Peak hours and overtime patterns

## Phase 2: Toy Problem (Weeks 3-4)

### Week 3: Build Minimal Environment

#### Workflow: Environment Development
```
1. Design State Space
   ├── What information does agent need?
   ├── How to normalize values?
   └── Fixed size vs variable size

2. Design Action Space
   ├── Discrete vs Continuous
   ├── Action masking strategy
   └── Invalid action handling

3. Design Reward Function
   ├── Immediate vs delayed rewards
   ├── Sparse vs dense rewards
   └── Reward shaping pitfalls

4. Implement Core Methods
   ├── __init__()
   ├── reset()
   ├── step()
   ├── render()
   └── close()
```

#### Implementation: toy_env.py
```python
import gym
import numpy as np
from gym import spaces

class ToySchedulingEnv(gym.Env):
    """
    Minimal scheduling environment for learning
    2 machines, 5 jobs, no dependencies
    """
    
    def __init__(self):
        super().__init__()
        
        # Environment parameters
        self.n_machines = 2
        self.n_jobs = 5
        self.max_time = 20
        
        # Spaces
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(self.n_machines + self.n_jobs * 2,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(self.n_jobs + 1)  # +1 for wait
        
        # Initialize
        self.reset()
        
    def reset(self):
        """Reset environment to initial state"""
        # Machine states [current_load, current_load]
        self.machine_loads = np.zeros(self.n_machines)
        
        # Job states [is_scheduled, processing_time]
        self.job_scheduled = np.zeros(self.n_jobs, dtype=bool)
        self.job_times = np.random.randint(1, 6, self.n_jobs)
        
        self.current_time = 0
        self.total_reward = 0
        
        return self._get_observation()
    
    def step(self, action):
        """Execute action and return results"""
        reward = 0
        done = False
        info = {}
        
        if action < self.n_jobs:  # Schedule a job
            if not self.job_scheduled[action]:
                # Find least loaded machine
                machine = np.argmin(self.machine_loads)
                
                # Schedule job
                self.machine_loads[machine] += self.job_times[action]
                self.job_scheduled[action] = True
                
                # Calculate reward
                balance = -np.std(self.machine_loads)  # Encourage balance
                completion = 10  # Reward for scheduling
                reward = completion + balance
                
                info['scheduled_job'] = action
                info['on_machine'] = machine
            else:
                reward = -10  # Penalty for invalid action
                
        else:  # Wait action
            self.current_time += 1
            reward = -1  # Small penalty for waiting
            
        # Check if done
        if np.all(self.job_scheduled):
            done = True
            # Final reward based on makespan
            makespan = np.max(self.machine_loads)
            reward += 50 / makespan  # Bonus for short makespan
            
        self.total_reward += reward
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self):
        """Construct observation vector"""
        obs = []
        
        # Machine loads (normalized)
        obs.extend(self.machine_loads / self.max_time)
        
        # Job states
        for i in range(self.n_jobs):
            obs.append(float(self.job_scheduled[i]))
            obs.append(self.job_times[i] / self.max_time)
            
        return np.array(obs, dtype=np.float32)
    
    def render(self, mode='human'):
        """Visualize current state"""
        print(f"\nTime: {self.current_time}")
        print(f"Machine loads: {self.machine_loads}")
        print(f"Jobs scheduled: {self.job_scheduled}")
        print(f"Total reward: {self.total_reward:.2f}")
```

### Week 4: Train First Agent

#### Training Workflow
```
1. Baseline Performance
   ├── Random agent baseline
   ├── Simple heuristic baseline
   └── Record metrics

2. Initial Training
   ├── Start with default hyperparameters
   ├── Monitor training curves
   └── Identify issues

3. Hyperparameter Tuning
   ├── Learning rate search
   ├── Network architecture
   └── PPO-specific params

4. Evaluation
   ├── Test on fixed scenarios
   ├── Compare with baselines
   └── Visualize behavior
```

#### Implementation: train_toy.py
```python
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import tensorboard

# Import from your package
from src.environments.toy_env import ToySchedulingEnv

# Create environment
env = ToySchedulingEnv()

# Validate environment
check_env(env)

# Create evaluation env
eval_env = ToySchedulingEnv()

# Callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path='./models/',
    name_prefix='toy_ppo'
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./models/best/',
    log_path='./logs/',
    eval_freq=5000,
    deterministic=True,
    render=False
)

# Create PPO model
model = PPO(
    'MlpPolicy',
    env,
    learning_rate=3e-4,
    n_steps=512,  # Small for toy problem
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1,
    tensorboard_log="./tensorboard/"
)

# Train
model.learn(
    total_timesteps=100_000,
    callback=[checkpoint_callback, eval_callback],
    tb_log_name="toy_run"
)

# Save final model
model.save("toy_ppo_final")

# Test trained model
obs = env.reset()
for _ in range(10):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        break
```

#### Running Training with UV
```bash
# Ensure you're in project root
cd /Users/carrickcheah/Project/ppo/app

# Run training script
uv run python src/training/train_toy.py

# Monitor with tensorboard
uv run tensorboard --logdir ./tensorboard/

# Run with different config
uv run python src/training/train_toy.py --config configs/toy_config.yaml

# Run tests after training
uv run pytest tests/test_toy_env.py -v
```

## Phase 3: Realistic Constraints (Weeks 5-8)

### Week 5: Add Job Dependencies

#### Workflow: Incremental Complexity
```
1. Add Single Feature
   ├── Implement in environment
   ├── Update state/action space
   └── Adjust rewards

2. Test Impact
   ├── Can agent still learn?
   ├── Training stability?
   └── Performance impact?

3. Debug Issues
   ├── Visualize failures
   ├── Adjust implementation
   └── Re-test
```

#### Implementation Updates
```python
class DependencySchedulingEnv(ToySchedulingEnv):
    """Extended environment with job dependencies"""
    
    def __init__(self):
        super().__init__()
        
        # Define job families and dependencies
        self.families = {
            'F1': ['J1_P01', 'J1_P02'],
            'F2': ['J2_P01', 'J2_P02', 'J2_P03'],
            'F3': ['J3_P01']  # Single process
        }
        
        # Flatten all jobs
        self.all_jobs = []
        self.job_to_family = {}
        self.job_dependencies = {}
        
        for family, processes in self.families.items():
            self.all_jobs.extend(processes)
            for i, job in enumerate(processes):
                self.job_to_family[job] = family
                if i > 0:
                    self.job_dependencies[job] = processes[i-1]
                    
        self.n_jobs = len(self.all_jobs)
        
        # Update spaces
        self._update_spaces()
        
    def _can_schedule(self, job_idx):
        """Check if job can be scheduled (dependencies met)"""
        job = self.all_jobs[job_idx]
        
        if job not in self.job_dependencies:
            return True  # No dependencies
            
        dep_job = self.job_dependencies[job]
        dep_idx = self.all_jobs.index(dep_job)
        
        return self.job_completed[dep_idx]
    
    def step(self, action):
        """Modified step with dependency checking"""
        if action < self.n_jobs:
            if not self._can_schedule(action):
                # Invalid action - dependency not met
                return self._get_observation(), -20, False, {'error': 'dependency'}
                
        return super().step(action)
```

### Week 6: Add Time Constraints

#### Time-Aware Environment
```python
class TimeConstrainedEnv(DependencySchedulingEnv):
    """Add working hours and overtime costs"""
    
    def __init__(self):
        super().__init__()
        
        # Time parameters
        self.day_length = 24
        self.work_start = 6.5   # 6:30 AM
        self.work_end = 17.5    # 5:30 PM
        self.break_times = [(12, 13)]  # Lunch break
        
        # Overtime multipliers
        self.normal_cost = 1.0
        self.overtime_cost = 1.5
        self.emergency_cost = 3.0
        
    def _get_time_cost(self, current_hour):
        """Calculate cost multiplier for current time"""
        hour_of_day = current_hour % self.day_length
        
        if self.work_start <= hour_of_day <= self.work_end:
            # Check if break time
            for break_start, break_end in self.break_times:
                if break_start <= hour_of_day < break_end:
                    return self.emergency_cost
            return self.normal_cost
        elif hour_of_day < 2 or hour_of_day > 22:  # Night shift
            return self.emergency_cost
        else:
            return self.overtime_cost
            
    def step(self, action):
        obs, reward, done, info = super().step(action)
        
        # Apply time-based cost
        if 'scheduled_job' in info:
            time_cost = self._get_time_cost(self.current_time)
            reward = reward / time_cost  # Reduce reward for expensive times
            
        return obs, reward, done, info
```

### Week 7: Scale to Medium Size

#### Performance Optimization Workflow
```
1. Profile Current Code
   ├── Identify bottlenecks
   ├── Memory usage
   └── Computation time

2. Optimize Critical Paths
   ├── Vectorize operations
   ├── Cache computations
   └── Parallel processing

3. Benchmark Improvements
   ├── Speed comparison
   ├── Maintain correctness
   └── Document changes
```

### Week 8: Benchmark Against Greedy

#### Comprehensive Benchmark Suite
```python
# benchmark.py
import time
import numpy as np
import pandas as pd
from collections import defaultdict

class BenchmarkSuite:
    """Compare RL agent with baseline algorithms"""
    
    def __init__(self, env_class, n_scenarios=100):
        self.env_class = env_class
        self.n_scenarios = n_scenarios
        self.results = defaultdict(list)
        
    def benchmark_random(self):
        """Random action baseline"""
        for _ in range(self.n_scenarios):
            env = self.env_class()
            obs = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            start_time = time.time()
            while not done:
                action = env.action_space.sample()
                obs, reward, done, _ = env.step(action)
                total_reward += reward
                steps += 1
                
            self.results['random'].append({
                'reward': total_reward,
                'steps': steps,
                'time': time.time() - start_time,
                'makespan': np.max(env.machine_loads)
            })
            
    def benchmark_greedy(self):
        """Simple greedy algorithm"""
        for _ in range(self.n_scenarios):
            env = self.env_class()
            obs = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            start_time = time.time()
            while not done:
                # Greedy: schedule shortest available job on least loaded machine
                available_jobs = [i for i in range(env.n_jobs) 
                                if not env.job_scheduled[i] and env._can_schedule(i)]
                
                if available_jobs:
                    # Pick shortest job
                    job_times = [env.job_times[i] for i in available_jobs]
                    action = available_jobs[np.argmin(job_times)]
                else:
                    action = env.n_jobs  # Wait
                    
                obs, reward, done, _ = env.step(action)
                total_reward += reward
                steps += 1
                
            self.results['greedy'].append({
                'reward': total_reward,
                'steps': steps,
                'time': time.time() - start_time,
                'makespan': np.max(env.machine_loads)
            })
            
    def benchmark_rl(self, model):
        """Benchmark trained RL model"""
        for _ in range(self.n_scenarios):
            env = self.env_class()
            obs = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            start_time = time.time()
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                total_reward += reward
                steps += 1
                
            self.results['rl'].append({
                'reward': total_reward,
                'steps': steps,
                'time': time.time() - start_time,
                'makespan': np.max(env.machine_loads)
            })
            
    def generate_report(self):
        """Generate comparison report"""
        df_results = pd.DataFrame()
        
        for method, results in self.results.items():
            df = pd.DataFrame(results)
            df['method'] = method
            df_results = pd.concat([df_results, df])
            
        # Calculate statistics
        summary = df_results.groupby('method').agg({
            'reward': ['mean', 'std'],
            'steps': ['mean', 'std'],
            'time': ['mean', 'std'],
            'makespan': ['mean', 'std']
        })
        
        print("=== Benchmark Results ===")
        print(summary)
        
        # Save detailed results
        df_results.to_csv('benchmark_results.csv', index=False)
        
        return summary
```

## Phase 4: Full Environment (Weeks 9-12)

### Week 9: Real Data Integration

#### Database Integration Workflow
```
1. Data Extraction
   ├── Connect to MariaDB
   ├── Extract schemas
   └── Sample data

2. Data Analysis
   ├── Job patterns
   ├── Machine characteristics
   └── Constraint mapping

3. Environment Design
   ├── State representation
   ├── Action mapping
   └── Reward alignment

4. Validation
   ├── Compare with real schedules
   ├── Constraint satisfaction
   └── Performance metrics
```

#### Implementation: production_env.py
```python
import mysql.connector
import numpy as np
from datetime import datetime, timedelta

class ProductionSchedulingEnv(gym.Env):
    """Full production environment with real constraints"""
    
    def __init__(self, db_config, planning_horizon_days=90):
        super().__init__()
        
        # Database connection
        self.db_config = db_config
        self.planning_horizon = planning_horizon_days
        
        # Load static data
        self._load_machines()
        self._load_process_sequences()
        self._load_time_constraints()
        
        # State/action spaces
        self._setup_spaces()
        
        # Reset
        self.reset()
        
    def _load_machines(self):
        """Load machine configurations from database"""
        conn = mysql.connector.connect(**self.db_config)
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT machine_id, machine_type, capacity,
                   setup_time_same, setup_time_different
            FROM machines
            WHERE status = 'ACTIVE'
        """)
        
        self.machines = {m['machine_id']: m for m in cursor.fetchall()}
        self.n_machines = len(self.machines)
        
        conn.close()
        
    def _load_process_sequences(self):
        """Extract family process sequences"""
        conn = mysql.connector.connect(**self.db_config)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DISTINCT family_name, process_order
            FROM production_jobs
            WHERE created_at > DATE_SUB(NOW(), INTERVAL 180 DAY)
            ORDER BY family_name, 
                     CAST(SUBSTRING_INDEX(process_order, 'P', -1) AS UNSIGNED)
        """)
        
        self.family_sequences = defaultdict(list)
        for family, process in cursor.fetchall():
            self.family_sequences[family].append(process)
            
        conn.close()
        
    def _load_current_jobs(self):
        """Load jobs that need scheduling"""
        conn = mysql.connector.connect(**self.db_config)
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT job_id, family_name, process_order,
                   processing_time, priority, lcd_date,
                   plan_date, machine_type
            FROM production_jobs
            WHERE status IN ('PENDING', 'READY')
            AND plan_date <= DATE_ADD(NOW(), INTERVAL %s DAY)
            ORDER BY priority DESC, plan_date ASC
            LIMIT 500
        """, (self.planning_horizon,))
        
        self.jobs = cursor.fetchall()
        self.n_jobs = len(self.jobs)
        
        # Calculate derived features
        for job in self.jobs:
            job['days_late'] = max(0, (datetime.now() - job['plan_date']).days)
            job['lcd_urgency'] = (job['lcd_date'] - datetime.now()).days
            
        conn.close()
        
    def _setup_spaces(self):
        """Define observation and action spaces"""
        # State vector size
        machine_features = 4  # utilization, avg_lateness, job_count, hours_busy
        job_features = 8      # priority, days_late, proc_time, etc.
        global_features = 10  # time, day_of_week, etc.
        
        state_size = (self.n_machines * machine_features + 
                     min(self.n_jobs, 100) * job_features + 
                     global_features)
        
        self.observation_space = gym.spaces.Box(
            low=0, high=1,
            shape=(state_size,),
            dtype=np.float32
        )
        
        # Actions: schedule job i on machine j or wait
        self.action_space = gym.spaces.Discrete(
            self.n_jobs + 1  # +1 for wait action
        )
```

### Week 10: State & Action Design

#### State Engineering Workflow
```
1. Feature Selection
   ├── Domain knowledge
   ├── Correlation analysis
   └── Dimensionality reduction

2. Normalization Strategy
   ├── Min-max scaling
   ├── Standardization
   └── Custom transforms

3. Temporal Features
   ├── Time encoding
   ├── Cyclical features
   └── Trend indicators

4. Validation
   ├── State coverage
   ├── Information content
   └── Stability
```

### Week 11: Reward Engineering

#### Reward Design Principles
```
1. Align with Business KPIs
   ├── On-time delivery rate
   ├── Machine utilization
   └── Overtime costs

2. Balance Multiple Objectives
   ├── Weighted sum
   ├── Hierarchical rewards
   └── Constraint penalties

3. Avoid Reward Hacking
   ├── Test edge cases
   ├── Monitor behavior
   └── Iterative refinement
```

### Week 12: Training at Scale

#### Distributed Training Setup
```python
# train_production.py
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

def make_env(rank, seed=0):
    """Create environment instance for parallel training"""
    def _init():
        env = ProductionSchedulingEnv(db_config)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

# Create parallel environments
n_envs = 8
env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

# Scaled hyperparameters
model = PPO(
    CustomSchedulingPolicy,
    env,
    learning_rate=linear_schedule(3e-4),
    n_steps=2048 * n_envs,  # Scale with envs
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    max_grad_norm=0.5,
    vf_coef=0.5,
    ent_coef=0.01,
    tensorboard_log="./logs/production/",
    device='cuda'  # Use GPU
)

# Training with curriculum
curriculum_stages = [
    {'timesteps': 1_000_000, 'job_limit': 50},
    {'timesteps': 2_000_000, 'job_limit': 100},
    {'timesteps': 5_000_000, 'job_limit': 200},
    {'timesteps': 10_000_000, 'job_limit': 500}
]

for stage in curriculum_stages:
    print(f"Training stage: {stage['job_limit']} jobs")
    env.env_method('set_job_limit', stage['job_limit'])
    model.learn(
        total_timesteps=stage['timesteps'],
        reset_num_timesteps=False,
        callback=[eval_callback, checkpoint_callback]
    )
```

## Phase 5: Validation & Safety (Weeks 13-14)

### Week 13: Comprehensive Testing

#### Test Suite Development
```python
# test_suite.py
class SchedulingTestSuite:
    """Comprehensive testing for RL scheduler"""
    
    def __init__(self, model, env_class):
        self.model = model
        self.env_class = env_class
        self.test_results = {}
        
    def test_constraint_satisfaction(self, n_episodes=100):
        """Test all hard constraints"""
        violations = {
            'dependency': 0,
            'machine_compatibility': 0,
            'time_overlap': 0,
            'working_hours': 0
        }
        
        for _ in range(n_episodes):
            env = self.env_class()
            obs = env.reset()
            done = False
            
            while not done:
                action, _ = self.model.predict(obs)
                obs, _, done, info = env.step(action)
                
                # Check violations
                if 'constraint_violation' in info:
                    violations[info['violation_type']] += 1
                    
        self.test_results['constraints'] = violations
        return all(v == 0 for v in violations.values())
        
    def test_stress_scenarios(self):
        """Test edge cases and stress scenarios"""
        scenarios = [
            {'name': 'all_urgent', 'setup': self._setup_all_urgent},
            {'name': 'machine_breakdown', 'setup': self._setup_breakdown},
            {'name': 'rush_order', 'setup': self._setup_rush_order},
            {'name': 'overload', 'setup': self._setup_overload}
        ]
        
        for scenario in scenarios:
            env = self.env_class()
            scenario['setup'](env)
            
            # Run episode
            obs = env.reset()
            done = False
            metrics = self._run_episode(env, obs)
            
            self.test_results[scenario['name']] = metrics
            
    def test_performance_metrics(self):
        """Measure KPIs across scenarios"""
        metrics = {
            'avg_lateness': [],
            'makespan': [],
            'utilization': [],
            'overtime_hours': [],
            'setup_time': []
        }
        
        for _ in range(100):
            env = self.env_class()
            schedule = self._generate_schedule(env)
            
            # Calculate metrics
            metrics['avg_lateness'].append(self._calc_avg_lateness(schedule))
            metrics['makespan'].append(self._calc_makespan(schedule))
            metrics['utilization'].append(self._calc_utilization(schedule))
            metrics['overtime_hours'].append(self._calc_overtime(schedule))
            metrics['setup_time'].append(self._calc_setup_time(schedule))
            
        # Summary statistics
        self.test_results['performance'] = {
            metric: {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
            for metric, values in metrics.items()
        }
        
    def generate_report(self):
        """Generate comprehensive test report"""
        report = []
        report.append("=== RL Scheduler Test Report ===\n")
        
        # Constraint satisfaction
        report.append("1. Constraint Satisfaction:")
        for constraint, violations in self.test_results['constraints'].items():
            status = "✓ PASS" if violations == 0 else f"✗ FAIL ({violations} violations)"
            report.append(f"   - {constraint}: {status}")
            
        # Stress tests
        report.append("\n2. Stress Test Results:")
        for scenario, metrics in self.test_results.items():
            if scenario.startswith('test_'):
                report.append(f"   - {scenario}: {metrics}")
                
        # Performance
        report.append("\n3. Performance Metrics:")
        for metric, stats in self.test_results['performance'].items():
            report.append(f"   - {metric}:")
            report.append(f"     Mean: {stats['mean']:.2f} (±{stats['std']:.2f})")
            report.append(f"     Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
            
        return "\n".join(report)
```

### Week 14: Safety Wrapper Implementation

#### Production Safety Architecture
```python
# safe_scheduler.py
import logging
from enum import Enum
from typing import List, Dict, Optional

class ScheduleStatus(Enum):
    VALID = "valid"
    INVALID = "invalid"
    PARTIAL = "partial"

class SafetyValidator:
    """Validate schedules against all constraints"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def validate_schedule(self, schedule: List[Dict]) -> ScheduleStatus:
        """Complete validation of proposed schedule"""
        
        checks = [
            self._check_dependencies,
            self._check_machine_compatibility,
            self._check_time_overlaps,
            self._check_working_hours,
            self._check_capacity_limits
        ]
        
        for check in checks:
            if not check(schedule):
                return ScheduleStatus.INVALID
                
        return ScheduleStatus.VALID
        
    def _check_dependencies(self, schedule):
        """Verify all job dependencies are respected"""
        scheduled_jobs = {job['job_id']: job for job in schedule}
        
        for job in schedule:
            if 'dependencies' in job:
                for dep_id in job['dependencies']:
                    if dep_id not in scheduled_jobs:
                        self.logger.error(f"Missing dependency {dep_id} for {job['job_id']}")
                        return False
                        
                    dep_job = scheduled_jobs[dep_id]
                    if dep_job['end_time'] > job['start_time']:
                        self.logger.error(f"Dependency violation: {dep_id} -> {job['job_id']}")
                        return False
                        
        return True

class SafeRLScheduler:
    """Production-safe RL scheduler with fallback"""
    
    def __init__(self, model_path: str, db_config: dict):
        self.rl_model = PPO.load(model_path)
        self.greedy_solver = GreedySolver(db_config)
        self.validator = SafetyValidator(db_config)
        self.logger = logging.getLogger(__name__)
        
        # Metrics
        self.rl_success_rate = 0.95  # Track success rate
        self.fallback_count = 0
        
    def schedule(self, jobs: List[Dict], confidence_threshold: float = 0.8):
        """Generate schedule with safety guarantees"""
        
        try:
            # Attempt RL scheduling
            if self.rl_success_rate > confidence_threshold:
                rl_schedule = self._rl_schedule(jobs)
                
                # Validate
                status = self.validator.validate_schedule(rl_schedule)
                
                if status == ScheduleStatus.VALID:
                    self._update_success_rate(True)
                    return rl_schedule
                else:
                    self.logger.warning("RL schedule invalid, using fallback")
                    self._update_success_rate(False)
                    
        except Exception as e:
            self.logger.error(f"RL scheduling failed: {e}")
            
        # Fallback to greedy
        self.fallback_count += 1
        return self.greedy_solver.solve(jobs)
        
    def _rl_schedule(self, jobs):
        """Generate schedule using RL model"""
        # Create environment with current jobs
        env = ProductionSchedulingEnv.from_jobs(jobs)
        obs = env.reset()
        
        schedule = []
        done = False
        
        while not done:
            action, _ = self.rl_model.predict(obs, deterministic=True)
            obs, _, done, info = env.step(action)
            
            if 'scheduled_job' in info:
                schedule.append(info['scheduled_job'])
                
        return schedule
        
    def _update_success_rate(self, success: bool):
        """Update rolling success rate"""
        # Exponential moving average
        alpha = 0.1
        self.rl_success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * self.rl_success_rate
```

## Phase 6: Production Deployment (Weeks 15-16)

### Week 15: Pilot Deployment

#### Deployment Workflow
```
1. Pre-deployment Checklist
   ├── Model artifacts ready
   ├── Monitoring setup
   ├── Rollback plan
   └── Team training

2. Shadow Mode Deployment
   ├── Run parallel to production
   ├── Compare outputs
   ├── Log all decisions
   └── No real impact

3. Limited Production
   ├── Start with 10% traffic
   ├── Low-priority jobs only
   ├── Monitor closely
   └── Daily reviews

4. Gradual Rollout
   ├── Increase percentage
   ├── Add job types
   ├── Expand time windows
   └── Full deployment
```

#### Monitoring Dashboard
```python
# monitoring.py
from flask import Flask, render_template
import plotly.graph_objs as go
import pandas as pd

app = Flask(__name__)

class SchedulerMonitor:
    """Real-time monitoring for RL scheduler"""
    
    def __init__(self, db_config):
        self.db_config = db_config
        self.metrics = {
            'lateness': [],
            'utilization': [],
            'fallback_rate': [],
            'constraint_violations': []
        }
        
    @app.route('/dashboard')
    def dashboard(self):
        """Main monitoring dashboard"""
        
        # Fetch current metrics
        current_stats = self._get_current_stats()
        
        # Create visualizations
        plots = {
            'lateness_trend': self._plot_lateness_trend(),
            'utilization_heatmap': self._plot_utilization_heatmap(),
            'rl_vs_greedy': self._plot_comparison(),
            'alerts': self._get_alerts()
        }
        
        return render_template('dashboard.html', 
                             stats=current_stats, 
                             plots=plots)
                             
    def _get_current_stats(self):
        """Fetch real-time statistics"""
        conn = mysql.connector.connect(**self.db_config)
        cursor = conn.cursor(dictionary=True)
        
        # Current performance
        cursor.execute("""
            SELECT 
                AVG(DATEDIFF(completion_date, lcd_date)) as avg_lateness,
                COUNT(CASE WHEN completion_date > lcd_date THEN 1 END) / COUNT(*) as late_ratio,
                AVG(machine_utilization) as avg_utilization
            FROM scheduled_jobs
            WHERE scheduled_date >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
        """)
        
        return cursor.fetchone()
        
    def _get_alerts(self):
        """Check for issues requiring attention"""
        alerts = []
        
        # Check fallback rate
        if self.metrics['fallback_rate'][-1] > 0.1:
            alerts.append({
                'level': 'warning',
                'message': 'High fallback rate detected',
                'value': f"{self.metrics['fallback_rate'][-1]:.1%}"
            })
            
        # Check constraint violations
        if self.metrics['constraint_violations'][-1] > 0:
            alerts.append({
                'level': 'error',
                'message': 'Constraint violations detected',
                'value': self.metrics['constraint_violations'][-1]
            })
            
        return alerts
```

### Week 16: Full Production Rollout

#### Final Deployment Checklist
- [ ] All tests passing (>99% constraint satisfaction)
- [ ] Performance metrics better than greedy
- [ ] Inference time < 10 seconds for 500 jobs
- [ ] Monitoring dashboard operational
- [ ] Rollback procedure tested
- [ ] Team trained on new system
- [ ] Documentation complete

## Workflows

### Daily Operational Workflow
```
Morning (8:00 AM):
├── Review overnight performance
├── Check alert dashboard
├── Address any fallbacks
└── Plan day's scheduling

Hourly:
├── Monitor real-time metrics
├── Check constraint satisfaction
├── Review RL vs greedy performance
└── Log any anomalies

End of Day (5:00 PM):
├── Daily performance report
├── Update success metrics
├── Plan next day's improvements
└── Backup model checkpoints
```

### Model Update Workflow
```
Weekly:
1. Collect week's data
2. Analyze failure cases
3. Retrain if needed
4. Test on validation set
5. A/B test if improved
6. Deploy if successful
```

### Incident Response Workflow
```
If constraint violation detected:
1. Immediate fallback to greedy
2. Log full context
3. Alert operations team
4. Investigate root cause
5. Update validation rules
6. Retrain if systematic issue
```

## Risk Management

### Technical Risks
| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Model fails to converge | Medium | High | Multiple algorithms, hyperparameter search |
| Constraint violations | Low | Critical | Hard validation layer, fallback system |
| Performance degradation | Medium | Medium | Continuous monitoring, quick rollback |
| Inference too slow | Low | High | Model optimization, caching strategies |

### Operational Risks
| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Team resistance | Medium | Medium | Training, gradual rollout, show benefits |
| Data quality issues | Medium | High | Data validation, cleaning pipelines |
| System integration | Low | Medium | API compatibility, testing |

## Success Metrics

### Phase Milestones
1. **Toy Problem (Week 4)**
   - [x] PPO converges reliably
   - [x] Beats random baseline by 50%
   - [x] Training time < 1 hour

2. **Realistic Scale (Week 8)**  
   - [x] Handles 50 jobs, 10 machines
   - [x] Respects all constraints
   - [x] Within 20% of greedy performance

3. **Full Environment (Week 12)**
   - [x] Processes 500 jobs successfully
   - [x] Reduces average lateness by 10%
   - [x] Inference time < 10 seconds

4. **Production Ready (Week 16)**
   - [x] 99.9% constraint satisfaction
   - [x] Beats greedy on all KPIs
   - [x] Successfully deployed

### KPI Targets
| Metric | Current (Greedy) | Target (RL) | Stretch Goal |
|--------|------------------|-------------|--------------|
| Avg Days Late | 25.9 | 20.0 | 15.0 |
| Schedule Time | 4.09s | <10s | <5s |
| Machine Utilization Std | High | -20% | -40% |
| On-time Delivery | ~60% | 75% | 85% |
| Overtime Hours | Baseline | -10% | -20% |

## Appendix: Code Templates

### Custom Policy Network
```python
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class SchedulingFeaturesExtractor(BaseFeaturesExtractor):
    """Custom feature extractor for scheduling state"""
    
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        
        n_input = observation_space.shape[0]
        
        # Separate processing for different feature types
        self.machine_encoder = nn.Sequential(
            nn.Linear(74 * 4, 128),  # 74 machines * 4 features
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        self.job_encoder = nn.Sequential(
            nn.Linear(100 * 8, 256),  # 100 jobs * 8 features
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        self.global_encoder = nn.Sequential(
            nn.Linear(10, 32),  # 10 global features
            nn.ReLU()
        )
        
        # Combine all
        self.combine = nn.Sequential(
            nn.Linear(64 + 128 + 32, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations):
        # Split observation
        machine_features = observations[:, :74*4]
        job_features = observations[:, 74*4:74*4+100*8]
        global_features = observations[:, -10:]
        
        # Encode separately
        machine_encoded = self.machine_encoder(machine_features)
        job_encoded = self.job_encoder(job_features)
        global_encoded = self.global_encoder(global_features)
        
        # Combine
        combined = torch.cat([machine_encoded, job_encoded, global_encoded], dim=1)
        return self.combine(combined)
```

### Hyperparameter Tuning
```python
import optuna
from stable_baselines3.common.evaluation import evaluate_policy

def objective(trial):
    """Optuna objective for hyperparameter search"""
    
    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    n_steps = trial.suggest_categorical('n_steps', [512, 1024, 2048, 4096])
    gamma = trial.suggest_uniform('gamma', 0.9, 0.999)
    gae_lambda = trial.suggest_uniform('gae_lambda', 0.9, 0.99)
    ent_coef = trial.suggest_loguniform('ent_coef', 1e-4, 1e-1)
    
    # Create model with suggested parameters
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        verbose=0
    )
    
    # Train
    model.learn(total_timesteps=100_000)
    
    # Evaluate
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10)
    
    return mean_reward

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("Best parameters:", study.best_params)
```

## Final Notes

This plan provides a structured path from zero RL knowledge to a production-ready scheduling system. Key success factors:

1. **Start Simple**: Don't jump to the full problem immediately
2. **Iterate Quickly**: Fail fast and learn from each iteration
3. **Measure Everything**: You can't improve what you don't measure
4. **Safety First**: Always have fallback options
5. **Gradual Deployment**: Build confidence through incremental rollout

Good luck with your implementation!