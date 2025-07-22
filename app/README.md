# Deep Reinforcement Learning Scheduling System

## Project Status: Ready for Production Deployment (Week 13 of 16)

### Phase 4 Achievement ✅ (PRODUCTION READY)
- **Curriculum Learning Success**: Scaled from 2 → 10 → 40 → 152 machines
- **Phase 4 Performance**: 49.2h makespan with 100% completion rate
- **Full Production Scale**: 152 machines, 172 jobs/batch (3 batches for 411 total)
- **State Compression**: Successfully reduced from 505 to 60 features
- **Sub-linear Scaling**: 3.8x machines → only 2.5x makespan increase
- **API & Safety**: FastAPI server and SafeScheduler implemented (July 21)

### Phase 5 Results (Hierarchical Action Space) ✅
- **Problem Solved**: Action space reduced from 46,400 → 465 (99% reduction)
- **Best Performance**: 31% job scheduling (98/320 jobs) at 300k steps
- **Key Learning**: Hierarchical approach validated but needs action masking
- **Limitation**: 90% invalid action rate throughout training
- **Decision**: Deploy Phase 4 model to production while researching improvements

## Project Location
This project is located at: `/Users/carrickcheah/Project/ppo/app`

## Table of Contents
1. [Overview & Architecture](#overview--architecture)
2. [Completed Phases](#completed-phases)
3. [Phase 4 Production Model](#phase-4-results-complete-)
4. [Phase 5 Results: Hierarchical Action Space](#phase-5-results-hierarchical-action-space-)
5. [Upcoming: Production Deployment](#upcoming-phases)
6. [Development Workflow](#development-workflow)
7. [Technical Implementation](#technical-implementation)
8. [Success Metrics](#success-metrics)

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

## Completed Phases

### Phase 1-2: Foundation & Toy Environment (Weeks 1-4) ✅
- **Achievement**: Built and trained PPO agent on 2 machines, 5 jobs
- **Performance**: 81.25% utilization, 20-33% better than random
- **Key Learning**: Basic RL concepts, PPO implementation, reward design

### Phase 3: Scaled Production (Weeks 5-8) ✅
- **Achievement**: Successfully scaled to 40 machines with real production data
- **Performance**: 
  - 10 machines: 86.3h makespan
  - 20 machines: 21.0h makespan  
  - 40 machines: 19.7h makespan (best result)
- **Key Features**: 
  - Boolean importance system (replaced 1-5 priority)
  - Machine type constraints
  - Setup time optimization
  - Break time constraints

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
│   ├── toy_config.yaml
│   ├── medium_config.yaml
│   ├── production_config.yaml
│   ├── scaled_production_config.yaml
│   └── phase4_config.yaml
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

## Current Status: Production Deployment Ready

### Phase 5 Complete: Hierarchical Action Space Research
- **Problem Solved**: Phase 4 limitation of max_valid_actions=200 prevented single-pass scheduling
- **Solution Implemented**: Hierarchical action space - job selection → machine selection
- **Technical Achievements**:
  - ✅ Two-stage decision making approach working
  - ✅ Reduced action space from 46,400 → 465 (99% reduction)
  - ✅ MultiDiscrete wrapper for SB3 compatibility
  - ✅ All 320 real jobs visible in single pass
  - ✅ Extended training to 750k timesteps
  
### Phase 5 Training Results
- **Best Performance**: 300k steps achieving 31% job scheduling (98/320 jobs)
- **Training Progression**:
  - 100k steps: 30% job scheduling (95 jobs)
  - 300k steps: 31% job scheduling (98 jobs) - PEAK
  - 550k steps: 27% job scheduling (85 jobs)
  - 750k steps: 15% job scheduling (48 jobs)
- **Key Finding**: 90% invalid action rate shows need for action masking

### Phase 5a Iterative Debugging Process
- **Dimension Fix**: Discovered and fixed job count mismatch (411 synthetic vs 320 real)
- **max_valid_actions Fix**: Removed 200 limit → set to 10,000 for full visibility
- **Hyperparameter Tuning**: Higher LR (0.001), entropy (0.1), exploration focus
- **Architecture Changes**: Tried 256x256 networks, reduced penalties
- **Result**: Model improved from 0% to 31% job scheduling but plateaued

### Final Decision
- **Deploy Phase 4 Model**: 49.2h makespan with 100% completion rate
- **Infrastructure Ready**: FastAPI server and SafeScheduler already implemented
- **Future Research**: Action masking and curriculum learning for Phase 5 approach

## Phase 4 Results (Complete) ✅

### Full Production Scale Achievement (152 Machines)
- **Performance**: 49.2h makespan with 100% completion rate
- **Scale**: 152 machines, 172 jobs/batch (3 batches for 411 total)
- **Model Location**: `app/models/full_production/final_model.zip`
- **Key Achievements**:
  - Successfully scaled from 40 to 152 machines
  - Implemented hierarchical state compression (505 → 60 features)
  - Achieved sub-linear scaling (3.8x machines → 2.5x makespan)
  - Maintained 100% job completion rate
  - API development complete (FastAPI server ready)

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

## Upcoming Phases

### Phase 6: Production Deployment (Weeks 14-16) 📅
- Connect Phase 4 model to existing API (already built)
- Comprehensive safety testing with SafeScheduler
- Shadow mode deployment (parallel to production)
- Gradual rollout (10% → 25% → 50% → 100%)
- Real-time monitoring dashboard
- Final performance validation

### Future Research: Enhanced Hierarchical Approach
- Implement sophisticated action masking in policy network
- Try curriculum learning (50 → 100 → 200 → 320 jobs)
- Explore imitation learning from historical schedules
- Research attention mechanisms for job-machine compatibility
- Target: <45h makespan with 100% completion

## Development Workflow

### Current Daily Cycle - Deployment Preparation
```
Morning (2-3 hours):
├── Review API performance tests
├── Validate SafeScheduler constraints
├── Check Phase 4 model integration
└── Prepare deployment checklist

Afternoon (3-4 hours):
├── Shadow mode testing setup
├── Performance benchmarking
├── Integration testing with database
└── Documentation updates

Evening (1 hour):
├── Monitor test results
├── Update deployment plan
└── Coordinate with operations team
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

## Technical Implementation

### Environment Setup with UV

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or on macOS with Homebrew
brew install uv

# Project initialization
cd /Users/carrickcheah/Project/ppo/app
uv init
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

### Quick Start
```bash
# Activate virtual environment
cd /Users/carrickcheah/Project/ppo/app
source .venv/bin/activate

# Install dependencies
uv sync

# Check current training status
uv run python check_training_status.py

# Resume Phase 4 training
uv run python src/training/train_full_production.py --resume
```

### Key Components Implemented

#### Environments
1. **ToySchedulingEnv** (`toy_env.py`): 2 machines, 5 jobs for learning basics
2. **MediumEnvBoolean** (`medium_env_boolean.py`): 10 machines with boolean importance
3. **ScaledProductionEnv** (`scaled_production_env.py`): 40 machines with full constraints
4. **FullProductionEnv** (`full_production_env.py`): 152 machines with state compression

#### Key Features
- **State Compression**: Hierarchical (60), Compressed (20), or Full (505) features
- **Boolean Importance**: Simplified from 1-5 priority system
- **Break Time Constraints**: Integrated with MariaDB `ai_breaktimes` table
- **Machine Type Handling**: 42 types compressed to 10 for state space
- **Transfer Learning**: Support for loading previous phase models
- **Parallel Training**: 8 concurrent environments for efficiency

### Database Integration
- **MariaDB Connection**: Using `pymysql` for production data
- **Tables Used**: 
  - `tbl_machine`: Machine configurations (151 machines)
  - `ai_breaktimes`: Break schedules and working hours
  - `ai_holidays`: Holiday calendar (future integration)
- **Data Pipeline**: Extract → Transform → Environment → Train

### Training Pipeline
1. **Curriculum Learning**: Start small, scale gradually
2. **Transfer Learning**: Load previous phase models
3. **Parallel Training**: Multiple environments for efficiency
4. **Checkpointing**: Save every 50,000 steps
5. **Evaluation**: Compare with Random, FirstFit, Priority baselines

### Current Models and Results
- **Phase 1-2**: Toy model achieved 81.25% utilization
- **Phase 3**: Best model at 40 machines - 19.7h makespan
- **Phase 4**: Full production - 49.2h makespan (100% completion)
- **Phase 5**: Hierarchical approach - training challenges

## Performance Benchmarks

| Phase | Scale | Best Result | Status |
|-------|-------|-------------|---------|
| Phase 1-2 | 2 machines, 5 jobs | 81.25% utilization | ✅ Complete |
| Phase 3 | 40 machines, 172 jobs | 19.7h makespan | ✅ Complete |
| Phase 4 | 152 machines, 172 jobs/batch | 49.2h makespan, 100% completion | ✅ Complete (PRODUCTION READY) |
| Phase 5 | 145 machines, 320 jobs | 31% job scheduling, 99% action reduction | ✅ Complete (Needs enhancement) |

### Key Technical Challenges Solved

1. **Observation Space Mismatch**: Fixed by calling `_update_observation_space()` in environment init
2. **Machine Type Handling**: Capped 42 types to 10 for fixed state size
3. **NaN Data Handling**: Added validation for database extraction
4. **Activation Function**: Switched from None to nn.Tanh for PPO
5. **Date Handling**: Fixed datetime object method calls

### Current Files Structure
- `src/environments/`: All environment implementations
- `src/training/`: Training scripts for each phase
- `src/utils/`: Database connectors, data parsers, visualizers
- `configs/`: YAML configuration files
- `models/`: Saved model checkpoints
- `data/`: Generated datasets and snapshots

## Future Work

### Planned Enhancements
1. **Dynamic Job Arrivals**: Handle real-time job submissions
2. **Machine Breakdowns**: Recovery and rescheduling strategies
3. **Multi-Objective Optimization**: Balance makespan, energy, quality
4. **Distributed Training**: Scale to multiple GPUs
5. **Online Learning**: Continuous improvement in production

## Monitoring and Operations

### Real-time Metrics
- **Training Progress**: TensorBoard logs in `./logs/`
- **Model Checkpoints**: Saved every 50k steps in `./models/`
- **Performance Tracking**: Episode rewards, makespan, utilization
- **Constraint Violations**: Logged for debugging

### Deployment Strategy
1. **Shadow Mode**: Run parallel to production without impact
2. **A/B Testing**: Compare RL vs greedy on subset of jobs
3. **Gradual Rollout**: 10% → 25% → 50% → 100% traffic
4. **Fallback Ready**: Automatic switch to greedy if constraints violated
5. **Monitoring Dashboard**: Real-time performance tracking

#### Final Deployment Checklist
- [ ] All tests passing (>99% constraint satisfaction)
- [ ] Performance metrics better than greedy
- [ ] Inference time < 10 seconds for 500 jobs
- [ ] Monitoring dashboard operational
- [ ] Rollback procedure tested
- [ ] Team trained on new system
- [ ] Documentation complete

## Risk Management

### Technical Risks
| Risk | Probability | Impact | Mitigation |
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
1. **Toy Problem (Week 4)** ✅
   - [x] PPO converges reliably
   - [x] Beats random baseline by 20-33%
   - [x] Training time < 1 hour

2. **Realistic Scale (Week 8)** ✅
   - [x] Handles 172 jobs, 40 machines
   - [x] Respects all constraints
   - [x] Achieved 19.7h makespan

3. **Full Environment (Week 12)** ✅
   - [x] Processes 411 jobs (172/batch)
   - [x] 152 machines integrated
   - [x] Training completed: 49.2h makespan
   - [x] API development complete

4. **Hierarchical Approach (Week 13)** ✅
   - [x] Action space reduced 99% (46,400 → 465)
   - [x] Single-pass visibility (all 320 jobs)
   - [x] Training to 750k steps completed
   - [x] Best: 31% job scheduling (needs action masking)

5. **Production Ready (Week 14-16)** 📅
   - [x] Phase 4 model ready (49.2h, 100% completion)
   - [x] API implemented (FastAPI server built)
   - [x] Safety wrapper implemented (SafeScheduler)
   - [ ] Shadow mode testing
   - [ ] Production deployment

### KPI Targets
| Metric | Phase 4 Result | Phase 5 Target | Stretch Goal |
|--------|----------------|----------------|--------------|
| Makespan | 49.2h | <45h | <40h |
| Completion Rate | 100% | 100% | 100% |
| Schedule Time | <1s | <2s | <1s |
| Action Space | 59,595 | 556 | 556 |
| Machine Utilization Std | High | -20% | -40% |
| On-time Delivery | ~60% | 75% | 85% |
| Overtime Hours | Baseline | -10% | -20% |

## Key Lessons Learned

1. **Curriculum Learning Works**: Scaling from 2 → 10 → 40 → 152 machines proved effective
2. **State Compression Critical**: Reduced 505 features to 60 without losing performance
3. **Boolean > Priority Levels**: Simplified importance system performed equally well
4. **Database Integration**: Real production data integration requires careful NaN handling
5. **Transfer Learning**: Important for preserving learned behaviors across phases

## Next Steps

1. **Deploy Phase 4 Model**: Connect to existing FastAPI server
2. **Shadow Mode Testing**: Run parallel to production system
3. **Performance Validation**: Compare with current OR-Tools greedy
4. **Gradual Production Rollout**: 10% → 50% → 100% traffic
5. **Research Phase 5 Enhancements**: Action masking, curriculum learning

---
*Project actively maintained. See z_MEMORY.md for detailed implementation history.*