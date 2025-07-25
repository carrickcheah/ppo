# Pure Deep Reinforcement Learning Scheduling System - TODO List

## Project Status: Phase 1, 1.5, 1.6, and 2 Complete - Ready for Phase 3 (Training)
Last Updated: 2025-07-24
**Objective: Build a pure DRL scheduler that learns everything from experience, like AI learning to play a game**

## Core Principles
- You control the game rules (when to play, what machines exist)
- PPO model focuses only on playing the game (which job on which machine)
- No hardcoded strategies - everything learned through rewards
- Simple interface: Raw data in → Schedule out

## Phase 1: Foundation - Build the Game Environment (Week 1) ✅ COMPLETE

### Step 1: Database Connection Layer ✅
- [x] Create simple MariaDB connector in `/app_2/src/data/`
- [x] Query pending jobs: job_id, family_id, sequence, machine_types, processing_time, lcd_date, is_important
- [x] Query machines: machine_id, machine_type
- [x] No validation, just raw data extraction

### Step 2: Define Game Rules Engine ✅
- [x] Hard Rule 1: Sequence checker (must follow 1→2→3 within family)
- [x] Hard Rule 2: Machine requirements (must use ALL specified machines simultaneously)
- [x] Hard Rule 3: No time overlap (one job per machine)
- [x] ~Hard Rule 4: Working hours~ (REMOVED - deployment only, not training)

### Step 3: Create Gymnasium Environment ✅
- [x] Build `SchedulingGameEnv` class
- [x] State: [jobs_status, machines_load, current_time, time_until_deadline]
- [x] Action: MultiDiscrete([n_jobs, n_machines])
- [x] Valid action masking (only show legal moves)

### Step 4: Implement Reward Function ✅
- [x] On-time completion: +100
- [x] Late delivery: -200
- [x] Important job bonus: +50
- [x] Sequence violation penalty: -500 (discovered after family complete)
- [x] Makespan efficiency: +10 per hour saved

### Step 5: Database Integration & Testing ✅
- [x] Update DBConnector for actual schema mapping
- [x] Test connection and data fetching (233 jobs, 145 machines)
- [x] Parse job families from DocRef_v field
- [x] Extract machine compatibility from Machine_v field
- [x] Handle complex working hours and break schedules

## Phase 1.5: Data Pipeline Fixes ✅ COMPLETE (2025-07-24)

### Data Processing Corrections
- [x] Fix processing time calculation: (JoQty_d / (CapQty_d * 60)) + (SetupTime_d / 60)
- [x] Parse Machine_v as list of ALL required machines (simultaneous occupation)
- [x] Use IsImportant column from tbl_jo_txn (not DifficultyLevel)
- [x] Generate proper job_id: DocRef_v + Task_v
- [x] Remove CycleTime_d (always 0, not used)
- [x] Test with real multi-machine jobs

## Phase 1.6: Environment Updates ✅ COMPLETE (2025-07-24)

### Multi-Machine Handling
- [x] Update environment to handle multi-machine occupation
- [x] Fix rules engine to check ALL required machines are free
- [x] Update action execution to block ALL required machines
- [x] Implement proper action masking for multi-machine constraints
- [x] Remove working hours from training environment
- [x] Test with jobs requiring 5+ machines simultaneously

## Phase 2: Build the PPO Player (Week 2) ✅ COMPLETE (2025-07-24)

### Step 1: Design Flexible State Representation ✅
- [x] Attention mechanism for variable number of jobs (10 to 1000+)
- [x] Job encoding: [sequence_progress, urgency, processing_time, type_embedding]
- [x] Machine encoding: [current_load, type, utilization_rate]
- [x] No fixed size limits

### Step 2: Implement PPO Architecture ✅
- [x] TransformerEncoder for jobs and machines
- [x] Actor network: Outputs action probabilities
- [x] Critic network: Estimates state value
- [x] Action masking layer: Only valid moves

### Step 3: Configure Training Pipeline ✅
- [x] Vectorized environments for parallel training
- [x] Experience buffer implementation
- [x] Hyperparameters: lr=3e-4, batch_size=64, n_epochs=10
- [x] Entropy coefficient: 0.01 (encourage exploration)

### Step 4: Create Training Loop ✅
- [x] Collect experiences through gameplay
- [x] Calculate advantages
- [x] Update policy and value networks
- [x] Track metrics: reward, makespan, on-time rate

## Phase 3: Training - Let it Learn to Play (Week 3) 🚧 IN PROGRESS

### Step 1: Data Preparation & Enhancement ✅ COMPLETE
- [] Create multiple data snapshots (rush orders, normal, heavy load)
- [] Generate 500+ job snapshot with extended planning horizon
- [] Create synthetic variations for edge cases
- [] Prepare multi-machine heavy scenarios (30% multi-machine jobs)

### Step 2: Extended Curriculum Learning (16 stages) 🚧 IN PROGRESS
#### Foundation Training (100k timesteps) ✅ COMPLETE
- [] Toy Easy: 5 jobs, 3 machines - Learn sequence rules
- [] Toy Normal: 10 jobs, 5 machines - Learn deadlines
- [] Toy Hard: 15 jobs, 5 machines - Learn priorities
- [] Toy Multi: 10 jobs, 8 machines - Learn multi-machine

#### Strategy Development (200k timesteps) 🚧 IN PROGRESS
- [] Small Balanced: 30 jobs, 15 machines - Balance objectives
- [] Small Rush: 50 jobs, 20 machines - Handle urgency 
- [ ] Small Bottleneck: 40 jobs, 10 machines - Manage constraints
- [ ] Small Complex: 50 jobs, 25 machines - Complex dependencies

#### Scale Training (300k timesteps)
- [ ] Medium Normal: 150 jobs, 40 machines
- [ ] Medium Stress: 200 jobs, 50 machines
- [ ] Large Intro: 300 jobs, 75 machines
- [ ] Large Advanced: 400 jobs, 100 machines

#### Production Mastery (400k timesteps)
- [ ] Production Warmup: 295 jobs, 145 machines (normal load)
- [ ] Production Rush: 295 jobs, 145 machines (urgent orders)
- [ ] Production Heavy: 500 jobs, 145 machines (overload)
- [ ] Production Expert: 500 jobs, 145 machines (mixed scenarios)

### Step 3: Specialized Training Activities
#### Scenario Variations
- [ ] Deadline pressure training (all urgent, cascading delays)
- [ ] Machine failure simulation (10% down, critical failures)
- [ ] Load pattern training (steady, burst, seasonal, chaotic)
- [ ] Adversarial scenarios (worst-case situations)

#### Multi-Objective Training
- [ ] Rotate reward profiles every 50k steps
- [ ] Deadline-focused vs efficiency-focused training
- [ ] Importance-aware reward shaping
- [ ] Create adaptable behavior across objectives

### Step 4: Advanced Training Techniques
#### Hyperparameter Schedule
- [ ] Dynamic learning rate: 5e-4 → 5e-5 over training
- [ ] Entropy decay: 0.02 → 0.001 for exploration → exploitation
- [ ] Clip range adjustment based on KL divergence

#### Ensemble Training
- [ ] Train 3 models with different seeds
- [ ] Compare different architectures
- [ ] Test ensemble decisions
- [ ] Select best performer

### Step 5: Continuous Evaluation & Improvement
#### Benchmark Testing (every 25k steps)
- [ ] Test against FIFO, EDD, SPT, Critical Ratio
- [ ] Run stress tests (1000 jobs, 200 machines)
- [ ] Measure constraint violations, on-time rate, makespan
- [ ] Document performance progression

#### Weakness Detection & Retraining
- [ ] Identify failure patterns
- [ ] Create targeted scenarios for weaknesses
- [ ] Implement continuous improvement loop
- [ ] Log discovered strategies

### Step 6: Monitoring & Analysis
#### Real-Time Monitoring
- [ ] Live dashboard with training metrics
- [ ] Strategy logger for interesting behaviors
- [ ] Performance tracker with alerts
- [ ] Training diary documentation

#### Analysis Reports
- [ ] Daily training summaries
- [ ] Breakthrough moment documentation
- [ ] Failure analysis and fixes
- [ ] Emergent strategy catalog

### Success Criteria
#### Minimum Requirements
- [ ] 95% constraint satisfaction
- [ ] 85% on-time delivery rate
- [ ] Handle 500+ jobs smoothly
- [ ] <100ms inference time

#### Excellence Targets
- [ ] 98% constraint satisfaction
- [ ] 95% on-time delivery
- [ ] Optimal makespan (within 5% theoretical best)
- [ ] Discover novel scheduling strategies

## Phase 4: Deployment - Simple API (Week 4)

### Working Hours Filter (NEW)
- [ ] Apply working hours as post-processing filter
- [ ] Shift jobs to valid time windows
- [ ] Maintain optimized sequence from model
- [ ] Different hours per day (Mon-Thu, Fri, Sat)
- [ ] Handle break times (lunch, tea, maintenance)

### Step 1: Create Inference Server
- [ ] Build FastAPI server at `/app_2/src/api/`
- [ ] Single endpoint: POST /schedule
- [ ] Input: Raw jobs from database
- [ ] Output: Complete schedule with timings

### Step 2: Connect to Front2
- [ ] Ensure API returns same format as current system
- [ ] No changes needed in frontend
- [ ] Real-time scheduling (<1 second for 1000 jobs)
- [ ] Test with existing visualization

### Step 3: Production Testing
- [ ] Start with small batches
- [ ] Compare with current system
- [ ] Measure improvements
- [ ] Gather feedback

### Step 4: Continuous Learning Setup
- [ ] Save production schedules
- [ ] Retrain weekly with real data
- [ ] Model improves over time
- [ ] Track performance metrics

## Phase 5: Pure AI Evolution (Week 5+)

### Step 1: Remove Training Wheels
- [ ] Reduce reward shaping gradually
- [ ] Let model discover more strategies
- [ ] Trust the learning process
- [ ] Document emergent behaviors

### Step 2: Advanced Architectures
- [ ] Experiment with larger transformers
- [ ] Try graph neural networks
- [ ] Explore hierarchical policies
- [ ] Test multi-agent systems

### Step 3: Performance Optimization
- [ ] Optimize inference speed
- [ ] Reduce memory footprint
- [ ] Implement model quantization
- [ ] Edge deployment options

### Step 4: Document AI Discoveries
- [ ] Strategies humans never tried
- [ ] Unexpected optimizations
- [ ] New scheduling patterns
- [ ] Lessons learned

## Completed Components ✅

### Environment Structure
```
/app_2/
├── src/
│   ├── environment/
│   │   ├── scheduling_game_env.py    # Game rules and physics ✅
│   │   ├── rules_engine.py           # Hard constraints ✅
│   │   └── reward_function.py        # Soft preferences ✅
│   ├── model/
│   │   ├── ppo_scheduler.py          # PPO player (Phase 2)
│   │   ├── transformer_policy.py     # Neural architecture (Phase 2)
│   │   └── action_masking.py         # Valid moves only (Phase 2)
│   ├── training/
│   │   ├── train.py                  # Main training loop (Phase 3)
│   │   ├── curriculum.py             # Progressive difficulty (Phase 3)
│   │   └── evaluate.py               # Performance testing (Phase 3)
│   ├── data/
│   │   ├── db_connector.py           # MariaDB interface ✅
│   │   └── data_loader.py            # Job/machine loading ✅
│   └── api/
│       ├── server.py                 # FastAPI application (Phase 4)
│       └── models.py                 # Request/response schemas (Phase 4)
├── configs/
│   ├── environment.yaml              # Game settings ✅
│   ├── training.yaml                 # Hyperparameters ✅
│   └── deployment.yaml               # Production config ✅
└── tests/
    ├── test_environment.py           # Rule validation ✅
    ├── test_db_connection.py         # Database testing ✅
    ├── test_model.py                 # Architecture tests (Phase 2)
    └── test_api.py                   # Integration tests (Phase 4)
```

## Success Metrics
- [ ] Handle 1000+ jobs without batching
- [ ] Learn sequence constraints without being told
- [ ] Discover priority patterns from LCD dates
- [ ] Achieve 95%+ on-time delivery through learning alone
- [ ] Zero hardcoded scheduling rules in final system
- [ ] API response time <1 second
- [ ] 25%+ improvement over manual scheduling

## Current Issues to Fix 🔧

### Small Rush 0% Utilization Problem
- **Issue**: Model learns to do nothing (0% utilization) to avoid penalties
- **Root Cause**: Reward structure penalizes late jobs more than idle time
- **Required Fixes**:
  - [ ] Add completion bonus (+1.0 per job) to reward function
  - [ ] Reduce late penalty magnitude or use graduated penalties
  - [ ] Increase entropy coefficient from 0.01 to 0.05+ for exploration
  - [ ] Create "rush_order" reward profile that tolerates some lateness
  - [ ] Fix observation space mismatch between training and testing
  - [ ] Add 'name' field to stage configurations

## Timeline Summary
- **Week 1**: Build game environment with rules ✅ COMPLETE
- **Week 1.5**: Refine data processing & environment ✅ COMPLETE (2025-07-24)
- **Week 2**: Create PPO player architecture ✅ COMPLETE (2025-07-24)
- **Week 3**: Train model to play the game 📋 NEXT
- **Week 4**: Deploy simple API
- **Week 5+**: Let AI evolve and discover

## Next Immediate Actions
1. ~~Set up `/app_2/` project structure~~ ✅
2. ~~Create database connection to fetch real jobs~~ ✅
3. ~~Build minimal game environment with rules~~ ✅
4. ~~Implement simple reward function~~ ✅
5. ~~Test with 10 jobs manually to verify rules~~ ✅
6. ~~Connect to production database and test~~ ✅
7. ~~Fix db_connector.py with correct processing time formula~~ ✅ (2025-07-24)
8. ~~Parse Machine_v as multi-machine requirements~~ ✅ (2025-07-24)
9. ~~Update environment for multi-machine handling~~ ✅ (2025-07-24)
10. ~~Design transformer architecture for variable job sizes~~ ✅ (2025-07-24)
11. ~~Implement PPO algorithm with action masking~~ ✅ (2025-07-24)
12. ~~Create training loop with curriculum learning~~ ✅ (2025-07-24)
13. ~~Run comprehensive tests on all components~~ ✅ (2025-07-24)
14. **NEXT**: Start training with curriculum learning (Phase 3)
15. **NEXT**: Monitor training metrics and adjust hyperparameters
16. **NEXT**: Deploy inference API when training completes

## Key Insights from Discussion
- **Multi-Machine Jobs**: Machine_v="57,64,65,66,74" means job needs ALL 5 machines simultaneously
- **Processing Time**: Use capacity formula when CapMin_d=1: (JoQty_d/(CapQty_d*60)) + (SetupTime_d/60)
- **Working Hours**: Apply as filter during deployment, not in training
- **Constraints**: Hard (sequence, machines, overlap) vs Soft (deadlines, importance, efficiency)

---
*Note: Critical understanding - jobs can require multiple machines working together. This fundamentally changes our action space and state representation.*