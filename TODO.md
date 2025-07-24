# Pure Deep Reinforcement Learning Scheduling System - TODO List

## Project Status: Phase 1, 1.5, and 1.6 Complete - Ready for Phase 2 (PPO Model)
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

## Phase 2: Build the PPO Player (Week 2) 📋 PENDING

### Step 1: Design Flexible State Representation
- [ ] Attention mechanism for variable number of jobs (10 to 1000+)
- [ ] Job encoding: [sequence_progress, urgency, processing_time, type_embedding]
- [ ] Machine encoding: [current_load, type, utilization_rate]
- [ ] No fixed size limits

### Step 2: Implement PPO Architecture
- [ ] TransformerEncoder for jobs and machines
- [ ] Actor network: Outputs action probabilities
- [ ] Critic network: Estimates state value
- [ ] Action masking layer: Only valid moves

### Step 3: Configure Training Pipeline
- [ ] Vectorized environments for parallel training
- [ ] Experience buffer implementation
- [ ] Hyperparameters: lr=3e-4, batch_size=64, n_epochs=10
- [ ] Entropy coefficient: 0.01 (encourage exploration)

### Step 4: Create Training Loop
- [ ] Collect experiences through gameplay
- [ ] Calculate advantages
- [ ] Update policy and value networks
- [ ] Track metrics: reward, makespan, on-time rate

## Phase 3: Training - Let it Learn to Play (Week 3)

### Step 1: Curriculum Learning Setup
- [ ] Level 1: 10 jobs, 5 machines (learn basics)
- [ ] Level 2: 50 jobs, 20 machines (learn strategies)
- [ ] Level 3: 200 jobs, 50 machines (learn scaling)
- [ ] Level 4: 1000+ jobs, 100+ machines (production scale)

### Step 2: Diverse Scenario Generation
- [ ] Tight deadline scenarios
- [ ] Machine bottleneck scenarios
- [ ] Mixed priority scenarios
- [ ] Various working hour patterns

### Step 3: Train and Monitor
- [ ] Run 1 million episodes
- [ ] Track learning curves
- [ ] Identify emergent strategies
- [ ] Document unexpected patterns

### Step 4: Validate Learned Behaviors
- [ ] Check if model learned sequence rules
- [ ] Verify deadline prioritization emerged
- [ ] Confirm load balancing discovered
- [ ] No manual intervention - pure learning

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

## Timeline Summary
- **Week 1**: Build game environment with rules ✅ COMPLETE
- **Week 1.5**: Refine data processing & environment ✅ COMPLETE (2025-07-24)
- **Week 2**: Create PPO player architecture 📋 NEXT
- **Week 3**: Train model to play the game
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
10. **NEXT**: Design transformer architecture for variable job sizes
11. **NEXT**: Implement PPO algorithm with action masking
12. **NEXT**: Create training loop with curriculum learning

## Key Insights from Discussion
- **Multi-Machine Jobs**: Machine_v="57,64,65,66,74" means job needs ALL 5 machines simultaneously
- **Processing Time**: Use capacity formula when CapMin_d=1: (JoQty_d/(CapQty_d*60)) + (SetupTime_d/60)
- **Working Hours**: Apply as filter during deployment, not in training
- **Constraints**: Hard (sequence, machines, overlap) vs Soft (deadlines, importance, efficiency)

---
*Note: Critical understanding - jobs can require multiple machines working together. This fundamentally changes our action space and state representation.*