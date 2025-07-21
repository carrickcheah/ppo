# Hierarchical Action Space Design Document

## Overview

This document provides the detailed technical design for the hierarchical action space solution that will enable the PPO scheduler to handle all 411+ jobs in a single pass, eliminating the need for batch processing.

## Problem Analysis

### Current Flat Action Space
```python
# Current implementation
action = job_idx * n_machines + machine_idx
# Example: Job 200 on Machine 25 = 200 * 30 + 25 = 6025

# Problem
Total actions = n_jobs * n_machines = 411 * 30 = 12,330
Environment limit = 200-888 actions
Result: Can only see ~172 jobs per episode
```

### Why Current Approach Fails
1. **Combinatorial Explosion**: Every job-machine pair is a unique action
2. **Sparse Valid Actions**: Most combinations are invalid
3. **Fixed Action Space**: Must allocate space for all possibilities
4. **Learning Inefficiency**: Network must learn 12,000+ action values

## Hierarchical Solution Design

### Core Concept
```python
# Hierarchical approach
Step 1: action_job = select_job(state)       # 0-410 (411 options)
Step 2: action_machine = select_machine(state, job)  # 0-29 (30 options)
Total action space = 411 + 30 = 441 actions (96% reduction!)
```

### Architecture Diagram
```
                    State (60 features)
                           │
                    ┌──────┴──────┐
                    │   Shared    │
                    │   Features   │
                    │   (256 dim)  │
                    └──────┬──────┘
                           │
                ┌─────────┴─────────┐
                │                   │
          ┌─────┴─────┐      ┌─────┴─────┐
          │  Job Head  │      │  Job Embed │
          │ (411 dim)  │      │  (64 dim)  │
          └─────┬─────┘      └─────┬─────┘
                │                     │
          Select Job ID         Concatenate
                                     │
                            ┌───────┴───────┐
                            │ Machine Head  │
                            │  (30 dim)     │
                            └───────┬───────┘
                                     │
                              Select Machine
```

## Implementation Details

### 1. Environment Interface

```python
class HierarchicalProductionEnv(FullProductionEnv):
    def __init__(self, n_machines: int, n_jobs: int, **kwargs):
        super().__init__(n_machines, n_jobs, **kwargs)
        
        # Override action space
        self.action_space = spaces.Dict({
            'job': spaces.Discrete(n_jobs),
            'machine': spaces.Discrete(n_machines)
        })
        
        # Compatibility matrix for fast lookup
        self.compatibility_matrix = self._build_compatibility_matrix()
        
        # Action masks
        self.job_mask = np.ones(n_jobs, dtype=bool)
        self.machine_masks = {}
        
    def _build_compatibility_matrix(self) -> np.ndarray:
        """Build job-machine compatibility matrix"""
        matrix = np.zeros((self.n_jobs, self.n_machines), dtype=bool)
        for job_idx, job in enumerate(self.jobs):
            for machine_idx, machine in enumerate(self.machines):
                if machine.machine_type in job.allowed_machine_types:
                    matrix[job_idx, machine_idx] = True
        return matrix
```

### 2. Action Processing

```python
def step(self, action: Dict[str, int]) -> Tuple:
    job_idx = action['job']
    machine_idx = action['machine']
    
    # Validate job selection
    if not self.job_mask[job_idx]:
        return self._invalid_action_result("Job already scheduled")
    
    # Validate machine selection
    if not self.compatibility_matrix[job_idx, machine_idx]:
        return self._invalid_action_result("Incompatible machine")
    
    # Check machine availability
    if not self._is_machine_available(machine_idx):
        return self._invalid_action_result("Machine busy")
    
    # Execute assignment
    self._assign_job_to_machine(job_idx, machine_idx)
    
    # Update masks
    self.job_mask[job_idx] = False
    self._update_machine_availability()
    
    # Calculate reward
    reward = self._calculate_hierarchical_reward(job_idx, machine_idx)
    
    # Check termination
    done = np.sum(self.job_mask) == 0  # All jobs scheduled
    
    return self.state, reward, done, False, self._get_info()
```

### 3. State Representation

```python
def _get_hierarchical_state(self) -> np.ndarray:
    """
    Enhanced state for hierarchical decisions
    Total: 60 + 20 = 80 features
    """
    # Base state (60 features)
    base_state = self._get_base_state()
    
    # Hierarchical additions (20 features)
    hierarchical_features = [
        # Job selection helpers (10)
        self._get_job_urgency_distribution(),      # 5 features
        self._get_job_complexity_distribution(),    # 5 features
        
        # Machine selection helpers (10)
        self._get_machine_load_distribution(),      # 5 features
        self._get_machine_compatibility_stats(),    # 5 features
    ]
    
    return np.concatenate([base_state] + hierarchical_features)
```

### 4. Action Masking

```python
def get_action_masks(self) -> Dict[str, np.ndarray]:
    """
    Dynamic action masking for valid actions only
    """
    return {
        'job': self.job_mask.copy(),
        'machine': self._get_current_machine_mask()
    }
    
def _get_current_machine_mask(self) -> np.ndarray:
    """
    Machine mask depends on last selected job
    """
    if self.last_selected_job is None:
        # No job selected yet, all machines masked
        return np.zeros(self.n_machines, dtype=bool)
    
    job_idx = self.last_selected_job
    # Compatible AND available machines
    return self.compatibility_matrix[job_idx] & self.machine_availability
```

### 5. Reward Design

```python
def _calculate_hierarchical_reward(self, job_idx: int, machine_idx: int) -> float:
    """
    Hierarchical reward structure
    """
    reward = 0.0
    
    # Job selection reward
    job = self.jobs[job_idx]
    if job.is_important:
        reward += 5.0  # Prioritize important jobs
    
    urgency = self._calculate_urgency(job)
    reward += urgency * 2.0  # Urgency bonus
    
    # Machine selection reward
    machine = self.machines[machine_idx]
    utilization = self._get_machine_utilization(machine_idx)
    
    # Balance load across machines
    avg_utilization = np.mean(self.machine_utilizations)
    balance_reward = -abs(utilization - avg_utilization) * 3.0
    reward += balance_reward
    
    # Efficiency reward
    if self._is_preferred_machine(job_idx, machine_idx):
        reward += 3.0
    
    # Setup time penalty
    setup_time = self._calculate_setup_time(machine_idx, job.family_id)
    reward -= setup_time * 0.5
    
    # Completion bonus
    if self._all_jobs_scheduled():
        makespan = self._calculate_makespan()
        reward += 100.0 / makespan  # Inversely proportional to makespan
    
    return reward
```

## Policy Network Architecture

### 1. Network Structure

```python
class HierarchicalPolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_jobs: int, n_machines: int):
        super().__init__()
        
        # Shared feature extractor
        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )
        
        # Job selection head
        self.job_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_jobs)
        )
        
        # Job embedding for machine selection
        self.job_embedding = nn.Embedding(n_jobs, 64)
        
        # Machine selection head
        self.machine_net = nn.Sequential(
            nn.Linear(256 + 64, 128),  # Features + job embedding
            nn.ReLU(),
            nn.Linear(128, n_machines)
        )
        
        # Value heads
        self.value_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
```

### 2. Forward Pass

```python
def forward(self, obs: torch.Tensor, job_mask: torch.Tensor = None, 
            selected_job: torch.Tensor = None, machine_mask: torch.Tensor = None):
    """
    Hierarchical forward pass
    """
    # Extract shared features
    features = self.shared_net(obs)
    
    # Job selection
    job_logits = self.job_net(features)
    if job_mask is not None:
        job_logits = job_logits.masked_fill(~job_mask, float('-inf'))
    job_probs = F.softmax(job_logits, dim=-1)
    
    # Machine selection (if job provided)
    if selected_job is not None:
        job_embed = self.job_embedding(selected_job)
        machine_input = torch.cat([features, job_embed], dim=-1)
        machine_logits = self.machine_net(machine_input)
        
        if machine_mask is not None:
            machine_logits = machine_logits.masked_fill(~machine_mask, float('-inf'))
        machine_probs = F.softmax(machine_logits, dim=-1)
    else:
        machine_probs = None
    
    # Value estimation
    value = self.value_net(features)
    
    return job_probs, machine_probs, value
```

## Training Strategy

### 1. Curriculum Learning

```python
class CurriculumSchedule:
    def __init__(self, stages: List[int], stage_timesteps: int):
        self.stages = stages  # [100, 250, 500] jobs
        self.stage_timesteps = stage_timesteps
        self.current_stage = 0
        
    def get_n_jobs(self, timestep: int) -> int:
        stage = min(timestep // self.stage_timesteps, len(self.stages) - 1)
        return self.stages[stage]
```

### 2. Exploration Strategy

```python
class HierarchicalExploration:
    def __init__(self, job_epsilon: float = 0.2, machine_epsilon: float = 0.1):
        self.job_epsilon = job_epsilon
        self.machine_epsilon = machine_epsilon
        
    def explore_job(self, job_probs: np.ndarray, mask: np.ndarray) -> int:
        if np.random.random() < self.job_epsilon:
            # Random valid job
            valid_jobs = np.where(mask)[0]
            return np.random.choice(valid_jobs)
        else:
            # Follow policy
            return np.argmax(job_probs)
    
    def explore_machine(self, machine_probs: np.ndarray, mask: np.ndarray, 
                       job_features: np.ndarray) -> int:
        if np.random.random() < self.machine_epsilon:
            # Smart exploration: prefer less loaded machines
            valid_machines = np.where(mask)[0]
            if len(valid_machines) > 0:
                # Weight by inverse utilization
                weights = 1.0 / (self.machine_utilizations[valid_machines] + 0.1)
                weights = weights / np.sum(weights)
                return np.random.choice(valid_machines, p=weights)
        return np.argmax(machine_probs)
```

## Performance Optimizations

### 1. Batch Processing for Training

```python
def collect_hierarchical_rollout(env, model, n_steps: int):
    """
    Efficient rollout collection for hierarchical actions
    """
    rollout = {
        'obs': [],
        'job_actions': [],
        'machine_actions': [],
        'rewards': [],
        'dones': [],
        'job_masks': [],
        'machine_masks': []
    }
    
    obs = env.reset()
    for _ in range(n_steps):
        # Get masks
        masks = env.get_action_masks()
        
        # Predict job
        job_action, job_log_prob = model.predict_job(
            obs, mask=masks['job'], deterministic=False
        )
        
        # Predict machine
        machine_action, machine_log_prob = model.predict_machine(
            obs, job_action, mask=masks['machine'], deterministic=False
        )
        
        # Step environment
        action = {'job': job_action, 'machine': machine_action}
        next_obs, reward, done, info = env.step(action)
        
        # Store transition
        rollout['obs'].append(obs)
        rollout['job_actions'].append(job_action)
        rollout['machine_actions'].append(machine_action)
        rollout['rewards'].append(reward)
        rollout['dones'].append(done)
        rollout['job_masks'].append(masks['job'])
        rollout['machine_masks'].append(masks['machine'])
        
        obs = next_obs
        if done:
            obs = env.reset()
    
    return rollout
```

### 2. Caching and Memoization

```python
class CompatibilityCache:
    """
    Cache job-machine compatibility for fast lookup
    """
    def __init__(self, jobs, machines):
        self.cache = {}
        self._build_cache(jobs, machines)
        
    def _build_cache(self, jobs, machines):
        for job_idx, job in enumerate(jobs):
            compatible = []
            for machine_idx, machine in enumerate(machines):
                if machine.machine_type in job.allowed_machine_types:
                    compatible.append(machine_idx)
            self.cache[job_idx] = np.array(compatible)
    
    def get_compatible_machines(self, job_idx: int) -> np.ndarray:
        return self.cache[job_idx]
```

## Advantages Over Current Approach

### 1. Scalability
| Metric | Current (Flat) | Hierarchical | Improvement |
|--------|----------------|--------------|-------------|
| Action Space | O(n×m) | O(n+m) | 96% reduction |
| Memory Usage | O(n×m) | O(n+m) | Linear scaling |
| Learning Speed | Slow | Fast | 3-5x faster |
| Job Visibility | 42% | 100% | Complete view |

### 2. Interpretability
- **Clear Decisions**: "Select Job X, then Machine Y"
- **Explainable**: Can trace why specific assignments made
- **Debuggable**: Separate job and machine selection logic

### 3. Flexibility
- **Easy Extensions**: Add more decision levels if needed
- **Constraint Handling**: Natural place for business rules
- **Online Learning**: Can update job/machine preferences separately

## Migration Strategy

### 1. Backward Compatibility
```python
class UnifiedScheduler:
    def __init__(self, hierarchical_model, batch_model):
        self.hierarchical_model = hierarchical_model
        self.batch_model = batch_model
        
    def schedule(self, jobs, machines, use_hierarchical=True):
        if use_hierarchical and len(jobs) <= 500:
            return self.hierarchical_model.schedule(jobs, machines)
        else:
            # Fall back to batch for very large problems
            return self.batch_model.schedule(jobs, machines)
```

### 2. A/B Testing Framework
```python
def compare_schedulers(jobs, machines, n_trials=10):
    hierarchical_makespans = []
    batch_makespans = []
    
    for _ in range(n_trials):
        # Hierarchical
        h_schedule = hierarchical_scheduler.schedule(jobs, machines)
        hierarchical_makespans.append(h_schedule.makespan)
        
        # Batch
        b_schedule = batch_scheduler.schedule(jobs, machines)
        batch_makespans.append(b_schedule.makespan)
    
    improvement = (np.mean(batch_makespans) - np.mean(hierarchical_makespans)) / np.mean(batch_makespans)
    return improvement * 100  # Percentage improvement
```

## Conclusion

The hierarchical action space design solves the fundamental limitation of Phase 4 while maintaining all its strengths. By decomposing the job-machine assignment into two sequential decisions, we achieve:

1. **Full job visibility** in every scheduling pass
2. **96% reduction** in action space complexity
3. **Faster learning** through clearer action-reward relationships
4. **Better scalability** to 1000+ job problems
5. **Maintain 100% completion** guarantee

This design provides the foundation for Phase 5 implementation and future enhancements like online learning and multi-objective optimization.