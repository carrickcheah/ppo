"""
Curriculum Learning Environment for Phase 3

Supports progressive training from toy to production scale.
"""

import gymnasium as gym
import numpy as np
import random
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class CurriculumSchedulingEnv(gym.Env):
    """Scheduling environment with curriculum learning support."""
    
    def __init__(
        self,
        stage_config: Dict[str, Any],
        snapshot_path: Optional[str] = None,
        reward_profile: str = "balanced",
        seed: Optional[int] = None
    ):
        """
        Initialize curriculum environment.
        
        Args:
            stage_config: Configuration for current curriculum stage
            snapshot_path: Path to data snapshot (for non-synthetic stages)
            reward_profile: Reward weighting profile
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        # Set random seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Stage configuration
        self.stage_config = stage_config
        self.n_jobs = stage_config['jobs']
        self.n_machines = stage_config['machines']
        self.stage_name = stage_config['name']
        self.data_source = stage_config['data_source']
        
        # Load data based on source
        if self.data_source == "synthetic":
            self.data = self._generate_synthetic_data()
        else:
            self.data = self._load_snapshot_data(snapshot_path)
            
        # Initialize empty machine_ids first
        self.machine_ids = []
        
        # Get machine count from data before reset
        if self.data_source == "synthetic":
            self.n_machines_actual = self.n_machines
        else:
            self.n_machines_actual = len(self.data.get('machines', []))
        
        # Initialize environment state
        self.reset()
        
        # Define action and observation spaces
        # Action: MultiDiscrete [job_index, machine_index]
        self.action_space = gym.spaces.MultiDiscrete([self.n_jobs, self.n_machines_actual])
        
        # Observation space includes:
        # - Job features (per job): status, sequences, deadlines, importance
        # - Machine features (per machine): current load, availability
        # - Global features: time, completion rate
        obs_dim = (
            self.n_jobs * 6 +  # Job features
            self.n_machines_actual * 2 +  # Machine features  
            3  # Global features
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Reward configuration
        self.reward_profile = reward_profile
        self.reward_weights = self._get_reward_weights(reward_profile)
        
        # Metrics tracking
        self.metrics = defaultdict(list)
        
    def _generate_synthetic_data(self) -> Dict[str, Any]:
        """Generate synthetic data for toy stages."""
        logger.info(f"Generating synthetic data for {self.n_jobs} jobs, {self.n_machines} machines")
        
        # Multi-machine ratio for certain stages
        multi_ratio = self.stage_config.get('multi_machine_ratio', 0.1)
        
        families = {}
        machines = []
        
        # Generate machines
        for i in range(self.n_machines):
            machines.append({
                'machine_id': i + 1,
                'machine_name': f'MACHINE_{i+1:02d}',
                'machine_type_id': (i % 3) + 1  # 3 machine types
            })
            
        # Generate job families
        for i in range(self.n_jobs):
            family_id = f"JOB_{i:03d}"
            
            # Random properties
            is_important = random.random() < 0.3
            lcd_days = random.randint(3, 10) if not is_important else random.randint(1, 5)
            n_sequences = random.randint(1, 3)
            
            # Generate tasks
            tasks = []
            for seq in range(1, n_sequences + 1):
                # Processing time (hours)
                processing_time = random.uniform(0.5, 5.0)
                
                # Capable machines
                if random.random() < multi_ratio and self.n_machines > 3:
                    # Multi-machine task
                    n_capable = random.randint(2, min(4, self.n_machines))
                    capable_machines = random.sample(range(1, self.n_machines + 1), n_capable)
                else:
                    # Single machine task
                    capable_machines = [random.randint(1, self.n_machines)]
                    
                tasks.append({
                    'sequence': seq,
                    'process_name': f'PROCESS_{seq}',
                    'processing_time': processing_time,
                    'capable_machines': capable_machines,
                    'status': 'pending'
                })
                
            families[family_id] = {
                'job_reference': family_id,
                'product': f'PRODUCT_{i % 5}',
                'is_important': is_important,
                'lcd_days_remaining': lcd_days,
                'total_sequences': n_sequences,
                'tasks': tasks
            }
            
        return {
            'families': families,
            'machines': machines,
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'snapshot_type': 'synthetic',
                'total_families': len(families),
                'total_tasks': sum(len(f['tasks']) for f in families.values())
            }
        }
        
    def _load_snapshot_data(self, snapshot_path: str) -> Dict[str, Any]:
        """Load data from snapshot file."""
        with open(snapshot_path, 'r') as f:
            data = json.load(f)
            
        # Handle subset selection for larger snapshots
        if 'subset' in self.data_source:
            data = self._create_subset(data)
            
        return data
        
    def _create_subset(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create subset of data for progressive training."""
        families = data['families']
        
        # Select subset of families
        family_ids = list(families.keys())
        
        # Prioritize based on data source type
        if 'rush' in self.data_source:
            # Prioritize urgent jobs
            family_ids.sort(key=lambda x: families[x]['lcd_days_remaining'])
        elif 'important' in self.data_source:
            # Prioritize important jobs
            family_ids.sort(key=lambda x: families[x]['is_important'], reverse=True)
        else:
            # Random selection
            random.shuffle(family_ids)
            
        # Select required number of jobs
        selected_ids = family_ids[:self.n_jobs]
        
        # Create subset
        subset_families = {fid: families[fid] for fid in selected_ids}
        
        # Update metadata
        data['families'] = subset_families
        data['metadata']['total_families'] = len(subset_families)
        data['metadata']['total_tasks'] = sum(
            len(f['tasks']) for f in subset_families.values()
        )
        
        return data
        
    def _get_reward_weights(self, profile: str) -> Dict[str, float]:
        """Get reward weights for given profile."""
        profiles = {
            'balanced': {
                'deadline': 0.4,
                'efficiency': 0.3,
                'importance': 0.3
            },
            'deadline_focused': {
                'deadline': 0.7,
                'efficiency': 0.2,
                'importance': 0.1
            },
            'efficiency_focused': {
                'deadline': 0.2,
                'efficiency': 0.6,
                'importance': 0.2
            },
            'importance_aware': {
                'deadline': 0.3,
                'efficiency': 0.2,
                'importance': 0.5
            }
        }
        return profiles.get(profile, profiles['balanced'])
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Initialize time
        self.current_time = 0.0
        self.max_time = 168.0  # One week in hours
        
        # Initialize job states
        self.job_states = {}
        for family_id, family_data in self.data['families'].items():
            self.job_states[family_id] = {
                'status': 'pending',
                'current_sequence': 1,
                'total_sequences': family_data['total_sequences'],
                'is_important': family_data['is_important'],
                'lcd_hours': family_data['lcd_days_remaining'] * 24,
                'tasks': family_data['tasks'].copy(),
                'start_time': None,
                'end_time': None,
                'assigned_machine': None
            }
            
        # Initialize machine states
        self.machine_states = {}
        for machine in self.data['machines']:
            self.machine_states[machine['machine_id']] = {
                'status': 'idle',
                'current_job': None,
                'available_at': 0.0,
                'total_busy_time': 0.0,
                'jobs_processed': 0
            }
            
        # Create machine ID mappings
        self.machine_ids = sorted([m['machine_id'] for m in self.data['machines']])
        self.machine_id_to_index = {mid: idx for idx, mid in enumerate(self.machine_ids)}
        self.index_to_machine_id = {idx: mid for mid, idx in self.machine_id_to_index.items()}
        
        # Create job index mapping
        self.job_id_to_index = {
            jid: idx for idx, jid in enumerate(self.data['families'].keys())
        }
        self.index_to_job_id = {
            idx: jid for jid, idx in self.job_id_to_index.items()
        }
        
        # Metrics
        self.episode_metrics = {
            'jobs_completed': 0,
            'jobs_late': 0,
            'important_jobs_late': 0,
            'total_tardiness': 0.0,
            'machine_utilization': 0.0,
            'makespan': 0.0
        }
        
        return self._get_observation(), {}
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action in environment."""
        job_idx, machine_idx = action
        
        # Convert indices to IDs
        job_id = self.index_to_job_id.get(job_idx)
        
        # Convert machine index to actual ID
        machine_id = self.index_to_machine_id.get(machine_idx, None)
        if machine_id is None:
            machine_id = machine_idx + 1  # Fallback for synthetic data
        
        reward = 0.0
        info = {'action_valid': False}
        
        # Validate action
        if job_id and self._is_valid_action(job_id, machine_id):
            # Execute action
            reward = self._execute_action(job_id, machine_id)
            info['action_valid'] = True
            
        # Update time (small penalty for invalid actions)
        if not info['action_valid']:
            self.current_time += 0.1
            reward = -0.1
            
        # Check termination
        terminated = self._is_terminated()
        truncated = self.current_time >= self.max_time
        
        # Calculate final metrics if episode ended
        if terminated or truncated:
            self._calculate_final_metrics()
            info['episode_metrics'] = self.episode_metrics.copy()
            
        obs = self._get_observation()
        
        return obs, reward, terminated, truncated, info
        
    def _is_valid_action(self, job_id: str, machine_id: int) -> bool:
        """Check if action is valid."""
        job_state = self.job_states.get(job_id)
        if not job_state or job_state['status'] != 'pending':
            return False
            
        # Check if machine is available
        machine_state = self.machine_states.get(machine_id)
        if not machine_state or machine_state['available_at'] > self.current_time:
            return False
            
        # Check if machine is capable
        current_seq = job_state['current_sequence']
        current_task = job_state['tasks'][current_seq - 1]
        
        if machine_id not in current_task['capable_machines']:
            return False
            
        return True
        
    def _execute_action(self, job_id: str, machine_id: int) -> float:
        """Execute valid action and return reward."""
        job_state = self.job_states[job_id]
        machine_state = self.machine_states[machine_id]
        
        # Get current task
        current_seq = job_state['current_sequence']
        current_task = job_state['tasks'][current_seq - 1]
        processing_time = current_task['processing_time']
        
        # Schedule job on machine
        start_time = max(self.current_time, machine_state['available_at'])
        end_time = start_time + processing_time
        
        # Update states
        job_state['status'] = 'in_progress'
        job_state['start_time'] = start_time
        job_state['assigned_machine'] = machine_id
        
        machine_state['status'] = 'busy'
        machine_state['current_job'] = job_id
        machine_state['available_at'] = end_time
        machine_state['total_busy_time'] += processing_time
        
        # Update current time
        self.current_time = start_time
        
        # Calculate immediate reward
        reward = self._calculate_reward(job_id, start_time, end_time)
        
        # Complete task
        self._complete_task(job_id, machine_id, end_time)
        
        return reward
        
    def _complete_task(self, job_id: str, machine_id: int, end_time: float):
        """Complete current task and update states."""
        job_state = self.job_states[job_id]
        machine_state = self.machine_states[machine_id]
        
        # Mark task as completed
        current_seq = job_state['current_sequence']
        job_state['tasks'][current_seq - 1]['status'] = 'completed'
        
        # Check if job is fully completed
        if current_seq >= job_state['total_sequences']:
            # Job completed
            job_state['status'] = 'completed'
            job_state['end_time'] = end_time
            
            # Update metrics
            self.episode_metrics['jobs_completed'] += 1
            
            # Check if late
            if end_time > job_state['lcd_hours']:
                self.episode_metrics['jobs_late'] += 1
                tardiness = end_time - job_state['lcd_hours']
                self.episode_metrics['total_tardiness'] += tardiness
                
                if job_state['is_important']:
                    self.episode_metrics['important_jobs_late'] += 1
        else:
            # Move to next sequence
            job_state['current_sequence'] += 1
            job_state['status'] = 'pending'
            job_state['assigned_machine'] = None
            
        # Update machine state
        machine_state['status'] = 'idle'
        machine_state['current_job'] = None
        machine_state['jobs_processed'] += 1
        
    def _calculate_reward(self, job_id: str, start_time: float, end_time: float) -> float:
        """Calculate reward for scheduling decision."""
        job_state = self.job_states[job_id]
        
        # Components
        deadline_reward = 0.0
        efficiency_reward = 0.0
        importance_reward = 0.0
        
        # Deadline component
        slack = job_state['lcd_hours'] - end_time
        if slack >= 0:
            # On time - reward proportional to safety margin
            deadline_reward = min(1.0, slack / 24.0)
        else:
            # Late - penalty proportional to tardiness
            deadline_reward = max(-1.0, slack / 24.0)
            
        # Efficiency component (minimize idle time)
        idle_time = start_time - self.current_time
        efficiency_reward = -min(1.0, idle_time / 4.0)
        
        # Importance component
        if job_state['is_important']:
            importance_reward = 0.5 if slack >= 0 else -0.5
            
        # Weighted sum
        weights = self.reward_weights
        total_reward = (
            weights['deadline'] * deadline_reward +
            weights['efficiency'] * efficiency_reward +
            weights['importance'] * importance_reward
        )
        
        return total_reward
        
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        obs = []
        
        # Job features
        for job_id in sorted(self.job_id_to_index.keys()):
            job_state = self.job_states[job_id]
            
            # Status encoding (pending=0, in_progress=1, completed=2)
            status_map = {'pending': 0, 'in_progress': 1, 'completed': 2}
            status = status_map[job_state['status']]
            
            # Current sequence progress
            seq_progress = job_state['current_sequence'] / job_state['total_sequences']
            
            # Time to deadline
            time_to_deadline = (job_state['lcd_hours'] - self.current_time) / 168.0
            
            # Importance flag
            importance = float(job_state['is_important'])
            
            # Processing time of current task
            if job_state['status'] != 'completed':
                current_task = job_state['tasks'][job_state['current_sequence'] - 1]
                processing_time = current_task['processing_time'] / 10.0
                n_capable_machines = len(current_task['capable_machines']) / self.n_machines_actual
            else:
                processing_time = 0.0
                n_capable_machines = 0.0
                
            obs.extend([
                status, seq_progress, time_to_deadline,
                importance, processing_time, n_capable_machines
            ])
            
        # Machine features
        for machine_id in self.machine_ids:
            machine_state = self.machine_states[machine_id]
            
            # Availability
            time_until_available = max(0, machine_state['available_at'] - self.current_time) / 10.0
            
            # Utilization
            utilization = machine_state['total_busy_time'] / max(1.0, self.current_time)
            
            obs.extend([time_until_available, utilization])
            
        # Global features
        time_progress = self.current_time / self.max_time
        completion_rate = self.episode_metrics['jobs_completed'] / max(1, self.n_jobs)
        late_rate = self.episode_metrics['jobs_late'] / max(1, self.episode_metrics['jobs_completed'])
        
        obs.extend([time_progress, completion_rate, late_rate])
        
        return np.array(obs, dtype=np.float32)
        
    def _is_terminated(self) -> bool:
        """Check if episode is terminated."""
        # All jobs completed
        all_completed = all(
            state['status'] == 'completed' 
            for state in self.job_states.values()
        )
        return all_completed
        
    def _calculate_final_metrics(self):
        """Calculate final episode metrics."""
        # Machine utilization
        total_busy = sum(m['total_busy_time'] for m in self.machine_states.values())
        total_possible = self.current_time * self.n_machines
        self.episode_metrics['machine_utilization'] = total_busy / max(1.0, total_possible)
        
        # Makespan
        completed_times = [
            job['end_time'] for job in self.job_states.values()
            if job['status'] == 'completed' and job['end_time'] is not None
        ]
        if completed_times:
            self.episode_metrics['makespan'] = max(completed_times)
            
    def render(self):
        """Render environment state (text-based)."""
        print(f"\n{'='*60}")
        print(f"Stage: {self.stage_name}")
        print(f"Time: {self.current_time:.1f} / {self.max_time:.1f} hours")
        print(f"Jobs: {self.episode_metrics['jobs_completed']} / {self.n_jobs} completed")
        print(f"Late jobs: {self.episode_metrics['jobs_late']}")
        print(f"Machine utilization: {self.episode_metrics['machine_utilization']:.1%}")
        print(f"{'='*60}")