"""
Curriculum Learning Environment for Phase 3
Fixed version that addresses all identified issues:
- Proper multi-sequence job handling
- Correct machine ID mapping (0-based to 1-based)
- Better reward structure for exploration
- Proper info dict keys
"""

import gymnasium as gym
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.environment.rules_engine import RulesEngine

logger = logging.getLogger(__name__)


class CurriculumSchedulingEnv(gym.Env):
    """Curriculum learning environment with proper multi-sequence handling."""
    
    def __init__(
        self,
        stage_config: Dict[str, Any],
        data_source: str = "synthetic",
        snapshot_path: Optional[str] = None,
        reward_profile: str = "balanced",
        seed: Optional[int] = None
    ):
        """
        Initialize curriculum environment.
        
        Args:
            stage_config: Configuration for the current stage
            data_source: "synthetic" or "snapshot"
            snapshot_path: Path to data snapshot if using real data
            reward_profile: Reward weighting profile
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
        
        # Stage configuration
        self.stage_config = stage_config
        self.stage_name = stage_config.get('name', 'unknown')
        self.n_jobs_target = stage_config.get('jobs', 10)
        self.n_machines_target = stage_config.get('machines', 5)
        self.data_source = data_source
        self.reward_profile = reward_profile
        
        # Load data
        if data_source == "synthetic":
            self.data = self._generate_synthetic_data()
        else:
            self.data = self._load_snapshot_data(snapshot_path)
        
        # Initialize environment state
        self.reset()
        
        # Define action and observation spaces
        # Action: [job_index, machine_index]
        self.action_space = gym.spaces.MultiDiscrete([self.n_jobs, self.n_machines])
        
        # Observation space: flatten all features
        # Per job: [status, current_seq, total_seq, lcd_remaining, is_important, processing_time]
        # Per machine: [availability_time, utilization]
        # Global: [current_time, completion_rate]
        obs_dim = (
            self.n_jobs * 6 +  # Job features
            self.n_machines * 2 +  # Machine features
            2  # Global features
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Rules engine for constraint checking
        rules_config = {
            'enforce_sequence': True,
            'enforce_compatibility': True,
            'enforce_no_overlap': True,
            'enforce_working_hours': False  # No working hours in training
        }
        self.rules_engine = RulesEngine(rules_config)
        
        # Reward configuration
        self.reward_weights = self._get_reward_weights(reward_profile)
        
        # Metrics tracking
        self.episode_metrics = defaultdict(list)
    
    def _generate_synthetic_data(self) -> Dict[str, Any]:
        """Generate synthetic data for toy stages."""
        # Use the synthetic data we created
        snapshot_name = f"snapshot_{self.stage_name.split()[0].lower()}.json"
        snapshot_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            snapshot_name
        )
        
        if os.path.exists(snapshot_path):
            with open(snapshot_path, 'r') as f:
                return json.load(f)
        else:
            # Fallback to generating on the fly
            logger.warning(f"Snapshot {snapshot_path} not found, generating synthetic data")
            return self._generate_synthetic_data_inline()
    
    def _generate_synthetic_data_inline(self) -> Dict[str, Any]:
        """Generate synthetic data inline if snapshot not found."""
        machines = []
        for i in range(self.n_machines_target):
            machines.append({
                'machine_id': i + 1,
                'machine_name': f'M{i+1:02d}',
                'machine_type_id': (i % 3) + 1
            })
        
        families = {}
        for i in range(self.n_jobs_target):
            family_id = f'JOB_{i:04d}'
            
            # Random properties
            is_important = np.random.random() < 0.3
            lcd_days = np.random.randint(1, 14) if is_important else np.random.randint(3, 21)
            n_sequences = np.random.randint(1, 4)
            
            # Multi-machine ratio from stage config
            multi_ratio = self.stage_config.get('multi_machine_ratio', 0.1)
            
            tasks = []
            for seq in range(1, n_sequences + 1):
                processing_time = np.random.uniform(0.5, 8.0)
                
                # Capable machines
                if np.random.random() < multi_ratio and self.n_machines_target > 3:
                    n_capable = np.random.randint(2, min(5, self.n_machines_target))
                    capable_machines = np.random.choice(
                        range(1, self.n_machines_target + 1),
                        size=n_capable,
                        replace=False
                    ).tolist()
                else:
                    capable_machines = [np.random.randint(1, self.n_machines_target + 1)]
                
                tasks.append({
                    'sequence': seq,
                    'process_name': f'PROCESS_{seq}',
                    'processing_time': round(processing_time, 2),
                    'capable_machines': capable_machines,
                    'status': 'pending'
                })
            
            families[family_id] = {
                'job_reference': family_id,
                'product': f'PRODUCT_{i % 10}',
                'is_important': is_important,
                'lcd_days_remaining': lcd_days,
                'total_sequences': n_sequences,
                'tasks': tasks
            }
        
        return {
            'families': families,
            'machines': machines,
            'metadata': {
                'snapshot_type': 'synthetic_inline',
                'total_families': len(families),
                'total_machines': len(machines)
            }
        }
    
    def _load_snapshot_data(self, snapshot_path: str) -> Dict[str, Any]:
        """Load data from snapshot file."""
        with open(snapshot_path, 'r') as f:
            data = json.load(f)
        
        # Filter to match target job/machine counts if needed
        if len(data['families']) > self.n_jobs_target:
            # Select subset of jobs
            family_ids = list(data['families'].keys())[:self.n_jobs_target]
            data['families'] = {fid: data['families'][fid] for fid in family_ids}
        
        if len(data['machines']) > self.n_machines_target:
            # Select subset of machines
            data['machines'] = data['machines'][:self.n_machines_target]
        
        return data
    
    def _get_reward_weights(self, profile: str) -> Dict[str, float]:
        """Get reward weights for given profile."""
        profiles = {
            'balanced': {
                'completion': 0.3,
                'deadline': 0.3,
                'efficiency': 0.2,
                'importance': 0.2
            },
            'deadline_focused': {
                'completion': 0.2,
                'deadline': 0.5,
                'efficiency': 0.2,
                'importance': 0.1
            },
            'learning': {
                'completion': 0.7,  # Focus on just completing jobs
                'deadline': 0.1,
                'efficiency': 0.1,
                'importance': 0.1
            },
            'rush_order': {
                'completion': 0.5,
                'deadline': 0.2,
                'efficiency': 0.1,
                'importance': 0.2
            },
            'efficiency': {
                'completion': 0.2,
                'deadline': 0.2,
                'efficiency': 0.5,
                'importance': 0.1
            }
        }
        return profiles.get(profile, profiles['balanced'])
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Initialize time
        self.current_time = 0.0
        self.max_time = 168.0  # One week in hours
        
        # Extract job and machine info
        self.families = self.data['families']
        self.machines = self.data['machines']
        self.n_jobs = len(self.families)
        self.n_machines = len(self.machines)
        
        # Create mappings
        self.job_ids = list(self.families.keys())
        self.job_id_to_idx = {jid: idx for idx, jid in enumerate(self.job_ids)}
        self.machine_ids = [m['machine_id'] for m in self.machines]
        self.machine_id_to_idx = {mid: idx for idx, mid in enumerate(self.machine_ids)}
        
        # Initialize job states
        self.job_states = {}
        for job_id, family in self.families.items():
            self.job_states[job_id] = {
                'status': 'pending',  # pending, in_progress, completed
                'current_sequence': 1,
                'total_sequences': family['total_sequences'],
                'is_important': family.get('is_important', False),
                'lcd_hours': family['lcd_days_remaining'] * 24,
                'tasks': family['tasks'].copy(),
                'start_time': None,
                'end_time': None,
                'completion_times': []  # Track when each sequence completes
            }
        
        # Initialize machine states
        self.machine_states = {}
        for machine in self.machines:
            self.machine_states[machine['machine_id']] = {
                'available_at': 0.0,
                'current_job': None,
                'utilization_time': 0.0,
                'jobs_processed': 0
            }
        
        # Episode metrics
        self.episode_metrics = {
            'jobs_completed': 0,
            'sequences_completed': 0,
            'total_sequences': sum(f['total_sequences'] for f in self.families.values()),
            'jobs_late': 0,
            'total_tardiness': 0.0,
            'makespan': 0.0,
            'machine_utilization': 0.0,
            'important_jobs_completed': 0,
            'important_jobs_late': 0
        }
        
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action in environment."""
        job_idx, machine_idx = action
        
        # Map indices to IDs
        if job_idx >= len(self.job_ids):
            # Invalid job index
            reward = -0.1
            info = {'action_valid': False, 'reason': f'Invalid job index {job_idx}'}
            obs = self._get_observation()
            return obs, reward, False, False, info
        
        job_id = self.job_ids[job_idx]
        
        # CRITICAL FIX: Map machine index to machine ID (0-based to 1-based)
        machine_id = self.machine_ids[machine_idx] if machine_idx < len(self.machine_ids) else None
        
        if machine_id is None:
            reward = -0.1
            info = {'action_valid': False, 'reason': f'Invalid machine index {machine_idx}'}
            obs = self._get_observation()
            return obs, reward, False, False, info
        
        # Check if action is valid
        is_valid, reason = self._is_valid_action(job_id, machine_id)
        
        if is_valid:
            # Execute the action
            reward = self._execute_action(job_id, machine_id)
            info = {
                'action_valid': True,
                'job_scheduled': job_id,
                'on_machine': machine_id,
                'sequence': self.job_states[job_id]['current_sequence'] - 1
            }
        else:
            # Invalid action
            reward = -0.1  # Small penalty
            info = {'action_valid': False, 'reason': reason}
            # Advance time slightly to prevent getting stuck
            self.current_time += 0.1
        
        # Check termination
        terminated = self._check_termination()
        truncated = self.current_time >= self.max_time
        
        # Calculate final metrics if episode ends
        if terminated or truncated:
            self._calculate_final_metrics()
            info['final_metrics'] = self.episode_metrics.copy()
        
        obs = self._get_observation()
        return obs, reward, terminated, truncated, info
    
    def _is_valid_action(self, job_id: str, machine_id: int) -> Tuple[bool, str]:
        """Check if scheduling job on machine is valid."""
        job_state = self.job_states.get(job_id)
        if not job_state:
            return False, f"Job {job_id} not found"
        
        if job_state['status'] == 'completed':
            return False, f"Job {job_id} already completed"
        
        if job_state['status'] == 'in_progress':
            return False, f"Job {job_id} already in progress"
        
        # Get current task
        current_seq = job_state['current_sequence']
        if current_seq > job_state['total_sequences']:
            return False, f"Job {job_id} has no more sequences"
        
        current_task = job_state['tasks'][current_seq - 1]
        
        # Check machine capability
        if machine_id not in current_task['capable_machines']:
            return False, f"Machine {machine_id} not capable for {job_id} seq {current_seq}. Needs: {current_task['capable_machines']}"
        
        # Check machine availability
        machine_state = self.machine_states.get(machine_id)
        if not machine_state:
            return False, f"Machine {machine_id} not found"
        
        if machine_state['available_at'] > self.current_time:
            return False, f"Machine {machine_id} busy until {machine_state['available_at']:.1f}"
        
        # For multi-machine jobs, check all required machines
        if len(current_task['capable_machines']) > 1:
            for req_machine_id in current_task['capable_machines']:
                req_machine_state = self.machine_states.get(req_machine_id)
                if req_machine_state and req_machine_state['available_at'] > self.current_time:
                    return False, f"Required machine {req_machine_id} busy until {req_machine_state['available_at']:.1f}"
        
        return True, "Valid"
    
    def _execute_action(self, job_id: str, machine_id: int) -> float:
        """Execute valid action and calculate reward."""
        job_state = self.job_states[job_id]
        current_seq = job_state['current_sequence']
        current_task = job_state['tasks'][current_seq - 1]
        
        # Calculate start and end times
        start_time = self.current_time
        
        # For multi-machine jobs, find latest available time
        required_machines = current_task['capable_machines']
        if len(required_machines) > 1:
            # Multi-machine job - find when ALL machines are available
            max_available = max(
                self.machine_states[mid]['available_at']
                for mid in required_machines
            )
            start_time = max(self.current_time, max_available)
        else:
            # Single machine job
            machine_state = self.machine_states[machine_id]
            start_time = max(self.current_time, machine_state['available_at'])
        
        end_time = start_time + current_task['processing_time']
        
        # Update job state
        job_state['status'] = 'in_progress'
        if job_state['start_time'] is None:
            job_state['start_time'] = start_time
        
        # Update machine states
        for req_machine_id in required_machines:
            machine_state = self.machine_states[req_machine_id]
            machine_state['available_at'] = end_time
            machine_state['current_job'] = job_id
            machine_state['utilization_time'] += current_task['processing_time']
            machine_state['jobs_processed'] += 1
        
        # Mark task as completed
        current_task['status'] = 'completed'
        job_state['completion_times'].append(end_time)
        
        # Check if this completes the job
        if current_seq >= job_state['total_sequences']:
            # Job fully completed
            job_state['status'] = 'completed'
            job_state['end_time'] = end_time
            self.episode_metrics['jobs_completed'] += 1
            
            # Check if late
            if end_time > job_state['lcd_hours']:
                self.episode_metrics['jobs_late'] += 1
                self.episode_metrics['total_tardiness'] += end_time - job_state['lcd_hours']
                if job_state['is_important']:
                    self.episode_metrics['important_jobs_late'] += 1
            
            if job_state['is_important']:
                self.episode_metrics['important_jobs_completed'] += 1
        else:
            # Move to next sequence
            job_state['current_sequence'] += 1
            job_state['status'] = 'pending'  # Ready for next sequence
        
        self.episode_metrics['sequences_completed'] += 1
        
        # Update current time
        self.current_time = start_time
        
        # Calculate reward
        reward = self._calculate_reward(job_id, start_time, end_time, current_seq)
        
        return reward
    
    def _calculate_reward(self, job_id: str, start_time: float, end_time: float, sequence: int) -> float:
        """Calculate reward for scheduling decision."""
        job_state = self.job_states[job_id]
        weights = self.reward_weights
        
        reward = 0.0
        
        # 1. Completion reward (most important for learning)
        completion_bonus = 10.0  # Base reward for any sequence
        if job_state['status'] == 'completed':
            # Extra bonus for completing entire job
            completion_bonus += 20.0
            # Progressive bonus based on how many jobs completed so far
            completion_bonus += self.episode_metrics['jobs_completed'] * 5.0
        
        reward += weights['completion'] * completion_bonus
        
        # 2. Deadline component
        slack = job_state['lcd_hours'] - end_time
        if slack >= 0:
            # On time - reward proportional to safety margin
            deadline_reward = min(10.0, slack / 24.0)
        else:
            # Late - mild penalty (don't discourage too much)
            deadline_reward = max(-5.0, slack / 48.0)
        
        reward += weights['deadline'] * deadline_reward
        
        # 3. Efficiency component
        # Reward for keeping machines busy
        idle_time = start_time - self.current_time
        efficiency_reward = -min(2.0, idle_time / 4.0)  # Small penalty for idle time
        
        reward += weights['efficiency'] * efficiency_reward
        
        # 4. Importance component
        if job_state['is_important']:
            importance_bonus = 5.0 if slack >= 0 else -2.0
            reward += weights['importance'] * importance_bonus
        
        # 5. Multi-machine bonus
        current_task = job_state['tasks'][sequence - 1]
        if len(current_task['capable_machines']) > 1:
            reward += 5.0  # Bonus for successfully scheduling multi-machine job
        
        # 6. Exploration bonus - reward for trying different jobs
        unique_jobs_scheduled = len(set(
            job_id 
            for job_id, state in self.job_states.items() 
            if state['current_sequence'] > 1 or state['status'] == 'completed'
        ))
        if unique_jobs_scheduled > 1:
            reward += unique_jobs_scheduled * 2.0
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector."""
        obs = []
        
        # Job features (6 per job)
        for job_id in self.job_ids:
            job_state = self.job_states[job_id]
            
            # 1. Status (0=pending, 0.5=in_progress, 1=completed)
            if job_state['status'] == 'pending':
                status = 0.0
            elif job_state['status'] == 'in_progress':
                status = 0.5
            else:
                status = 1.0
            
            # 2. Sequence progress
            seq_progress = (job_state['current_sequence'] - 1) / job_state['total_sequences']
            
            # 3. Total sequences (normalized)
            total_seq_norm = job_state['total_sequences'] / 5.0
            
            # 4. Time to deadline (normalized)
            time_to_deadline = (job_state['lcd_hours'] - self.current_time) / 168.0
            
            # 5. Is important
            is_important = float(job_state['is_important'])
            
            # 6. Current task processing time (normalized)
            if job_state['current_sequence'] <= job_state['total_sequences']:
                current_task = job_state['tasks'][job_state['current_sequence'] - 1]
                proc_time = current_task['processing_time'] / 10.0
            else:
                proc_time = 0.0
            
            obs.extend([status, seq_progress, total_seq_norm, time_to_deadline, is_important, proc_time])
        
        # Machine features (2 per machine)
        for machine_id in self.machine_ids:
            machine_state = self.machine_states[machine_id]
            
            # 1. Availability (normalized)
            time_until_available = max(0, machine_state['available_at'] - self.current_time) / 10.0
            
            # 2. Utilization rate
            if self.current_time > 0:
                utilization = machine_state['utilization_time'] / self.current_time
            else:
                utilization = 0.0
            
            obs.extend([time_until_available, utilization])
        
        # Global features (2)
        # 1. Current time (normalized)
        time_progress = self.current_time / self.max_time
        
        # 2. Completion rate
        completion_rate = self.episode_metrics['jobs_completed'] / max(1, self.n_jobs)
        
        obs.extend([time_progress, completion_rate])
        
        return np.array(obs, dtype=np.float32)
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        # Terminate when all jobs are completed
        all_completed = all(
            state['status'] == 'completed'
            for state in self.job_states.values()
        )
        return all_completed
    
    def _calculate_final_metrics(self):
        """Calculate final episode metrics."""
        # Machine utilization
        total_utilization = sum(
            state['utilization_time']
            for state in self.machine_states.values()
        )
        total_possible = self.current_time * self.n_machines
        self.episode_metrics['machine_utilization'] = (
            total_utilization / total_possible if total_possible > 0 else 0
        )
        
        # Makespan
        completed_times = [
            state['end_time']
            for state in self.job_states.values()
            if state['end_time'] is not None
        ]
        if completed_times:
            self.episode_metrics['makespan'] = max(completed_times)
    
    def render(self):
        """Render current environment state."""
        print(f"\n{'='*60}")
        print(f"Stage: {self.stage_name}")
        print(f"Time: {self.current_time:.1f} / {self.max_time} hours")
        print(f"Jobs: {self.episode_metrics['jobs_completed']} / {self.n_jobs} completed")
        print(f"Sequences: {self.episode_metrics['sequences_completed']} / {self.episode_metrics['total_sequences']}")
        print(f"Late jobs: {self.episode_metrics['jobs_late']}")
        print(f"Machine utilization: {self.episode_metrics['machine_utilization']:.1%}")
        print(f"{'='*60}\n")