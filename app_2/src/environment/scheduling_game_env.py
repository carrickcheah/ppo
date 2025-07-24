"""
Pure DRL Scheduling Game Environment

This environment represents scheduling as a game where:
- State: Current job/machine status and time
- Action: Which job to schedule on which machine
- Rules: Hard constraints (sequence, compatibility, no overlap, working hours)
- Reward: Learned preferences (urgency, importance, efficiency)

No hardcoded strategies - the AI learns everything through experience.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class SchedulingGameEnv(gym.Env):
    """
    Pure game environment for scheduling - no hardcoded strategies.
    
    The environment enforces physics (hard rules) but doesn't encode
    any scheduling strategies. All strategies emerge from learning.
    """
    
    def __init__(
        self,
        jobs: List[Dict[str, Any]],
        machines: List[Dict[str, Any]], 
        working_hours: Dict[str, Any],
        config: Dict[str, Any]
    ):
        """
        Initialize the scheduling game.
        
        Args:
            jobs: List of job dictionaries from database
            machines: List of machine dictionaries from database
            working_hours: Working hours configuration
            config: Environment configuration from YAML
        """
        super().__init__()
        
        self.jobs = jobs
        self.machines = machines
        self.working_hours = working_hours
        self.config = config
        
        # Parse job families and sequences
        self._parse_job_structure()
        
        # Action space: MultiDiscrete([n_jobs, n_machines])
        self.n_jobs = len(self.jobs)
        self.n_machines = len(self.machines)
        self.action_space = spaces.MultiDiscrete([self.n_jobs, self.n_machines])
        
        # Observation space: flexible size for variable jobs
        # We'll use padding/masking in the model to handle variable sizes
        max_jobs = config.get('max_jobs', 2000)
        max_machines = config.get('max_machines', 200)
        
        # State representation per job:
        # [is_available, sequence_progress, urgency_score, processing_time, 
        #  machine_compatibility (one-hot), is_important]
        job_state_dim = 5 + max_machines  
        
        # State representation per machine:
        # [current_load, time_until_free, machine_type, utilization_rate]
        machine_state_dim = 4
        
        # Global state: [current_time, time_progress]
        global_state_dim = 2
        
        obs_dim = (max_jobs * job_state_dim + 
                   max_machines * machine_state_dim + 
                   global_state_dim)
        
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Environment state
        self.current_time = None
        self.machine_schedules = None
        self.completed_jobs = None
        self.job_assignments = None
        
    def _parse_job_structure(self):
        """Parse jobs into families and identify sequences."""
        self.families = {}
        self.job_to_family = {}
        self.job_sequences = {}
        
        for job in self.jobs:
            job_id = job['job_id']
            
            # Extract family ID and sequence from job_id
            # Format: FAMILYID_MACHINE-SEQ/TOTAL
            parts = job_id.split('_')
            if len(parts) >= 2:
                family_id = parts[0]
                
                # Extract sequence info
                seq_info = parts[-1]  # e.g., "1/3"
                if '/' in seq_info:
                    seq_num, total_seq = seq_info.split('/')
                    try:
                        sequence = int(seq_num)
                        total = int(total_seq)
                    except:
                        sequence = 1
                        total = 1
                else:
                    sequence = 1
                    total = 1
            else:
                family_id = job_id
                sequence = 1
                total = 1
                
            # Store family info
            if family_id not in self.families:
                self.families[family_id] = {
                    'jobs': [],
                    'total_sequences': total
                }
            
            self.families[family_id]['jobs'].append(job)
            self.job_to_family[job_id] = family_id
            self.job_sequences[job_id] = sequence
            
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Initialize time
        self.current_time = 0.0
        self.total_time_horizon = self.config.get('time_horizon', 168.0)  # 1 week default
        
        # Initialize machine schedules (list of scheduled jobs per machine)
        self.machine_schedules = [[] for _ in range(self.n_machines)]
        
        # Track completed jobs
        self.completed_jobs = set()
        
        # Track job assignments
        self.job_assignments = {}
        
        # Calculate initial state
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray):
        """
        Execute action in environment.
        
        Args:
            action: [job_index, machine_index]
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        job_idx = action[0]
        machine_idx = action[1]
        
        # Validate action
        is_valid, reason = self._is_action_valid(job_idx, machine_idx)
        
        if not is_valid:
            # Invalid action penalty
            reward = self.config.get('invalid_action_penalty', -20.0)
            info = {
                'invalid_action': True,
                'reason': reason
            }
            observation = self._get_observation()
            terminated = self._is_done()
            truncated = False
            return observation, reward, terminated, truncated, info
        
        # Execute valid action
        job = self.jobs[job_idx]
        machine = self.machines[machine_idx]
        
        # For multi-machine jobs, we need ALL required machines
        required_machines = job.get('required_machines', [])
        if len(required_machines) > 1:
            # This is a multi-machine job - validate all machines are available
            return self._schedule_multi_machine_job(job_idx, machine_idx)
        
        # Single machine job - proceed as normal
        # Calculate start time (after previous jobs on this machine)
        if self.machine_schedules[machine_idx]:
            last_job = self.machine_schedules[machine_idx][-1]
            start_time = last_job['end_time']
        else:
            start_time = self.current_time
            
        # Check family dependencies
        family_id = self.job_to_family[job['job_id']]
        job_sequence = self.job_sequences[job['job_id']]
        
        # Must wait for previous sequence in family
        family_ready_time = self._get_family_ready_time(family_id, job_sequence)
        start_time = max(start_time, family_ready_time)
        
        # Apply working hours constraints
        start_time = self._adjust_for_working_hours(start_time)
        
        # Calculate end time
        processing_time = job['processing_time']
        end_time = start_time + processing_time
        
        # Schedule the job
        scheduled_job = {
            'job': job,
            'job_idx': job_idx,
            'machine_idx': machine_idx,
            'start_time': start_time,
            'end_time': end_time
        }
        
        self.machine_schedules[machine_idx].append(scheduled_job)
        self.completed_jobs.add(job_idx)
        self.job_assignments[job_idx] = scheduled_job
        
        # Update current time
        self.current_time = max(self.current_time, start_time)
        
        # Calculate reward (let the AI learn what's good)
        reward = self._calculate_reward(job, machine, start_time, end_time)
        
        # Get new state
        observation = self._get_observation()
        
        # Check if done
        terminated = self._is_done()
        truncated = self.current_time >= self.total_time_horizon
        
        # Compile info
        info = {
            'scheduled_job': job['job_id'],
            'on_machine': machine['machine_name'],
            'start_time': start_time,
            'end_time': end_time,
            'valid_action': True
        }
        
        return observation, reward, terminated, truncated, info
    
    def _is_action_valid(self, job_idx: int, machine_idx: int) -> Tuple[bool, str]:
        """
        Check if action is valid according to game rules (physics).
        
        These are hard constraints that can't be violated:
        1. Job not already scheduled
        2. Sequence constraints within family
        3. Machine compatibility
        4. Working hours
        """
        # Check bounds
        if job_idx >= self.n_jobs or machine_idx >= self.n_machines:
            return False, "Action out of bounds"
            
        # Check if job already scheduled
        if job_idx in self.completed_jobs:
            return False, "Job already scheduled"
            
        job = self.jobs[job_idx]
        machine = self.machines[machine_idx]
        
        # Check sequence constraints
        family_id = self.job_to_family[job['job_id']]
        job_sequence = self.job_sequences[job['job_id']]
        
        # Must complete previous sequences first
        for other_job_idx, other_job in enumerate(self.jobs):
            if other_job_idx == job_idx:
                continue
                
            other_family = self.job_to_family[other_job['job_id']]
            if other_family == family_id:
                other_sequence = self.job_sequences[other_job['job_id']]
                if other_sequence < job_sequence and other_job_idx not in self.completed_jobs:
                    return False, f"Must complete sequence {other_sequence} first"
        
        # Check machine compatibility
        # Job has 'required_machines' which contains DB machine IDs
        required_machines = job.get('required_machines', [])
        machine_db_id = machine.get('db_machine_id')
        
        if required_machines and machine_db_id not in required_machines:
            return False, f"Machine {machine_db_id} not in required machines {required_machines}"
            
        return True, "Valid"
    
    def _get_family_ready_time(self, family_id: str, sequence: int) -> float:
        """Get earliest time this sequence can start based on family dependencies."""
        ready_time = 0.0
        
        # Find completion time of previous sequence
        for job_idx, assignment in self.job_assignments.items():
            job = self.jobs[job_idx]
            if self.job_to_family[job['job_id']] == family_id:
                job_seq = self.job_sequences[job['job_id']]
                if job_seq == sequence - 1:
                    ready_time = max(ready_time, assignment['end_time'])
                    
        return ready_time
    
    def _adjust_for_working_hours(self, start_time: float) -> float:
        """
        Adjust start time to respect working hours.
        
        NOTE: Per user request, working hours are NOT enforced during training.
        This is a deployment-only constraint. The AI learns pure scheduling
        without being constrained by specific working patterns.
        """
        # Working hours disabled for training - return as-is
        return start_time
    
    def _calculate_reward(self, job: Dict, machine: Dict, start_time: float, end_time: float) -> float:
        """
        Calculate reward for the action.
        
        The reward function is where the AI learns what constitutes good scheduling.
        We don't hardcode priorities - the AI discovers them.
        """
        reward = 0.0
        
        # Base reward for completing any job
        reward += self.config.get('completion_reward', 10.0)
        
        # Let AI learn importance
        if job.get('is_important', False):
            reward += self.config.get('importance_bonus', 20.0)
        
        # Let AI learn about deadlines
        if 'lcd_date' in job:
            days_until_deadline = job.get('lcd_days_remaining', 30)
            # More reward for jobs closer to deadline
            urgency_factor = max(0, 1.0 - days_until_deadline / 30.0)
            reward += urgency_factor * self.config.get('urgency_multiplier', 50.0)
        
        # Let AI learn about efficiency
        # Waiting time penalty
        wait_time = start_time - self.current_time
        reward -= wait_time * self.config.get('wait_penalty', 0.1)
        
        # Machine utilization bonus
        # AI learns to balance load across machines
        machine_load = len(self.machine_schedules[machine['machine_id']])
        avg_load = sum(len(schedule) for schedule in self.machine_schedules) / self.n_machines
        if machine_load <= avg_load:
            reward += self.config.get('balance_bonus', 5.0)
        
        # Time penalty to encourage faster completion
        reward -= self.config.get('time_penalty', 0.1)
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current state observation.
        
        State includes:
        - Job states (availability, progress, urgency, etc.)
        - Machine states (load, availability, etc.)
        - Global state (time progress)
        """
        # This is a simplified version - in production would be more sophisticated
        # Key is to provide enough info for AI to learn patterns
        
        obs = []
        
        # Job states
        for job_idx, job in enumerate(self.jobs):
            # Is job available to schedule?
            is_available = 1.0 if self._is_job_available(job_idx) else 0.0
            
            # Sequence progress in family
            family_id = self.job_to_family[job['job_id']]
            sequence_progress = self._get_family_progress(family_id)
            
            # Urgency (normalized days to deadline)
            urgency = min(1.0, max(0.0, 1.0 - job.get('lcd_days_remaining', 30) / 30.0))
            
            # Processing time (normalized)
            proc_time = job['processing_time'] / 10.0  # Assume max 10 hours
            
            # Is important
            is_important = 1.0 if job.get('is_important', False) else 0.0
            
            obs.extend([is_available, sequence_progress, urgency, proc_time, is_important])
        
        # Machine states  
        for machine_idx in range(self.n_machines):
            # Current load
            load = len(self.machine_schedules[machine_idx])
            normalized_load = min(1.0, load / 50.0)  # Assume max 50 jobs per machine
            
            # Time until free
            if self.machine_schedules[machine_idx]:
                last_job = self.machine_schedules[machine_idx][-1]
                time_until_free = max(0, last_job['end_time'] - self.current_time)
            else:
                time_until_free = 0.0
            normalized_time = min(1.0, time_until_free / 24.0)  # Normalize to 24 hours
            
            obs.extend([normalized_load, normalized_time])
        
        # Global state
        time_progress = min(1.0, self.current_time / self.total_time_horizon)
        obs.append(time_progress)
        
        # Pad to fixed size
        obs = np.array(obs, dtype=np.float32)
        if len(obs) < self.observation_space.shape[0]:
            obs = np.pad(obs, (0, self.observation_space.shape[0] - len(obs)))
        
        return obs[:self.observation_space.shape[0]]
    
    def _is_job_available(self, job_idx: int) -> bool:
        """Check if job is available to schedule now."""
        if job_idx in self.completed_jobs:
            return False
            
        # Check sequence dependencies
        job = self.jobs[job_idx]
        family_id = self.job_to_family[job['job_id']]
        job_sequence = self.job_sequences[job['job_id']]
        
        # Check if previous sequences are done
        for other_idx, other_job in enumerate(self.jobs):
            if other_idx == job_idx:
                continue
            other_family = self.job_to_family[other_job['job_id']] 
            if other_family == family_id:
                other_sequence = self.job_sequences[other_job['job_id']]
                if other_sequence < job_sequence and other_idx not in self.completed_jobs:
                    return False
                    
        return True
    
    def _schedule_multi_machine_job(self, job_idx: int, primary_machine_idx: int):
        """
        Schedule a job that requires multiple machines simultaneously.
        
        For multi-machine jobs:
        - ALL required machines must be available at the same time
        - All machines are occupied for the entire duration
        - The job appears in all machine schedules
        """
        job = self.jobs[job_idx]
        required_machines = job.get('required_machines', [])
        
        # Map DB machine IDs to environment indices
        required_machine_indices = []
        for db_id in required_machines:
            for m_idx, m in enumerate(self.machines):
                if m.get('db_machine_id') == db_id:
                    required_machine_indices.append(m_idx)
                    break
        
        # Validate we found all required machines
        if len(required_machine_indices) != len(required_machines):
            reward = self.config.get('invalid_action_penalty', -20.0)
            info = {
                'invalid_action': True,
                'reason': 'Could not find all required machines'
            }
            observation = self._get_observation()
            terminated = self._is_done()
            return observation, reward, terminated, False, info
        
        # Find the earliest time ALL machines are available
        start_time = self.current_time
        for m_idx in required_machine_indices:
            if self.machine_schedules[m_idx]:
                last_job = self.machine_schedules[m_idx][-1]
                start_time = max(start_time, last_job['end_time'])
        
        # Check family dependencies
        family_id = self.job_to_family[job['job_id']]
        job_sequence = self.job_sequences[job['job_id']]
        family_ready_time = self._get_family_ready_time(family_id, job_sequence)
        start_time = max(start_time, family_ready_time)
        
        # Apply working hours constraints
        start_time = self._adjust_for_working_hours(start_time)
        
        # Calculate end time
        processing_time = job['processing_time']
        end_time = start_time + processing_time
        
        # Schedule the job on ALL required machines
        scheduled_job = {
            'job': job,
            'job_idx': job_idx,
            'machine_indices': required_machine_indices,  # All machines used
            'start_time': start_time,
            'end_time': end_time,
            'is_multi_machine': True
        }
        
        # Add to all machine schedules
        for m_idx in required_machine_indices:
            self.machine_schedules[m_idx].append(scheduled_job)
        
        self.completed_jobs.add(job_idx)
        self.job_assignments[job_idx] = scheduled_job
        
        # Update current time
        self.current_time = max(self.current_time, start_time)
        
        # Calculate reward - multi-machine jobs might get bonus
        reward = self._calculate_reward(job, self.machines[primary_machine_idx], start_time, end_time)
        if len(required_machines) > 1:
            # Bonus for successfully scheduling complex multi-machine job
            reward += self.config.get('multi_machine_bonus', 10.0)
        
        # Get new state
        observation = self._get_observation()
        terminated = self._is_done()
        
        # Compile info
        info = {
            'scheduled_job': job['job_id'],
            'on_machines': [self.machines[m_idx]['machine_name'] for m_idx in required_machine_indices],
            'start_time': start_time,
            'end_time': end_time,
            'valid_action': True,
            'multi_machine': True,
            'num_machines': len(required_machine_indices)
        }
        
        return observation, reward, terminated, False, info
    
    def _check_multi_machine_availability(self, job_idx: int) -> bool:
        """
        Check if all required machines for a multi-machine job can be scheduled together.
        
        Returns True if all machines will be available at the same time.
        """
        job = self.jobs[job_idx]
        required_machines = job.get('required_machines', [])
        
        # Map DB IDs to machine indices
        required_indices = []
        for db_id in required_machines:
            for m_idx, m in enumerate(self.machines):
                if m.get('db_machine_id') == db_id:
                    required_indices.append(m_idx)
                    break
        
        # Check if we found all machines
        if len(required_indices) != len(required_machines):
            return False
        
        # For simplicity, return True - the actual scheduling will handle timing
        # In a more sophisticated version, we could check future availability
        return True
    
    def _get_family_progress(self, family_id: str) -> float:
        """Get completion progress for a family."""
        family_jobs = self.families[family_id]['jobs']
        completed = sum(1 for job in family_jobs 
                       if self._get_job_index(job['job_id']) in self.completed_jobs)
        return completed / len(family_jobs) if family_jobs else 0.0
    
    def _get_job_index(self, job_id: str) -> Optional[int]:
        """Get job index from job ID."""
        for idx, job in enumerate(self.jobs):
            if job['job_id'] == job_id:
                return idx
        return None
    
    def _is_done(self) -> bool:
        """Check if all jobs are scheduled."""
        return len(self.completed_jobs) == self.n_jobs
    
    def _get_info(self) -> Dict[str, Any]:
        """Get environment info."""
        return {
            'n_jobs': self.n_jobs,
            'n_machines': self.n_machines,
            'completed_jobs': len(self.completed_jobs),
            'current_time': self.current_time
        }
    
    def get_action_mask(self) -> np.ndarray:
        """
        Get mask of valid actions for current state.
        
        For multi-machine jobs:
        - Action is valid if choosing ANY of the required machines
        - Environment will automatically schedule on ALL required machines
        
        Returns:
            Boolean mask of shape (n_jobs, n_machines)
        """
        mask = np.zeros((self.n_jobs, self.n_machines), dtype=bool)
        
        for job_idx in range(self.n_jobs):
            # Skip if already scheduled
            if job_idx in self.completed_jobs:
                continue
                
            job = self.jobs[job_idx]
            required_machines = job.get('required_machines', [])
            
            if len(required_machines) > 1:
                # Multi-machine job - check if ALL required machines will be available
                # Mark as valid if choosing ANY of the required machines
                for machine_idx in range(self.n_machines):
                    machine_db_id = self.machines[machine_idx].get('db_machine_id')
                    if machine_db_id in required_machines:
                        # Check basic validity (sequence, etc)
                        is_valid, _ = self._is_action_valid(job_idx, machine_idx)
                        if is_valid:
                            # Additionally check if ALL required machines can be scheduled together
                            all_available = self._check_multi_machine_availability(job_idx)
                            mask[job_idx, machine_idx] = all_available
            else:
                # Single machine job - normal validation
                for machine_idx in range(self.n_machines):
                    is_valid, _ = self._is_action_valid(job_idx, machine_idx)
                    mask[job_idx, machine_idx] = is_valid
                
        return mask.flatten()
    
    def render(self, mode: str = 'human'):
        """Render current state."""
        if mode == 'human':
            print(f"\n=== Scheduling Game State ===")
            print(f"Time: {self.current_time:.1f} / {self.total_time_horizon:.1f}")
            print(f"Completed: {len(self.completed_jobs)} / {self.n_jobs} jobs")
            print(f"Active machines: {sum(1 for s in self.machine_schedules if s)}")
            
            # Show next available jobs
            available_jobs = [i for i in range(self.n_jobs) if self._is_job_available(i)]
            print(f"\nAvailable jobs: {len(available_jobs)}")
            if available_jobs:
                print("Next 5 jobs:")
                for i in available_jobs[:5]:
                    job = self.jobs[i]
                    imp = "[IMPORTANT]" if job.get('is_important') else ""
                    days = job.get('lcd_days_remaining', '?')
                    print(f"  {job['job_id']} {imp} (LCD: {days} days)")