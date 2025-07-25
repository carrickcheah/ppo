"""
Phase 3 Curriculum Environment with REAL Production Data
Loads from real data snapshots created by ingest_real_data.py
Implements all fixes: machine ID mapping, reward structure, action validity
"""

import os
import json
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from datetime import datetime, timedelta
import gymnasium as gym
from gymnasium import spaces

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CurriculumEnvironmentReal(gym.Env):
    """Curriculum learning environment using REAL production data."""
    
    def __init__(self, 
                 stage_name: str,
                 data_dir: str = "/Users/carrickcheah/Project/ppo/app_2/data",
                 reward_config: Optional[Dict] = None,
                 max_steps: int = 5000,
                 verbose: bool = False):
        """
        Initialize curriculum environment with real data.
        
        Args:
            stage_name: Name of curriculum stage (e.g., 'toy_easy', 'production_expert')
            data_dir: Directory containing real data snapshots
            reward_config: Custom reward configuration
            max_steps: Maximum steps per episode
            verbose: Enable detailed logging
        """
        super().__init__()
        
        self.stage_name = stage_name
        self.data_dir = data_dir
        self.verbose = verbose
        self.max_steps = max_steps
        self.current_step = 0
        
        # Load stage data
        self._load_stage_data()
        
        # Initialize reward configuration
        self.reward_config = reward_config or self._get_default_reward_config()
        
        # Define observation and action spaces
        self._setup_spaces()
        
        # Initialize state
        self.reset()
        
    def _load_stage_data(self):
        """Load real production data for the specified stage."""
        snapshot_path = os.path.join(self.data_dir, f"stage_{self.stage_name}_real_data.json")
        
        if not os.path.exists(snapshot_path):
            raise FileNotFoundError(f"Stage data not found: {snapshot_path}")
        
        with open(snapshot_path, 'r') as f:
            data = json.load(f)
        
        # Validate it's real data
        if data.get('metadata', {}).get('data_source') != 'REAL_PRODUCTION_DATABASE':
            raise ValueError(f"Stage {self.stage_name} does not contain real production data!")
        
        # Extract data
        self.families = data['families']
        self.machines = data['machines']
        self.stage_config = data.get('stage_config', {})
        
        # Create mappings
        self.family_ids = list(self.families.keys())
        self.machine_ids = [m['machine_id'] for m in self.machines]
        self.machine_id_to_idx = {m_id: idx for idx, m_id in enumerate(self.machine_ids)}
        
        # Calculate total tasks
        self.total_tasks = sum(len(f['tasks']) for f in self.families.values())
        
        if self.verbose:
            logger.info(f"Loaded stage '{self.stage_name}' with REAL data:")
            logger.info(f"  - {len(self.families)} job families")
            logger.info(f"  - {self.total_tasks} total tasks")
            logger.info(f"  - {len(self.machines)} machines")
            sample_jobs = list(self.families.keys())[:3]
            logger.info(f"  - Sample job IDs: {sample_jobs} (Real production jobs)")
    
    def _get_default_reward_config(self) -> Dict:
        """Get default reward configuration based on stage."""
        # Base rewards
        config = {
            'on_time_bonus': 50.0,
            'late_penalty_per_day': -10.0,
            'important_multiplier': 2.0,
            'utilization_bonus': 0.1,
            'makespan_bonus': 0.05,
            'invalid_action_penalty': -5.0,
            'completion_reward': 50.0,  # CRITICAL: Reward for completing ANY job
            'action_bonus': 5.0,  # NEW: Small reward for taking valid actions
            'idle_penalty': -0.1,  # Small penalty for idle time
            'sequence_violation_penalty': -100.0,
        }
        
        # Adjust based on stage profile
        if 'rush' in self.stage_name:
            config['on_time_bonus'] = 100.0
            config['late_penalty_per_day'] = -5.0  # More tolerant of lateness
            config['completion_reward'] = 75.0  # Higher completion bonus
        elif 'bottleneck' in self.stage_name:
            config['utilization_bonus'] = 0.5
            config['makespan_bonus'] = 0.1
        elif 'learning' in self.stage_config.get('reward_profile', ''):
            config['completion_reward'] = 100.0  # Very high for learning
            config['action_bonus'] = 10.0
            config['invalid_action_penalty'] = -2.0  # Gentle penalty
        
        return config
    
    def _setup_spaces(self):
        """Define observation and action spaces."""
        # Observation space
        obs_size = self._calculate_obs_size()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )
        
        # Action space: MultiDiscrete([n_jobs, n_machines])
        self.action_space = spaces.MultiDiscrete([len(self.families), len(self.machines)])
        
        if self.verbose:
            logger.info(f"Observation space: {self.observation_space}")
            logger.info(f"Action space: {self.action_space}")
    
    def _calculate_obs_size(self) -> int:
        """Calculate observation vector size."""
        # Job features: 6 per job (sequence_progress, total_sequences, days_remaining, 
        #                         is_important, total_processing_time, n_capable_machines)
        job_features = len(self.families) * 6
        
        # Machine features: 3 per machine (utilization, scheduled_until, machine_type)
        machine_features = len(self.machines) * 3
        
        # Global features: 5 (current_time, completed_ratio, avg_lateness, 
        #                    important_pending, total_idle_time)
        global_features = 5
        
        return job_features + machine_features + global_features
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Reset time and step counter
        self.current_time = 0.0
        self.current_step = 0
        
        # Reset scheduling state
        self.scheduled_jobs = set()
        self.completed_jobs = set()
        self.machine_schedules = {m_id: [] for m_id in self.machine_ids}
        self.job_assignments = {}
        self.job_start_times = {}
        self.job_end_times = {}
        
        # Reset family progress tracking
        self.family_progress = {}
        for family_id, family in self.families.items():
            self.family_progress[family_id] = {
                'completed_sequences': 0,
                'total_sequences': family['total_sequences'],
                'next_sequence': 1,
                'tasks': {task['sequence']: task for task in family['tasks']}
            }
        
        # Calculate initial state
        obs = self._get_observation()
        info = {'valid_actions': self._get_valid_actions()}
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action and return results."""
        job_idx, machine_idx = action
        
        # CRITICAL FIX: Map machine index to machine ID (0-based to 1-based)
        machine_id = self.machine_ids[machine_idx] if machine_idx < len(self.machine_ids) else None
        
        # Initialize info dict
        info = {
            'action_valid': False,  # CRITICAL: Use 'action_valid' not 'valid_action'
            'scheduled_job': None,
            'reason': None
        }
        
        # Validate action
        if job_idx >= len(self.family_ids) or machine_idx >= len(self.machine_ids):
            reward = self.reward_config['invalid_action_penalty']
            info['reason'] = 'Invalid indices'
        else:
            family_id = self.family_ids[job_idx]
            family = self.families[family_id]
            family_prog = self.family_progress[family_id]
            
            # Check if family has pending tasks
            if family_prog['completed_sequences'] >= family_prog['total_sequences']:
                reward = self.reward_config['invalid_action_penalty']
                info['reason'] = 'Family already completed'
            else:
                # Get next task to schedule
                next_seq = family_prog['next_sequence']
                task = family_prog['tasks'].get(next_seq)
                
                if not task:
                    reward = self.reward_config['invalid_action_penalty']
                    info['reason'] = f'Task sequence {next_seq} not found'
                else:
                    # Check if machine is capable
                    if machine_id not in task['capable_machines']:
                        reward = self.reward_config['invalid_action_penalty']
                        info['reason'] = f'Machine {machine_id} not capable for task'
                    else:
                        # Schedule the job
                        reward = self._schedule_job(family_id, task, machine_id)
                        info['action_valid'] = True
                        info['scheduled_job'] = f"{family_id}_seq{next_seq}"
        
        # Update time
        self.current_time += 0.1  # Small time increment
        self.current_step += 1
        
        # Check termination
        done = self._is_done()
        truncated = self.current_step >= self.max_steps
        
        # Calculate final reward adjustments
        if done:
            reward += self._calculate_final_reward()
        
        # Get new observation
        obs = self._get_observation()
        info['valid_actions'] = self._get_valid_actions()
        
        return obs, reward, done, truncated, info
    
    def _schedule_job(self, family_id: str, task: Dict, machine_id: int) -> float:
        """Schedule a job on the specified machine(s)."""
        # For multi-machine jobs
        if 'Machine_v' in task and ',' in str(task.get('Machine_v', '')):
            # This is a multi-machine job - schedule on ALL required machines
            required_machines = task['capable_machines']
            
            # Find earliest time when ALL machines are available
            earliest_start = 0.0
            for req_machine_id in required_machines:
                machine_available = self._get_machine_available_time(req_machine_id)
                earliest_start = max(earliest_start, machine_available)
            
            # Schedule on ALL machines
            job_key = f"{family_id}_seq{task['sequence']}"
            end_time = earliest_start + task['processing_time']
            
            for req_machine_id in required_machines:
                self.machine_schedules[req_machine_id].append({
                    'job': job_key,
                    'start': earliest_start,
                    'end': end_time
                })
            
            # Record assignment
            self.job_assignments[job_key] = {
                'machines': required_machines,
                'start': earliest_start,
                'end': end_time
            }
        else:
            # Single machine job
            earliest_start = max(self.current_time, self._get_machine_available_time(machine_id))
            job_key = f"{family_id}_seq{task['sequence']}"
            end_time = earliest_start + task['processing_time']
            
            # Schedule on machine
            self.machine_schedules[machine_id].append({
                'job': job_key,
                'start': earliest_start,
                'end': end_time
            })
            
            # Record assignment
            self.job_assignments[job_key] = {
                'machines': [machine_id],
                'start': earliest_start,
                'end': end_time
            }
        
        # Update family progress
        self.family_progress[family_id]['completed_sequences'] += 1
        self.family_progress[family_id]['next_sequence'] += 1
        self.scheduled_jobs.add(job_key)
        
        # Mark as completed if all sequences done
        if self.family_progress[family_id]['completed_sequences'] >= self.family_progress[family_id]['total_sequences']:
            self.completed_jobs.add(family_id)
        
        # Calculate immediate reward
        reward = self._calculate_immediate_reward(family_id, task, earliest_start, end_time)
        
        return reward
    
    def _get_machine_available_time(self, machine_id: int) -> float:
        """Get the earliest time a machine is available."""
        if not self.machine_schedules[machine_id]:
            return self.current_time
        
        # Sort by end time and get the last job's end time
        sorted_schedule = sorted(self.machine_schedules[machine_id], key=lambda x: x['end'])
        return sorted_schedule[-1]['end']
    
    def _calculate_immediate_reward(self, family_id: str, task: Dict, start_time: float, end_time: float) -> float:
        """Calculate immediate reward for scheduling a job."""
        family = self.families[family_id]
        
        # Base completion reward - CRITICAL for preventing "do nothing" behavior
        reward = self.reward_config['completion_reward']
        
        # Action bonus for taking ANY valid action
        reward += self.reward_config['action_bonus']
        
        # On-time bonus/penalty
        days_until_lcd = family['lcd_days_remaining'] - (end_time / 24.0)
        if days_until_lcd >= 0:
            reward += self.reward_config['on_time_bonus']
        else:
            reward += self.reward_config['late_penalty_per_day'] * abs(days_until_lcd)
        
        # Important job bonus
        if family['is_important']:
            reward *= self.reward_config['important_multiplier']
        
        # Utilization bonus
        idle_time = start_time - self.current_time
        if idle_time > 0:
            reward += self.reward_config['idle_penalty'] * idle_time
        
        return reward
    
    def _is_done(self) -> bool:
        """Check if episode is complete."""
        # Episode is done when all families are completed
        return len(self.completed_jobs) >= len(self.families)
    
    def _calculate_final_reward(self) -> float:
        """Calculate final reward at episode end."""
        final_reward = 0.0
        
        # Calculate overall metrics
        total_late = 0
        total_on_time = 0
        makespan = 0.0
        
        for family_id, family in self.families.items():
            if family_id in self.completed_jobs:
                # Get family completion time
                family_jobs = [job for job in self.job_assignments.keys() if job.startswith(family_id)]
                if family_jobs:
                    completion_time = max(self.job_assignments[job]['end'] for job in family_jobs)
                    makespan = max(makespan, completion_time)
                    
                    # Check if on time
                    days_late = (completion_time / 24.0) - family['lcd_days_remaining']
                    if days_late <= 0:
                        total_on_time += 1
                    else:
                        total_late += 1
        
        # Makespan bonus
        if makespan > 0:
            final_reward += self.reward_config['makespan_bonus'] * (self.total_tasks * 10 - makespan)
        
        # Overall performance bonus
        completion_rate = len(self.completed_jobs) / len(self.families)
        final_reward += completion_rate * 100
        
        return final_reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector."""
        obs = []
        
        # Job features
        for family_id in self.family_ids:
            family = self.families[family_id]
            progress = self.family_progress[family_id]
            
            # Features: sequence_progress, total_sequences, days_remaining,
            #          is_important, total_processing_time, n_capable_machines
            sequence_progress = progress['completed_sequences'] / progress['total_sequences']
            days_remaining = max(0, family['lcd_days_remaining'] - self.current_time / 24.0)
            total_processing = sum(task['processing_time'] for task in family['tasks'])
            avg_capable = np.mean([len(task['capable_machines']) for task in family['tasks']])
            
            obs.extend([
                sequence_progress,
                progress['total_sequences'],
                days_remaining,
                float(family['is_important']),
                total_processing,
                avg_capable
            ])
        
        # Machine features
        for machine_id in self.machine_ids:
            # Calculate utilization
            if self.current_time > 0:
                busy_time = sum(job['end'] - job['start'] for job in self.machine_schedules[machine_id])
                utilization = busy_time / self.current_time
            else:
                utilization = 0.0
            
            # When available
            available_time = self._get_machine_available_time(machine_id)
            time_until_available = max(0, available_time - self.current_time)
            
            # Machine type (normalized)
            machine = next(m for m in self.machines if m['machine_id'] == machine_id)
            machine_type = machine['machine_type_id'] / 100.0  # Normalize
            
            obs.extend([utilization, time_until_available, machine_type])
        
        # Global features
        completed_ratio = len(self.completed_jobs) / len(self.families) if self.families else 0
        scheduled_ratio = len(self.scheduled_jobs) / self.total_tasks if self.total_tasks > 0 else 0
        
        # Average lateness
        total_lateness = 0
        for family_id in self.completed_jobs:
            family = self.families[family_id]
            family_jobs = [job for job in self.job_assignments.keys() if job.startswith(family_id)]
            if family_jobs:
                completion_time = max(self.job_assignments[job]['end'] for job in family_jobs)
                days_late = (completion_time / 24.0) - family['lcd_days_remaining']
                total_lateness += max(0, days_late)
        
        avg_lateness = total_lateness / len(self.completed_jobs) if self.completed_jobs else 0
        
        # Important jobs pending
        important_pending = sum(1 for fid in self.family_ids 
                              if fid not in self.completed_jobs and self.families[fid]['is_important'])
        
        # Total idle time
        total_idle = sum(self._get_machine_available_time(m_id) - self.current_time 
                        for m_id in self.machine_ids) / len(self.machine_ids)
        
        obs.extend([
            self.current_time / 1000.0,  # Normalize time
            completed_ratio,
            avg_lateness / 10.0,  # Normalize
            important_pending / len(self.families) if self.families else 0,
            total_idle / 100.0  # Normalize
        ])
        
        return np.array(obs, dtype=np.float32)
    
    def _get_valid_actions(self) -> List[Tuple[int, int]]:
        """Get list of valid actions in current state."""
        valid_actions = []
        
        for job_idx, family_id in enumerate(self.family_ids):
            progress = self.family_progress[family_id]
            
            # Skip completed families
            if progress['completed_sequences'] >= progress['total_sequences']:
                continue
            
            # Get next task
            next_seq = progress['next_sequence']
            task = progress['tasks'].get(next_seq)
            
            if task:
                # Check each capable machine
                for machine_id in task['capable_machines']:
                    if machine_id in self.machine_id_to_idx:
                        machine_idx = self.machine_id_to_idx[machine_id]
                        valid_actions.append((job_idx, machine_idx))
        
        return valid_actions
    
    def render(self, mode='human'):
        """Render current state."""
        if not self.verbose:
            return
        
        print(f"\n=== Step {self.current_step} | Time: {self.current_time:.1f}h ===")
        print(f"Completed: {len(self.completed_jobs)}/{len(self.families)} families")
        print(f"Scheduled: {len(self.scheduled_jobs)}/{self.total_tasks} tasks")
        
        # Show machine utilization
        for machine_id in self.machine_ids[:5]:  # Show first 5 machines
            available = self._get_machine_available_time(machine_id)
            status = "BUSY" if available > self.current_time else "IDLE"
            print(f"Machine {machine_id}: {status} (available at {available:.1f}h)")
        
        print("...")