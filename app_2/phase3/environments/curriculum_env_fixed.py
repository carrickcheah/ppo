"""
FIXED Curriculum Environment - WITH PROPER NO-ACTION
"""

import os
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CurriculumEnvironmentFixed(gym.Env):
    """Fixed environment with REAL no-action support."""
    
    def __init__(self,
                 stage_name: str,
                 data_dir: str = "/Users/carrickcheah/Project/ppo/app_2/data",
                 reward_config: Optional[Dict] = None,
                 max_steps: int = 100,  # Shorter episodes for faster learning
                 verbose: bool = False):
        super().__init__()
        
        self.stage_name = stage_name
        self.data_dir = data_dir
        self.max_steps = max_steps
        self.verbose = verbose
        
        # Load stage data
        self._load_stage_data()
        
        # CRITICAL FIX: Add +1 to action space for NO-ACTION
        # Last index = no-action
        n_families = len(self.families)
        n_machines = len(self.machines)
        
        self.action_space = spaces.MultiDiscrete([n_families + 1, n_machines + 1])
        
        # Observation space
        obs_size = n_families * 6 + n_machines * 3 + 5
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        
        # Reward config with PROPER penalties
        self.reward_config = reward_config or {
            'no_action_penalty': -2.0,      # Penalty for doing nothing
            'invalid_action_penalty': -5.0,  # Penalty for invalid
            'action_bonus': 20.0,           # Bonus for valid scheduling
            'completion_reward': 100.0,     # Completing a job
            'on_time_bonus': 50.0,         # Meeting deadline
            'late_penalty_per_day': -10.0,  # Missing deadline
            'utilization_bonus': 1.0,      # Machine utilization
            'idle_penalty': -0.1           # Idle time penalty
        }
        
        if verbose:
            logger.info(f"FIXED Environment for {stage_name}:")
            logger.info(f"  Action space: {self.action_space} (includes NO-ACTION)")
            logger.info(f"  Jobs: {n_families}, Machines: {n_machines}")
            logger.info(f"  NO-ACTION is: [{n_families}, {n_machines}]")
        
        self.reset()
    
    def _load_stage_data(self):
        """Load real production data."""
        # Try clean data first, fallback to original
        clean_path = os.path.join(self.data_dir, f"stage_{self.stage_name}_clean_data.json")
        original_path = os.path.join(self.data_dir, f"stage_{self.stage_name}_real_data.json")
        
        if os.path.exists(clean_path):
            snapshot_path = clean_path
            if self.verbose:
                logger.info(f"Using CLEAN data: {clean_path}")
        else:
            snapshot_path = original_path
        
        if not os.path.exists(snapshot_path):
            raise FileNotFoundError(f"Stage data not found: {snapshot_path}")
        
        with open(snapshot_path, 'r') as f:
            data = json.load(f)
        
        self.families = data['families']
        self.family_ids = list(self.families.keys())
        self.machines = data['machines']
        self.machine_ids = [m['machine_id'] for m in self.machines]
        
        # Count total tasks
        self.total_tasks = sum(f['total_sequences'] for f in self.families.values())
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment."""
        super().reset(seed=seed)
        
        # Initialize state
        self.current_time = 0.0
        self.steps = 0
        self.scheduled_jobs = set()
        self.completed_jobs = set()
        self.job_assignments = {}
        self.machine_schedules = {m: [] for m in self.machine_ids}
        
        # Family progress tracking
        self.family_progress = {}
        for fid, family in self.families.items():
            self.family_progress[fid] = {
                'completed_sequences': 0,
                'next_sequence': 1,
                'total_sequences': family['total_sequences'],
                'tasks': {i+1: family['tasks'][i] for i in range(len(family['tasks']))}
            }
        
        obs = self._get_observation()
        info = {'valid_actions': self._get_valid_actions()}
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action with PROPER no-action handling."""
        job_idx, machine_idx = action
        self.steps += 1
        
        info = {
            'action_valid': False,
            'action_type': 'unknown',
            'scheduled_job': None
        }
        
        # CRITICAL: Check for NO-ACTION
        if job_idx == len(self.family_ids) or machine_idx == len(self.machine_ids):
            # This is NO-ACTION
            reward = self.reward_config['no_action_penalty']
            info['action_type'] = 'no_action'
            info['reason'] = 'No action taken'
            
            # Time still advances
            self.current_time += 0.1
            
        elif job_idx >= len(self.family_ids) or machine_idx >= len(self.machine_ids):
            # Invalid action
            reward = self.reward_config['invalid_action_penalty']
            info['action_type'] = 'invalid'
            info['reason'] = 'Invalid indices'
            
        else:
            # Valid indices - try to schedule
            family_id = self.family_ids[job_idx]
            machine_id = self.machine_ids[machine_idx]
            
            # Try to schedule
            reward = self._try_schedule(family_id, machine_id, info)
        
        # Get observation
        obs = self._get_observation()
        
        # Check termination
        done = (self.steps >= self.max_steps or 
                len(self.completed_jobs) == len(self.families))
        truncated = False
        
        # Add final reward if done
        if done:
            reward += self._calculate_final_reward()
        
        return obs, reward, done, truncated, info
    
    def _try_schedule(self, family_id: str, machine_id: int, info: Dict) -> float:
        """Try to schedule a job."""
        family = self.families[family_id]
        progress = self.family_progress[family_id]
        
        # Check if family done
        if progress['completed_sequences'] >= progress['total_sequences']:
            info['action_type'] = 'invalid'
            info['reason'] = 'Family completed'
            return self.reward_config['invalid_action_penalty']
        
        # Get next task
        next_seq = progress['next_sequence']
        task = progress['tasks'][next_seq]
        
        # Check machine capability
        if machine_id not in task['capable_machines']:
            info['action_type'] = 'invalid'
            info['reason'] = 'Machine not capable'
            return self.reward_config['invalid_action_penalty']
        
        # Find start time
        machine_available = self._get_machine_available_time(machine_id)
        start_time = max(self.current_time, machine_available)
        
        # Check dependencies
        if next_seq > 1:
            prev_key = f"{family_id}_seq{next_seq-1}"
            if prev_key in self.job_assignments:
                prev_end = self.job_assignments[prev_key]['end']
                start_time = max(start_time, prev_end)
            else:
                info['action_type'] = 'invalid'
                info['reason'] = 'Previous sequence not complete'
                return self.reward_config['invalid_action_penalty']
        
        # Schedule the job
        end_time = start_time + task['processing_time']
        job_key = f"{family_id}_seq{next_seq}"
        
        self.job_assignments[job_key] = {
            'machines': [machine_id],
            'start': start_time,
            'end': end_time
        }
        
        self.machine_schedules[machine_id].append({
            'job': job_key,
            'start': start_time,
            'end': end_time
        })
        
        self.scheduled_jobs.add(job_key)
        
        # Update progress
        progress['next_sequence'] += 1
        progress['completed_sequences'] += 1
        
        if progress['completed_sequences'] >= progress['total_sequences']:
            self.completed_jobs.add(family_id)
        
        # Update time
        self.current_time = max(self.current_time, start_time + 0.1)
        
        # Calculate reward
        reward = self.reward_config['action_bonus']
        
        # Add completion bonus
        if progress['completed_sequences'] >= progress['total_sequences']:
            reward += self.reward_config['completion_reward']
        
        # Check deadline
        days_until_lcd = family['lcd_days_remaining'] - (end_time / 24.0)
        if days_until_lcd >= 0:
            reward += self.reward_config['on_time_bonus']
        else:
            reward += self.reward_config['late_penalty_per_day'] * abs(days_until_lcd)
        
        info['action_valid'] = True
        info['action_type'] = 'schedule'
        info['scheduled_job'] = job_key
        
        return reward
    
    def _get_machine_available_time(self, machine_id: int) -> float:
        """Get when machine will be available."""
        if not self.machine_schedules[machine_id]:
            return 0.0
        return max(job['end'] for job in self.machine_schedules[machine_id])
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        obs = []
        
        # Family features
        for fid in self.family_ids:
            family = self.families[fid]
            progress = self.family_progress[fid]
            
            completion_rate = progress['completed_sequences'] / progress['total_sequences']
            days_remaining = family['lcd_days_remaining'] - self.current_time / 24.0
            
            obs.extend([
                completion_rate,
                progress['total_sequences'],
                days_remaining,
                float(family.get('is_important', 0)),
                len(self.scheduled_jobs),  # Simplified
                self.current_time
            ])
        
        # Machine features
        for mid in self.machine_ids:
            available_time = self._get_machine_available_time(mid)
            utilization = len(self.machine_schedules[mid]) / max(1, self.steps)
            
            obs.extend([
                utilization,
                max(0, available_time - self.current_time),
                1.0  # Simplified machine type
            ])
        
        # Global features
        obs.extend([
            len(self.completed_jobs) / len(self.families),
            len(self.scheduled_jobs) / self.total_tasks,
            self.current_time / 100.0,
            self.steps / self.max_steps,
            1.0  # Simplified
        ])
        
        return np.array(obs, dtype=np.float32)
    
    def _get_valid_actions(self) -> List[Tuple[int, int]]:
        """Get list of valid actions INCLUDING no-action."""
        valid = []
        
        # Check all job-machine pairs
        for j_idx, fid in enumerate(self.family_ids):
            progress = self.family_progress[fid]
            
            if progress['completed_sequences'] < progress['total_sequences']:
                next_seq = progress['next_sequence']
                task = progress['tasks'][next_seq]
                
                # Check dependencies
                can_schedule = True
                if next_seq > 1:
                    prev_key = f"{fid}_seq{next_seq-1}"
                    if prev_key not in self.job_assignments:
                        can_schedule = False
                
                if can_schedule:
                    for m_idx, mid in enumerate(self.machine_ids):
                        if mid in task['capable_machines']:
                            valid.append((j_idx, m_idx))
        
        # ALWAYS add no-action as valid
        valid.append((len(self.family_ids), len(self.machine_ids)))
        
        return valid
    
    def _calculate_final_reward(self) -> float:
        """Calculate final episode reward."""
        # Only reward actual scheduling performance
        completion_rate = len(self.completed_jobs) / len(self.families)
        scheduling_rate = len(self.scheduled_jobs) / self.total_tasks
        
        return (completion_rate * 200 + scheduling_rate * 100)