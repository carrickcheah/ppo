"""
Improved Small Bottleneck Environment with better time scaling and action masking
"""

import os
import numpy as np
from typing import Dict, Optional, List, Tuple
from .base_strategy_env import BaseStrategyEnvironment
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete


class ImprovedBottleneckEnvironment(BaseStrategyEnvironment):
    """
    Improved bottleneck environment with:
    - Better time scaling (1 hour increments instead of 0.1)
    - Action masking to prevent invalid actions
    - Adjusted reward structure
    - Better episode length for proper evaluation
    """
    
    def __init__(self, verbose: bool = False):
        """Initialize Improved Bottleneck environment."""
        
        # Path to data file
        data_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data',
            'small_bottleneck_data.json'
        )
        
        # Improved reward configuration
        reward_config = {
            'no_action_penalty': -2.0,      # Increased to discourage waiting
            'invalid_action_penalty': -50.0, # Much higher to strongly discourage
            'valid_schedule_reward': 50.0,   # Higher base reward
            'sequence_completion': 100.0,    # Much higher sequence bonus
            'family_completion': 200.0,      # Much higher family bonus
            'on_time_bonus': 50.0,          # Good on-time bonus
            'late_penalty_per_day': -5.0,   # Moderate late penalty
            'utilization_bonus': 20.0,      # Bonus for high utilization
            'progress_bonus': 30.0          # New: bonus for making progress
        }
        
        # Initialize tracking first
        self.max_concurrent_jobs = 0
        self.total_wait_time = 0
        self.last_progress_step = 0
        self.time_increment = 1.0  # 1 hour instead of 0.1
        self.valid_pairs = {}  # Initialize empty
        
        # Initialize with longer episode length
        super().__init__(
            scenario_name='improved_bottleneck',
            data_file=data_file,
            reward_config=reward_config,
            max_steps=500,  # Increased for better time coverage
            verbose=verbose
        )
        
        # Now we can initialize machine tracking and compute valid pairs
        self.machine_idle_time = {m: 0 for m in self.machine_ids}
        
        # Pre-compute valid job-machine pairs for efficiency
        self._precompute_valid_pairs()
    
    def _precompute_valid_pairs(self):
        """Pre-compute which job-machine pairs are valid."""
        self.valid_pairs = {}
        
        for fid, family in self.families.items():
            family_idx = self.family_ids.index(fid)
            self.valid_pairs[family_idx] = []
            
            # Get first task (sequence 1) to check capable machines
            first_task = family['tasks'][0] if family.get('tasks') else None
            if first_task and 'capable_machines' in first_task:
                for machine_id in first_task['capable_machines']:
                    if machine_id in self.machine_ids:
                        machine_idx = self.machine_ids.index(machine_id)
                        self.valid_pairs[family_idx].append(machine_idx)
    
    def step(self, action):
        """Override step with improved time handling and rewards."""
        prev_time = self.current_time
        prev_scheduled = len(self.scheduled_jobs)
        
        # Get base step results
        obs, reward, done, truncated, info = super().step(action)
        
        # Override time increment
        if not done and not truncated:
            self.current_time = prev_time + self.time_increment
        
        # Add progress bonus if we scheduled something new
        if len(self.scheduled_jobs) > prev_scheduled:
            progress_bonus = self.reward_config.get('progress_bonus', 30.0)
            # Decay progress bonus if we haven't made progress recently
            steps_since_progress = self.steps - self.last_progress_step
            if steps_since_progress > 10:
                progress_bonus *= 0.5
            reward += progress_bonus
            self.last_progress_step = self.steps
            info['progress_bonus'] = progress_bonus
        
        # Track utilization and add bonus
        if info.get('action_valid') and info.get('action_type') == 'schedule':
            utilization = self._calculate_utilization()
            if utilization > 0.7:  # Good utilization
                utilization_bonus = self.reward_config.get('utilization_bonus', 20.0)
                utilization_bonus *= utilization  # Scale by actual utilization
                reward += utilization_bonus
                info['utilization_bonus'] = utilization_bonus
        
        # Penalize if no progress for too long
        if self.steps - self.last_progress_step > 20:
            reward -= 5.0  # Additional penalty for being stuck
        
        return obs, reward, done, truncated, info
    
    def _get_valid_actions(self) -> List[Tuple[int, int]]:
        """Get list of valid actions with pre-computed pairs."""
        valid_actions = []
        
        for family_idx, family_id in enumerate(self.family_ids):
            progress = self.family_progress[family_id]
            
            # Skip if family is complete
            if progress['completed_sequences'] >= progress['total_sequences']:
                continue
            
            # Get valid machines for this family
            valid_machines = self.valid_pairs.get(family_idx, [])
            
            for machine_idx in valid_machines:
                machine_id = self.machine_ids[machine_idx]
                
                # Check if machine is available soon enough
                machine_available = self._get_machine_available_time(machine_id)
                if machine_available - self.current_time > 48:  # Skip if too far in future
                    continue
                
                valid_actions.append((family_idx, machine_idx))
        
        # Always add NO-ACTION as an option
        valid_actions.append((len(self.family_ids), len(self.machine_ids)))
        
        return valid_actions
    
    def _is_action_valid(self, family_idx: int, machine_idx: int) -> bool:
        """Check if action is valid using pre-computed pairs."""
        # Check NO-ACTION
        if family_idx == len(self.family_ids) or machine_idx == len(self.machine_ids):
            return True
        
        # Check bounds
        if family_idx >= len(self.family_ids) or machine_idx >= len(self.machine_ids):
            return False
        
        # Check if machine is valid for this family
        return machine_idx in self.valid_pairs.get(family_idx, [])
    
    def _calculate_utilization(self) -> float:
        """Calculate current machine utilization."""
        if self.current_time == 0:
            return 0.0
        
        busy_time = 0
        total_time = 0
        
        for machine_id in self.machine_ids:
            # Calculate total busy time for this machine
            for job in self.machine_schedules.get(machine_id, []):
                if job['start'] < self.current_time:
                    busy_end = min(job['end'], self.current_time)
                    busy_time += busy_end - job['start']
            
            total_time += self.current_time
        
        return busy_time / total_time if total_time > 0 else 0.0
    
    def get_action_mask(self) -> np.ndarray:
        """Get action mask for valid actions."""
        mask = np.zeros(self.action_space.nvec, dtype=bool)
        
        # Mark all valid actions
        valid_actions = self._get_valid_actions()
        for family_idx, machine_idx in valid_actions:
            if family_idx < self.action_space.nvec[0] and machine_idx < self.action_space.nvec[1]:
                mask[family_idx, machine_idx] = True
        
        return mask
    
    def _get_info(self) -> Dict:
        """Get environment info with improved metrics."""
        info = super()._get_info() if hasattr(super(), '_get_info') else {}
        
        # Calculate throughput
        throughput = len(self.scheduled_jobs) / max(1, self.current_time) * 24  # Jobs per day
        
        # Calculate true utilization
        utilization = self._calculate_utilization()
        
        info.update({
            'scenario': 'improved_bottleneck',
            'scheduled_jobs': len(self.scheduled_jobs),
            'total_tasks': self.total_tasks,
            'completion_rate': len(self.scheduled_jobs) / self.total_tasks if self.total_tasks > 0 else 0,
            'machine_utilization': utilization,
            'throughput_per_day': throughput,
            'current_time_days': self.current_time / 24.0,
            'valid_actions_count': len(self._get_valid_actions()),
            'time_increment': self.time_increment
        })
        
        return info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment with improved tracking."""
        obs, info = super().reset(seed, options)
        
        # Reset improved tracking
        self.max_concurrent_jobs = 0
        self.total_wait_time = 0
        self.machine_idle_time = {m: 0 for m in self.machine_ids}
        self.last_progress_step = 0
        self.time_increment = 1.0
        
        # Add action mask to info
        info['action_mask'] = self.get_action_mask()
        
        return obs, info