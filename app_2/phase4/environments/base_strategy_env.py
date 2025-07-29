"""
Base Strategy Environment for Phase 4
Inherits from the curriculum environment and adds strategy-specific features
"""

import os
import sys
import json
import numpy as np
import gymnasium as gym
from typing import Dict, List, Tuple, Optional, Any

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from phase3.environments.curriculum_env_truly_fixed import CurriculumEnvironmentTrulyFixed


class BaseStrategyEnvironment(CurriculumEnvironmentTrulyFixed):
    """
    Base class for strategy-specific environments.
    Provides common functionality and metrics tracking.
    """
    
    def __init__(self,
                 scenario_name: str,
                 data_file: str,
                 reward_config: Optional[Dict] = None,
                 max_steps: int = 200,
                 verbose: bool = False):
        """
        Initialize strategy environment.
        
        Args:
            scenario_name: Name of the scenario (e.g., 'small_balanced')
            data_file: Path to the JSON data file
            reward_config: Custom reward configuration
            max_steps: Maximum steps per episode
            verbose: Whether to print debug info
        """
        self.scenario_name = scenario_name
        self.data_file = data_file
        
        # Don't call parent init yet - we need to load our data first
        self.verbose = verbose
        self.max_steps = max_steps
        
        # Load strategy-specific data
        self._load_strategy_data()
        
        # Now we can initialize parent with our data
        # We'll override the data loading method
        self._init_complete = True
        
        # Set up action and observation spaces
        n_families = len(self.families)
        n_machines = len(self.machines)
        
        self.family_ids = list(self.families.keys())
        self.machine_ids = [m['machine_id'] for m in self.machines]
        
        # Action space: [family_index, machine_index]
        self.action_space = gym.spaces.MultiDiscrete([n_families + 1, n_machines + 1])
        
        # Observation space
        obs_size = n_families * 6 + n_machines * 3 + 5
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        
        # Default reward config with strategy adjustments
        self.reward_config = reward_config or self._get_default_reward_config()
        
        # Strategy-specific metrics
        self.strategy_metrics = {
            'total_episodes': 0,
            'successful_episodes': 0,
            'avg_completion_rate': 0.0,
            'avg_makespan': 0.0,
            'avg_lateness': 0.0,
            'best_completion_rate': 0.0
        }
        
        # Initialize environment state
        self.reset()
        
        if verbose:
            print(f"\nStrategy Environment: {scenario_name}")
            print(f"Jobs: {n_families}, Machines: {n_machines}")
            print(f"Scenario: {self.scenario_data.get('description', 'N/A')}")
    
    def _load_strategy_data(self):
        """Load strategy-specific data from JSON file."""
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        
        with open(self.data_file, 'r') as f:
            self.scenario_data = json.load(f)
        
        self.families = self.scenario_data['families']
        self.machines = self.scenario_data['machines']
        
        # Count total tasks
        self.total_tasks = sum(
            family.get('total_sequences', len(family.get('tasks', [])))
            for family in self.families.values()
        )
    
    def _load_stage_data(self):
        """Override parent's data loading - we already loaded our data."""
        # This prevents parent from trying to load stage data
        pass
    
    def _get_default_reward_config(self) -> Dict[str, float]:
        """Get default reward configuration for strategy environments."""
        return {
            'no_action_penalty': -1.0,
            'invalid_action_penalty': -10.0,
            'valid_schedule_reward': 15.0,    # Higher than toy stages
            'sequence_completion': 25.0,       # Higher sequence bonus
            'family_completion': 60.0,         # Higher family bonus
            'on_time_bonus': 20.0,            # Higher on-time bonus
            'late_penalty_per_day': -2.0      # Moderate late penalty
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment and track metrics."""
        # Track previous episode metrics if any
        if hasattr(self, 'scheduled_jobs') and self.scheduled_jobs:
            self._update_episode_metrics()
        
        # Standard reset
        self.current_time = 0.0
        self.steps = 0
        self.scheduled_jobs = set()
        self.completed_jobs = set()
        self.job_assignments = {}
        self.machine_schedules = {m: [] for m in self.machine_ids}
        
        # Family progress tracking
        self.family_progress = {}
        for fid, family in self.families.items():
            total_seq = family.get('total_sequences', len(family.get('tasks', [])))
            self.family_progress[fid] = {
                'completed_sequences': 0,
                'next_sequence': 1,
                'total_sequences': total_seq,
                'tasks': {i+1: task for i, task in enumerate(family.get('tasks', []))}
            }
        
        # Strategy-specific tracking
        self.episode_start_time = 0.0
        self.late_jobs = 0
        self.on_time_jobs = 0
        
        obs = self._get_observation()
        info = {
            'valid_actions': self._get_valid_actions(),
            'scenario': self.scenario_name
        }
        
        return obs, info
    
    def _update_episode_metrics(self):
        """Update strategy metrics at episode end."""
        self.strategy_metrics['total_episodes'] += 1
        
        # Calculate completion rate
        completion_rate = len(self.scheduled_jobs) / self.total_tasks if self.total_tasks > 0 else 0
        
        # Update best
        if completion_rate > self.strategy_metrics['best_completion_rate']:
            self.strategy_metrics['best_completion_rate'] = completion_rate
        
        # Update running average
        n = self.strategy_metrics['total_episodes']
        old_avg = self.strategy_metrics['avg_completion_rate']
        self.strategy_metrics['avg_completion_rate'] = (old_avg * (n-1) + completion_rate) / n
        
        # Track success (>70% completion)
        if completion_rate >= 0.7:
            self.strategy_metrics['successful_episodes'] += 1
        
        # Calculate makespan if jobs were scheduled
        if self.job_assignments:
            makespan = max(
                assignment['end']
                for assignment in self.job_assignments.values()
            )
            old_makespan = self.strategy_metrics['avg_makespan']
            self.strategy_metrics['avg_makespan'] = (old_makespan * (n-1) + makespan) / n
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of strategy metrics."""
        success_rate = (
            self.strategy_metrics['successful_episodes'] / 
            self.strategy_metrics['total_episodes']
            if self.strategy_metrics['total_episodes'] > 0 else 0
        )
        
        return {
            'scenario': self.scenario_name,
            'total_episodes': self.strategy_metrics['total_episodes'],
            'success_rate': success_rate,
            'avg_completion': self.strategy_metrics['avg_completion_rate'],
            'best_completion': self.strategy_metrics['best_completion_rate'],
            'avg_makespan': self.strategy_metrics['avg_makespan']
        }
    
    def render(self, mode='human'):
        """Render current state."""
        if mode == 'human':
            print(f"\n--- {self.scenario_name} Step {self.steps} ---")
            print(f"Scheduled: {len(self.scheduled_jobs)}/{self.total_tasks}")
            print(f"On-time: {self.on_time_jobs}, Late: {self.late_jobs}")
            print(f"Current time: {self.current_time:.1f} hours")