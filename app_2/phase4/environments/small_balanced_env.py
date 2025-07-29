"""
Small Balanced Environment
30 jobs, 15 machines - Tests general scheduling with balanced workload
"""

import os
from typing import Dict, Optional
from .base_strategy_env import BaseStrategyEnvironment


class SmallBalancedEnvironment(BaseStrategyEnvironment):
    """
    Balanced scenario with mix of urgent and normal deadlines.
    Tests general scheduling ability without extreme constraints.
    """
    
    def __init__(self, verbose: bool = False):
        """Initialize Small Balanced environment."""
        
        # Path to data file
        data_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data',
            'small_balanced_data.json'
        )
        
        # Balanced reward configuration
        reward_config = {
            'no_action_penalty': -1.0,
            'invalid_action_penalty': -10.0,
            'valid_schedule_reward': 15.0,
            'sequence_completion': 25.0,
            'family_completion': 60.0,
            'on_time_bonus': 20.0,
            'late_penalty_per_day': -2.0  # Moderate penalty
        }
        
        super().__init__(
            scenario_name='small_balanced',
            data_file=data_file,
            reward_config=reward_config,
            max_steps=200,
            verbose=verbose
        )
    
    def _get_info(self) -> Dict:
        """Get environment info with balanced metrics."""
        info = super()._get_info() if hasattr(super(), '_get_info') else {}
        
        # Add balanced-specific metrics
        info.update({
            'scenario': 'small_balanced',
            'scheduled_jobs': len(self.scheduled_jobs),
            'total_tasks': self.total_tasks,
            'completion_rate': len(self.scheduled_jobs) / self.total_tasks if self.total_tasks > 0 else 0,
            'on_time_jobs': self.on_time_jobs,
            'late_jobs': self.late_jobs,
            'machine_utilization': self._calculate_utilization()
        })
        
        return info
    
    def _calculate_utilization(self) -> float:
        """Calculate average machine utilization."""
        if self.current_time == 0:
            return 0.0
        
        total_busy_time = 0
        for machine_jobs in self.machine_schedules.values():
            for job in machine_jobs:
                total_busy_time += job['end'] - job['start']
        
        total_capacity = self.current_time * len(self.machine_ids)
        return total_busy_time / total_capacity if total_capacity > 0 else 0.0