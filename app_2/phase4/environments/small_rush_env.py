"""
Small Rush Environment
50 jobs, 20 machines - Tests prioritization under tight deadlines
"""

import os
from typing import Dict, Optional
from .base_strategy_env import BaseStrategyEnvironment


class SmallRushEnvironment(BaseStrategyEnvironment):
    """
    Rush scenario with many urgent jobs and tight deadlines.
    Tests ability to prioritize and handle time pressure.
    """
    
    def __init__(self, verbose: bool = False):
        """Initialize Small Rush environment."""
        
        # Path to data file
        data_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data',
            'small_rush_data.json'
        )
        
        # Rush-optimized reward configuration
        reward_config = {
            'no_action_penalty': -2.0,      # Higher penalty for wasting time
            'invalid_action_penalty': -15.0, # Higher penalty for mistakes
            'valid_schedule_reward': 20.0,   # Higher reward for progress
            'sequence_completion': 30.0,     # Higher sequence bonus
            'family_completion': 80.0,       # Big bonus for completion
            'on_time_bonus': 40.0,          # Double bonus for on-time
            'late_penalty_per_day': -5.0    # Higher late penalty
        }
        
        super().__init__(
            scenario_name='small_rush',
            data_file=data_file,
            reward_config=reward_config,
            max_steps=250,  # More steps for more jobs
            verbose=verbose
        )
        
        # Rush-specific tracking
        self.critical_jobs = 0
        self.critical_jobs_scheduled = 0
        self._identify_critical_jobs()
    
    def _identify_critical_jobs(self):
        """Identify jobs with very tight deadlines (<3 days)."""
        self.critical_families = set()
        for fid, family in self.families.items():
            if family.get('lcd_days_remaining', 999) < 3:
                self.critical_families.add(fid)
                self.critical_jobs += family.get('total_sequences', 1)
    
    def step(self, action):
        """Override step to track critical job scheduling."""
        obs, reward, done, truncated, info = super().step(action)
        
        # Check if we scheduled a critical job
        if info.get('action_valid') and info.get('scheduled_job'):
            job_family = '_'.join(info['scheduled_job'].split('_')[:-1])
            if job_family in self.critical_families:
                self.critical_jobs_scheduled += 1
                # Bonus for scheduling critical jobs
                reward += 10.0
                info['critical_job_scheduled'] = True
        
        return obs, reward, done, truncated, info
    
    def _get_info(self) -> Dict:
        """Get environment info with rush metrics."""
        info = super()._get_info() if hasattr(super(), '_get_info') else {}
        
        # Add rush-specific metrics
        critical_rate = (
            self.critical_jobs_scheduled / self.critical_jobs 
            if self.critical_jobs > 0 else 1.0
        )
        
        info.update({
            'scenario': 'small_rush',
            'scheduled_jobs': len(self.scheduled_jobs),
            'total_tasks': self.total_tasks,
            'completion_rate': len(self.scheduled_jobs) / self.total_tasks if self.total_tasks > 0 else 0,
            'critical_jobs': self.critical_jobs,
            'critical_scheduled': self.critical_jobs_scheduled,
            'critical_completion_rate': critical_rate,
            'avg_lateness': self._calculate_avg_lateness()
        })
        
        return info
    
    def _calculate_avg_lateness(self) -> float:
        """Calculate average lateness of scheduled jobs."""
        if not self.job_assignments:
            return 0.0
        
        total_lateness = 0
        late_count = 0
        
        for job_key, assignment in self.job_assignments.items():
            family_id = '_'.join(job_key.split('_')[:-1])
            if family_id in self.families:
                family = self.families[family_id]
                deadline = family.get('lcd_days_remaining', 999) * 24
                
                if assignment['end'] > deadline:
                    lateness = (assignment['end'] - deadline) / 24  # Convert to days
                    total_lateness += lateness
                    late_count += 1
        
        return total_lateness / late_count if late_count > 0 else 0.0