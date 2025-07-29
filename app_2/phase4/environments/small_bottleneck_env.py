"""
Small Bottleneck Environment  
40 jobs, 10 machines - Tests resource allocation with constrained capacity
"""

import os
from typing import Dict, Optional
from .base_strategy_env import BaseStrategyEnvironment


class SmallBottleneckEnvironment(BaseStrategyEnvironment):
    """
    Bottleneck scenario with high job-to-machine ratio (4:1).
    Tests efficient resource allocation and machine utilization.
    """
    
    def __init__(self, verbose: bool = False):
        """Initialize Small Bottleneck environment."""
        
        # Path to data file
        data_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data',
            'small_bottleneck_data.json'
        )
        
        # Bottleneck-optimized reward configuration
        reward_config = {
            'no_action_penalty': -0.5,      # Small penalty - waiting might be necessary
            'invalid_action_penalty': -10.0,
            'valid_schedule_reward': 25.0,   # Higher reward for any progress
            'sequence_completion': 35.0,     # Higher sequence bonus
            'family_completion': 70.0,       
            'on_time_bonus': 15.0,          # Lower - focus on throughput
            'late_penalty_per_day': -1.5,   # Lower - some lateness expected
            'utilization_bonus': 5.0        # New: bonus for high utilization
        }
        
        super().__init__(
            scenario_name='small_bottleneck',
            data_file=data_file,
            reward_config=reward_config,
            max_steps=300,  # More steps needed due to bottleneck
            verbose=verbose
        )
        
        # Bottleneck-specific tracking
        self.max_concurrent_jobs = 0
        self.total_wait_time = 0
        self.machine_idle_time = {m: 0 for m in self.machine_ids}
    
    def step(self, action):
        """Override step to track bottleneck metrics."""
        prev_time = self.current_time
        
        obs, reward, done, truncated, info = super().step(action)
        
        # Track machine utilization
        if info.get('action_valid') and info.get('action_type') == 'schedule':
            # Count concurrent jobs
            concurrent = sum(
                1 for jobs in self.machine_schedules.values() 
                if any(job['start'] <= self.current_time <= job['end'] for job in jobs)
            )
            self.max_concurrent_jobs = max(self.max_concurrent_jobs, concurrent)
            
            # Utilization bonus
            utilization = self._calculate_utilization()
            if utilization > 0.8:  # High utilization
                utilization_bonus = self.reward_config.get('utilization_bonus', 5.0)
                reward += utilization_bonus
                info['utilization_bonus'] = utilization_bonus
        
        # Track idle time
        time_step = self.current_time - prev_time
        if time_step > 0:
            for machine_id in self.machine_ids:
                if not self._is_machine_busy(machine_id, prev_time, self.current_time):
                    self.machine_idle_time[machine_id] += time_step
        
        return obs, reward, done, truncated, info
    
    def _is_machine_busy(self, machine_id: int, start_time: float, end_time: float) -> bool:
        """Check if machine is busy during time interval."""
        for job in self.machine_schedules.get(machine_id, []):
            if job['start'] < end_time and job['end'] > start_time:
                return True
        return False
    
    def _calculate_utilization(self) -> float:
        """Calculate current machine utilization."""
        if self.current_time == 0:
            return 0.0
        
        busy_machines = 0
        for machine_id in self.machine_ids:
            if self._is_machine_busy(machine_id, self.current_time - 0.1, self.current_time):
                busy_machines += 1
        
        return busy_machines / len(self.machine_ids)
    
    def _get_info(self) -> Dict:
        """Get environment info with bottleneck metrics."""
        info = super()._get_info() if hasattr(super(), '_get_info') else {}
        
        # Calculate throughput
        throughput = len(self.scheduled_jobs) / self.current_time if self.current_time > 0 else 0
        
        # Average machine utilization
        total_idle = sum(self.machine_idle_time.values())
        total_time = self.current_time * len(self.machine_ids)
        avg_utilization = 1 - (total_idle / total_time) if total_time > 0 else 0
        
        info.update({
            'scenario': 'small_bottleneck',
            'scheduled_jobs': len(self.scheduled_jobs),
            'total_tasks': self.total_tasks,
            'completion_rate': len(self.scheduled_jobs) / self.total_tasks if self.total_tasks > 0 else 0,
            'job_machine_ratio': len(self.families) / len(self.machine_ids),
            'max_concurrent_jobs': self.max_concurrent_jobs,
            'avg_utilization': avg_utilization,
            'throughput': throughput,
            'current_utilization': self._calculate_utilization()
        })
        
        return info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment and bottleneck tracking."""
        obs, info = super().reset(seed, options)
        
        # Reset bottleneck-specific tracking
        self.max_concurrent_jobs = 0
        self.total_wait_time = 0
        self.machine_idle_time = {m: 0 for m in self.machine_ids}
        
        return obs, info