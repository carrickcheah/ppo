"""
Reward Function for Scheduling Game

This module defines the reward signals that guide the AI's learning.
Unlike hard rules, these are preferences that the AI discovers through experience.

The AI will learn:
- To prioritize urgent jobs (without being told what "urgent" means)
- To value important jobs (discovering this from is_important flag)
- To balance machine loads (emergent behavior for efficiency)
- To minimize makespan (discovering global optimization)
"""

from typing import Dict, List, Optional, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


class RewardFunction:
    """
    Calculates rewards for scheduling actions.
    
    This is where the AI learns what constitutes "good" scheduling.
    We provide signals, not strategies.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize reward function.
        
        Args:
            config: Reward configuration from YAML
        """
        self.config = config
        
        # Reward components from config
        self.completion_reward = config.get('completion_reward', 10.0)
        self.importance_bonus = config.get('importance_bonus', 20.0)
        self.urgency_multiplier = config.get('urgency_multiplier', 50.0)
        self.efficiency_bonus = config.get('efficiency_bonus', 5.0)
        self.balance_bonus = config.get('balance_bonus', 5.0)
        self.makespan_penalty = config.get('makespan_penalty', 0.1)
        self.wait_penalty = config.get('wait_penalty', 0.1)
        self.invalid_action_penalty = config.get('invalid_action_penalty', -20.0)
        
        # Episode completion bonuses
        self.all_jobs_bonus = config.get('all_jobs_bonus', 100.0)
        self.efficiency_multiplier = config.get('efficiency_multiplier', 50.0)
        
    def calculate_step_reward(
        self,
        job: Dict,
        machine: Dict,
        start_time: float,
        end_time: float,
        current_time: float,
        machine_schedules: List[List[Dict]],
        completed_jobs: int,
        total_jobs: int,
        makespan: float
    ) -> float:
        """
        Calculate reward for a single scheduling action.
        
        Args:
            job: Job that was scheduled
            machine: Machine it was scheduled on
            start_time: When job starts
            end_time: When job ends
            current_time: Current simulation time
            machine_schedules: Current schedules for all machines
            completed_jobs: Number of completed jobs
            total_jobs: Total number of jobs
            makespan: Current makespan
            
        Returns:
            Reward value
        """
        reward = 0.0
        
        # 1. Base reward for scheduling any job
        reward += self.completion_reward
        
        # 2. Importance signal (AI learns to prioritize these)
        if job.get('is_important', False):
            reward += self.importance_bonus
            
        # 3. Urgency signal (AI learns deadline pressure)
        if 'lcd_days_remaining' in job:
            days_remaining = job['lcd_days_remaining']
            # Normalize: 0 days = max urgency (1.0), 30+ days = no urgency (0.0)
            urgency_factor = max(0.0, min(1.0, 1.0 - days_remaining / 30.0))
            urgency_reward = urgency_factor * self.urgency_multiplier
            reward += urgency_reward
            
            # Extra signal for critical deadlines
            if days_remaining < 7:
                reward += (7 - days_remaining) * 2.0
                
        # 4. Efficiency signals (AI learns to optimize)
        
        # Wait time penalty (encourages keeping machines busy)
        wait_time = start_time - current_time
        reward -= wait_time * self.wait_penalty
        
        # Machine utilization (AI learns load balancing)
        machine_idx = machine['machine_id']
        machine_load = len(machine_schedules[machine_idx])
        
        # Calculate average load
        total_scheduled = sum(len(schedule) for schedule in machine_schedules)
        n_machines = len(machine_schedules)
        avg_load = total_scheduled / n_machines if n_machines > 0 else 0
        
        # Reward for using less loaded machines
        if machine_load <= avg_load:
            reward += self.balance_bonus
        else:
            # Slight penalty for overloading
            reward -= (machine_load - avg_load) * 0.5
            
        # 5. Progress reward (encourages completion)
        progress = completed_jobs / total_jobs if total_jobs > 0 else 0
        reward += progress * 2.0
        
        # 6. Makespan consideration (global efficiency)
        if makespan > 0:
            # Penalty for extending makespan too much
            makespan_extension = end_time - makespan
            if makespan_extension > 0:
                reward -= makespan_extension * self.makespan_penalty
                
        return reward
    
    def calculate_invalid_action_penalty(self) -> float:
        """Get penalty for invalid actions."""
        return self.invalid_action_penalty
    
    def calculate_episode_bonus(
        self,
        completed_jobs: int,
        total_jobs: int,
        makespan: float,
        total_processing_time: float,
        n_machines: int,
        late_jobs: int,
        important_jobs_on_time: int,
        total_important_jobs: int
    ) -> float:
        """
        Calculate bonus reward at end of episode.
        
        Args:
            completed_jobs: Number of jobs completed
            total_jobs: Total number of jobs
            makespan: Final makespan
            total_processing_time: Sum of all processing times
            n_machines: Number of machines
            late_jobs: Number of jobs completed after deadline
            important_jobs_on_time: Important jobs meeting deadline
            total_important_jobs: Total important jobs
            
        Returns:
            Episode bonus reward
        """
        bonus = 0.0
        
        # 1. Completion bonus (only if all jobs done)
        if completed_jobs == total_jobs:
            bonus += self.all_jobs_bonus
            
            # 2. Efficiency bonus (theoretical vs actual)
            theoretical_min = total_processing_time / n_machines
            if makespan > 0:
                efficiency = theoretical_min / makespan
                bonus += efficiency * self.efficiency_multiplier
                
            # 3. On-time delivery bonus
            on_time_rate = 1.0 - (late_jobs / total_jobs)
            bonus += on_time_rate * 50.0
            
            # 4. Important jobs bonus
            if total_important_jobs > 0:
                important_success_rate = important_jobs_on_time / total_important_jobs
                bonus += important_success_rate * 30.0
                
        else:
            # Partial completion penalty
            completion_rate = completed_jobs / total_jobs
            bonus -= (1.0 - completion_rate) * 50.0
            
        return bonus
    
    def calculate_intermediate_bonus(
        self,
        milestone: str,
        **kwargs
    ) -> float:
        """
        Calculate bonus for reaching certain milestones.
        
        Args:
            milestone: Type of milestone reached
            **kwargs: Additional context
            
        Returns:
            Bonus reward
        """
        bonus = 0.0
        
        if milestone == 'family_completed':
            # Bonus for completing all jobs in a family
            family_size = kwargs.get('family_size', 1)
            is_important = kwargs.get('is_important', False)
            bonus = 5.0 * family_size
            if is_important:
                bonus *= 2.0
                
        elif milestone == 'half_jobs_complete':
            # Bonus for reaching 50% completion
            bonus = 20.0
            
        elif milestone == 'all_urgent_scheduled':
            # Bonus for scheduling all urgent jobs (deadline < 7 days)
            n_urgent = kwargs.get('n_urgent_jobs', 0)
            bonus = 5.0 * n_urgent
            
        return bonus
    
    def shape_reward(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        base_reward: float
    ) -> float:
        """
        Apply reward shaping to guide learning.
        
        This can help the AI learn faster by providing additional signals.
        Should be used carefully to avoid biasing the learning.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            base_reward: Base reward calculated
            
        Returns:
            Shaped reward
        """
        # For now, just return base reward
        # Can add potential-based shaping later if needed
        return base_reward
    
    def get_reward_info(
        self,
        reward: float,
        job: Dict,
        machine: Dict
    ) -> Dict[str, Any]:
        """
        Get detailed breakdown of reward components for logging.
        
        Args:
            reward: Total reward
            job: Job that was scheduled
            machine: Machine used
            
        Returns:
            Dictionary with reward component breakdown
        """
        info = {
            'total_reward': reward,
            'job_id': job.get('job_id', 'unknown'),
            'machine_name': machine.get('machine_name', 'unknown'),
            'is_important': job.get('is_important', False),
            'lcd_days_remaining': job.get('lcd_days_remaining', None),
            'reward_components': {
                'completion': self.completion_reward,
                'importance': self.importance_bonus if job.get('is_important') else 0,
                'urgency': 0,  # Calculated dynamically
                'efficiency': 0,  # Calculated dynamically
            }
        }
        
        # Add urgency component if applicable
        if 'lcd_days_remaining' in job:
            days = job['lcd_days_remaining']
            urgency_factor = max(0.0, min(1.0, 1.0 - days / 30.0))
            info['reward_components']['urgency'] = urgency_factor * self.urgency_multiplier
            
        return info