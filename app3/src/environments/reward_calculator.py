"""
Reward calculator for scheduling environment.
Calculates rewards based on on-time delivery, utilization, and constraint violations.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class RewardCalculator:
    """Calculates rewards for scheduling actions."""
    
    def __init__(
        self,
        on_time_reward: float = 100.0,
        early_bonus_per_day: float = 50.0,
        late_penalty_per_day: float = -100.0,
        sequence_violation_penalty: float = -500.0,
        utilization_bonus: float = 10.0,
        action_taken_bonus: float = 5.0,
        idle_penalty: float = -1.0
    ):
        """
        Initialize reward calculator with configurable weights.
        
        Args:
            on_time_reward: Reward for completing task on time
            early_bonus_per_day: Bonus per day early
            late_penalty_per_day: Penalty per day late
            sequence_violation_penalty: Penalty for sequence violations
            utilization_bonus: Bonus for machine utilization
            action_taken_bonus: Small bonus for taking any valid action
            idle_penalty: Small penalty for idle time
        """
        self.on_time_reward = on_time_reward
        self.early_bonus_per_day = early_bonus_per_day
        self.late_penalty_per_day = late_penalty_per_day
        self.sequence_violation_penalty = sequence_violation_penalty
        self.utilization_bonus = utilization_bonus
        self.action_taken_bonus = action_taken_bonus
        self.idle_penalty = idle_penalty
        
        # Track cumulative metrics
        self.total_reward = 0
        self.tasks_completed = 0
        self.on_time_tasks = 0
        self.late_tasks = 0
        self.early_tasks = 0
        
    def calculate_step_reward(
        self,
        task_scheduled: Optional[Dict],
        current_time: float,
        machine_utilization: float,
        action_valid: bool,
        loader
    ) -> float:
        """
        Calculate reward for a single step.
        
        Args:
            task_scheduled: Dict with task info if scheduled, None if idle
            current_time: Current simulation time
            machine_utilization: Current machine utilization rate
            action_valid: Whether action was valid
            loader: SnapshotLoader for accessing task/family data
            
        Returns:
            Step reward
        """
        reward = 0.0
        
        if not action_valid:
            # Invalid action attempted
            reward += self.idle_penalty
            return reward
            
        if task_scheduled is None:
            # No task scheduled (idle)
            reward += self.idle_penalty
            return reward
            
        # Valid action taken - small bonus to encourage action
        reward += self.action_taken_bonus
        
        # Get task and family info
        task_idx = task_scheduled['task_idx']
        task = loader.tasks[task_idx]
        family = loader.families[task.family_id]
        
        # Calculate deadline reward/penalty
        task_end_time = task_scheduled['end_time']
        deadline_reward = self._calculate_deadline_reward(
            task_end_time, family.lcd_days_remaining, current_time
        )
        reward += deadline_reward
        
        # Utilization bonus
        reward += self.utilization_bonus * machine_utilization
        
        # Update metrics
        self.tasks_completed += 1
        if deadline_reward > 0:
            if deadline_reward > self.on_time_reward:
                self.early_tasks += 1
            else:
                self.on_time_tasks += 1
        else:
            self.late_tasks += 1
            
        self.total_reward += reward
        
        return reward
        
    def _calculate_deadline_reward(
        self,
        task_end_time: float,
        lcd_days_remaining: int,
        current_time: float
    ) -> float:
        """
        Calculate reward based on task completion vs deadline.
        
        Args:
            task_end_time: When task will complete
            lcd_days_remaining: Days until LCD from data
            current_time: Current simulation time
            
        Returns:
            Deadline-based reward
        """
        # Convert LCD days to hours
        lcd_hours = lcd_days_remaining * 24
        deadline_time = current_time + lcd_hours
        
        # Calculate days early/late
        time_diff = deadline_time - task_end_time
        days_diff = time_diff / 24.0
        
        if days_diff >= 0:
            # On time or early
            if days_diff > 0:
                # Early completion bonus
                return self.on_time_reward + (self.early_bonus_per_day * days_diff)
            else:
                # Exactly on time
                return self.on_time_reward
        else:
            # Late completion penalty
            return self.late_penalty_per_day * abs(days_diff)
            
    def calculate_final_reward(
        self,
        all_tasks_scheduled: bool,
        total_makespan: float,
        avg_utilization: float,
        constraint_violations: List[str],
        loader
    ) -> float:
        """
        Calculate final reward at episode end.
        
        Args:
            all_tasks_scheduled: Whether all tasks were scheduled
            total_makespan: Total time to complete all tasks
            avg_utilization: Average machine utilization
            constraint_violations: List of constraint violations
            loader: SnapshotLoader for data access
            
        Returns:
            Final episode reward
        """
        reward = 0.0
        
        # Penalty for constraint violations
        reward += len(constraint_violations) * self.sequence_violation_penalty
        
        # Bonus for completing all tasks
        if all_tasks_scheduled:
            completion_rate = self.tasks_completed / len(loader.tasks)
            reward += self.on_time_reward * completion_rate
            
        # Efficiency bonus based on makespan
        if total_makespan > 0:
            # Theoretical minimum makespan (sum of all processing times / n_machines)
            total_processing = sum(t.processing_time for t in loader.tasks)
            theoretical_min = total_processing / len(loader.machines)
            efficiency = theoretical_min / total_makespan
            reward += self.utilization_bonus * efficiency * 100
            
        # Utilization bonus
        reward += self.utilization_bonus * avg_utilization * 100
        
        return reward
        
    def get_metrics(self) -> Dict:
        """Get reward metrics for logging."""
        return {
            'total_reward': self.total_reward,
            'tasks_completed': self.tasks_completed,
            'on_time_tasks': self.on_time_tasks,
            'early_tasks': self.early_tasks,
            'late_tasks': self.late_tasks,
            'on_time_rate': self.on_time_tasks / max(self.tasks_completed, 1),
            'early_rate': self.early_tasks / max(self.tasks_completed, 1),
            'late_rate': self.late_tasks / max(self.tasks_completed, 1)
        }
        
    def reset(self):
        """Reset metrics for new episode."""
        self.total_reward = 0
        self.tasks_completed = 0
        self.on_time_tasks = 0
        self.late_tasks = 0
        self.early_tasks = 0
        
    def calculate_shaped_reward(
        self,
        prev_state: Dict,
        curr_state: Dict,
        action: int,
        loader
    ) -> float:
        """
        Calculate shaped reward to encourage good intermediate behavior.
        
        Args:
            prev_state: Previous environment state
            curr_state: Current environment state
            action: Action taken
            loader: SnapshotLoader
            
        Returns:
            Shaped reward component
        """
        shaped = 0.0
        
        # Reward for reducing number of unscheduled urgent tasks
        prev_urgent = prev_state.get('urgent_unscheduled', 0)
        curr_urgent = curr_state.get('urgent_unscheduled', 0)
        if curr_urgent < prev_urgent:
            shaped += 10.0  # Bonus for scheduling urgent task
            
        # Reward for maintaining sequence progress
        prev_sequences = prev_state.get('sequences_in_progress', 0)
        curr_sequences = curr_state.get('sequences_in_progress', 0)
        if curr_sequences > prev_sequences:
            shaped += 5.0  # Bonus for starting new sequence
            
        # Penalty for leaving machines idle when tasks available
        if action == -1:  # No-op action
            available_tasks = curr_state.get('available_tasks', 0)
            if available_tasks > 0:
                shaped -= 2.0  # Penalty for unnecessary idle
                
        return shaped


class AdaptiveRewardCalculator(RewardCalculator):
    """Adaptive reward calculator that adjusts weights based on performance."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.episode_history = []
        self.adaptation_rate = 0.1
        
    def adapt_weights(self):
        """Adapt reward weights based on recent performance."""
        if len(self.episode_history) < 10:
            return  # Need enough history
            
        recent = self.episode_history[-10:]
        avg_on_time = np.mean([ep['on_time_rate'] for ep in recent])
        avg_late = np.mean([ep['late_rate'] for ep in recent])
        
        # If too many late tasks, increase late penalty
        if avg_late > 0.3:
            self.late_penalty_per_day *= (1 + self.adaptation_rate)
            logger.info(f"Increased late penalty to {self.late_penalty_per_day}")
            
        # If doing well, can reduce penalties slightly
        if avg_on_time > 0.8:
            self.late_penalty_per_day *= (1 - self.adaptation_rate * 0.5)
            logger.info(f"Reduced late penalty to {self.late_penalty_per_day}")
            
    def end_episode(self):
        """Record episode metrics and adapt."""
        metrics = self.get_metrics()
        self.episode_history.append(metrics)
        self.adapt_weights()
        self.reset()


if __name__ == "__main__":
    # Test reward calculator
    calc = RewardCalculator()
    
    # Test deadline reward calculation
    reward = calc._calculate_deadline_reward(
        task_end_time=100,  # Task ends at hour 100
        lcd_days_remaining=5,  # LCD is 5 days away
        current_time=0  # Current time is 0
    )
    print(f"Early completion (20 hours early): {reward}")
    
    reward = calc._calculate_deadline_reward(
        task_end_time=120,  # Task ends at hour 120
        lcd_days_remaining=5,  # LCD is 5 days away
        current_time=0  # Current time is 0
    )
    print(f"On-time completion: {reward}")
    
    reward = calc._calculate_deadline_reward(
        task_end_time=144,  # Task ends at hour 144
        lcd_days_remaining=5,  # LCD is 5 days away
        current_time=0  # Current time is 0
    )
    print(f"Late completion (24 hours late): {reward}")
    
    # Test metrics
    print(f"\nInitial metrics: {calc.get_metrics()}")