"""
Fixed implementation of gradual break constraints environment.
Properly enforces break constraints at different levels.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from src.environments.scaled_production_env import ScaledProductionEnv


class GradualBreaksEnv(ScaledProductionEnv):
    """Environment with properly enforced gradual break constraints."""
    
    def __init__(self, break_level: str = 'none', **kwargs):
        """
        Initialize with specific break level.
        
        Args:
            break_level: 'none', 'tea', 'tea_lunch', or 'full'
        """
        self.break_level = break_level
        
        # Set use_break_constraints based on level
        if break_level == 'none':
            kwargs['use_break_constraints'] = False
        else:
            kwargs['use_break_constraints'] = True
            
        super().__init__(**kwargs)
        
        # Override break configuration after init
        if self.break_level != 'none':
            self._configure_custom_breaks()
    
    def _configure_custom_breaks(self):
        """Configure breaks based on level."""
        if self.break_level == 'tea':
            # Only morning and afternoon tea (30 min total per day)
            self.daily_breaks = [
                {'start': 10.0, 'end': 10.25, 'name': 'Morning Tea'},
                {'start': 15.0, 'end': 15.25, 'name': 'Afternoon Tea'}
            ]
            self.use_weekends = False  # No weekends for tea only
            
        elif self.break_level == 'tea_lunch':
            # Tea breaks + lunch (2 hours total per day)
            self.daily_breaks = [
                {'start': 10.0, 'end': 10.25, 'name': 'Morning Tea'},
                {'start': 12.0, 'end': 13.0, 'name': 'Lunch'},
                {'start': 15.0, 'end': 15.25, 'name': 'Afternoon Tea'}
            ]
            self.use_weekends = False  # No weekends yet
            
        elif self.break_level == 'full':
            # All breaks including dinner and weekends
            self.daily_breaks = [
                {'start': 10.0, 'end': 10.25, 'name': 'Morning Tea'},
                {'start': 12.0, 'end': 13.0, 'name': 'Lunch'},
                {'start': 15.0, 'end': 15.25, 'name': 'Afternoon Tea'},
                {'start': 18.0, 'end': 19.0, 'name': 'Dinner'}
            ]
            self.use_weekends = True  # Enable weekends
    
    def _is_break_time(self, time: float) -> bool:
        """Check if a given time falls within a break period."""
        if self.break_level == 'none':
            return False
            
        # Check daily breaks
        hour_of_day = time % 24
        for break_period in self.daily_breaks:
            if break_period['start'] <= hour_of_day < break_period['end']:
                return True
        
        # Check weekends if enabled
        if self.use_weekends:
            day_of_week = int(time / 24) % 7
            # Saturday noon to Monday 6am
            if day_of_week == 5 and hour_of_day >= 12:  # Saturday afternoon
                return True
            elif day_of_week == 6:  # Sunday all day
                return True
            elif day_of_week == 0 and hour_of_day < 6:  # Monday early morning
                return True
                
        return False
    
    def _find_next_available_time(self, start_time: float) -> float:
        """Find the next available time after breaks."""
        current_time = start_time
        
        while self._is_break_time(current_time):
            hour_of_day = current_time % 24
            day_start = int(current_time / 24) * 24
            
            # Skip to end of current break
            moved = False
            for break_period in self.daily_breaks:
                if break_period['start'] <= hour_of_day < break_period['end']:
                    current_time = day_start + break_period['end']
                    moved = True
                    break
            
            # Handle weekend breaks
            if not moved and self.use_weekends:
                day_of_week = int(current_time / 24) % 7
                if day_of_week == 5 and hour_of_day >= 12:
                    # Skip to Monday 6am
                    days_to_monday = 2
                    current_time = int(current_time / 24) * 24 + days_to_monday * 24 + 6
                elif day_of_week == 6:
                    # Skip to Monday 6am
                    current_time = int(current_time / 24) * 24 + 24 + 6
                elif day_of_week == 0 and hour_of_day < 6:
                    # Skip to 6am
                    current_time = day_start + 6
            
            # Safety check to avoid infinite loop
            if current_time > start_time + 72:  # Max 3 days skip
                break
                
        return current_time
    
    def _can_schedule_job(self, machine_idx: int, start_time: float, 
                         duration: float) -> Tuple[bool, float]:
        """
        Check if job can be scheduled and return adjusted start time.
        
        Returns:
            (can_schedule, adjusted_start_time)
        """
        # Find next available start time
        adjusted_start = self._find_next_available_time(start_time)
        
        # Check if job would span breaks
        end_time = adjusted_start + duration
        current = adjusted_start
        
        while current < end_time:
            if self._is_break_time(current):
                # Job would span a break, need to delay
                adjusted_start = self._find_next_available_time(current)
                end_time = adjusted_start + duration
                current = adjusted_start
                
                # Safety check
                if adjusted_start > start_time + 168:  # Max 1 week delay
                    return False, start_time
            else:
                current += 0.25  # Check every 15 minutes
        
        return True, adjusted_start
    
    def step(self, action: int):
        """Override step to enforce break constraints."""
        if not self.use_break_constraints or self.break_level == 'none':
            # No breaks, use parent implementation
            return super().step(action)
        
        # Get the selected job and machine
        if action >= len(self.valid_actions):
            return self._get_obs(), 0, True, False, {'error': 'Invalid action'}
        
        family_idx, task_idx = self.valid_actions[action]
        family = self.families[family_idx]
        task = family['tasks'][task_idx]
        
        # Find best machine (simplified from parent)
        best_machine = None
        best_start_time = float('inf')
        
        for machine_idx, machine in enumerate(self.machines):
            if task['machine_type'] == machine['machine_type_id']:
                # Get machine availability
                available_time = machine.get('available_time', 0)
                
                # Check setup time
                setup_time = 0
                if machine.get('last_family') != family_idx:
                    setup_time = 0.5  # Default setup time
                
                # Total start time including setup
                start_time = available_time + setup_time
                
                # Check if we can schedule with breaks
                can_schedule, adjusted_start = self._can_schedule_job(
                    machine_idx, start_time, task['processing_time']
                )
                
                if can_schedule and adjusted_start < best_start_time:
                    best_machine = machine_idx
                    best_start_time = adjusted_start
        
        if best_machine is None:
            # No valid machine found
            return self._get_obs(), -10, False, False, {'error': 'No valid machine'}
        
        # Schedule the job
        machine = self.machines[best_machine]
        
        # Update machine state
        machine['available_time'] = best_start_time + task['processing_time']
        machine['last_family'] = family_idx
        machine['total_busy_time'] = machine.get('total_busy_time', 0) + task['processing_time']
        
        # Mark task as completed
        task['completed'] = True
        task['assigned_machine'] = best_machine
        task['start_time'] = best_start_time
        task['end_time'] = best_start_time + task['processing_time']
        
        # Update family and task counts
        family['completed_tasks'] += 1
        self.completed_task_count += 1
        
        # Update makespan
        self.current_makespan = max(self.current_makespan, task['end_time'])
        
        # Calculate reward
        time_penalty = -0.1
        completion_bonus = 10 if self.completed_task_count >= self.total_tasks else 0
        
        # Bonus for efficient break scheduling
        break_efficiency_bonus = 0
        if self.break_level != 'none':
            # Reward for scheduling just before breaks
            hour_before_break = (best_start_time + task['processing_time']) % 24
            for break_period in self.daily_breaks:
                if abs(hour_before_break - break_period['start']) < 0.25:
                    break_efficiency_bonus = 0.5  # Finished just before break
                    break
        
        reward = time_penalty + completion_bonus + break_efficiency_bonus
        
        # Check if done
        done = self.completed_task_count >= self.total_tasks
        
        if done:
            self.episode_makespan = self.current_makespan
        
        # Remove completed action
        self.valid_actions.remove((family_idx, task_idx))
        
        # Get info
        info = {
            'makespan': self.current_makespan,
            'completed_tasks': self.completed_task_count,
            'total_tasks': self.total_tasks,
            'break_level': self.break_level
        }
        
        return self._get_obs(), reward, done, False, info
    
    def reset(self, **kwargs):
        """Reset and log break configuration."""
        obs, info = super().reset(**kwargs)
        
        # Add break info
        total_break_hours = 0
        if self.break_level != 'none':
            daily_break_hours = sum(b['end'] - b['start'] for b in self.daily_breaks)
            total_break_hours = daily_break_hours * 5  # 5 work days
            
            if self.use_weekends:
                total_break_hours += 42  # Weekend hours
        
        info['break_hours_per_week'] = total_break_hours
        info['break_level'] = self.break_level
        
        return obs, info