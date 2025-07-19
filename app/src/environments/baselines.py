"""
Baseline scheduling policies for comparison.
"""

import numpy as np
from typing import Dict, Any, List


class BaselinePolicy:
    """Base class for baseline policies."""
    
    def get_action(self, observation: np.ndarray, info: Dict[str, Any]) -> int:
        """Get action based on policy."""
        raise NotImplementedError


class RandomPolicy(BaselinePolicy):
    """Random selection from valid actions."""
    
    def get_action(self, observation: np.ndarray, info: Dict[str, Any]) -> int:
        valid_actions = info.get('valid_actions', [])
        if valid_actions:
            return np.random.choice(valid_actions)
        else:
            # Return wait action (last action)
            return info.get('action_space_size', 101) - 1


class FirstFitPolicy(BaselinePolicy):
    """First-fit policy: select first valid job."""
    
    def get_action(self, observation: np.ndarray, info: Dict[str, Any]) -> int:
        valid_actions = info.get('valid_actions', [])
        if valid_actions:
            # Return the first valid action (excluding wait)
            wait_action = info.get('action_space_size', 101) - 1
            non_wait_actions = [a for a in valid_actions if a != wait_action]
            if non_wait_actions:
                return non_wait_actions[0]
            else:
                return wait_action
        else:
            return info.get('action_space_size', 101) - 1


class PriorityPolicy(BaselinePolicy):
    """Priority-based policy: select highest priority valid job."""
    
    def get_action(self, observation: np.ndarray, info: Dict[str, Any]) -> int:
        valid_actions = info.get('valid_actions', [])
        if not valid_actions:
            return info.get('action_space_size', 101) - 1
            
        # Get job priorities from info
        job_priorities = info.get('job_priorities', {})
        wait_action = info.get('action_space_size', 101) - 1
        
        # Find highest priority valid action
        best_action = wait_action
        best_priority = float('inf')
        
        for action in valid_actions:
            if action != wait_action and action in job_priorities:
                priority = job_priorities[action]
                if priority < best_priority:
                    best_priority = priority
                    best_action = action
                    
        return best_action


class EarliestDueDatePolicy(BaselinePolicy):
    """EDD policy: select job with earliest due date."""
    
    def get_action(self, observation: np.ndarray, info: Dict[str, Any]) -> int:
        valid_actions = info.get('valid_actions', [])
        if not valid_actions:
            return info.get('action_space_size', 101) - 1
            
        # Get job due dates from info
        job_due_dates = info.get('job_due_dates', {})
        wait_action = info.get('action_space_size', 101) - 1
        
        # Find earliest due date
        best_action = wait_action
        earliest_date = float('inf')
        
        for action in valid_actions:
            if action != wait_action and action in job_due_dates:
                due_date = job_due_dates[action]
                if due_date < earliest_date:
                    earliest_date = due_date
                    best_action = action
                    
        return best_action


class ShortestProcessingTimePolicy(BaselinePolicy):
    """SPT policy: select job with shortest processing time."""
    
    def get_action(self, observation: np.ndarray, info: Dict[str, Any]) -> int:
        valid_actions = info.get('valid_actions', [])
        if not valid_actions:
            return info.get('action_space_size', 101) - 1
            
        # Get job processing times from info
        job_processing_times = info.get('job_processing_times', {})
        wait_action = info.get('action_space_size', 101) - 1
        
        # Find shortest processing time
        best_action = wait_action
        shortest_time = float('inf')
        
        for action in valid_actions:
            if action != wait_action and action in job_processing_times:
                proc_time = job_processing_times[action]
                if proc_time < shortest_time:
                    shortest_time = proc_time
                    best_action = action
                    
        return best_action