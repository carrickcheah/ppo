"""
Environment that encourages scheduling ALL jobs by proper termination
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from phase3.environments.curriculum_env_truly_fixed import CurriculumEnvironmentTrulyFixed


class ScheduleAllEnvironment(CurriculumEnvironmentTrulyFixed):
    """
    Modified environment that terminates when all TASKS are scheduled.
    
    Key changes:
    1. Episode ends when all individual tasks are scheduled (not families)
    2. Better rewards for scheduling
    3. Completion bonus when all tasks scheduled
    """
    
    def __init__(self, stage_name='toy_normal', data_dir="/Users/carrickcheah/Project/ppo/app_2/data", verbose=True):
        super().__init__(stage_name, data_dir=data_dir, verbose=verbose)
        
        # Better reward config
        self.reward_config = {
            'no_action_penalty': -2.0,      # Small penalty
            'invalid_action_penalty': -10.0,
            'valid_schedule_reward': 30.0,   # High base reward
            'sequence_completion': 20.0,
            'family_completion': 50.0,
            'on_time_bonus': 20.0,
            'late_penalty_per_day': -1.0    # Small late penalty
        }
        
        if verbose:
            print(f"\nScheduleAllEnvironment: Episode ends when all {self.total_tasks} tasks scheduled!")
    
    def step(self, action):
        """Override step to handle proper termination."""
        obs, reward, done, truncated, info = super().step(action)
        
        # Check if all TASKS (not families) are scheduled
        if len(self.scheduled_jobs) >= self.total_tasks:
            done = True
            # Big completion bonus!
            completion_bonus = 200.0
            reward += completion_bonus
            info['completion_bonus'] = completion_bonus
            info['all_tasks_scheduled'] = True
            if self.verbose:
                print(f"\n✓ ALL {self.total_tasks} TASKS SCHEDULED! Bonus: {completion_bonus}")
        
        # Also terminate if no more valid scheduling actions
        # (only no-action remains)
        valid_actions = self._get_valid_actions()
        no_action = (len(self.family_ids), len(self.machine_ids))
        if len(valid_actions) == 1 and valid_actions[0] == no_action:
            done = True
            info['no_valid_actions'] = True
            if self.verbose:
                print(f"\nNo more valid actions. Scheduled {len(self.scheduled_jobs)}/{self.total_tasks}")
        
        return obs, reward, done, truncated, info