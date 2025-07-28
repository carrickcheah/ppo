"""
Environment with better reward structure that encourages scheduling even late jobs
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from phase3.environments.curriculum_env_truly_fixed import CurriculumEnvironmentTrulyFixed


class BetterRewardEnvironment(CurriculumEnvironmentTrulyFixed):
    """
    Modified environment that encourages scheduling ALL jobs, even if late.
    
    Key changes:
    1. Always give positive reward for scheduling (even if late)
    2. Scale late penalty by how late it is (not binary)
    3. Big bonus for completing all jobs
    """
    
    def __init__(self, stage_name='toy_normal', data_dir="/Users/carrickcheah/Project/ppo/app_2/data", verbose=True):
        super().__init__(stage_name, data_dir=data_dir, verbose=verbose)
        # Override reward config with better values
        self.reward_config = {
            'no_action_penalty': -5.0,      # Increased penalty for doing nothing
            'invalid_action_penalty': -10.0,
            'valid_schedule_reward': 20.0,   # Higher base reward
            'sequence_completion': 10.0,
            'family_completion': 30.0,
            'on_time_bonus': 10.0,
            'late_penalty_per_day': -0.5    # Much smaller late penalty!
        }
    
