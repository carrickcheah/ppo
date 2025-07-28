"""
Action-Masked Environment Wrapper for Toy Stages
Provides action masks to MaskablePPO for efficient learning
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib.common.wrappers import ActionMasker


class ToyStageActionMasker(ActionMasker):
    """
    Action masker that uses the environment's _get_valid_actions method.
    Inherits from sb3-contrib's ActionMasker for compatibility.
    """
    
    def __init__(self, env):
        # First wrap the env to convert MultiDiscrete to Discrete
        wrapped_env = MultiDiscreteToDiscreteWrapper(env)
        super().__init__(wrapped_env, self._get_action_mask)
        
        # Store reference to original env for accessing _get_valid_actions
        self.original_env = env
        self.n_jobs = env.action_space.nvec[0]  # Including no-action
        self.n_machines = env.action_space.nvec[1]  # Including no-action
    
    def _get_action_mask(self, env) -> np.ndarray:
        """Get action mask from environment's valid actions."""
        # Get valid (job, machine) pairs
        valid_actions = self.original_env._get_valid_actions()
        
        # Create mask for flattened action space
        n_actions = self.n_jobs * self.n_machines
        mask = np.zeros(n_actions, dtype=bool)
        
        for job_idx, machine_idx in valid_actions:
            # Convert to flattened index
            flat_idx = job_idx * self.n_machines + machine_idx
            mask[flat_idx] = True
        
        return mask


class MultiDiscreteToDiscreteWrapper(gym.Wrapper):
    """
    Converts MultiDiscrete action space to Discrete for MaskablePPO.
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        if not isinstance(env.action_space, spaces.MultiDiscrete):
            raise ValueError("Expected MultiDiscrete action space")
        
        # Store original space dimensions
        self.n_jobs = env.action_space.nvec[0]
        self.n_machines = env.action_space.nvec[1]
        
        # Create flattened Discrete action space
        self.action_space = spaces.Discrete(self.n_jobs * self.n_machines)
    
    def step(self, action: int):
        """Convert flattened action to MultiDiscrete and step."""
        # Convert flattened action back to (job, machine)
        job_idx = action // self.n_machines
        machine_idx = action % self.n_machines
        multi_action = np.array([job_idx, machine_idx])
        
        return self.env.step(multi_action)
    
    def reset(self, **kwargs):
        """Reset the environment."""
        return self.env.reset(**kwargs)