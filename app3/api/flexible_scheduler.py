"""
Flexible PPO scheduling service that handles different dataset sizes.
"""

import os
import sys
import time
import numpy as np
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from src.environments.scheduling_env import SchedulingEnv
from api.models import JobTask, MachineTask, ScheduleStatistics


class FlexibleScheduler:
    """Scheduler that adapts to different observation sizes."""
    
    def __init__(self, model_path: str, expected_obs_size: int = 2402):
        """Initialize with a model and expected observation size."""
        self.model = PPO.load(model_path)
        self.expected_obs_size = expected_obs_size
        
    def pad_observation(self, obs: np.ndarray) -> np.ndarray:
        """Pad observation to expected size."""
        current_size = obs.shape[0]
        if current_size == self.expected_obs_size:
            return obs
        elif current_size < self.expected_obs_size:
            # Pad with zeros
            padding = np.zeros(self.expected_obs_size - current_size)
            return np.concatenate([obs, padding])
        else:
            # Truncate if too large
            return obs[:self.expected_obs_size]
    
    def predict(self, obs: np.ndarray, deterministic: bool = True):
        """Predict action with padded observation."""
        padded_obs = self.pad_observation(obs)
        action, states = self.model.predict(padded_obs, deterministic=deterministic)
        return action, states
    
    def schedule_with_padding(
        self,
        env: SchedulingEnv,
        max_steps: int = 10000,
        deterministic: bool = True
    ) -> Dict:
        """Schedule jobs with observation padding."""
        obs, info = env.reset()
        
        done = False
        steps = 0
        scheduled_count = 0
        
        while not done and steps < max_steps:
            # Get action with padded observation
            action, _ = self.predict(obs, deterministic=deterministic)
            
            # Ensure action is within valid range
            if action >= env.action_space.n:
                action = 0  # Default to first action if out of range
            
            # Apply action masking if available
            if 'action_mask' in info:
                mask = info['action_mask']
                if action < len(mask) and not mask[action]:
                    # Find first valid action
                    valid_actions = np.where(mask)[0]
                    if len(valid_actions) > 0:
                        action = valid_actions[0]
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            steps += 1
            
            # Track progress
            if info.get('tasks_scheduled', 0) > scheduled_count:
                scheduled_count = info['tasks_scheduled']
                if steps % 100 == 0:
                    print(f"  Progress: {scheduled_count}/{info['total_tasks']} tasks scheduled")
        
        return env.get_final_schedule(), info