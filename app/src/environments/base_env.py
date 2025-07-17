"""Base environment class for scheduling problems."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional
import gymnasium as gym
import numpy as np


class BaseSchedulingEnv(gym.Env, ABC):
    """Abstract base class for all scheduling environments.
    
    This class defines the common interface and shared functionality
    for scheduling environments of varying complexity.
    """
    
    def __init__(self, n_machines: int, n_jobs: int, seed: Optional[int] = None):
        """Initialize base scheduling environment.
        
        Args:
            n_machines: Number of machines available
            n_jobs: Number of jobs to schedule
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.n_machines = n_machines
        self.n_jobs = n_jobs
        self.np_random = None
        
        if seed is not None:
            self.seed(seed)
            
        # Track metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    @abstractmethod
    def _get_observation(self) -> np.ndarray:
        """Get current observation of the environment state.
        
        Returns:
            Observation array normalized to [0, 1]
        """
        pass
    
    @abstractmethod
    def _calculate_reward(self, action: int, valid_action: bool) -> float:
        """Calculate reward for the given action.
        
        Args:
            action: The action taken
            valid_action: Whether the action was valid
            
        Returns:
            Reward value
        """
        pass
    
    @abstractmethod
    def _is_done(self) -> bool:
        """Check if episode is complete.
        
        Returns:
            True if episode should end
        """
        pass
    
    def seed(self, seed: int = None) -> list:
        """Set random seed for reproducibility."""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def get_action_mask(self) -> np.ndarray:
        """Get mask of valid actions in current state.
        
        Returns:
            Boolean array where True indicates valid action
        """
        # Default implementation - override in subclasses
        return np.ones(self.action_space.n, dtype=bool)
    
    def get_info(self) -> Dict[str, Any]:
        """Get additional information about current state.
        
        Returns:
            Dictionary with debugging/analysis information
        """
        return {
            'episode_reward': self.current_episode_reward,
            'episode_length': self.current_episode_length,
            'action_mask': self.get_action_mask()
        }
    
    def normalize_value(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize a value to [0, 1] range.
        
        Args:
            value: Value to normalize
            min_val: Minimum possible value
            max_val: Maximum possible value
            
        Returns:
            Normalized value in [0, 1]
        """
        if max_val == min_val:
            return 0.0
        return (value - min_val) / (max_val - min_val)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action in environment.
        
        Args:
            action: Action to execute
            
        Returns:
            observation: New environment state
            reward: Reward for this step
            terminated: Whether episode ended (success/failure)
            truncated: Whether episode was cut off (time limit)
            info: Additional information
        """
        # Update episode tracking
        self.current_episode_length += 1
        
        # Execute action (implemented by subclass)
        obs, reward, terminated, truncated, info = self._step_impl(action)
        
        # Track reward
        self.current_episode_reward += reward
        
        # Reset episode tracking if done
        if terminated or truncated:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0
            
        return obs, reward, terminated, truncated, info
    
    @abstractmethod
    def _step_impl(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Implementation of step logic by subclass."""
        pass
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional reset options
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Reset episode tracking
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
        # Reset implementation (by subclass)
        obs = self._reset_impl(options)
        info = self.get_info()
        
        return obs, info
    
    @abstractmethod
    def _reset_impl(self, options: Optional[Dict] = None) -> np.ndarray:
        """Implementation of reset logic by subclass."""
        pass
    
    def render(self):
        """Render the environment (optional)."""
        pass
    
    def close(self):
        """Clean up resources."""
        pass