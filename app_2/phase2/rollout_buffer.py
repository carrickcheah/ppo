"""
Rollout Buffer for PPO Training

Stores experiences during environment interaction and computes
advantages using Generalized Advantage Estimation (GAE).
"""

import numpy as np
import torch
from typing import Optional, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class RolloutBuffer:
    """
    Buffer for storing rollout experiences and computing advantages.
    
    Handles:
    - Experience storage during rollouts
    - GAE advantage computation
    - Return calculation
    - Batch generation for training
    """
    
    def __init__(self,
                 buffer_size: int,
                 observation_shape: Tuple[int, ...],
                 action_shape: Tuple[int, ...],
                 n_envs: int = 1,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 device: str = 'cpu'):
        """
        Initialize rollout buffer.
        
        Args:
            buffer_size: Number of steps to store per environment
            observation_shape: Shape of observations
            action_shape: Shape of actions (usually scalar for our case)
            n_envs: Number of parallel environments
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            device: Device for tensors
        """
        self.buffer_size = buffer_size
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.n_envs = n_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        
        self.pos = 0
        self.full = False
        
        # Initialize buffers
        self.observations = np.zeros((buffer_size, n_envs, *observation_shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size, n_envs, *action_shape), dtype=np.int64)
        self.rewards = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.values = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size, n_envs), dtype=np.float32)
        
        # Optional: store action masks
        self.action_masks = None
        self.store_action_masks = False
        
        # For storing job/machine counts per episode
        self.episode_info = []
        
    def reset(self):
        """Reset the buffer."""
        self.pos = 0
        self.full = False
        self.episode_info.clear()
        
    def add(self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            value: np.ndarray,
            log_prob: np.ndarray,
            action_mask: Optional[np.ndarray] = None,
            infos: Optional[List[Dict]] = None):
        """
        Add a transition to the buffer.
        
        Args:
            obs: Observations (n_envs, *obs_shape)
            action: Actions taken (n_envs,)
            reward: Rewards received (n_envs,)
            done: Done flags (n_envs,)
            value: Value estimates (n_envs,)
            log_prob: Log probabilities of actions (n_envs,)
            action_mask: Optional action masks (n_envs, n_actions)
            infos: Optional info dictionaries from environments
        """
        if self.pos == 0 and self.action_masks is None and action_mask is not None:
            # Initialize action mask buffer on first add
            self.store_action_masks = True
            mask_shape = action_mask.shape[1:]  # Remove env dimension
            self.action_masks = np.zeros((self.buffer_size, self.n_envs, *mask_shape), dtype=bool)
            
        # Store data
        self.observations[self.pos] = obs.copy()
        self.actions[self.pos] = action.copy()
        self.rewards[self.pos] = reward.copy()
        self.dones[self.pos] = done.copy()
        self.values[self.pos] = value.copy()
        self.log_probs[self.pos] = log_prob.copy()
        
        if self.store_action_masks and action_mask is not None:
            self.action_masks[self.pos] = action_mask.copy()
            
        # Store episode info if provided
        if infos is not None:
            for info in infos:
                if 'n_jobs' in info and 'n_machines' in info:
                    self.episode_info.append({
                        'n_jobs': info['n_jobs'],
                        'n_machines': info['n_machines'],
                        'step': self.pos
                    })
                    
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            
    def compute_returns_and_advantages(self, last_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute returns and advantages using GAE.
        
        Args:
            last_values: Value estimates for the last states (n_envs,)
            
        Returns:
            advantages: Computed advantages
            returns: Computed returns
        """
        # Initialize advantages
        advantages = np.zeros_like(self.rewards)
        last_gae_lam = 0
        
        # Work backwards through buffer
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step]
                next_values = self.values[step + 1]
                
            # TD error
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            
            # GAE
            advantages[step] = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            last_gae_lam = advantages[step]
            
        # Compute returns
        returns = advantages + self.values
        
        return advantages, returns
    
    def get(self, batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Get all data from buffer for training.
        
        Args:
            batch_size: If provided, yield batches of this size
            
        Returns:
            Dictionary containing all buffer data as tensors
        """
        assert self.full or self.pos > 0, "Buffer is empty"
        
        # Determine actual buffer size
        buffer_len = self.buffer_size if self.full else self.pos
        
        # Flatten across environments and time
        n_samples = buffer_len * self.n_envs
        
        # Get data slices
        obs = self.observations[:buffer_len]
        actions = self.actions[:buffer_len]
        log_probs = self.log_probs[:buffer_len]
        advantages = self.advantages[:buffer_len]
        returns = self.returns[:buffer_len]
        values = self.values[:buffer_len]
        
        # Reshape to (n_samples, ...)
        obs = obs.reshape(n_samples, *self.observation_shape)
        actions = actions.reshape(n_samples, *self.action_shape)
        log_probs = log_probs.reshape(n_samples)
        advantages = advantages.reshape(n_samples)
        returns = returns.reshape(n_samples)
        values = values.reshape(n_samples)
        
        # Convert to tensors
        data = {
            'obs': torch.from_numpy(obs).to(self.device),
            'actions': torch.from_numpy(actions).to(self.device),
            'log_probs': torch.from_numpy(log_probs).to(self.device),
            'advantages': torch.from_numpy(advantages).to(self.device),
            'returns': torch.from_numpy(returns).to(self.device),
            'values': torch.from_numpy(values).to(self.device)
        }
        
        # Add action masks if stored
        if self.store_action_masks:
            action_masks = self.action_masks[:buffer_len]
            mask_shape = action_masks.shape[2:]  # Get mask dimensions
            action_masks = action_masks.reshape(n_samples, *mask_shape)
            data['action_masks'] = torch.from_numpy(action_masks).to(self.device)
            
        return data
    
    def get_batches(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """
        Get data in batches for mini-batch training.
        
        Args:
            batch_size: Size of each batch
            
        Yields:
            Batches of data
        """
        data = self.get()
        n_samples = len(data['obs'])
        
        # Generate random indices
        indices = np.random.permutation(n_samples)
        
        batches = []
        for start_idx in range(0, n_samples, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            
            batch = {}
            for key, tensor in data.items():
                batch[key] = tensor[batch_indices]
                
            batches.append(batch)
            
        return batches
    
    def add_computed_values(self, advantages: np.ndarray, returns: np.ndarray):
        """
        Add computed advantages and returns to buffer.
        
        Args:
            advantages: Computed advantages
            returns: Computed returns
        """
        self.advantages = advantages
        self.returns = returns
        
    def get_episode_statistics(self) -> Dict[str, float]:
        """
        Get statistics about episodes in the buffer.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            'mean_episode_reward': np.mean(self.rewards),
            'std_episode_reward': np.std(self.rewards),
            'mean_value': np.mean(self.values),
            'n_episodes': np.sum(self.dones)
        }
        
        if self.episode_info:
            n_jobs = [info['n_jobs'] for info in self.episode_info]
            n_machines = [info['n_machines'] for info in self.episode_info]
            stats['mean_n_jobs'] = np.mean(n_jobs)
            stats['mean_n_machines'] = np.mean(n_machines)
            
        return stats