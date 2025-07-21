"""
MultiDiscrete Hierarchical Production Environment for Phase 5

This environment wraps the HierarchicalProductionEnv to provide
a MultiDiscrete action space compatible with Stable Baselines3 PPO.

Action space: MultiDiscrete([n_jobs, n_machines])
- action[0] = job index to schedule
- action[1] = machine index to assign to
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
import gymnasium as gym
from gymnasium import spaces
import logging

from .hierarchical_production_env import HierarchicalProductionEnv

logger = logging.getLogger(__name__)


class MultiDiscreteHierarchicalEnv(HierarchicalProductionEnv):
    """
    MultiDiscrete version of hierarchical production environment.
    
    Converts Dict action space to MultiDiscrete for SB3 compatibility
    while maintaining all hierarchical decision-making benefits.
    """
    
    def __init__(
        self,
        n_machines: int = 152,
        n_jobs: int = 500,
        invalid_action_penalty: float = -20.0,
        **kwargs
    ):
        # Store parameters before parent init
        self.invalid_action_penalty = invalid_action_penalty
        
        # Initialize parent with Dict action space
        super().__init__(
            n_machines=n_machines,
            n_jobs=n_jobs,
            **kwargs
        )
        
        # Override action space to MultiDiscrete
        # [n_jobs, n_machines] - job selection then machine selection
        # Use actual numbers from loaded data if available
        actual_n_jobs = len(self.jobs) if hasattr(self, 'jobs') and self.jobs is not None else n_jobs
        actual_n_machines = len(self.machines) if hasattr(self, 'machines') and self.machines is not None else n_machines
        self.action_space = spaces.MultiDiscrete([actual_n_jobs, actual_n_machines])
        
        # Track invalid actions for monitoring
        self.invalid_action_count = 0
        self.total_actions = 0
        
        logger.info(f"Initialized MultiDiscreteHierarchicalEnv")
        logger.info(f"Action space: MultiDiscrete([{n_jobs}, {n_machines}])")
        logger.info(f"Total action combinations: {n_jobs * n_machines} (hierarchical: {n_jobs + n_machines})")
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment and tracking variables."""
        obs, info = super().reset(seed=seed, options=options)
        
        # Reset tracking
        self.invalid_action_count = 0
        self.total_actions = 0
        
        # Add MultiDiscrete-specific info
        info['action_space_type'] = 'MultiDiscrete'
        info['invalid_action_penalty'] = self.invalid_action_penalty
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute MultiDiscrete action.
        
        Args:
            action: np.array([job_idx, machine_idx])
            
        Returns:
            Standard gym step returns
        """
        # Validate action format
        if not isinstance(action, (np.ndarray, list)) or len(action) != 2:
            raise ValueError(f"Action must be array of length 2, got {action}")
        
        # Extract job and machine indices
        job_idx = int(action[0])
        machine_idx = int(action[1])
        
        # Track total actions
        self.total_actions += 1
        
        # Convert to Dict format for parent class
        dict_action = {'job': job_idx, 'machine': machine_idx}
        
        # Check if this is a valid action before passing to parent
        is_valid, invalid_reason = self._check_action_validity(job_idx, machine_idx)
        
        if not is_valid:
            # Invalid action - return penalty without modifying state
            self.invalid_action_count += 1
            
            obs = self._get_hierarchical_state() if self.use_hierarchical_features else self._get_observation()
            reward = self.invalid_action_penalty
            done = False
            truncated = False
            
            info = {
                'invalid_action': True,
                'invalid_reason': invalid_reason,
                'job_idx': job_idx,
                'machine_idx': machine_idx,
                'invalid_action_rate': self.invalid_action_count / self.total_actions,
                'scheduled_count': self.scheduled_count,
                'action_masks': self.get_action_masks()
            }
            
            return obs, reward, done, truncated, info
        
        # Valid action - pass to parent
        obs, reward, done, truncated, info = super().step(dict_action)
        
        # Add MultiDiscrete-specific info
        info['invalid_action_rate'] = self.invalid_action_count / self.total_actions
        info['action_type'] = 'MultiDiscrete'
        
        return obs, reward, done, truncated, info
    
    def _check_action_validity(self, job_idx: int, machine_idx: int) -> Tuple[bool, Optional[str]]:
        """
        Check if the action is valid without modifying environment state.
        
        Returns:
            (is_valid, reason_if_invalid)
        """
        # Check job index bounds
        if job_idx < 0 or job_idx >= self.n_jobs:
            return False, f"Invalid job index: {job_idx} (must be 0-{self.n_jobs-1})"
        
        # Check if job already scheduled
        if hasattr(self, 'job_mask') and not self.job_mask[job_idx]:
            return False, f"Job {job_idx} already scheduled"
        
        # Check machine index bounds
        if machine_idx < 0 or machine_idx >= self.n_machines:
            return False, f"Invalid machine index: {machine_idx} (must be 0-{self.n_machines-1})"
        
        # Check compatibility
        if self.compatibility_matrix is not None:
            if job_idx < self.compatibility_matrix.shape[0]:
                if not self.compatibility_matrix[job_idx, machine_idx]:
                    return False, f"Job {job_idx} incompatible with machine {machine_idx}"
        
        return True, None
    
    def get_valid_action_mask(self) -> np.ndarray:
        """
        Get a flattened mask of all valid actions for MultiDiscrete space.
        
        Returns:
            Boolean array of shape (n_jobs * n_machines,) indicating valid actions
        """
        # Get hierarchical masks
        masks = self.get_action_masks()
        job_mask = masks['job']
        machine_masks = masks['machine']
        
        # Create flattened mask for all combinations
        valid_mask = np.zeros((self.n_jobs, self.n_machines), dtype=bool)
        
        for job_idx in range(len(job_mask)):
            if job_mask[job_idx] and job_idx < len(machine_masks):
                valid_mask[job_idx, :] = machine_masks[job_idx]
        
        return valid_mask.flatten()
    
    def get_valid_actions_info(self) -> dict:
        """
        Get information about valid actions for debugging and monitoring.
        """
        masks = self.get_action_masks()
        job_mask = masks['job']
        machine_masks = masks['machine']
        
        n_valid_jobs = np.sum(job_mask)
        n_valid_combinations = 0
        
        for job_idx in range(len(job_mask)):
            if job_mask[job_idx] and job_idx < len(machine_masks):
                n_valid_combinations += np.sum(machine_masks[job_idx])
        
        return {
            'n_valid_jobs': int(n_valid_jobs),
            'n_total_jobs': self.n_jobs,
            'n_valid_combinations': int(n_valid_combinations),
            'n_total_combinations': self.n_jobs * self.n_machines,
            'valid_ratio': n_valid_combinations / (self.n_jobs * self.n_machines),
            'scheduled_jobs': self.scheduled_count
        }
    
    def render(self) -> Optional[np.ndarray]:
        """Render with MultiDiscrete-specific information."""
        if self.render_mode == "human":
            print(f"\n=== MultiDiscrete Hierarchical Production Environment ===")
            print(f"Time: {self.current_time:.1f}")
            print(f"Scheduled: {self.scheduled_count}/{self.n_jobs} jobs")
            print(f"Action space: MultiDiscrete([{self.n_jobs}, {self.n_machines}])")
            
            if self.total_actions > 0:
                print(f"Invalid action rate: {self.invalid_action_count}/{self.total_actions} "
                      f"({100 * self.invalid_action_count / self.total_actions:.1f}%)")
            
            # Get valid actions info
            valid_info = self.get_valid_actions_info()
            print(f"\nValid actions: {valid_info['n_valid_combinations']}/"
                  f"{valid_info['n_total_combinations']} "
                  f"({100 * valid_info['valid_ratio']:.1f}%)")
            
            # Show machine utilization
            print("\nMachine utilization:")
            utilizations = self._calculate_machine_utilizations()
            for m_idx in range(min(5, self.n_machines)):
                print(f"  Machine {m_idx}: {utilizations[m_idx]:.1%}")
        
        return None