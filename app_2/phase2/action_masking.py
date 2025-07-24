"""
Action Masking for Invalid Actions

Handles the conversion of environment action masks to tensor format
and ensures proper gradient flow through valid actions only.
"""

import torch
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ActionMasking:
    """
    Handles action masking for the scheduling environment.
    
    Key responsibilities:
    - Convert environment masks to tensor format
    - Apply masks to action logits
    - Ensure numerical stability
    - Handle multi-machine job constraints
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize action masking.
        
        Args:
            device: Device to place tensors on
        """
        self.device = device
        self.eps = 1e-8
        self.mask_value = -1e9  # Large negative value for masked actions
        
    def env_mask_to_tensor(self, 
                          env_mask: np.ndarray,
                          batch_size: int = 1) -> torch.Tensor:
        """
        Convert environment action mask to tensor.
        
        Args:
            env_mask: Boolean mask from environment.get_action_mask()
                     Shape: (n_jobs * n_machines,) for single env
                     or (batch_size, n_jobs * n_machines) for vectorized
            batch_size: Number of parallel environments
            
        Returns:
            Tensor mask (batch_size, n_actions) with True for valid actions
        """
        if env_mask.ndim == 1:
            # Single environment
            mask = torch.from_numpy(env_mask).to(self.device)
            mask = mask.unsqueeze(0).repeat(batch_size, 1)
        else:
            # Vectorized environment
            mask = torch.from_numpy(env_mask).to(self.device)
            
        return mask.bool()
    
    def apply_mask_to_logits(self,
                            logits: torch.Tensor,
                            mask: torch.Tensor) -> torch.Tensor:
        """
        Apply mask to action logits before softmax.
        
        Args:
            logits: Action logits (batch_size, n_actions)
            mask: Boolean mask (batch_size, n_actions), True for valid
            
        Returns:
            Masked logits with -inf for invalid actions
        """
        # Apply large negative value to invalid actions
        masked_logits = logits.masked_fill(~mask, self.mask_value)
        
        # Check if any batch has no valid actions
        valid_actions_per_batch = mask.sum(dim=-1)
        if (valid_actions_per_batch == 0).any():
            logger.warning(f"Some batches have no valid actions! Valid counts: {valid_actions_per_batch}")
            # In this case, unmask all actions for those batches to avoid NaN
            no_valid_mask = valid_actions_per_batch == 0
            if no_valid_mask.any():
                masked_logits[no_valid_mask] = logits[no_valid_mask]
        
        return masked_logits
    
    def get_valid_action_indices(self, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get indices of valid actions for each batch.
        
        Args:
            mask: Boolean mask (batch_size, n_actions)
            
        Returns:
            valid_indices: List of valid action indices per batch
            n_valid: Number of valid actions per batch
        """
        batch_size = mask.shape[0]
        n_valid = mask.sum(dim=-1)
        
        # Get indices of valid actions
        valid_indices = []
        for i in range(batch_size):
            batch_valid = torch.where(mask[i])[0]
            valid_indices.append(batch_valid)
            
        return valid_indices, n_valid
    
    def sample_masked_action(self,
                           action_probs: torch.Tensor,
                           mask: torch.Tensor,
                           deterministic: bool = False) -> torch.Tensor:
        """
        Sample actions from masked probability distribution.
        
        Args:
            action_probs: Action probabilities (batch_size, n_actions)
            mask: Boolean mask (batch_size, n_actions)
            deterministic: If True, select argmax; if False, sample
            
        Returns:
            Selected actions (batch_size,)
        """
        batch_size = action_probs.shape[0]
        
        # Ensure only valid actions have non-zero probability
        masked_probs = action_probs * mask.float()
        
        # Renormalize
        masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        
        if deterministic:
            # Select action with highest probability
            actions = masked_probs.argmax(dim=-1)
        else:
            # Sample from distribution
            actions = torch.multinomial(masked_probs, num_samples=1).squeeze(-1)
            
        return actions
    
    def decode_action(self, 
                     action_idx: int,
                     n_jobs: int,
                     n_machines: int) -> Tuple[int, int]:
        """
        Decode flattened action index to (job, machine) pair.
        
        Args:
            action_idx: Flattened action index
            n_jobs: Number of jobs
            n_machines: Number of machines
            
        Returns:
            (job_idx, machine_idx)
        """
        job_idx = action_idx // n_machines
        machine_idx = action_idx % n_machines
        
        return job_idx, machine_idx
    
    def encode_action(self,
                     job_idx: int,
                     machine_idx: int,
                     n_machines: int) -> int:
        """
        Encode (job, machine) pair to flattened action index.
        
        Args:
            job_idx: Job index
            machine_idx: Machine index
            n_machines: Number of machines
            
        Returns:
            Flattened action index
        """
        return job_idx * n_machines + machine_idx
    
    def create_multimachine_mask(self,
                                jobs: list,
                                machines: list,
                                current_mask: torch.Tensor) -> torch.Tensor:
        """
        Refine mask for multi-machine job constraints.
        
        For jobs requiring multiple machines, ensure that selecting
        any required machine is valid only if ALL required machines
        are available.
        
        Args:
            jobs: List of job dictionaries with 'required_machines'
            machines: List of machine dictionaries
            current_mask: Current action mask
            
        Returns:
            Refined mask considering multi-machine constraints
        """
        # This would be implemented based on the specific
        # multi-machine handling logic in the environment
        # For now, return the current mask unchanged
        return current_mask
    
    def get_action_entropy(self,
                          action_probs: torch.Tensor,
                          mask: torch.Tensor) -> torch.Tensor:
        """
        Calculate entropy of masked action distribution.
        
        Args:
            action_probs: Action probabilities (batch_size, n_actions)
            mask: Boolean mask (batch_size, n_actions)
            
        Returns:
            Entropy per batch (batch_size,)
        """
        # Only consider valid actions for entropy
        masked_probs = action_probs * mask.float()
        
        # Avoid log(0)
        masked_probs = masked_probs.clamp(min=self.eps)
        
        # Calculate entropy: -sum(p * log(p))
        entropy = -(masked_probs * torch.log(masked_probs)).sum(dim=-1)
        
        return entropy