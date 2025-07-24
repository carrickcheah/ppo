"""
PPO (Proximal Policy Optimization) Implementation for Scheduling

Core PPO algorithm with:
- Actor-Critic architecture using transformer policy
- Advantage estimation with GAE
- Policy clipping for stable updates
- Value function clipping
- Entropy bonus for exploration
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional, Union
import logging

from .state_encoder import StateEncoder
from .transformer_policy import TransformerSchedulingPolicy
from .action_masking import ActionMasking

logger = logging.getLogger(__name__)


class PPOScheduler(nn.Module):
    """
    PPO implementation for scheduling tasks.
    
    Combines state encoding, transformer policy, and PPO training logic
    into a single trainable model.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize PPO scheduler.
        
        Args:
            config: Configuration dictionary containing:
                - model: Model architecture config
                - ppo: PPO algorithm config
                - training: Training config
        """
        super().__init__()
        
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))
        
        # Initialize components
        self.state_encoder = StateEncoder(config['model']).to(self.device)
        self.policy = TransformerSchedulingPolicy(config['model']).to(self.device)
        self.action_masking = ActionMasking(device=self.device)
        
        # PPO hyperparameters
        ppo_config = config['ppo']
        self.clip_range = ppo_config.get('clip_range', 0.2)
        self.clip_range_vf = ppo_config.get('clip_range_vf', None)
        self.ent_coef = ppo_config.get('ent_coef', 0.01)
        self.vf_coef = ppo_config.get('vf_coef', 0.5)
        self.max_grad_norm = ppo_config.get('max_grad_norm', 0.5)
        self.normalize_advantage = ppo_config.get('normalize_advantage', True)
        
        # Optimizer
        self.learning_rate = ppo_config.get('learning_rate', 3e-4)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
    def forward(self,
                obs: Union[np.ndarray, torch.Tensor],
                action_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
                n_jobs: Optional[int] = None,
                n_machines: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            obs: Observation from environment
            action_mask: Valid actions mask
            n_jobs: Number of jobs (required for parsing obs)
            n_machines: Number of machines (required for parsing obs)
            
        Returns:
            action_probs: Probability distribution over actions
            value: Estimated state value
            action_logits: Raw logits (for training)
        """
        # Convert to tensor if needed
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.device)
        
        # Ensure batch dimension
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
            
        # Convert action mask to tensor
        if action_mask is not None:
            if isinstance(action_mask, np.ndarray):
                action_mask = self.action_masking.env_mask_to_tensor(action_mask)
            elif isinstance(action_mask, torch.Tensor):
                action_mask = action_mask.to(self.device)
                
        # Encode state
        encoded_state = self.state_encoder.encode_from_env_observation(
            obs, n_jobs or 81, n_machines or 145  # Default sizes
        )
        
        # Get policy outputs
        action_logits, value = self.policy(encoded_state, action_mask)
        
        # Get action probabilities
        action_probs = self.policy.get_action_probs(encoded_state, action_mask)
        
        return action_probs, value, action_logits
    
    def get_action(self,
                   obs: np.ndarray,
                   action_mask: Optional[np.ndarray] = None,
                   deterministic: bool = False,
                   n_jobs: Optional[int] = None,
                   n_machines: Optional[int] = None) -> Tuple[int, float, float]:
        """
        Get action for a single environment.
        
        Args:
            obs: Environment observation
            action_mask: Valid actions
            deterministic: Whether to use deterministic policy
            n_jobs: Number of jobs
            n_machines: Number of machines
            
        Returns:
            action: Selected action index
            value: Estimated state value
            log_prob: Log probability of selected action
        """
        with torch.no_grad():
            action_probs, value, _ = self.forward(obs, action_mask, n_jobs, n_machines)
            
            # Sample action
            if action_mask is not None:
                mask_tensor = self.action_masking.env_mask_to_tensor(action_mask)
            else:
                mask_tensor = torch.ones_like(action_probs).bool()
                
            action = self.action_masking.sample_masked_action(
                action_probs, mask_tensor, deterministic
            )
            
            # Calculate log probability
            log_prob = torch.log(action_probs.gather(1, action.unsqueeze(1)) + 1e-8)
            
        return action.item(), value.item(), log_prob.item()
    
    def evaluate_actions(self,
                        obs: torch.Tensor,
                        actions: torch.Tensor,
                        action_masks: Optional[torch.Tensor] = None,
                        n_jobs: Optional[int] = None,
                        n_machines: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate given actions for PPO loss calculation.
        
        Args:
            obs: Batch of observations
            actions: Batch of actions taken
            action_masks: Batch of action masks
            n_jobs: Number of jobs
            n_machines: Number of machines
            
        Returns:
            values: Estimated state values
            log_probs: Log probabilities of actions
            entropy: Entropy of action distribution
        """
        action_probs, values, _ = self.forward(obs, action_masks, n_jobs, n_machines)
        
        # Get log probabilities of selected actions
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)) + 1e-8).squeeze(1)
        
        # Calculate entropy
        if action_masks is not None:
            entropy = self.action_masking.get_action_entropy(action_probs, action_masks)
        else:
            entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1)
            
        return values.squeeze(-1), log_probs, entropy.mean()
    
    def compute_ppo_loss(self,
                        obs: torch.Tensor,
                        actions: torch.Tensor,
                        old_log_probs: torch.Tensor,
                        advantages: torch.Tensor,
                        returns: torch.Tensor,
                        old_values: torch.Tensor,
                        action_masks: Optional[torch.Tensor] = None,
                        n_jobs: Optional[int] = None,
                        n_machines: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Compute PPO loss components.
        
        Args:
            obs: Batch of observations
            actions: Actions taken
            old_log_probs: Log probabilities from rollout
            advantages: Computed advantages
            returns: Computed returns
            old_values: Value estimates from rollout
            action_masks: Valid action masks
            n_jobs: Number of jobs
            n_machines: Number of machines
            
        Returns:
            Dictionary containing:
                - policy_loss: Policy loss
                - value_loss: Value function loss
                - entropy_loss: Entropy bonus (negative for maximization)
                - total_loss: Combined loss
                - approx_kl: Approximate KL divergence
                - clip_fraction: Fraction of clipped ratios
        """
        # Normalize advantages if configured
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        # Get current policy evaluation
        values, log_probs, entropy = self.evaluate_actions(
            obs, actions, action_masks, n_jobs, n_machines
        )
        
        # Calculate ratio for PPO
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Calculate policy loss with clipping
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Calculate value loss
        if self.clip_range_vf is not None:
            # Clip value function
            values_clipped = old_values + torch.clamp(
                values - old_values, -self.clip_range_vf, self.clip_range_vf
            )
            value_loss_unclipped = (values - returns) ** 2
            value_loss_clipped = (values_clipped - returns) ** 2
            value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
        else:
            value_loss = 0.5 * ((values - returns) ** 2).mean()
            
        # Entropy loss (negative because we want to maximize entropy)
        entropy_loss = -entropy
        
        # Total loss
        total_loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
        
        # Calculate additional metrics
        with torch.no_grad():
            approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
            clip_fraction = ((ratio - 1).abs() > self.clip_range).float().mean()
            
        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss,
            'total_loss': total_loss,
            'approx_kl': approx_kl,
            'clip_fraction': clip_fraction
        }
    
    def update(self,
               rollout_data: Dict[str, torch.Tensor],
               n_epochs: int = 10,
               batch_size: int = 64,
               n_jobs: Optional[int] = None,
               n_machines: Optional[int] = None) -> Dict[str, float]:
        """
        Update policy using collected rollout data.
        
        Args:
            rollout_data: Dictionary containing rollout data
            n_epochs: Number of epochs to train
            batch_size: Batch size for updates
            n_jobs: Number of jobs
            n_machines: Number of machines
            
        Returns:
            Dictionary of training metrics
        """
        # Extract data
        obs = rollout_data['obs']
        actions = rollout_data['actions']
        old_log_probs = rollout_data['log_probs']
        advantages = rollout_data['advantages']
        returns = rollout_data['returns']
        old_values = rollout_data['values']
        action_masks = rollout_data.get('action_masks')
        
        # Training metrics
        metrics = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'approx_kl': [],
            'clip_fraction': []
        }
        
        n_samples = len(obs)
        indices = np.arange(n_samples)
        
        for epoch in range(n_epochs):
            # Shuffle data
            np.random.shuffle(indices)
            
            # Train in mini-batches
            for start_idx in range(0, n_samples, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                
                batch_data = {
                    'obs': obs[batch_indices],
                    'actions': actions[batch_indices],
                    'old_log_probs': old_log_probs[batch_indices],
                    'advantages': advantages[batch_indices],
                    'returns': returns[batch_indices],
                    'old_values': old_values[batch_indices]
                }
                
                if action_masks is not None:
                    batch_data['action_masks'] = action_masks[batch_indices]
                    
                # Compute loss
                losses = self.compute_ppo_loss(**batch_data, n_jobs=n_jobs, n_machines=n_machines)
                
                # Backward pass
                self.optimizer.zero_grad()
                losses['total_loss'].backward()
                
                # Gradient clipping
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                    
                self.optimizer.step()
                
                # Record metrics
                for key in metrics:
                    if key in losses:
                        metrics[key].append(losses[key].item())
                        
        # Average metrics
        avg_metrics = {key: np.mean(values) for key, values in metrics.items()}
        
        return avg_metrics
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'state_encoder_state_dict': self.state_encoder.state_dict(),
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
        logger.info(f"Model saved to {path}")
        
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.state_encoder.load_state_dict(checkpoint['state_encoder_state_dict'])
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Model loaded from {path}")