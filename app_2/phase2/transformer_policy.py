"""
Transformer-based Policy Network for Scheduling

Uses self-attention and cross-attention to handle variable-sized inputs
and learn complex relationships between jobs and machines.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math
import logging

logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % n_heads == 0
        
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = math.sqrt(self.head_dim)
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply multi-head attention.
        
        Args:
            query: (batch_size, seq_len_q, embed_dim)
            key: (batch_size, seq_len_k, embed_dim)
            value: (batch_size, seq_len_v, embed_dim)
            mask: (batch_size, seq_len_q, seq_len_k) or None
            
        Returns:
            Output tensor (batch_size, seq_len_q, embed_dim)
        """
        batch_size, seq_len_q = query.shape[:2]
        seq_len_k = key.shape[1]
        
        # Project and reshape for multi-head
        Q = self.q_proj(query).view(batch_size, seq_len_q, self.n_heads, self.head_dim)
        K = self.k_proj(key).view(batch_size, seq_len_k, self.n_heads, self.head_dim)
        V = self.v_proj(value).view(batch_size, seq_len_k, self.n_heads, self.head_dim)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # (batch, n_heads, seq_len_q, head_dim)
        K = K.transpose(1, 2)  # (batch, n_heads, seq_len_k, head_dim)
        V = V.transpose(1, 2)  # (batch, n_heads, seq_len_v, head_dim)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1)  # Add head dimension
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len_q, self.embed_dim)
        
        # Final projection
        output = self.out_proj(attn_output)
        
        return output


class TransformerBlock(nn.Module):
    """Transformer encoder block with self-attention and feedforward."""
    
    def __init__(self, embed_dim: int, n_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(embed_dim, n_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply transformer block.
        
        Args:
            x: Input tensor (batch_size, seq_len, embed_dim)
            mask: Attention mask (batch_size, seq_len, seq_len)
            
        Returns:
            Output tensor (batch_size, seq_len, embed_dim)
        """
        # Self-attention with residual
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        
        # Feedforward with residual
        ff_output = self.feedforward(x)
        x = self.norm2(x + ff_output)
        
        return x


class TransformerSchedulingPolicy(nn.Module):
    """
    Transformer-based policy network for scheduling.
    
    Architecture:
    1. Job encoder: Self-attention over jobs
    2. Machine encoder: Self-attention over machines
    3. Cross-attention: Jobs attend to machines
    4. Action decoder: Output (job, machine) probabilities
    5. Value head: Estimate state value
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.embed_dim = config.get('embed_dim', 256)
        self.n_heads = config.get('n_heads', 8)
        self.n_layers = config.get('n_layers', 6)
        self.ff_dim = config.get('ff_dim', 1024)
        self.dropout = config.get('dropout', 0.1)
        
        # Job encoder
        self.job_encoder = nn.ModuleList([
            TransformerBlock(self.embed_dim, self.n_heads, self.ff_dim, self.dropout)
            for _ in range(self.n_layers // 2)
        ])
        
        # Machine encoder
        self.machine_encoder = nn.ModuleList([
            TransformerBlock(self.embed_dim, self.n_heads, self.ff_dim, self.dropout)
            for _ in range(self.n_layers // 2)
        ])
        
        # Cross-attention layers
        self.cross_attention = MultiHeadAttention(self.embed_dim, self.n_heads, self.dropout)
        self.cross_norm = nn.LayerNorm(self.embed_dim)
        
        # Action decoder
        self.action_decoder = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.ff_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.ff_dim, 1)  # Score for (job, machine) pair
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.ff_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.ff_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, 
                encoded_state: Dict[str, torch.Tensor],
                action_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through policy network.
        
        Args:
            encoded_state: Dictionary from StateEncoder containing:
                - job_embeddings: (batch_size, n_jobs, embed_dim)
                - machine_embeddings: (batch_size, n_machines, embed_dim)
                - global_embedding: (batch_size, embed_dim)
                - job_mask: (batch_size, n_jobs)
                - machine_mask: (batch_size, n_machines)
            action_mask: Valid actions mask (batch_size, n_jobs * n_machines)
            
        Returns:
            action_logits: (batch_size, n_jobs * n_machines)
            value: (batch_size, 1)
        """
        job_embeddings = encoded_state['job_embeddings']
        machine_embeddings = encoded_state['machine_embeddings']
        global_embedding = encoded_state['global_embedding']
        job_mask = encoded_state.get('job_mask')
        machine_mask = encoded_state.get('machine_mask')
        
        batch_size = job_embeddings.shape[0]
        n_jobs = job_embeddings.shape[1]
        n_machines = machine_embeddings.shape[1]
        
        # Encode jobs with self-attention
        job_features = job_embeddings
        for layer in self.job_encoder:
            job_features = layer(job_features, self._create_padding_mask(job_mask))
        
        # Encode machines with self-attention
        machine_features = machine_embeddings
        for layer in self.machine_encoder:
            machine_features = layer(machine_features, self._create_padding_mask(machine_mask))
        
        # Cross-attention: jobs attend to machines
        # This learns which machines are suitable for each job
        cross_mask = self._create_cross_attention_mask(job_mask, machine_mask)
        job_machine_features = self.cross_attention(
            job_features, machine_features, machine_features, cross_mask
        )
        job_machine_features = self.cross_norm(job_features + job_machine_features)
        
        # Compute action scores for all (job, machine) pairs
        action_logits = self._compute_action_logits(
            job_machine_features, machine_features, n_jobs, n_machines
        )
        
        # Apply action mask if provided
        if action_mask is not None:
            action_logits = action_logits.masked_fill(~action_mask, -1e9)
        
        # Compute value using global features
        global_features = global_embedding.unsqueeze(1)  # (batch, 1, embed_dim)
        
        # Aggregate job features for value estimation
        if job_mask is not None:
            job_mask_expanded = job_mask.unsqueeze(-1).float()
            job_features_masked = job_features * job_mask_expanded
            job_features_mean = job_features_masked.sum(dim=1) / job_mask_expanded.sum(dim=1).clamp(min=1)
        else:
            job_features_mean = job_features.mean(dim=1)
        
        # Combine with global features
        value_features = job_features_mean + global_embedding
        value = self.value_head(value_features)
        
        return action_logits, value
    
    def _create_padding_mask(self, mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Create attention mask from padding mask."""
        if mask is None:
            return None
        
        # Convert padding mask to attention mask
        # mask: (batch_size, seq_len) -> (batch_size, seq_len, seq_len)
        batch_size, seq_len = mask.shape
        mask = mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
        mask = mask & mask.transpose(1, 2)
        
        return mask
    
    def _create_cross_attention_mask(self,
                                    job_mask: Optional[torch.Tensor],
                                    machine_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Create mask for cross-attention between jobs and machines."""
        if job_mask is None or machine_mask is None:
            return None
        
        # job_mask: (batch_size, n_jobs)
        # machine_mask: (batch_size, n_machines)
        # output: (batch_size, n_jobs, n_machines)
        
        batch_size = job_mask.shape[0]
        n_jobs = job_mask.shape[1]
        n_machines = machine_mask.shape[1]
        
        job_mask = job_mask.unsqueeze(2).expand(batch_size, n_jobs, n_machines)
        machine_mask = machine_mask.unsqueeze(1).expand(batch_size, n_jobs, n_machines)
        
        return job_mask & machine_mask
    
    def _compute_action_logits(self,
                              job_features: torch.Tensor,
                              machine_features: torch.Tensor,
                              n_jobs: int,
                              n_machines: int) -> torch.Tensor:
        """
        Compute logits for all (job, machine) pairs.
        
        Returns:
            Flattened logits (batch_size, n_jobs * n_machines)
        """
        batch_size = job_features.shape[0]
        
        # Expand features for all combinations
        job_features_expanded = job_features.unsqueeze(2).expand(
            batch_size, n_jobs, n_machines, self.embed_dim
        )
        machine_features_expanded = machine_features.unsqueeze(1).expand(
            batch_size, n_jobs, n_machines, self.embed_dim
        )
        
        # Concatenate job and machine features
        combined_features = torch.cat([job_features_expanded, machine_features_expanded], dim=-1)
        
        # Compute scores
        action_scores = self.action_decoder(combined_features).squeeze(-1)
        
        # Flatten to match action space
        action_logits = action_scores.view(batch_size, n_jobs * n_machines)
        
        return action_logits
    
    def get_action_probs(self, 
                        encoded_state: Dict[str, torch.Tensor],
                        action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get action probabilities with proper masking."""
        action_logits, _ = self.forward(encoded_state, action_mask)
        
        # Apply softmax to get probabilities
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Ensure masked actions have exactly 0 probability
        if action_mask is not None:
            action_probs = action_probs * action_mask.float()
            # Renormalize
            action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        
        return action_probs