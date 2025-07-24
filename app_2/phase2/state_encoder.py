"""
State Encoder for Scheduling Environment

Converts raw environment observations into structured tensor representations
suitable for the transformer policy network. Handles variable-sized inputs
with padding and masking.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class StateEncoder(nn.Module):
    """
    Encodes the scheduling environment state into embeddings.
    
    Handles:
    - Variable number of jobs (10-1000+)
    - Variable number of machines (5-200+)
    - Multi-machine job requirements
    - Padding and masking for transformer
    """
    
    def __init__(self, config: Dict):
        """
        Initialize state encoder.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        
        self.config = config
        self.job_feature_dim = config.get('job_feature_dim', 16)
        self.machine_feature_dim = config.get('machine_feature_dim', 16)
        self.embed_dim = config.get('embed_dim', 256)
        self.max_machines_per_job = config.get('max_machines_per_job', 10)
        
        # Job feature encoders
        self.job_encoder = nn.Sequential(
            nn.Linear(self.job_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.embed_dim)
        )
        
        # Machine feature encoders
        self.machine_encoder = nn.Sequential(
            nn.Linear(self.machine_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.embed_dim)
        )
        
        # Machine type embedding
        self.machine_type_embedding = nn.Embedding(
            num_embeddings=50,  # Max machine types
            embedding_dim=32
        )
        
        # Multi-machine encoding
        self.machine_set_encoder = nn.Sequential(
            nn.Linear(self.max_machines_per_job * 32, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Global context encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(3, 64),  # current_time, progress, completion_rate
            nn.ReLU(),
            nn.Linear(64, self.embed_dim)
        )
        
    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Encode raw observations into structured features.
        
        Args:
            obs: Dictionary containing:
                - jobs: Job features (batch_size, n_jobs, job_features)
                - machines: Machine features (batch_size, n_machines, machine_features)
                - global_state: Global features (batch_size, global_features)
                - job_mask: Valid jobs mask (batch_size, n_jobs)
                - machine_mask: Valid machines mask (batch_size, n_machines)
                
        Returns:
            Dictionary containing:
                - job_embeddings: (batch_size, n_jobs, embed_dim)
                - machine_embeddings: (batch_size, n_machines, embed_dim)
                - global_embedding: (batch_size, embed_dim)
                - job_mask: (batch_size, n_jobs)
                - machine_mask: (batch_size, n_machines)
        """
        batch_size = obs['jobs'].shape[0]
        n_jobs = obs['jobs'].shape[1]
        n_machines = obs['machines'].shape[1]
        
        # Encode jobs
        job_features = self._extract_job_features(obs['jobs'])
        job_embeddings = self.job_encoder(job_features)
        
        # Encode machines
        machine_features = self._extract_machine_features(obs['machines'])
        machine_embeddings = self.machine_encoder(machine_features)
        
        # Encode global state
        global_embedding = self.global_encoder(obs['global_state'])
        
        return {
            'job_embeddings': job_embeddings,
            'machine_embeddings': machine_embeddings,
            'global_embedding': global_embedding,
            'job_mask': obs.get('job_mask'),
            'machine_mask': obs.get('machine_mask')
        }
    
    def _extract_job_features(self, jobs: torch.Tensor) -> torch.Tensor:
        """
        Extract and process job features.
        
        Features include:
        - is_available
        - sequence_progress
        - urgency_score
        - processing_time
        - is_important
        - required_machines encoding
        """
        # In production, this would parse the flattened observation
        # For now, assume jobs tensor already contains these features
        return jobs
    
    def _extract_machine_features(self, machines: torch.Tensor) -> torch.Tensor:
        """
        Extract and process machine features.
        
        Features include:
        - current_load
        - time_until_free
        - machine_type (embedded)
        - utilization_rate
        """
        # In production, this would parse the flattened observation
        # For now, assume machines tensor already contains these features
        return machines
    
    def encode_from_env_observation(self, 
                                   env_obs: np.ndarray,
                                   n_jobs: int,
                                   n_machines: int) -> Dict[str, torch.Tensor]:
        """
        Convert raw environment observation to encoded state.
        
        Args:
            env_obs: Flattened observation from environment
            n_jobs: Number of jobs in this instance
            n_machines: Number of machines in this instance
            
        Returns:
            Encoded state dictionary
        """
        # Parse the flattened observation
        # This is environment-specific and would need proper implementation
        # based on the exact observation format
        
        # Placeholder implementation
        batch_size = 1 if env_obs.ndim == 1 else env_obs.shape[0]
        
        # Create dummy structured observations
        obs = {
            'jobs': torch.zeros(batch_size, n_jobs, self.job_feature_dim),
            'machines': torch.zeros(batch_size, n_machines, self.machine_feature_dim),
            'global_state': torch.zeros(batch_size, 3),
            'job_mask': torch.ones(batch_size, n_jobs, dtype=torch.bool),
            'machine_mask': torch.ones(batch_size, n_machines, dtype=torch.bool)
        }
        
        return self.forward(obs)