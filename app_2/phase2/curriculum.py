"""
Curriculum Learning for Progressive Training

Manages training stages from simple to complex scenarios,
gradually increasing difficulty as the model improves.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import yaml

logger = logging.getLogger(__name__)


@dataclass
class CurriculumStage:
    """Definition of a curriculum stage."""
    name: str
    timesteps: Optional[int]  # None means continue indefinitely
    env_config: Dict  # Environment configuration for this stage
    success_threshold: float = 0.8  # Success rate to advance
    min_timesteps: int = 10000  # Minimum timesteps before advancing
    

class CurriculumManager:
    """
    Manages curriculum learning progression.
    
    Tracks performance and advances through stages automatically
    based on success metrics.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize curriculum manager.
        
        Args:
            config_path: Path to curriculum configuration YAML
        """
        self.stages: List[CurriculumStage] = []
        self.current_stage_idx = 0
        self.stage_timesteps = 0
        self.total_timesteps = 0
        self.performance_history = []
        
        if config_path:
            self.load_config(config_path)
        else:
            self._setup_default_stages()
            
    def _setup_default_stages(self):
        """Setup default curriculum stages."""
        self.stages = [
            CurriculumStage(
                name="toy",
                timesteps=100000,
                env_config={
                    'max_jobs': 10,
                    'max_machines': 5,
                    'complexity': 'simple'
                },
                success_threshold=0.9,
                min_timesteps=50000
            ),
            CurriculumStage(
                name="small",
                timesteps=300000,
                env_config={
                    'max_jobs': 50,
                    'max_machines': 20,
                    'complexity': 'moderate'
                },
                success_threshold=0.85,
                min_timesteps=100000
            ),
            CurriculumStage(
                name="medium",
                timesteps=500000,
                env_config={
                    'max_jobs': 200,
                    'max_machines': 50,
                    'complexity': 'complex'
                },
                success_threshold=0.8,
                min_timesteps=200000
            ),
            CurriculumStage(
                name="large",
                timesteps=1000000,
                env_config={
                    'max_jobs': 500,
                    'max_machines': 100,
                    'complexity': 'very_complex'
                },
                success_threshold=0.75,
                min_timesteps=400000
            ),
            CurriculumStage(
                name="production",
                timesteps=None,  # Continue indefinitely
                env_config={
                    'max_jobs': None,  # No limit
                    'max_machines': None,  # No limit
                    'complexity': 'production'
                },
                success_threshold=0.0,  # Never advance
                min_timesteps=0
            )
        ]
        
    def load_config(self, config_path: str):
        """Load curriculum configuration from YAML."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        self.stages = []
        for stage_config in config['curriculum']['stages']:
            stage = CurriculumStage(
                name=stage_config['name'],
                timesteps=stage_config.get('timesteps'),
                env_config=stage_config['env_config'],
                success_threshold=stage_config.get('success_threshold', 0.8),
                min_timesteps=stage_config.get('min_timesteps', 10000)
            )
            self.stages.append(stage)
            
    def get_current_stage(self) -> CurriculumStage:
        """Get current curriculum stage."""
        return self.stages[self.current_stage_idx]
    
    def get_env_config(self) -> Dict:
        """Get environment configuration for current stage."""
        stage = self.get_current_stage()
        config = stage.env_config.copy()
        
        # Add stage-specific parameters
        config['curriculum_stage'] = stage.name
        config['stage_progress'] = self.get_stage_progress()
        
        return config
    
    def update(self, timesteps: int, metrics: Dict[str, float]) -> bool:
        """
        Update curriculum progress.
        
        Args:
            timesteps: Number of timesteps completed
            metrics: Performance metrics from training
            
        Returns:
            True if advanced to next stage
        """
        self.stage_timesteps += timesteps
        self.total_timesteps += timesteps
        
        # Record performance
        self.performance_history.append({
            'stage': self.get_current_stage().name,
            'timesteps': self.total_timesteps,
            'metrics': metrics.copy()
        })
        
        # Check if should advance
        if self.should_advance(metrics):
            return self.advance_stage()
            
        return False
    
    def should_advance(self, metrics: Dict[str, float]) -> bool:
        """
        Check if should advance to next stage.
        
        Args:
            metrics: Current performance metrics
            
        Returns:
            True if should advance
        """
        stage = self.get_current_stage()
        
        # Check if at final stage
        if self.current_stage_idx >= len(self.stages) - 1:
            return False
            
        # Check minimum timesteps
        if self.stage_timesteps < stage.min_timesteps:
            return False
            
        # Check if exceeded stage timesteps
        if stage.timesteps is not None and self.stage_timesteps >= stage.timesteps:
            return True
            
        # Check performance threshold
        success_rate = self._calculate_success_rate(metrics)
        if success_rate >= stage.success_threshold:
            logger.info(f"Success rate {success_rate:.2f} >= threshold {stage.success_threshold}")
            return True
            
        return False
    
    def advance_stage(self) -> bool:
        """
        Advance to next curriculum stage.
        
        Returns:
            True if advanced, False if already at final stage
        """
        if self.current_stage_idx >= len(self.stages) - 1:
            return False
            
        old_stage = self.get_current_stage()
        self.current_stage_idx += 1
        self.stage_timesteps = 0
        new_stage = self.get_current_stage()
        
        logger.info(f"Advanced from stage '{old_stage.name}' to '{new_stage.name}'")
        logger.info(f"New configuration: {new_stage.env_config}")
        
        return True
    
    def _calculate_success_rate(self, metrics: Dict[str, float]) -> float:
        """
        Calculate success rate from metrics.
        
        Args:
            metrics: Performance metrics
            
        Returns:
            Success rate between 0 and 1
        """
        # Define success based on multiple factors
        factors = []
        
        # On-time delivery rate
        if 'on_time_rate' in metrics:
            factors.append(metrics['on_time_rate'])
            
        # Normalized reward
        if 'mean_reward' in metrics and 'max_possible_reward' in metrics:
            normalized_reward = metrics['mean_reward'] / metrics['max_possible_reward']
            factors.append(max(0, min(1, normalized_reward)))
            
        # Efficiency (inverse of makespan)
        if 'efficiency' in metrics:
            factors.append(metrics['efficiency'])
            
        # Average success rate
        if factors:
            return np.mean(factors)
        else:
            # Fallback: use episode reward
            if 'episode_reward' in metrics:
                # Assume reward > 0 is success
                return min(1.0, max(0.0, metrics['episode_reward'] / 100.0))
            return 0.0
    
    def get_stage_progress(self) -> float:
        """
        Get progress through current stage.
        
        Returns:
            Progress from 0 to 1
        """
        stage = self.get_current_stage()
        if stage.timesteps is None:
            return 0.0  # Infinite stage
            
        return min(1.0, self.stage_timesteps / stage.timesteps)
    
    def get_overall_progress(self) -> float:
        """
        Get overall curriculum progress.
        
        Returns:
            Progress from 0 to 1
        """
        # Simple linear progress through stages
        stage_weight = 1.0 / len(self.stages)
        completed_stages = self.current_stage_idx
        current_progress = self.get_stage_progress()
        
        return completed_stages * stage_weight + current_progress * stage_weight
    
    def get_difficulty_multiplier(self) -> float:
        """
        Get difficulty multiplier for current stage.
        
        Used to scale rewards or adjust environment difficulty.
        
        Returns:
            Multiplier >= 1.0
        """
        # Increase difficulty as we progress
        base_multiplier = 1.0 + self.current_stage_idx * 0.5
        
        # Add stage progress component
        stage_progress = self.get_stage_progress()
        stage_multiplier = 1.0 + stage_progress * 0.2
        
        return base_multiplier * stage_multiplier
    
    def should_use_mixed_stages(self) -> bool:
        """
        Check if should mix difficulties for robustness.
        
        Returns:
            True if should occasionally sample from previous stages
        """
        # After reaching medium stage, occasionally train on easier scenarios
        return self.current_stage_idx >= 2
    
    def sample_stage_config(self) -> Tuple[Dict, str]:
        """
        Sample a stage configuration, possibly from previous stages.
        
        Returns:
            (env_config, stage_name)
        """
        if self.should_use_mixed_stages() and np.random.random() < 0.2:
            # 20% chance to sample from previous stages
            stage_idx = np.random.randint(0, self.current_stage_idx)
            stage = self.stages[stage_idx]
            return stage.env_config, stage.name
        else:
            # Use current stage
            stage = self.get_current_stage()
            return stage.env_config, stage.name
    
    def get_summary(self) -> Dict:
        """Get curriculum summary statistics."""
        return {
            'current_stage': self.get_current_stage().name,
            'stage_idx': self.current_stage_idx,
            'total_stages': len(self.stages),
            'stage_timesteps': self.stage_timesteps,
            'total_timesteps': self.total_timesteps,
            'stage_progress': self.get_stage_progress(),
            'overall_progress': self.get_overall_progress(),
            'difficulty_multiplier': self.get_difficulty_multiplier()
        }
    
    def save_progress(self, path: str):
        """Save curriculum progress."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'current_stage_idx': self.current_stage_idx,
                'stage_timesteps': self.stage_timesteps,
                'total_timesteps': self.total_timesteps,
                'performance_history': self.performance_history
            }, f)
            
    def load_progress(self, path: str):
        """Load curriculum progress."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.current_stage_idx = data['current_stage_idx']
            self.stage_timesteps = data['stage_timesteps']
            self.total_timesteps = data['total_timesteps']
            self.performance_history = data['performance_history']