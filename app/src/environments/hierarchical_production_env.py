"""
Hierarchical Production Environment for Phase 5

This environment solves the action space limitation by decomposing
the job-machine assignment into two sequential decisions:
1. Select which job to schedule
2. Select which machine to assign it to

This reduces the action space from O(n_jobs × n_machines) to O(n_jobs + n_machines)
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, Any
import gymnasium as gym
from gymnasium import spaces
import logging

from .full_production_env import FullProductionEnv

logger = logging.getLogger(__name__)


class HierarchicalProductionEnv(FullProductionEnv):
    """
    Hierarchical action space version of the production environment.
    
    Key differences from FullProductionEnv:
    - Action space is Dict with 'job' and 'machine' keys
    - Two-stage decision making
    - Enhanced state representation
    - More efficient action masking
    """
    
    def __init__(
        self,
        n_machines: int = 152,
        n_jobs: int = 500,
        data_file: str = None,
        snapshot_file: str = None,
        max_episode_steps: int = 2000,
        max_valid_actions: int = 200,
        use_break_constraints: bool = True,
        use_holiday_constraints: bool = True,
        seed: Optional[int] = None,
        state_compression: str = "hierarchical_enhanced",
        **kwargs
    ):
        # Initialize parent class first
        super().__init__(
            n_machines=n_machines,
            n_jobs=n_jobs,
            data_file=data_file,
            snapshot_file=snapshot_file,
            max_episode_steps=max_episode_steps,
            max_valid_actions=max_valid_actions,
            use_break_constraints=use_break_constraints,
            use_holiday_constraints=use_holiday_constraints,
            seed=seed,
            state_compression="hierarchical",  # Use base compression for parent
            **kwargs
        )
        
        # Override action space with hierarchical structure
        self.action_space = spaces.Dict({
            'job': spaces.Discrete(n_jobs),
            'machine': spaces.Discrete(n_machines)
        })
        
        # Compatibility matrix for fast lookup
        self.compatibility_matrix = None
        self._build_compatibility_matrix()
        
        # Action masks for both stages
        self.job_mask = np.ones(n_jobs, dtype=bool)
        self.machine_masks = {}  # Will store machine masks for each job
        
        # Cache for performance
        self.compatibility_cache = {}
        
        # Tracking for hierarchical decisions
        self.last_selected_job = None
        self.decision_stage = 'job'  # 'job' or 'machine'
        
        # Enhanced state features
        self.use_hierarchical_features = kwargs.get('use_hierarchical_features', True)
        if self.use_hierarchical_features:
            # Add 20 hierarchical features to base 60
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(80,), dtype=np.float32
            )
        
        # Initialize scheduling tracking
        self.scheduled_jobs = [None] * n_jobs
        self.scheduled_count = 0
        self.machine_available_times = np.zeros(n_machines)
        
        logger.info(f"Initialized HierarchicalProductionEnv with {n_jobs} jobs and {n_machines} machines")
        logger.info(f"Action space reduced from {n_jobs * n_machines} to {n_jobs + n_machines}")
    
    def _build_compatibility_matrix(self) -> None:
        """
        Build job-machine compatibility matrix for efficient lookup.
        Matrix[i,j] = True if job i can be processed on machine j.
        """
        if self.jobs is None or self.machines is None:
            # Will be built after reset when jobs/machines are created
            return
            
        actual_n_jobs = len(self.jobs)
        actual_n_machines = len(self.machines)
        self.compatibility_matrix = np.zeros((actual_n_jobs, actual_n_machines), dtype=bool)
        
        for job_idx in range(actual_n_jobs):
            job = self.jobs[job_idx]
            for machine_idx in range(actual_n_machines):
                machine = self.machines[machine_idx]
                # Check if machine is capable for this job
                # Handle both dict and object formats
                if isinstance(job, dict):
                    capable_machines = job.get('capable_machines', [])
                    allowed_types = job.get('allowed_machine_types', [])
                else:
                    capable_machines = getattr(job, 'capable_machines', [])
                    allowed_types = getattr(job, 'allowed_machine_types', [])
                
                # First check capable_machines (machine IDs)
                if capable_machines:
                    # Get machine ID
                    machine_id = machine.get('machine_id', machine_idx) if isinstance(machine, dict) else getattr(machine, 'machine_id', machine_idx)
                    if machine_id in capable_machines:
                        self.compatibility_matrix[job_idx, machine_idx] = True
                # Then check allowed_types if no capable_machines
                elif allowed_types:
                    machine_type = machine.get('machine_type', 0) if isinstance(machine, dict) else getattr(machine, 'machine_type', 0)
                    if machine_type in allowed_types:
                        self.compatibility_matrix[job_idx, machine_idx] = True
                else:
                    # Fallback: all machines compatible
                    self.compatibility_matrix[job_idx, machine_idx] = True
        
        # Log compatibility statistics
        avg_compatible = np.mean(np.sum(self.compatibility_matrix, axis=1))
        logger.debug(f"Average compatible machines per job: {avg_compatible:.1f}")
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment with hierarchical initialization."""
        # Initialize tracking variables before parent reset
        self.scheduled_count = 0
        self.current_time = 0
        
        obs, info = super().reset(seed=seed, options=options)
        
        # Build compatibility matrix with actual jobs/machines
        self._build_compatibility_matrix()
        
        # Reset hierarchical tracking (after parent reset which sets n_jobs)
        # Use the actual number of jobs from loaded data if available
        actual_n_jobs = len(self.jobs) if hasattr(self, 'jobs') and self.jobs is not None else self.n_jobs
        self.job_mask = np.ones(actual_n_jobs, dtype=bool)
        self.machine_masks = {}
        self.last_selected_job = None
        self.decision_stage = 'job'
        
        # Reinitialize scheduled_jobs with correct size
        self.scheduled_jobs = [None] * actual_n_jobs
        self.machine_available_times = np.zeros(len(self.machines) if self.machines else self.n_machines)
        
        # Clear caches
        self.compatibility_cache.clear()
        
        # Get hierarchical state if enabled
        if self.use_hierarchical_features:
            obs = self._get_hierarchical_state()
        
        # Add action masks to info
        info['action_masks'] = self.get_action_masks()
        
        return obs, info
    
    def step(self, action: Dict[str, int]) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute hierarchical action.
        
        Args:
            action: Dict with 'job' and 'machine' keys
            
        Returns:
            Standard gym step returns
        """
        if not isinstance(action, dict) or 'job' not in action or 'machine' not in action:
            raise ValueError(f"Action must be dict with 'job' and 'machine' keys, got {action}")
        
        job_idx = action['job']
        machine_idx = action['machine']
        
        # Validate job selection
        actual_n_jobs = len(self.jobs) if hasattr(self, 'jobs') and self.jobs else self.n_jobs
        if job_idx < 0 or job_idx >= actual_n_jobs:
            return self._invalid_action_result(f"Invalid job index: {job_idx}")
        
        if not self.job_mask[job_idx]:
            return self._invalid_action_result(f"Job {job_idx} already scheduled")
        
        # Validate machine selection
        actual_n_machines = len(self.machines)
        if machine_idx < 0 or machine_idx >= actual_n_machines:
            return self._invalid_action_result(f"Invalid machine index: {machine_idx}")
        
        if not self.compatibility_matrix[job_idx, machine_idx]:
            return self._invalid_action_result(f"Job {job_idx} incompatible with machine {machine_idx}")
        
        # Check machine availability
        job = self.jobs[job_idx]
        machine = self.machines[machine_idx]
        
        current_time = self.current_time
        machine_available_time = self.machine_available_times[machine_idx]
        
        if current_time < machine_available_time:
            # Machine busy - this is actually valid, just need to wait
            start_time = machine_available_time
        else:
            start_time = current_time
        
        # Calculate setup time if applicable
        setup_time = 0
        if hasattr(self, 'use_setup_times') and self.use_setup_times:
            # Handle both dict and object formats
            family_id = job.get('family_id', None) if isinstance(job, dict) else getattr(job, 'family_id', None)
            if family_id:
                last_family = getattr(machine, 'last_family', None)
                if last_family and last_family != family_id:
                    setup_time = getattr(self, 'setup_time', 0.5)
        
        # Execute assignment
        actual_start_time = start_time + setup_time
        # Handle both dict and object formats for processing time
        processing_time = job.get('processing_time', 1.0) if isinstance(job, dict) else getattr(job, 'processing_time', 1.0)
        end_time = actual_start_time + processing_time
        
        # Update environment state
        self.scheduled_jobs[job_idx] = {
            'machine_id': machine_idx,
            'start_time': actual_start_time,
            'end_time': end_time,
            'setup_time': setup_time
        }
        
        # Update machine state
        self.machine_available_times[machine_idx] = end_time
        if hasattr(machine, 'last_family'):
            family_id = job.get('family_id', None) if isinstance(job, dict) else getattr(job, 'family_id', None)
            if family_id:
                machine.last_family = family_id
        
        # Update tracking
        self.scheduled_count += 1
        self.job_mask[job_idx] = False
        self.last_selected_job = job_idx
        
        # Calculate hierarchical reward
        reward = self._calculate_hierarchical_reward(job_idx, machine_idx, setup_time)
        
        # Check termination
        done = self.scheduled_count >= self.n_jobs
        
        # Get next state
        if self.use_hierarchical_features:
            next_obs = self._get_hierarchical_state()
        else:
            next_obs = self._get_observation()
        
        # Prepare info
        info = {
            'scheduled_count': self.scheduled_count,
            'job_idx': job_idx,
            'machine_idx': machine_idx,
            'start_time': actual_start_time,
            'setup_time': setup_time,
            'makespan': self._calculate_makespan() if done else 0,
            'action_masks': self.get_action_masks()
        }
        
        # Log progress
        if self.scheduled_count % 50 == 0:
            logger.info(f"Scheduled {self.scheduled_count}/{self.n_jobs} jobs")
        
        return next_obs, reward, done, False, info
    
    def _invalid_action_result(self, reason: str) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Handle invalid action with logging."""
        logger.debug(f"Invalid action: {reason}")
        obs = self._get_hierarchical_state() if self.use_hierarchical_features else self._get_observation()
        reward = -20.0  # Invalid action penalty
        info = {
            'invalid_action': True,
            'reason': reason,
            'action_masks': self.get_action_masks()
        }
        return obs, reward, False, False, info
    
    def _calculate_hierarchical_reward(self, job_idx: int, machine_idx: int, setup_time: float) -> float:
        """
        Calculate reward for hierarchical action.
        
        Rewards both good job selection and good machine selection.
        """
        reward = 0.0
        job = self.jobs[job_idx]
        
        # Job selection rewards
        # Handle both dict and object formats
        if isinstance(job, dict):
            is_important = job.get('is_important', False)
            lcd_days = job.get('lcd_days_remaining', 30)
        else:
            is_important = getattr(job, 'is_important', False)
            lcd_days = getattr(job, 'lcd_days_remaining', 30)
            
        if is_important:
            reward += 5.0  # Important job bonus
        
        # Urgency reward (based on LCD)
        urgency = max(0, 30 - lcd_days) / 30  # 0-1 scale
        reward += urgency * 2.0
        
        # Machine selection rewards
        # Load balancing
        machine_utilizations = self._calculate_machine_utilizations()
        avg_utilization = np.mean(machine_utilizations)
        machine_utilization = machine_utilizations[machine_idx]
        
        # Penalize deviation from average (encourage balance)
        balance_penalty = -abs(machine_utilization - avg_utilization) * 3.0
        reward += balance_penalty
        
        # Setup time penalty
        reward -= setup_time * 0.5
        
        # Efficiency bonus for preferred machines
        compatible_machines = self.compatibility_matrix[job_idx]
        n_compatible = np.sum(compatible_machines)
        if n_compatible > 0 and n_compatible < self.n_machines:
            # This is a constrained job, reward for using compatible machine
            reward += 3.0
        
        # Progress reward
        progress = self.scheduled_count / self.n_jobs
        reward += progress * 2.0
        
        # Completion bonus
        if self.scheduled_count >= self.n_jobs:
            makespan = self._calculate_makespan()
            # Bonus inversely proportional to makespan
            reward += 100.0 / max(makespan, 1.0)
        
        return reward
    
    def _calculate_machine_utilizations(self) -> np.ndarray:
        """Calculate current utilization for each machine."""
        utilizations = np.zeros(self.n_machines)
        
        for job_idx, schedule in enumerate(self.scheduled_jobs):
            if schedule is not None:
                machine_idx = schedule['machine_id']
                job = self.jobs[job_idx]
                # Handle both dict and object formats
                processing_time = job.get('processing_time', 1.0) if isinstance(job, dict) else getattr(job, 'processing_time', 1.0)
                utilizations[machine_idx] += processing_time
        
        # Normalize by total time if needed
        total_time = np.max(self.machine_available_times)
        if total_time > 0:
            utilizations = utilizations / total_time
        
        return utilizations
    
    def _get_hierarchical_state(self) -> np.ndarray:
        """
        Get enhanced state representation for hierarchical decisions.
        
        Adds 20 features to the base 60 for better decision making.
        """
        # Get base state (60 features)
        base_state = self._get_observation()
        
        # Additional hierarchical features (20)
        hierarchical_features = []
        
        # Job urgency distribution (5 features)
        urgency_dist = self._get_job_urgency_distribution()
        hierarchical_features.extend(urgency_dist)
        
        # Job complexity distribution (5 features)
        complexity_dist = self._get_job_complexity_distribution()
        hierarchical_features.extend(complexity_dist)
        
        # Machine load distribution (5 features)
        load_dist = self._get_machine_load_distribution()
        hierarchical_features.extend(load_dist)
        
        # Machine compatibility statistics (5 features)
        compat_stats = self._get_machine_compatibility_stats()
        hierarchical_features.extend(compat_stats)
        
        # Combine all features
        hierarchical_state = np.concatenate([
            base_state,
            np.array(hierarchical_features, dtype=np.float32)
        ])
        
        return hierarchical_state
    
    def _get_job_urgency_distribution(self) -> List[float]:
        """Get distribution of job urgencies (5 bins)."""
        urgencies = []
        for idx in range(self.n_jobs):
            if self.job_mask[idx]:  # Only unscheduled jobs
                job = self.jobs[idx]
                # Handle both dict and object formats
                if isinstance(job, dict):
                    urgencies.append(job.get('lcd_days_remaining', 30))
                else:
                    urgencies.append(getattr(job, 'lcd_days_remaining', 30))
        
        if not urgencies:
            return [0.0] * 5
        
        # Create 5-bin histogram
        bins = np.linspace(0, 60, 6)
        hist, _ = np.histogram(urgencies, bins=bins)
        hist = hist / max(np.sum(hist), 1)  # Normalize
        
        return hist.tolist()
    
    def _get_job_complexity_distribution(self) -> List[float]:
        """Get distribution of job complexities (5 bins)."""
        complexities = []
        for idx in range(self.n_jobs):
            if self.job_mask[idx]:
                job = self.jobs[idx]
                # Complexity = processing time * (1 / n_compatible_machines)
                n_compatible = np.sum(self.compatibility_matrix[idx])
                # Handle both dict and object formats
                processing_time = job.get('processing_time', 1.0) if isinstance(job, dict) else getattr(job, 'processing_time', 1.0)
                complexity = processing_time * (self.n_machines / max(n_compatible, 1))
                complexities.append(complexity)
        
        if not complexities:
            return [0.0] * 5
        
        # Create 5-bin histogram
        bins = np.linspace(0, max(complexities) + 1, 6)
        hist, _ = np.histogram(complexities, bins=bins)
        hist = hist / max(np.sum(hist), 1)
        
        return hist.tolist()
    
    def _get_machine_load_distribution(self) -> List[float]:
        """Get distribution of machine loads (5 bins)."""
        loads = self.machine_available_times - self.current_time
        loads = np.maximum(loads, 0)  # Only positive loads
        
        # Create 5-bin histogram
        if np.max(loads) > 0:
            bins = np.linspace(0, np.max(loads), 6)
            hist, _ = np.histogram(loads, bins=bins)
            hist = hist / max(np.sum(hist), 1)
        else:
            hist = np.array([1.0, 0, 0, 0, 0])  # All machines free
        
        return hist.tolist()
    
    def _get_machine_compatibility_stats(self) -> List[float]:
        """Get machine compatibility statistics."""
        unscheduled_mask = self.job_mask
        if np.sum(unscheduled_mask) == 0:
            return [0.0] * 5
        
        # For unscheduled jobs, calculate compatibility stats
        unscheduled_compat = self.compatibility_matrix[unscheduled_mask]
        
        stats = [
            np.mean(np.sum(unscheduled_compat, axis=1)),  # Avg compatible machines per job
            np.std(np.sum(unscheduled_compat, axis=1)),   # Std of compatible machines
            np.mean(np.sum(unscheduled_compat, axis=0)),  # Avg compatible jobs per machine
            np.std(np.sum(unscheduled_compat, axis=0)),   # Std of compatible jobs
            np.sum(unscheduled_compat) / (np.sum(unscheduled_mask) * self.n_machines)  # Overall compatibility ratio
        ]
        
        # Normalize to 0-1 range
        stats = np.array(stats)
        stats[0] /= self.n_machines  # Avg machines per job
        stats[1] /= self.n_machines  # Std machines per job
        stats[2] /= np.sum(unscheduled_mask)  # Avg jobs per machine
        stats[3] /= np.sum(unscheduled_mask)  # Std jobs per machine
        # stats[4] already in 0-1 range
        
        return np.clip(stats, 0, 1).tolist()
    
    def get_action_masks(self) -> Dict[str, np.ndarray]:
        """
        Get current valid actions for both job and machine selection.
        
        Returns:
            Dictionary with 'job' and 'machine' boolean masks
        """
        # Job mask: which jobs can be selected
        job_mask = self.job_mask.copy()
        
        # Machine mask: depends on the current context
        # For training, we need masks for all possible job selections
        actual_n_jobs = len(job_mask)
        machine_masks = np.zeros((actual_n_jobs, self.n_machines), dtype=bool)
        
        for job_idx in range(actual_n_jobs):
            if job_mask[job_idx]:
                # This job can be selected, check which machines are compatible
                if self.compatibility_matrix is not None and job_idx < self.compatibility_matrix.shape[0]:
                    compatible = self.compatibility_matrix[job_idx]
                else:
                    # If no compatibility matrix or out of bounds, assume all machines compatible
                    compatible = np.ones(self.n_machines, dtype=bool)
                # Machine must be compatible AND we must be able to schedule on it
                # (In practice, we always allow scheduling even if machine is busy)
                machine_masks[job_idx] = compatible
        
        return {
            'job': job_mask,
            'machine': machine_masks
        }
    
    def get_info(self) -> dict:
        """Override to avoid parent's get_action_mask call."""
        info = {
            'current_time': self.current_time,
            'scheduled_count': self.scheduled_count,
            'total_jobs': self.n_jobs,
            'action_masks': self.get_action_masks()
        }
        if self.scheduled_count >= self.n_jobs:
            info['makespan'] = self._calculate_makespan()
        return info
    
    def render(self) -> Optional[np.ndarray]:
        """Render current state (console or return RGB array)."""
        if self.render_mode == "human":
            print(f"\n=== Hierarchical Production Environment ===")
            print(f"Time: {self.current_time:.1f}")
            print(f"Scheduled: {self.scheduled_count}/{self.n_jobs} jobs")
            print(f"Action space: {self.n_jobs} + {self.n_machines} = {self.n_jobs + self.n_machines} (vs {self.n_jobs * self.n_machines} flat)")
            
            # Show machine utilization
            print("\nMachine utilization:")
            for m_idx in range(min(5, self.n_machines)):
                util = self._calculate_machine_utilizations()[m_idx]
                print(f"  Machine {m_idx}: {util:.1%}")
            
            # Show recent assignments
            print("\nRecent assignments:")
            recent = [(idx, s) for idx, s in enumerate(self.scheduled_jobs) if s is not None][-5:]
            for job_idx, schedule in recent:
                job = self.jobs[job_idx]
                print(f"  Job {job_idx} -> Machine {schedule['machine_id']} at time {schedule['start_time']:.1f}")
        
        return None