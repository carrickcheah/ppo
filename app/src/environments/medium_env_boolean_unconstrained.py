"""
Medium scheduling environment with boolean importance (unconstrained version).
No sorting - PPO decides when to schedule important vs normal jobs.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
import gymnasium as gym
from gymnasium import spaces
import json
from datetime import datetime
import logging
import random

from .base_env import BaseSchedulingEnv

logger = logging.getLogger(__name__)


class MediumBooleanUnconstrainedSchedulingEnv(BaseSchedulingEnv):
    """
    Medium complexity scheduling environment using boolean importance WITHOUT sorting.
    
    Key differences from constrained version:
    - No sorting of valid actions
    - PPO learns when to prioritize important jobs
    - Tracks when PPO chooses normal jobs over important ones
    """
    
    def __init__(self,
                 n_machines: int = 10,
                 data_file: str = '/Users/carrickcheah/Project/ppo/app/data/parsed_production_data_boolean.json',
                 max_episode_steps: int = 500,
                 max_valid_actions: int = 100,
                 seed: Optional[int] = None):
        """Initialize unconstrained boolean environment."""
        # Load production data
        with open(data_file, 'r') as f:
            self.families_data = json.load(f)
        
        n_jobs = sum(len(f['tasks']) for f in self.families_data.values())
        super().__init__(n_machines, n_jobs, seed)
        
        self.n_families = len(self.families_data)
        self.max_episode_steps = max_episode_steps
        self.max_valid_actions = max_valid_actions
        
        # Set random seed for shuffling
        self.rng = np.random.RandomState(seed)
        
        # Define observation space (same as constrained)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(n_machines + 4 * self.n_families + 1,),
            dtype=np.float32
        )
        
        # Action space: can select any valid action
        self.action_space = spaces.Discrete(max_valid_actions)
        
        # Environment state
        self.machine_loads = None
        self.family_progress = None
        self.completed_tasks = None
        self.current_step = None
        self.current_time = None
        self.valid_actions = None
        self.episode_makespan = None
        
        # Parse families
        self.family_ids = list(self.families_data.keys())
        self.family_idx_map = {fid: idx for idx, fid in enumerate(self.family_ids)}
        
        # Track behavior
        self.importance_violations = 0  # Times normal job chosen over important
        self.urgency_justified = 0      # Violations justified by urgency
        
    def _reset_impl(self, options: Optional[Dict] = None) -> np.ndarray:
        """Reset environment to initial state."""
        # Reset machine loads
        self.machine_loads = np.zeros(self.n_machines, dtype=np.float32)
        
        # Reset family progress
        self.family_progress = np.zeros(self.n_families, dtype=np.float32)
        
        # Reset completed tasks tracking
        self.completed_tasks = {
            fid: set() for fid in self.family_ids
        }
        
        # Reset time and statistics
        self.current_step = 0
        self.current_time = 0.0
        self.episode_makespan = 0.0
        self.importance_violations = 0
        self.urgency_justified = 0
        
        # Get initial valid actions WITHOUT sorting
        self.valid_actions = self._get_valid_actions()
        
        return self._get_observation()
    
    def _get_valid_actions(self) -> List[Tuple[str, int, Dict]]:
        """
        Get valid actions WITHOUT sorting.
        Returns in random order to avoid position bias.
        """
        valid = []
        
        for family_id, family_data in self.families_data.items():
            # Find next uncompleted task in sequence
            for task_idx, task in enumerate(family_data['tasks']):
                task_seq = task['sequence']
                
                # Skip if already completed
                if task_seq in self.completed_tasks[family_id]:
                    continue
                
                # Check dependencies
                dependencies_met = True
                for other_task in family_data['tasks']:
                    if other_task['sequence'] < task_seq:
                        if other_task['sequence'] not in self.completed_tasks[family_id]:
                            dependencies_met = False
                            break
                
                if dependencies_met:
                    valid.append((family_id, task_idx, task))
                    break
        
        # CRITICAL: Shuffle to remove any bias
        self.rng.shuffle(valid)
        
        return valid
    
    def _calculate_urgency(self, family_id: str) -> float:
        """Calculate urgency score based on LCD date."""
        family = self.families_data[family_id]
        lcd_date = datetime.fromisoformat(family['lcd_date'])
        base_date = datetime(2024, 1, 15)
        
        days_to_lcd = (lcd_date - base_date).days
        return days_to_lcd
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation (same structure as constrained)."""
        # Normalize machine loads
        max_load = max(self.machine_loads) if np.any(self.machine_loads > 0) else 1.0
        normalized_loads = self.machine_loads / max_load
        
        # Family progress
        family_progress = np.zeros(self.n_families)
        for idx, family_id in enumerate(self.family_ids):
            family = self.families_data[family_id]
            total_tasks = len(family['tasks'])
            completed = len(self.completed_tasks[family_id])
            family_progress[idx] = completed / total_tasks if total_tasks > 0 else 1.0
        
        # Family importance (boolean as float)
        family_importance = np.zeros(self.n_families)
        for idx, family_id in enumerate(self.family_ids):
            family_importance[idx] = 1.0 if self.families_data[family_id]['is_important'] else 0.0
        
        # Family urgency
        family_urgency = np.zeros(self.n_families)
        for idx, family_id in enumerate(self.family_ids):
            urgency = self._calculate_urgency(family_id)
            family_urgency[idx] = 1.0 - min(urgency / 90.0, 1.0)
        
        # Next job ready
        next_job_ready = np.zeros(self.n_families)
        for valid_action in self.valid_actions:
            family_idx = self.family_idx_map[valid_action[0]]
            next_job_ready[family_idx] = 1.0
        
        # Time progress
        time_progress = min(self.current_step / self.max_episode_steps, 1.0)
        
        # Concatenate
        obs = np.concatenate([
            normalized_loads,
            family_progress,
            family_importance,
            family_urgency,
            next_job_ready,
            [time_progress]
        ])
        
        return obs.astype(np.float32)
    
    def _step_impl(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action in environment."""
        reward = 0.0
        info = {}
        
        # Validate action
        if action >= len(self.valid_actions):
            reward = -20.0
            info['invalid_action'] = True
            self.current_step += 1
            
            obs = self._get_observation()
            terminated = False
            truncated = self.current_step >= self.max_episode_steps
            
            return obs, reward, terminated, truncated, info
        
        # Get selected job
        family_id, task_idx, task = self.valid_actions[action]
        family = self.families_data[family_id]
        
        # Track if PPO violated importance ordering
        selected_important = family['is_important']
        important_available = any(
            self.families_data[fid]['is_important'] 
            for fid, _, _ in self.valid_actions
        )
        
        if not selected_important and important_available:
            self.importance_violations += 1
            info['importance_violation'] = True
            
            # Check if justified by urgency
            urgency = self._calculate_urgency(family_id)
            if urgency < 7:  # Very urgent
                self.urgency_justified += 1
                info['urgency_justified'] = True
        
        # Choose machine
        capable_machines = task['capable_machines']
        machine_loads_on_capable = [
            self.machine_loads[m] for m in capable_machines
        ]
        best_machine_idx = np.argmin(machine_loads_on_capable)
        selected_machine = capable_machines[best_machine_idx]
        
        # Schedule job
        processing_time = task['processing_time']
        start_time = self.machine_loads[selected_machine]
        end_time = start_time + processing_time
        
        # Update state
        self.machine_loads[selected_machine] = end_time
        self.completed_tasks[family_id].add(task['sequence'])
        self.current_time = max(self.current_time, end_time)
        self.episode_makespan = max(self.episode_makespan, end_time)
        self.current_step += 1
        
        # Calculate rewards
        
        # Base completion
        reward += 10.0
        
        # Importance bonus (simple and clear)
        if family['is_important']:
            reward += 20.0  # Strong signal for important jobs
        
        # Load balancing
        load_variance = np.var(self.machine_loads)
        max_variance = (np.mean(self.machine_loads) ** 2) * self.n_machines
        if max_variance > 0:
            balance_score = 1.0 - (load_variance / max_variance)
            reward += balance_score * 5.0
        
        # Urgency bonus
        urgency = self._calculate_urgency(family_id)
        if urgency < 7:
            reward += 10.0
        elif urgency < 14:
            reward += 5.0
        
        # Time penalty
        reward -= 0.1
        
        # Store info
        info['scheduled_job'] = f"{family_id}-{task['sequence']}"
        info['on_machine'] = selected_machine
        info['processing_time'] = processing_time
        info['is_important'] = family['is_important']
        info['urgency_days'] = urgency
        info['family_progress'] = len(self.completed_tasks[family_id]) / len(family['tasks'])
        
        # Check completion
        total_completed = sum(len(completed) for completed in self.completed_tasks.values())
        all_done = total_completed >= self.n_jobs
        
        if all_done:
            reward += 50.0
            
            # Efficiency bonus
            total_work = sum(
                task['processing_time'] 
                for family in self.families_data.values()
                for task in family['tasks']
            )
            theoretical_min = total_work / self.n_machines
            efficiency = theoretical_min / self.episode_makespan
            reward += efficiency * 30.0
            
            info['makespan'] = self.episode_makespan
            info['efficiency'] = efficiency
            info['all_tasks_completed'] = True
            info['importance_violations'] = self.importance_violations
            info['urgency_justified'] = self.urgency_justified
        
        # Update valid actions (shuffled)
        self.valid_actions = self._get_valid_actions()
        
        # Get new observation
        obs = self._get_observation()
        
        terminated = all_done
        truncated = self.current_step >= self.max_episode_steps
        
        return obs, reward, terminated, truncated, info
    
    def get_action_mask(self) -> np.ndarray:
        """Get mask of valid actions."""
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[:len(self.valid_actions)] = True
        return mask
    
    def render(self, mode: str = 'human'):
        """Render current environment state."""
        if mode == 'human':
            print(f"\n=== Step {self.current_step}/{self.max_episode_steps} ===")
            print(f"Current time: {self.current_time:.1f}h")
            print(f"Machine loads: {self.machine_loads}")
            
            # Stats
            print(f"\nImportance violations: {self.importance_violations}")
            print(f"Urgency justified: {self.urgency_justified}")
            
            # Valid actions
            print(f"\nValid actions: {len(self.valid_actions)}")
            if self.valid_actions and len(self.valid_actions) >= 3:
                print("Sample (random order):")
                for i in range(min(3, len(self.valid_actions))):
                    fid, _, task = self.valid_actions[i]
                    family = self.families_data[fid]
                    imp = "❗" if family['is_important'] else "  "
                    urgency = self._calculate_urgency(fid)
                    print(f"{imp}{i}. {fid}-{task['sequence']} ({urgency:.0f} days)")
    
    def _calculate_reward(self, action: int, valid_action: bool) -> float:
        """Calculate reward (implemented in _step_impl)."""
        return 0.0
    
    def _is_done(self) -> bool:
        """Check if episode is complete."""
        total_completed = sum(len(completed) for completed in self.completed_tasks.values())
        return total_completed >= self.n_jobs or self.current_step >= self.max_episode_steps