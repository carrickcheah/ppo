"""
Medium scheduling environment WITHOUT sorting constraints.
Allows PPO to explore all valid actions freely and potentially discover better strategies.
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


class MediumUnconstrainedSchedulingEnv(BaseSchedulingEnv):
    """
    Medium complexity scheduling environment without pre-sorting.
    
    Key differences from hybrid version:
    - No sorting of valid actions by priority/LCD
    - PPO sees ALL valid actions in random order
    - Larger action space to handle variable valid actions
    - PPO learns priority/urgency trade-offs through rewards only
    
    This tests if PPO can discover better strategies than rigid rules.
    """
    
    def __init__(self,
                 n_machines: int = 10,
                 data_file: str = '/Users/carrickcheah/Project/ppo/app/data/parsed_production_data.json',
                 max_episode_steps: int = 500,
                 max_valid_actions: int = 200,  # Maximum possible valid actions
                 seed: Optional[int] = None):
        """
        Initialize unconstrained environment.
        
        Args:
            n_machines: Number of machines
            data_file: Path to parsed production data
            max_episode_steps: Maximum steps per episode
            max_valid_actions: Maximum possible valid actions at any step
            seed: Random seed
        """
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
        
        # Define observation space (same as hybrid)
        # [machine_loads(10), family_progress(50), family_priorities(50), 
        #  family_urgency(50), next_job_ready(50), current_time(1)]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(n_machines + 4 * self.n_families + 1,),
            dtype=np.float32
        )
        
        # Action space: can select any valid action (up to max)
        self.action_space = spaces.Discrete(max_valid_actions)
        
        # Environment state
        self.machine_loads = None
        self.family_progress = None
        self.completed_tasks = None
        self.current_step = None
        self.current_time = None
        self.valid_actions = None
        self.episode_makespan = None
        
        # Parse families into indexed structure
        self.family_ids = list(self.families_data.keys())
        self.family_idx_map = {fid: idx for idx, fid in enumerate(self.family_ids)}
        
        # Track statistics
        self.priority_violations = 0  # Times PPO chose lower priority over higher
        self.urgency_wins = 0  # Times urgent job was prioritized
        
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
        
        # Reset time
        self.current_step = 0
        self.current_time = 0.0
        self.episode_makespan = 0.0
        
        # Reset statistics
        self.priority_violations = 0
        self.urgency_wins = 0
        
        # Get initial valid actions WITHOUT sorting
        self.valid_actions = self._get_valid_actions()
        
        return self._get_observation()
    
    def _get_valid_actions(self) -> List[Tuple[str, int, Dict]]:
        """
        Get valid actions WITHOUT sorting.
        Returns list of (family_id, task_idx, task_info) tuples in RANDOM order.
        """
        valid = []
        
        for family_id, family_data in self.families_data.items():
            # Find next uncompleted task in sequence
            for task_idx, task in enumerate(family_data['tasks']):
                task_seq = task['sequence']
                
                # Skip if already completed
                if task_seq in self.completed_tasks[family_id]:
                    continue
                
                # Check dependencies: all lower sequences must be complete
                dependencies_met = True
                for other_task in family_data['tasks']:
                    if other_task['sequence'] < task_seq:
                        if other_task['sequence'] not in self.completed_tasks[family_id]:
                            dependencies_met = False
                            break
                
                if dependencies_met:
                    valid.append((family_id, task_idx, task))
                    break  # Only one task per family can be ready
        
        # CRITICAL CHANGE: Shuffle instead of sort!
        # This ensures PPO doesn't learn position bias
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
        """Get current observation (same as hybrid version)."""
        # Normalize machine loads
        max_load = max(self.machine_loads) if np.any(self.machine_loads > 0) else 1.0
        normalized_loads = self.machine_loads / max_load
        
        # Family progress (% of tasks completed)
        family_progress = np.zeros(self.n_families)
        for idx, family_id in enumerate(self.family_ids):
            family = self.families_data[family_id]
            total_tasks = len(family['tasks'])
            completed = len(self.completed_tasks[family_id])
            family_progress[idx] = completed / total_tasks if total_tasks > 0 else 1.0
        
        # Family priorities (normalized)
        family_priorities = np.zeros(self.n_families)
        for idx, family_id in enumerate(self.family_ids):
            priority = self.families_data[family_id]['priority']
            family_priorities[idx] = (6 - priority) / 5.0  # Invert and normalize
        
        # Family urgency (normalized days to LCD)
        family_urgency = np.zeros(self.n_families)
        for idx, family_id in enumerate(self.family_ids):
            urgency = self._calculate_urgency(family_id)
            # Normalize to [0, 1] where 1 is most urgent
            family_urgency[idx] = 1.0 - min(urgency / 90.0, 1.0)
        
        # Next job ready (binary)
        next_job_ready = np.zeros(self.n_families)
        for valid_action in self.valid_actions:
            family_idx = self.family_idx_map[valid_action[0]]
            next_job_ready[family_idx] = 1.0
        
        # Current time progress
        time_progress = min(self.current_step / self.max_episode_steps, 1.0)
        
        # Concatenate all features
        obs = np.concatenate([
            normalized_loads,
            family_progress,
            family_priorities,
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
            # Invalid action
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
        
        # Track if PPO violated priority ordering
        selected_priority = family['priority']
        best_available_priority = min(
            self.families_data[fid]['priority'] 
            for fid, _, _ in self.valid_actions
        )
        if selected_priority > best_available_priority:
            self.priority_violations += 1
            info['priority_violation'] = True
            
            # Check if it was due to urgency
            urgency = self._calculate_urgency(family_id)
            if urgency < 14:  # Less than 2 weeks
                self.urgency_wins += 1
                info['urgency_override'] = True
        
        # Choose machine (same logic as hybrid)
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
        
        # Calculate rewards (same as hybrid to maintain consistency)
        
        # Base completion reward
        reward += 10.0
        
        # Priority bonus (higher for higher priority jobs)
        priority_bonus = (6 - family['priority']) * 3.0
        reward += priority_bonus
        
        # Load balancing reward
        load_variance = np.var(self.machine_loads)
        max_variance = (np.mean(self.machine_loads) ** 2) * self.n_machines
        if max_variance > 0:
            balance_score = 1.0 - (load_variance / max_variance)
            reward += balance_score * 5.0
        
        # Urgency bonus
        urgency = self._calculate_urgency(family_id)
        if urgency < 7:  # Less than a week
            reward += 10.0
        elif urgency < 14:  # Less than 2 weeks
            reward += 5.0
        
        # Time penalty to encourage efficiency
        reward -= 0.1
        
        # Store info
        info['scheduled_job'] = f"{family_id}-{task['sequence']}"
        info['on_machine'] = selected_machine
        info['processing_time'] = processing_time
        info['priority'] = family['priority']
        info['urgency_days'] = urgency
        info['family_progress'] = len(self.completed_tasks[family_id]) / len(family['tasks'])
        
        # Check if all tasks completed
        total_completed = sum(len(completed) for completed in self.completed_tasks.values())
        all_done = total_completed >= self.n_jobs
        
        if all_done:
            # Episode completion bonus
            reward += 50.0
            
            # Makespan bonus (compare to simple bound)
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
            info['priority_violations'] = self.priority_violations
            info['urgency_wins'] = self.urgency_wins
        
        # Update valid actions for next step (shuffled)
        self.valid_actions = self._get_valid_actions()
        
        # Get new observation
        obs = self._get_observation()
        
        # Episode ends when all tasks done or max steps
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
            
            # Family progress
            print("\nFamily progress:")
            for idx, family_id in enumerate(self.family_ids[:5]):  # First 5
                family = self.families_data[family_id]
                progress = len(self.completed_tasks[family_id]) / len(family['tasks'])
                print(f"  {family_id} (P{family['priority']}): {progress:.0%}")
            
            # Valid actions
            print(f"\nValid actions available: {len(self.valid_actions)}")
            print(f"Priority violations so far: {self.priority_violations}")
            print(f"Urgency overrides: {self.urgency_wins}")
            
            if self.valid_actions and len(self.valid_actions) >= 3:
                print("Sample of available actions (random order):")
                for i in range(min(3, len(self.valid_actions))):
                    fid, _, task = self.valid_actions[i]
                    family = self.families_data[fid]
                    urgency = self._calculate_urgency(fid)
                    print(f"  {i}. {fid}-{task['sequence']} (P{family['priority']}, {urgency:.0f} days to LCD)")
    
    def _calculate_reward(self, action: int, valid_action: bool) -> float:
        """Calculate reward (implemented in _step_impl)."""
        return 0.0
    
    def _is_done(self) -> bool:
        """Check if episode is complete."""
        total_completed = sum(len(completed) for completed in self.completed_tasks.values())
        return total_completed >= self.n_jobs or self.current_step >= self.max_episode_steps