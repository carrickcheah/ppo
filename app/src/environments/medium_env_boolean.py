"""
Medium scheduling environment with boolean importance (constrained version).
Sorts by importance first (True before False), then by LCD date.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
import gymnasium as gym
from gymnasium import spaces
import json
from datetime import datetime
import logging

from .base_env import BaseSchedulingEnv

logger = logging.getLogger(__name__)


class MediumBooleanSchedulingEnv(BaseSchedulingEnv):
    """
    Medium complexity scheduling environment using boolean importance.
    - Constrained: Sorts by importance → LCD date
    - is_important: True/False (from database in production)
    
    State Space (simplified):
        - Machine loads (10 values)
        - Family progress (50 values)
        - Family importance (50 boolean values as float)
        - Family urgency (50 values)
        - Next job ready (50 values)
        Total: 211 dimensions
    """
    
    def __init__(self,
                 n_machines: int = 10,
                 data_file: str = '/Users/carrickcheah/Project/ppo/app/data/parsed_production_data_boolean.json',
                 max_episode_steps: int = 500,
                 top_k_actions: int = 10,
                 seed: Optional[int] = None):
        """Initialize boolean importance environment."""
        # Load production data
        with open(data_file, 'r') as f:
            self.families_data = json.load(f)
        
        n_jobs = sum(len(f['tasks']) for f in self.families_data.values())
        super().__init__(n_machines, n_jobs, seed)
        
        self.n_families = len(self.families_data)
        self.max_episode_steps = max_episode_steps
        self.top_k_actions = top_k_actions
        
        # Define observation space
        # [machine_loads(10), family_progress(50), family_importance(50), 
        #  family_urgency(50), next_job_ready(50), current_time(1)]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(n_machines + 4 * self.n_families + 1,),
            dtype=np.float32
        )
        
        # Action space: index into filtered valid jobs
        self.action_space = spaces.Discrete(top_k_actions)
        
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
        
        # Get initial valid actions
        self.valid_actions = self._get_valid_actions()
        
        return self._get_observation()
    
    def _get_valid_actions(self) -> List[Tuple[str, int, Dict]]:
        """
        Get valid actions respecting hard constraints.
        Sorts by: is_important (True first) → LCD urgency
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
        
        # Sort by importance (True=0, False=1 for sorting) then LCD urgency
        valid.sort(key=lambda x: (
            0 if self.families_data[x[0]]['is_important'] else 1,  # Important first
            self._calculate_urgency(x[0])                          # Then by urgency
        ))
        
        return valid
    
    def _calculate_urgency(self, family_id: str) -> float:
        """Calculate urgency score based on LCD date."""
        family = self.families_data[family_id]
        lcd_date = datetime.fromisoformat(family['lcd_date'])
        base_date = datetime(2024, 1, 15)
        
        days_to_lcd = (lcd_date - base_date).days
        return days_to_lcd
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
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
        
        # Family importance (boolean as float)
        family_importance = np.zeros(self.n_families)
        for idx, family_id in enumerate(self.family_ids):
            family_importance[idx] = 1.0 if self.families_data[family_id]['is_important'] else 0.0
        
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
            family_importance,  # Boolean importance instead of 1-5 priority
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
        
        # Choose machine (simple heuristic: least loaded)
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
        
        # Base completion reward
        reward += 10.0
        
        # Importance bonus (much simpler than 1-5 priority!)
        if family['is_important']:
            reward += 20.0  # Big bonus for important jobs
        
        # Load balancing reward
        load_variance = np.var(self.machine_loads)
        max_variance = (np.mean(self.machine_loads) ** 2) * self.n_machines
        if max_variance > 0:
            balance_score = 1.0 - (load_variance / max_variance)
            reward += balance_score * 5.0
        
        # Urgency bonus (still relevant)
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
        info['is_important'] = family['is_important']
        info['urgency_days'] = urgency
        info['family_progress'] = len(self.completed_tasks[family_id]) / len(family['tasks'])
        
        # Check if all tasks completed
        total_completed = sum(len(completed) for completed in self.completed_tasks.values())
        all_done = total_completed >= self.n_jobs
        
        if all_done:
            # Episode completion bonus
            reward += 50.0
            
            # Makespan bonus
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
        
        # Update valid actions for next step
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
        mask[:min(len(self.valid_actions), self.top_k_actions)] = True
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
                imp_marker = "❗" if family['is_important'] else "  "
                print(f"{imp_marker}{family_id}: {progress:.0%}")
            
            # Valid actions
            print(f"\nValid actions available: {len(self.valid_actions)}")
            if self.valid_actions:
                print("Top 3 options:")
                for i, (fid, _, task) in enumerate(self.valid_actions[:3]):
                    family = self.families_data[fid]
                    imp_marker = "❗" if family['is_important'] else "  "
                    urgency = self._calculate_urgency(fid)
                    print(f"{imp_marker}{i+1}. {fid}-{task['sequence']} ({urgency:.0f} days)")
    
    def _calculate_reward(self, action: int, valid_action: bool) -> float:
        """Calculate reward (implemented in _step_impl)."""
        return 0.0
    
    def _is_done(self) -> bool:
        """Check if episode is complete."""
        total_completed = sum(len(completed) for completed in self.completed_tasks.values())
        return total_completed >= self.n_jobs or self.current_step >= self.max_episode_steps