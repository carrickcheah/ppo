"""
Gym-compatible scheduling environment for PPO training.
Manages state, actions, rewards, and episode logic.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Add parent directory to path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.snapshot_loader import SnapshotLoader
from environments.constraint_validator import ConstraintValidator
from environments.reward_calculator import RewardCalculator

logger = logging.getLogger(__name__)


class SchedulingEnv(gym.Env):
    """Production scheduling environment for PPO training."""
    
    def __init__(
        self,
        snapshot_path: str,
        max_steps: int = 1000,
        planning_horizon: float = 720.0,  # 30 days in hours
        reward_config: Optional[Dict] = None
    ):
        """
        Initialize scheduling environment.
        
        Args:
            snapshot_path: Path to JSON snapshot file
            max_steps: Maximum steps per episode
            planning_horizon: Planning horizon in hours
            reward_config: Optional reward configuration
        """
        super().__init__()
        
        # Load data
        self.loader = SnapshotLoader(snapshot_path)
        self.validator = ConstraintValidator(self.loader)
        
        # Initialize reward calculator
        reward_config = reward_config or {}
        self.reward_calc = RewardCalculator(**reward_config)
        
        # Environment parameters
        self.max_steps = max_steps
        self.planning_horizon = planning_horizon
        
        # Define action and observation spaces
        self.n_tasks = len(self.loader.tasks)
        self.n_machines = len(self.loader.machines)
        
        # Action space: Select which task to schedule next
        self.action_space = spaces.Discrete(self.n_tasks)
        
        # Observation space: Concatenated features
        # Task features (n_tasks * 6) + Machine features (n_machines * 3) + Global features (5)
        obs_dim = (self.n_tasks * 6) + (self.n_machines * 3) + 5
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        
        # Episode state
        self.current_time = 0.0
        self.steps = 0
        self.scheduled_tasks = set()
        self.machine_schedules = {}  # machine -> [(start, end), ...]
        self.task_schedules = {}  # task_idx -> (start, end, machine)
        
        # Metrics
        self.episode_reward = 0.0
        self.tasks_completed = 0
        
        logger.info(f"Environment initialized with {self.n_tasks} tasks and {self.n_machines} machines")
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment for new episode.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset episode state
        self.current_time = 0.0
        self.steps = 0
        self.scheduled_tasks.clear()
        self.machine_schedules.clear()
        self.task_schedules.clear()
        
        # Reset data loader state
        self.loader.reset()
        
        # Reset reward calculator
        self.reward_calc.reset()
        
        # Reset metrics
        self.episode_reward = 0.0
        self.tasks_completed = 0
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute action in environment.
        
        Args:
            action: Task index to schedule
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        self.steps += 1
        
        # Get action mask
        action_mask = self.validator.get_action_mask(
            self.current_time,
            self.machine_schedules,
            self.scheduled_tasks
        )
        
        # Validate action
        if action < 0 or action >= self.n_tasks or not action_mask[action]:
            # Invalid action - small penalty and continue
            reward = self.reward_calc.calculate_step_reward(
                None, self.current_time, self._get_utilization(), False, self.loader
            )
            obs = self._get_observation()
            info = self._get_info()
            info['action_valid'] = False
            
            # Advance time slightly to prevent infinite loops
            self.current_time += 0.1
            
            terminated = self._is_terminated()
            truncated = self.steps >= self.max_steps
            
            self.episode_reward += reward
            
            return obs, reward, terminated, truncated, info
            
        # Execute valid action
        task_scheduled = self._schedule_task(action)
        
        # Calculate reward
        reward = self.reward_calc.calculate_step_reward(
            task_scheduled,
            self.current_time,
            self._get_utilization(),
            True,
            self.loader
        )
        
        # Update time to next decision point
        self._update_time()
        
        # Get new observation
        obs = self._get_observation()
        
        # Check termination
        terminated = self._is_terminated()
        truncated = self.steps >= self.max_steps
        
        # Get info
        info = self._get_info()
        info['action_valid'] = True
        info['task_scheduled'] = task_scheduled
        
        self.episode_reward += reward
        
        return obs, reward, terminated, truncated, info
        
    def _schedule_task(self, task_idx: int) -> Dict:
        """
        Schedule a task on appropriate machine.
        
        Args:
            task_idx: Index of task to schedule
            
        Returns:
            Dict with scheduling info
        """
        task = self.loader.tasks[task_idx]
        family = self.loader.families[task.family_id]
        
        # Get available machine
        machine = self.validator.get_machine_for_task(
            task_idx, self.current_time, self.machine_schedules
        )
        
        if machine is None:
            return None  # Should not happen if validation works
            
        # Calculate start time considering BOTH machine availability AND sequence dependencies
        machine_available = self._get_machine_next_available(machine)
        sequence_available = self.current_time
        
        # Check if previous sequence must complete first
        if task.sequence > 1:
            prev_seq_key = (family.family_id, task.sequence - 1)
            if prev_seq_key in self.loader.task_by_family_seq:
                prev_task = self.loader.task_by_family_seq[prev_seq_key]
                # Find when previous task ends
                for t_idx, (t_start, t_end, t_machine) in self.task_schedules.items():
                    if self.loader.tasks[t_idx] == prev_task:
                        sequence_available = t_end
                        break
        
        # Task can only start when BOTH machine is free AND sequence is ready
        start_time = max(machine_available, sequence_available, self.current_time)
        end_time = start_time + task.processing_time
        
        # Update machine schedule
        if machine not in self.machine_schedules:
            self.machine_schedules[machine] = []
        self.machine_schedules[machine].append((start_time, end_time))
        
        # Update task schedule
        self.task_schedules[task_idx] = (start_time, end_time, machine)
        
        # Mark task as scheduled
        task.is_scheduled = True
        task.start_time = start_time
        task.end_time = end_time
        task.machine_used = machine
        
        # Mark sequence as complete in family
        family.mark_sequence_complete(task.sequence)
        
        # Add to scheduled tasks
        self.scheduled_tasks.add(task_idx)
        self.tasks_completed += 1
        
        return {
            'task_idx': task_idx,
            'family_id': task.family_id,
            'sequence': task.sequence,
            'machine': machine,
            'start_time': start_time,
            'end_time': end_time,
            'processing_time': task.processing_time
        }
        
    def _get_machine_next_available(self, machine: str) -> float:
        """Get next available time for a machine."""
        if machine not in self.machine_schedules or not self.machine_schedules[machine]:
            return max(self.current_time, 0.0)
            
        # Get end time of last scheduled task
        last_end = max(end for _, end in self.machine_schedules[machine])
        return max(self.current_time, last_end)
        
    def _update_time(self):
        """Update current time to next decision point."""
        # Find next time when a machine becomes free or task becomes available
        next_times = [self.current_time + 1.0]  # Default small increment
        
        # Add machine completion times
        for machine, schedule in self.machine_schedules.items():
            for start, end in schedule:
                if end > self.current_time:
                    next_times.append(end)
                    
        # Move to nearest future time
        future_times = [t for t in next_times if t > self.current_time]
        if future_times:
            self.current_time = min(future_times)
        else:
            self.current_time += 1.0
            
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Task features
        task_features = self.loader.get_task_features(self.current_time)
        task_flat = task_features.flatten()
        
        # Machine features
        machine_features = self.loader.get_machine_features(self.machine_schedules)
        machine_flat = machine_features.flatten()
        
        # Global features
        global_features = np.array([
            self.current_time / self.planning_horizon,  # Time progress
            len(self.scheduled_tasks) / self.n_tasks,  # Completion progress
            self._get_utilization(),  # Current utilization
            self._count_urgent_unscheduled() / max(self.n_tasks, 1),  # Urgent tasks ratio
            self.steps / self.max_steps  # Step progress
        ], dtype=np.float32)
        
        # Concatenate all features
        obs = np.concatenate([task_flat, machine_flat, global_features])
        
        return obs.astype(np.float32)
        
    def _get_utilization(self) -> float:
        """Calculate current machine utilization."""
        if not self.machine_schedules:
            return 0.0
            
        total_busy = 0.0
        for schedule in self.machine_schedules.values():
            for start, end in schedule:
                total_busy += (end - start)
                
        total_possible = self.current_time * self.n_machines
        if total_possible == 0:
            return 0.0
            
        return min(total_busy / total_possible, 1.0)
        
    def _count_urgent_unscheduled(self) -> int:
        """Count urgent tasks not yet scheduled."""
        count = 0
        for task in self.loader.tasks:
            if not task.is_scheduled:
                family = self.loader.families[task.family_id]
                if family.is_urgent:
                    count += 1
        return count
        
    def _is_terminated(self) -> bool:
        """Check if episode is terminated."""
        # Episode ends when all tasks scheduled
        return len(self.scheduled_tasks) >= self.n_tasks
        
    def _get_info(self) -> Dict[str, Any]:
        """Get info dict for current state."""
        # Get action mask for next step
        action_mask = self.validator.get_action_mask(
            self.current_time,
            self.machine_schedules,
            self.scheduled_tasks
        )
        
        info = {
            'current_time': self.current_time,
            'tasks_scheduled': len(self.scheduled_tasks),
            'total_tasks': self.n_tasks,
            'utilization': self._get_utilization(),
            'episode_reward': self.episode_reward,
            'action_mask': action_mask,
            'valid_actions': action_mask.sum(),
            'urgent_unscheduled': self._count_urgent_unscheduled()
        }
        
        return info
        
    def render(self):
        """Render environment state (text-based)."""
        print(f"\n=== Step {self.steps} | Time: {self.current_time:.1f}h ===")
        print(f"Tasks: {len(self.scheduled_tasks)}/{self.n_tasks} scheduled")
        print(f"Utilization: {self._get_utilization():.2%}")
        print(f"Episode Reward: {self.episode_reward:.2f}")
        
        # Show recent schedules
        recent = sorted(self.task_schedules.items(), key=lambda x: x[1][0])[-5:]
        if recent:
            print("\nRecent schedules:")
            for task_idx, (start, end, machine) in recent:
                task = self.loader.tasks[task_idx]
                print(f"  {task.family_id}-{task.sequence} on {machine}: {start:.1f}-{end:.1f}")
                
    def get_final_schedule(self) -> Dict:
        """Get final schedule for visualization."""
        schedule = {
            'tasks': [],
            'machines': [],
            'metrics': {}
        }
        
        # Add task schedules
        for task_idx, (start, end, machine) in self.task_schedules.items():
            task = self.loader.tasks[task_idx]
            family = self.loader.families[task.family_id]
            
            schedule['tasks'].append({
                'task_id': f"{task.family_id}-{task.sequence}",
                'family_id': task.family_id,
                'sequence': task.sequence,
                'machine': machine,
                'start': start,
                'end': end,
                'processing_time': task.processing_time,
                'lcd_days': family.lcd_days_remaining,
                'is_urgent': family.is_urgent
            })
            
        # Add machine schedules
        for machine, tasks in self.machine_schedules.items():
            schedule['machines'].append({
                'machine': machine,
                'tasks': tasks,
                'utilization': sum(end - start for start, end in tasks) / self.current_time if self.current_time > 0 else 0
            })
            
        # Add metrics
        schedule['metrics'] = {
            'total_time': self.current_time,
            'tasks_scheduled': len(self.scheduled_tasks),
            'total_tasks': self.n_tasks,
            'avg_utilization': self._get_utilization(),
            'episode_reward': self.episode_reward,
            **self.reward_calc.get_metrics()
        }
        
        return schedule


if __name__ == "__main__":
    # Test environment
    env = SchedulingEnv("data/10_jobs.json", max_steps=100)
    
    # Test reset
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Initial info: {info}")
    
    # Test a few steps
    done = False
    for i in range(10):
        # Get valid actions
        valid_actions = np.where(info['action_mask'])[0]
        
        if len(valid_actions) > 0:
            # Choose first valid action
            action = valid_actions[0]
            print(f"\nStep {i}: Taking action {action}")
            
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")
            
            if terminated or truncated:
                break
        else:
            print(f"\nStep {i}: No valid actions available")
            break
            
    # Get final schedule
    schedule = env.get_final_schedule()
    print(f"\nFinal metrics: {schedule['metrics']}")