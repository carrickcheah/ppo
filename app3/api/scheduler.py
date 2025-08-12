"""
PPO scheduling service using Stable Baselines3 models.
"""

import os
import sys
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from src.environments.scheduling_env import SchedulingEnv
from api.flexible_scheduler import FlexibleScheduler
from src.utils.config import get_reward_config, get_environment_config
from api.models import (
    JobTask, MachineTask, ScheduleStatistics,
    DatasetType, ModelType
)


class PPOSchedulerService:
    """Service for scheduling jobs using trained PPO models."""
    
    def __init__(self):
        """Initialize the scheduler service."""
        self.models_cache = {}
        self.base_path = Path("/Users/carrickcheah/Project/ppo/app3")
        
    def get_model_path(self, model_name: str) -> str:
        """Get the path to a model checkpoint by name."""
        # Check if it's a direct path to best_model.zip
        if "/" in model_name:
            # Nested model like "sb3_500k/stage_1"
            return str(self.base_path / f"checkpoints/{model_name}/best_model.zip")
        else:
            # Direct model name
            return str(self.base_path / f"checkpoints/{model_name}/best_model.zip")
    
    def get_dataset_path(self, dataset_type: DatasetType) -> str:
        """Get the path to a dataset."""
        dataset_paths = {
            DatasetType.JOBS_10: "data/10_jobs.json",
            DatasetType.JOBS_20: "data/20_jobs.json",
            DatasetType.JOBS_40: "data/40_jobs.json",
            DatasetType.JOBS_60: "data/60_jobs.json",
            DatasetType.JOBS_80: "data/80_jobs.json",
            DatasetType.JOBS_100: "data/100_jobs.json",
            DatasetType.JOBS_150: "data/150_jobs.json",
            DatasetType.JOBS_180: "data/180_jobs.json",
            DatasetType.JOBS_200: "data/200_jobs.json",
            DatasetType.JOBS_250: "data/250_jobs.json",
            DatasetType.JOBS_300: "data/300_jobs.json",
            DatasetType.JOBS_330: "data/330_jobs.json",
            DatasetType.JOBS_380: "data/380_jobs.json",
            DatasetType.JOBS_400: "data/400_jobs.json",
            DatasetType.JOBS_430: "data/430_jobs.json",
            DatasetType.JOBS_450: "data/450_jobs.json",
            DatasetType.JOBS_500: "data/500_jobs.json"
        }
        return str(self.base_path / dataset_paths[dataset_type])
    
    def load_model(self, model_name: str) -> PPO:
        """Load a PPO model from checkpoint."""
        if model_name not in self.models_cache:
            model_path = self.get_model_path(model_name)
            
            # Check if model exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            # Load SB3 models (assuming all auto-detected models are SB3)
            self.models_cache[model_name] = PPO.load(model_path)
            
        return self.models_cache[model_name]
    
    def schedule_jobs(
        self,
        dataset_type: DatasetType,
        model_name: str,
        deterministic: bool = True,
        max_steps: int = 10000
    ) -> Dict:
        """
        Schedule jobs using a PPO model with flexible observation handling.
        
        Returns:
            Dict containing jobs, machines, and statistics
        """
        start_time = time.time()
        
        # Get model path
        model_path = self.get_model_path(model_name)
        
        # Create environment
        dataset_path = self.get_dataset_path(dataset_type)
        # Load configs
        env_cfg = get_environment_config()
        reward_cfg = get_reward_config()

        env = SchedulingEnv(
            snapshot_path=dataset_path,
            max_steps=env_cfg.get("max_steps_per_episode", max_steps),
            planning_horizon=float(env_cfg.get("planning_horizon", 720.0)),
            reward_config=reward_cfg
        )
        
        # Use flexible scheduler for any dataset size
        # The expected size is 2402 for models trained on 100 jobs (327 tasks)
        # Formula: (n_tasks * 6) + (n_machines * 3) + 5
        # For 100 jobs: (327 * 6) + (145 * 3) + 5 = 1962 + 435 + 5 = 2402
        expected_size = 2402
        
        # Create flexible scheduler
        scheduler = FlexibleScheduler(model_path, expected_size)
        
        # Run scheduling with padding
        schedule, info = scheduler.schedule_with_padding(
            env, max_steps, deterministic
        )
        
        inference_time = time.time() - start_time
        
        # Process results
        jobs = self._process_job_tasks(schedule, env)
        machines = self._process_machine_tasks(schedule, env)
        statistics = self._calculate_statistics(
            schedule, env, info, inference_time
        )
        
        return {
            "jobs": jobs,
            "machines": machines,
            "statistics": statistics
        }
    
    def _process_job_tasks(self, schedule: Dict, env: SchedulingEnv) -> List[JobTask]:
        """Process scheduled tasks into job format for visualization."""
        jobs = []
        
        for task_data in schedule.get('tasks', []):
            # Get task info from data
            family_id = task_data['family_id']
            sequence = task_data['sequence']
            
            # Find the task object
            task = None
            for t in env.loader.tasks:
                if t.family_id == family_id and t.sequence == sequence:
                    task = t
                    break
            
            if not task:
                continue
                
            family = env.loader.families[task.family_id]
            
            # Calculate deadline status and color
            lcd_hours = family.lcd_days_remaining * 24
            days_to_deadline = (lcd_hours - task_data['end']) / 24
            
            if days_to_deadline < 0:
                color = '#FF0000'  # Red - Late
            elif days_to_deadline < 1:
                color = '#FFA500'  # Orange - Warning (<24h)
            elif days_to_deadline < 3:
                color = '#FFFF00'  # Yellow - Caution (<72h)
            else:
                color = '#00FF00'  # Green - OK (>72h)
            
            # Extract sequence info from process name
            process_parts = task.process_name.split('_')
            if len(process_parts) > 1 and '/' in process_parts[-1]:
                seq_info = process_parts[-1]
                current_seq, total_seq = seq_info.split('/')
            else:
                # Count total sequences for this family
                total_seq = len([t for t in env.loader.tasks if t.family_id == task.family_id])
                current_seq = task.sequence
            
            # Create shorter, cleaner task label
            # Just use family ID and sequence info
            if '/' in task.process_name:
                # Extract just the sequence part
                seq_part = task.process_name.split('_')[-1] if '_' in task.process_name else task.process_name
                task_label = f"{task.family_id}_{seq_part}"
            else:
                task_label = f"{task.family_id}_seq{task.sequence}"
            
            jobs.append(JobTask(
                job_id=task.family_id,
                task_label=task_label,
                sequence=task.sequence,
                start=task_data['start'],
                end=task_data['end'],
                duration=task_data['processing_time'],
                machine=task_data['machine'],
                color=color,
                lcd_hours=lcd_hours,
                days_to_deadline=days_to_deadline,
                process_name=task.process_name
            ))
        
        # Sort by family and sequence for proper display
        jobs.sort(key=lambda x: (x.job_id, x.sequence))
        
        return jobs
    
    def _process_machine_tasks(self, schedule: Dict, env: SchedulingEnv) -> List[MachineTask]:
        """Process scheduled tasks into machine format for visualization."""
        machine_dict = {}
        
        # Group tasks by machine
        for task_data in schedule.get('tasks', []):
            machine_id = task_data['machine']
            
            if machine_id not in machine_dict:
                # Get machine name from loader
                # Machines is a list of dicts or strings
                machine_name = f"Machine_{machine_id}"
                if isinstance(env.loader.machines, list) and len(env.loader.machines) > 0:
                    # If machines are strings, use the machine_id directly
                    if isinstance(env.loader.machines[0], str):
                        machine_name = str(machine_id)
                    # If machines are dicts with name field
                    elif isinstance(env.loader.machines[0], dict):
                        for m in env.loader.machines:
                            if str(m.get('machine_id', m.get('id', ''))) == str(machine_id):
                                machine_name = m.get('name', m.get('machine_name', f"Machine_{machine_id}"))
                                break
                
                machine_dict[machine_id] = {
                    'machine_id': machine_id,
                    'machine_name': machine_name,
                    'tasks': [],
                    'total_busy_time': 0.0
                }
            
            # Get task info from data
            family_id = task_data['family_id']
            sequence = task_data['sequence']
            
            # Find the task object
            task = None
            for t in env.loader.tasks:
                if t.family_id == family_id and t.sequence == sequence:
                    task = t
                    break
            
            if not task:
                continue
                
            family = env.loader.families[task.family_id]
            
            # Calculate color
            lcd_hours = family.lcd_days_remaining * 24
            days_to_deadline = (lcd_hours - task_data['end']) / 24
            
            if days_to_deadline < 0:
                color = '#FF0000'
            elif days_to_deadline < 1:
                color = '#FFA500'
            elif days_to_deadline < 3:
                color = '#FFFF00'
            else:
                color = '#00FF00'
            
            task_label = f"{task.family_id}_{task.process_name}"
            
            machine_dict[machine_id]['tasks'].append(JobTask(
                job_id=task.family_id,
                task_label=task_label,
                sequence=task.sequence,
                start=task_data['start'],
                end=task_data['end'],
                duration=task_data['processing_time'],
                machine=machine_id,
                color=color,
                lcd_hours=lcd_hours,
                days_to_deadline=days_to_deadline,
                process_name=task.process_name
            ))
            
            machine_dict[machine_id]['total_busy_time'] += task_data['processing_time']
        
        # Calculate utilization and create MachineTask objects
        machines = []
        makespan = max([t['end'] for t in schedule.get('tasks', [])], default=0)
        
        for machine_data in machine_dict.values():
            utilization = (
                machine_data['total_busy_time'] / makespan * 100
                if makespan > 0 else 0
            )
            
            machines.append(MachineTask(
                machine_id=machine_data['machine_id'],
                machine_name=machine_data['machine_name'],
                tasks=sorted(machine_data['tasks'], key=lambda x: x.start),
                utilization=utilization,
                total_busy_time=machine_data['total_busy_time']
            ))
        
        return sorted(machines, key=lambda x: x.machine_name)
    
    def _calculate_statistics(
        self,
        schedule: Dict,
        env: SchedulingEnv,
        info: Dict,
        inference_time: float
    ) -> ScheduleStatistics:
        """Calculate scheduling performance statistics."""
        tasks = schedule.get('tasks', [])
        
        if not tasks:
            return ScheduleStatistics(
                total_tasks=info.get('total_tasks', 0),
                scheduled_tasks=0,
                completion_rate=0.0,
                on_time_tasks=0,
                late_tasks=0,
                on_time_rate=0.0,
                average_tardiness=0.0,
                makespan=0.0,
                machine_utilization=0.0,
                total_reward=env.episode_reward,
                inference_time=inference_time
            )
        
        # Calculate metrics
        total_tasks = info.get('total_tasks', len(env.loader.tasks))
        scheduled_tasks = len(tasks)
        completion_rate = (scheduled_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        # Count on-time and late tasks
        on_time_tasks = 0
        late_tasks = 0
        total_tardiness = 0.0
        
        for task_data in tasks:
            # Get task info from data
            family_id = task_data['family_id']
            sequence = task_data['sequence']
            
            # Find the task object
            task = None
            for t in env.loader.tasks:
                if t.family_id == family_id and t.sequence == sequence:
                    task = t
                    break
            
            if not task:
                continue
                
            family = env.loader.families[task.family_id]
            
            lcd_hours = family.lcd_days_remaining * 24
            tardiness = max(0, task_data['end'] - lcd_hours)
            
            if tardiness > 0:
                late_tasks += 1
                total_tardiness += tardiness
            else:
                on_time_tasks += 1
        
        on_time_rate = (on_time_tasks / scheduled_tasks * 100) if scheduled_tasks > 0 else 0
        average_tardiness = total_tardiness / late_tasks if late_tasks > 0 else 0
        
        # Calculate makespan and utilization
        makespan = max([t['end'] for t in tasks], default=0)
        
        total_processing = sum(t['processing_time'] for t in tasks)
        n_machines = len(env.loader.machines)
        machine_utilization = (
            (total_processing / n_machines / makespan * 100)
            if makespan > 0 and n_machines > 0 else 0
        )
        
        return ScheduleStatistics(
            total_tasks=total_tasks,
            scheduled_tasks=scheduled_tasks,
            completion_rate=completion_rate,
            on_time_tasks=on_time_tasks,
            late_tasks=late_tasks,
            on_time_rate=on_time_rate,
            average_tardiness=average_tardiness,
            makespan=makespan,
            machine_utilization=machine_utilization,
            total_reward=env.episode_reward,
            inference_time=inference_time
        )


# Global scheduler instance
scheduler_service = PPOSchedulerService()