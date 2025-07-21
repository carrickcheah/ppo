"""
PPO Scheduler Module

This module implements the actual scheduling logic using the trained PPO model
and the FullProductionEnv environment.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np

from .models import Job, ScheduledJob, Machine

logger = logging.getLogger(__name__)

# Try to import the full production environment
try:
    from ..environments.full_production_env import FullProductionEnv
    from ..environments.break_time_constraints import BreakTimeConstraints
    FULL_ENV_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import FullProductionEnv: {e}")
    FULL_ENV_AVAILABLE = False
    FullProductionEnv = None
    BreakTimeConstraints = None


class PPOScheduler:
    """
    Scheduler that uses the trained PPO model to generate production schedules.
    """
    
    def __init__(self, ppo_model, env_config: Dict[str, Any] = None):
        """
        Initialize the scheduler with a trained PPO model.
        
        Args:
            ppo_model: Trained Stable Baselines3 PPO model
            env_config: Configuration for the environment
        """
        if not FULL_ENV_AVAILABLE:
            raise ImportError("FullProductionEnv is not available. Cannot use PPO scheduler.")
            
        self.model = ppo_model
        self.env_config = env_config or {
            'n_machines': 152,
            'use_break_constraints': True,
            'use_holiday_constraints': True,
            'state_compression': 'hierarchical'
        }
        self.break_constraints = BreakTimeConstraints() if BreakTimeConstraints else None
        
    def create_environment(self, jobs: List[Job], machines: List[Machine], 
                         schedule_start: datetime) -> FullProductionEnv:
        """
        Create a FullProductionEnv instance with the given jobs and machines.
        
        Args:
            jobs: List of jobs to schedule
            machines: List of available machines
            schedule_start: Start time for the schedule
            
        Returns:
            Configured FullProductionEnv instance
        """
        # Convert jobs to environment format
        env_jobs = []
        for job in jobs:
            env_job = {
                'id': job.job_id,
                'family_id': job.family_id,
                'sequence': job.sequence,
                'processing_time': job.processing_time,
                'machine_types': job.machine_types,
                'is_important': job.is_important,
                'lcd_date': job.lcd_date,
                'setup_time': job.setup_time or 0.3
            }
            env_jobs.append(env_job)
        
        # Convert machines to environment format
        env_machines = []
        for machine in machines:
            env_machine = {
                'id': machine.machine_id,
                'name': machine.machine_name,
                'type': machine.machine_type,
                'initial_load': machine.current_load or 0.0
            }
            env_machines.append(env_machine)
        
        # Create environment configuration
        config = self.env_config.copy()
        config['jobs'] = env_jobs
        config['machines'] = env_machines
        config['schedule_start'] = schedule_start
        config['n_machines'] = len(machines)
        
        # Create and return environment
        env = FullProductionEnv(**config)
        return env
    
    def schedule(self, jobs: List[Job], machines: List[Machine], 
                schedule_start: datetime) -> Tuple[List[ScheduledJob], Dict[str, Any]]:
        """
        Generate a schedule using the PPO model.
        
        Args:
            jobs: List of jobs to schedule
            machines: List of available machines
            schedule_start: Start time for the schedule
            
        Returns:
            Tuple of (scheduled_jobs, metrics)
        """
        logger.info(f"Starting PPO scheduling for {len(jobs)} jobs on {len(machines)} machines")
        
        # Create environment
        env = self.create_environment(jobs, machines, schedule_start)
        
        # Reset environment
        obs, info = env.reset()
        
        # Run PPO model to generate schedule
        scheduled_jobs = []
        done = False
        step_count = 0
        total_reward = 0
        
        while not done and step_count < env.max_steps:
            # Get action from PPO model
            action, _states = self.model.predict(obs, deterministic=True)
            
            # Take action in environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1
            
            # Check if a job was scheduled in this step
            if 'scheduled_job' in info:
                job_info = info['scheduled_job']
                
                # Convert to ScheduledJob
                start_hours = job_info['start_time']
                end_hours = job_info['end_time']
                
                # Convert hours to datetime
                start_dt = schedule_start + timedelta(hours=start_hours)
                end_dt = schedule_start + timedelta(hours=end_hours)
                
                scheduled_job = ScheduledJob(
                    job_id=job_info['job_id'],
                    machine_id=job_info['machine_id'],
                    machine_name=job_info['machine_name'],
                    start_time=start_hours,
                    end_time=end_hours,
                    start_datetime=start_dt,
                    end_datetime=end_dt,
                    setup_time_included=job_info.get('setup_time', 0.0)
                )
                scheduled_jobs.append(scheduled_job)
        
        # Extract metrics from final info
        metrics = {
            'makespan': info.get('makespan', 0.0),
            'total_jobs': len(jobs),
            'scheduled_jobs': len(scheduled_jobs),
            'completion_rate': (len(scheduled_jobs) / len(jobs) * 100) if jobs else 0.0,
            'average_utilization': info.get('avg_utilization', 0.0) * 100,
            'total_setup_time': info.get('total_setup_time', 0.0),
            'important_jobs_on_time': self._calculate_on_time_rate(scheduled_jobs, jobs),
            'total_reward': total_reward,
            'steps_taken': step_count
        }
        
        logger.info(f"PPO scheduling complete: {len(scheduled_jobs)}/{len(jobs)} jobs scheduled, "
                   f"makespan={metrics['makespan']:.1f}h")
        
        return scheduled_jobs, metrics
    
    def _calculate_on_time_rate(self, scheduled_jobs: List[ScheduledJob], 
                               original_jobs: List[Job]) -> float:
        """
        Calculate the percentage of important jobs completed on time.
        
        Args:
            scheduled_jobs: List of scheduled jobs
            original_jobs: Original job list with LCD dates
            
        Returns:
            Percentage of important jobs meeting their deadlines
        """
        # Create job lookup
        job_dict = {job.job_id: job for job in original_jobs}
        
        important_total = 0
        important_on_time = 0
        
        for sched_job in scheduled_jobs:
            if sched_job.job_id in job_dict:
                orig_job = job_dict[sched_job.job_id]
                if orig_job.is_important:
                    important_total += 1
                    if sched_job.end_datetime <= orig_job.lcd_date:
                        important_on_time += 1
        
        if important_total == 0:
            return 100.0
        
        return (important_on_time / important_total) * 100


class MockScheduler:
    """
    Mock scheduler for testing when PPO model is not available.
    """
    
    def schedule(self, jobs: List[Job], machines: List[Machine], 
                schedule_start: datetime) -> Tuple[List[ScheduledJob], Dict[str, Any]]:
        """
        Generate a mock schedule using a simple first-fit algorithm.
        
        Args:
            jobs: List of jobs to schedule
            machines: List of available machines
            schedule_start: Start time for the schedule
            
        Returns:
            Tuple of (scheduled_jobs, metrics)
        """
        logger.info("Using mock scheduler (PPO not connected)")
        
        # Simple first-fit scheduling
        machine_loads = {m.machine_id: 0.0 for m in machines}
        scheduled_jobs = []
        
        for job in jobs:
            # Find compatible machine with lowest load
            compatible_machines = [
                m for m in machines 
                if m.machine_type in job.machine_types
            ]
            
            if not compatible_machines:
                logger.warning(f"No compatible machines for job {job.job_id}")
                continue
            
            # Select machine with minimum load
            selected_machine = min(
                compatible_machines, 
                key=lambda m: machine_loads[m.machine_id]
            )
            
            # Schedule job
            start_time = machine_loads[selected_machine.machine_id]
            end_time = start_time + job.processing_time + (job.setup_time or 0.3)
            
            start_dt = schedule_start + timedelta(hours=start_time)
            end_dt = schedule_start + timedelta(hours=end_time)
            
            scheduled_job = ScheduledJob(
                job_id=job.job_id,
                machine_id=selected_machine.machine_id,
                machine_name=selected_machine.machine_name,
                start_time=start_time,
                end_time=end_time,
                start_datetime=start_dt,
                end_datetime=end_dt,
                setup_time_included=job.setup_time or 0.3
            )
            
            scheduled_jobs.append(scheduled_job)
            machine_loads[selected_machine.machine_id] = end_time
        
        # Calculate metrics
        makespan = max(machine_loads.values()) if machine_loads else 0.0
        
        metrics = {
            'makespan': makespan,
            'total_jobs': len(jobs),
            'scheduled_jobs': len(scheduled_jobs),
            'completion_rate': (len(scheduled_jobs) / len(jobs) * 100) if jobs else 0.0,
            'average_utilization': 65.0,  # Mock value
            'total_setup_time': len(scheduled_jobs) * 0.3,
            'important_jobs_on_time': 85.0,  # Mock value
            'algorithm_used': 'mock_first_fit'
        }
        
        return scheduled_jobs, metrics