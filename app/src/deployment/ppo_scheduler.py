"""
PPO Scheduler for production deployment
Handles batch scheduling to work around environment limitations
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from stable_baselines3 import PPO
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PPOBatchScheduler:
    """
    Production scheduler that uses trained PPO model in batches.
    Works around the 172-job limitation by scheduling in rounds.
    """
    
    def __init__(self, model_path: str = "models/full_production/real_data/final_real_model.zip"):
        """Initialize scheduler with trained model."""
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            # Try alternative paths
            alt_paths = [
                "models/full_production/fixed/final_fixed_model.zip",
                "models/full_production/final_model.zip"
            ]
            for alt in alt_paths:
                if Path(alt).exists():
                    self.model_path = Path(alt)
                    break
        
        logger.info(f"Loading PPO model from {self.model_path}")
        self.model = PPO.load(str(self.model_path))
        self.batch_size = 170  # Maximum jobs per batch
        
    def schedule_jobs(self, jobs: List[Dict], machines: List[Dict]) -> Dict[str, Any]:
        """
        Schedule jobs using PPO model in batches.
        
        Args:
            jobs: List of job dictionaries
            machines: List of machine dictionaries
            
        Returns:
            Schedule dictionary with assignments and metrics
        """
        logger.info(f"Scheduling {len(jobs)} jobs across {len(machines)} machines")
        
        # Divide jobs into batches
        batches = self._create_batches(jobs)
        logger.info(f"Created {len(batches)} batches")
        
        # Schedule each batch
        all_assignments = []
        total_makespan = 0
        
        for i, batch in enumerate(batches):
            logger.info(f"Processing batch {i+1}/{len(batches)} with {len(batch)} jobs")
            
            # Create mini-environment for this batch
            env = self._create_batch_environment(batch, machines)
            
            # Run PPO model
            batch_schedule = self._run_model_on_batch(env, batch)
            
            # Adjust times based on previous batches
            if all_assignments:
                # Offset start times by previous makespan
                for assignment in batch_schedule['assignments']:
                    assignment['start_time'] += total_makespan
                    assignment['end_time'] += total_makespan
            
            all_assignments.extend(batch_schedule['assignments'])
            total_makespan = max(total_makespan, batch_schedule['makespan'])
        
        # Combine results
        final_schedule = {
            'assignments': all_assignments,
            'makespan': total_makespan,
            'jobs_scheduled': len(all_assignments),
            'total_jobs': len(jobs),
            'completion_rate': len(all_assignments) / len(jobs),
            'batches_used': len(batches)
        }
        
        logger.info(f"Scheduling complete: {final_schedule['jobs_scheduled']}/{len(jobs)} jobs")
        logger.info(f"Total makespan: {final_schedule['makespan']:.1f}h")
        
        return final_schedule
    
    def _create_batches(self, jobs: List[Dict]) -> List[List[Dict]]:
        """Divide jobs into batches of max 170 jobs each."""
        batches = []
        
        # Sort jobs by priority and LCD date
        sorted_jobs = sorted(jobs, key=lambda j: (
            j.get('priority', 3),
            j.get('lcd_date', '2099-12-31')
        ))
        
        # Create batches
        for i in range(0, len(sorted_jobs), self.batch_size):
            batch = sorted_jobs[i:i + self.batch_size]
            batches.append(batch)
            
        return batches
    
    def _create_batch_environment(self, jobs: List[Dict], machines: List[Dict]):
        """Create a minimal environment for batch scheduling."""
        # Import here to avoid circular imports
        from src.environments.full_production_env import FullProductionEnv
        
        # Create environment with batch data
        env = FullProductionEnv(
            n_machines=len(machines),
            n_jobs=len(jobs),
            max_valid_actions=200,
            max_episode_steps=2000,
            state_compression="hierarchical",
            use_break_constraints=True,
            use_holiday_constraints=True
        )
        
        # Override with actual job/machine data
        env.jobs = jobs
        env.machines = machines
        
        return env
    
    def _run_model_on_batch(self, env, jobs: List[Dict]) -> Dict[str, Any]:
        """Run PPO model on a batch of jobs."""
        obs, info = env.reset()
        terminated = False
        truncated = False
        steps = 0
        
        while not (terminated or truncated) and steps < 2000:
            # Get action from model
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
        
        # Extract schedule from environment
        assignments = []
        if hasattr(env, 'schedule'):
            for job_id, (machine_id, start_time, end_time) in env.schedule.items():
                assignments.append({
                    'job_id': job_id,
                    'machine_id': machine_id,
                    'start_time': start_time,
                    'end_time': end_time
                })
        
        makespan = info.get('makespan', 0)
        
        return {
            'assignments': assignments,
            'makespan': makespan,
            'steps': steps
        }


class SimplePPOScheduler:
    """
    Simplified scheduler for API integration.
    Returns mock results for now.
    """
    
    def __init__(self):
        self.model_loaded = True
        
    def schedule(self, jobs: List[Dict], machines: List[Dict]) -> Dict[str, Any]:
        """Simple scheduling that assigns jobs to machines."""
        assignments = []
        machine_loads = {m['machine_id']: 0.0 for m in machines}
        
        # Simple round-robin with load balancing
        for i, job in enumerate(jobs[:172]):  # Limit to what model can handle
            # Find least loaded compatible machine
            compatible_machines = [
                m for m in machines 
                if m.get('machine_type_id', 1) in job.get('allowed_machine_types', [1])
            ]
            
            if not compatible_machines:
                compatible_machines = machines  # Fallback
            
            # Pick least loaded
            best_machine = min(compatible_machines, 
                             key=lambda m: machine_loads[m['machine_id']])
            
            start_time = machine_loads[best_machine['machine_id']]
            end_time = start_time + job.get('processing_time', 1.0)
            
            assignments.append({
                'job_id': job.get('job_id', f"job_{i}"),
                'machine_id': best_machine['machine_id'],
                'start_time': start_time,
                'end_time': end_time
            })
            
            machine_loads[best_machine['machine_id']] = end_time
        
        makespan = max(machine_loads.values()) if machine_loads else 0
        
        return {
            'assignments': assignments,
            'makespan': makespan,
            'jobs_scheduled': len(assignments),
            'total_jobs': len(jobs),
            'completion_rate': len(assignments) / len(jobs) if jobs else 0
        }