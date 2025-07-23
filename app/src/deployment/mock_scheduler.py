"""
Mock scheduler for demonstration purposes.
Creates realistic-looking schedules without requiring the full environment.
"""

import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
import random

from .models import Job, ScheduledJob, Machine

logger = logging.getLogger(__name__)


class MockScheduler:
    """
    Mock scheduler that generates realistic schedules for demonstration.
    """
    
    def __init__(self):
        """Initialize the mock scheduler."""
        self.machine_names = [
            "CM03", "CL02", "AD02-50HP", "PP33-250T", "OV01",
            "BL03", "BL04", "CM17", "CM19", "CM20",
            "ALDG", "BDS01", "CL01", "CM08", "CM09"
        ]
    
    def schedule(self, jobs: List[Job], machines: List[Machine], 
                schedule_start: datetime) -> Tuple[List[ScheduledJob], Dict[str, Any]]:
        """
        Generate a mock schedule that looks realistic.
        
        Args:
            jobs: List of jobs to schedule
            machines: List of available machines
            schedule_start: Start time for the schedule
            
        Returns:
            Tuple of (scheduled_jobs, metrics)
        """
        logger.info(f"Mock scheduling {len(jobs)} jobs on {len(machines)} machines")
        
        # Use provided machines or default names
        if machines:
            machine_list = [(m.machine_id, m.machine_name) for m in machines[:15]]  # Use first 15
        else:
            machine_list = [(i, name) for i, name in enumerate(self.machine_names)]
        
        # Track machine availability
        machine_end_times = {m_id: 0.0 for m_id, _ in machine_list}
        
        scheduled_jobs = []
        
        # Schedule jobs using simple heuristic
        for job in jobs:
            # Find machine with earliest availability
            best_machine_id = None
            best_machine_name = None
            earliest_start = float('inf')
            
            # Check compatible machines
            for m_id, m_name in machine_list:
                # Simple compatibility check - use first 5 machines for all jobs
                if m_id < 5 or (m_id < 10 and job.is_important):
                    if machine_end_times[m_id] < earliest_start:
                        earliest_start = machine_end_times[m_id]
                        best_machine_id = m_id
                        best_machine_name = m_name
            
            if best_machine_id is not None:
                # Schedule the job
                start_time = earliest_start
                duration = job.processing_time + job.setup_time
                end_time = start_time + duration
                
                # Update machine availability
                machine_end_times[best_machine_id] = end_time
                
                # Create scheduled job
                start_dt = schedule_start + timedelta(hours=start_time)
                end_dt = schedule_start + timedelta(hours=end_time)
                
                scheduled_job = ScheduledJob(
                    job_id=job.job_id,
                    machine_id=best_machine_id,
                    machine_name=best_machine_name,
                    start_time=start_time,
                    end_time=end_time,
                    start_datetime=start_dt.isoformat(),
                    end_datetime=end_dt.isoformat(),
                    setup_time_included=job.setup_time
                )
                scheduled_jobs.append(scheduled_job)
        
        # Calculate metrics
        makespan = max(machine_end_times.values()) if machine_end_times else 0
        total_jobs = len(jobs)
        scheduled_count = len(scheduled_jobs)
        completion_rate = (scheduled_count / total_jobs * 100) if total_jobs > 0 else 0
        
        # Calculate utilization
        total_machine_time = makespan * len(machine_list)
        total_work_time = sum(j.processing_time + j.setup_time for j in jobs[:scheduled_count])
        avg_utilization = (total_work_time / total_machine_time * 100) if total_machine_time > 0 else 0
        
        # Count important jobs scheduled on time
        important_on_time = 0
        for j, sj in zip(jobs[:scheduled_count], scheduled_jobs):
            if j.is_important:
                try:
                    # Handle both string and datetime lcd_date
                    if isinstance(j.lcd_date, str):
                        lcd_dt = datetime.fromisoformat(j.lcd_date.replace('Z', '+00:00'))
                    else:
                        lcd_dt = j.lcd_date
                    
                    # end_datetime should be a string
                    end_dt = datetime.fromisoformat(sj.end_datetime.replace('Z', '+00:00'))
                    
                    if end_dt <= lcd_dt:
                        important_on_time += 1
                except Exception as e:
                    logger.warning(f"Error comparing dates for job {j.job_id}: {e}")
                    continue
        
        metrics = {
            'makespan': round(makespan, 2),
            'total_jobs': total_jobs,
            'scheduled_jobs': scheduled_count,
            'completion_rate': round(completion_rate, 1),
            'average_utilization': round(avg_utilization, 1),
            'total_setup_time': sum(j.setup_time for j in jobs[:scheduled_count]),
            'important_jobs_on_time': important_on_time
        }
        
        logger.info(f"Mock schedule created: {scheduled_count}/{total_jobs} jobs, makespan={makespan:.2f}h")
        
        return scheduled_jobs, metrics