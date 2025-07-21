"""
Safe Scheduler Wrapper for Production PPO Scheduler

This module provides a safety wrapper around the PPO scheduler to ensure
all generated schedules meet production constraints and quality standards.
It validates schedules, detects anomalies, and provides detailed reporting.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict

from .models import Job, Machine, ScheduledJob
from .scheduler import PPOScheduler

logger = logging.getLogger(__name__)


class ScheduleValidationError(Exception):
    """Raised when schedule validation fails."""
    pass


class SafeScheduler:
    """
    Safety wrapper for the PPO scheduler that ensures all constraints are met.
    
    This wrapper performs:
    - Pre-scheduling validation
    - Post-scheduling verification
    - Anomaly detection
    - Constraint compliance checking
    - Performance monitoring
    """
    
    def __init__(self, ppo_scheduler: PPOScheduler, 
                 strict_mode: bool = True,
                 max_makespan_hours: float = 60.0,
                 min_utilization: float = 0.5):
        """
        Initialize the safe scheduler wrapper.
        
        Args:
            ppo_scheduler: The underlying PPO scheduler
            strict_mode: If True, raise errors on violations. If False, log warnings.
            max_makespan_hours: Maximum acceptable makespan (anomaly threshold)
            min_utilization: Minimum acceptable utilization (anomaly threshold)
        """
        self.scheduler = ppo_scheduler
        self.strict_mode = strict_mode
        self.max_makespan_hours = max_makespan_hours
        self.min_utilization = min_utilization
        
        # Constraint parameters
        self.break_duration = 0.5  # 30 minutes
        self.break_interval = 4.0  # Every 4 hours
        self.working_hours_per_day = 16.5  # 6:30 AM - 11:00 PM
        self.days_per_week = 7
        
        logger.info(f"SafeScheduler initialized in {'strict' if strict_mode else 'permissive'} mode")
    
    def validate_inputs(self, jobs: List[Job], machines: List[Machine]) -> Dict[str, Any]:
        """
        Validate input data before scheduling.
        
        Args:
            jobs: List of jobs to validate
            machines: List of machines to validate
            
        Returns:
            Validation report with any issues found
            
        Raises:
            ScheduleValidationError: If critical validation fails in strict mode
        """
        validation_report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Validate jobs
        if not jobs:
            validation_report['errors'].append("No jobs provided for scheduling")
            validation_report['valid'] = False
        else:
            # Check for duplicate job IDs
            job_ids = [job.job_id for job in jobs]
            if len(job_ids) != len(set(job_ids)):
                validation_report['errors'].append("Duplicate job IDs found")
                validation_report['valid'] = False
            
            # Check job constraints
            for job in jobs:
                # Validate processing time
                if job.processing_time <= 0:
                    validation_report['errors'].append(
                        f"Job {job.job_id} has invalid processing time: {job.processing_time}"
                    )
                    validation_report['valid'] = False
                
                # Check if job has compatible machines
                if not job.machine_types:
                    validation_report['errors'].append(
                        f"Job {job.job_id} has no compatible machine types"
                    )
                    validation_report['valid'] = False
                
                # Warn about unrealistic processing times
                if job.processing_time > 24:
                    validation_report['warnings'].append(
                        f"Job {job.job_id} has unusually long processing time: {job.processing_time}h"
                    )
        
        # Validate machines
        if not machines:
            validation_report['errors'].append("No machines provided for scheduling")
            validation_report['valid'] = False
        else:
            # Check machine types coverage
            required_types = set()
            for job in jobs:
                required_types.update(job.machine_types)
            
            available_types = {m.machine_type for m in machines}
            missing_types = required_types - available_types
            
            if missing_types:
                validation_report['errors'].append(
                    f"Missing machine types: {missing_types}"
                )
                validation_report['valid'] = False
        
        # Calculate statistics
        validation_report['statistics'] = {
            'total_jobs': len(jobs),
            'total_machines': len(machines),
            'unique_machine_types': len(available_types) if machines else 0,
            'important_jobs': sum(1 for j in jobs if j.is_important),
            'total_processing_time': sum(j.processing_time for j in jobs)
        }
        
        # Raise error in strict mode if validation failed
        if not validation_report['valid'] and self.strict_mode:
            error_msg = "; ".join(validation_report['errors'])
            raise ScheduleValidationError(f"Input validation failed: {error_msg}")
        
        return validation_report
    
    def verify_schedule(self, scheduled_jobs: List[ScheduledJob], 
                       original_jobs: List[Job],
                       machines: List[Machine],
                       schedule_start: datetime) -> Dict[str, Any]:
        """
        Verify the generated schedule meets all constraints.
        
        Args:
            scheduled_jobs: The generated schedule
            original_jobs: Original job list
            machines: Available machines
            schedule_start: Schedule start time
            
        Returns:
            Verification report with constraint violations
        """
        verification_report = {
            'valid': True,
            'constraint_violations': [],
            'warnings': [],
            'metrics': {}
        }
        
        # Create lookup dictionaries
        job_dict = {job.job_id: job for job in original_jobs}
        machine_dict = {m.machine_id: m for m in machines}
        
        # Check all jobs are scheduled
        scheduled_job_ids = {sj.job_id for sj in scheduled_jobs}
        unscheduled_jobs = set(job_dict.keys()) - scheduled_job_ids
        
        if unscheduled_jobs:
            verification_report['constraint_violations'].append({
                'type': 'incomplete_schedule',
                'message': f"{len(unscheduled_jobs)} jobs not scheduled",
                'job_ids': list(unscheduled_jobs)
            })
            verification_report['valid'] = False
        
        # Check for overlapping jobs on same machine
        machine_schedules = defaultdict(list)
        for sj in scheduled_jobs:
            machine_schedules[sj.machine_id].append(sj)
        
        for machine_id, jobs in machine_schedules.items():
            # Sort by start time
            jobs.sort(key=lambda x: x.start_time)
            
            for i in range(len(jobs) - 1):
                if jobs[i].end_time > jobs[i + 1].start_time:
                    verification_report['constraint_violations'].append({
                        'type': 'job_overlap',
                        'message': f"Jobs {jobs[i].job_id} and {jobs[i + 1].job_id} overlap on machine {machine_id}",
                        'details': {
                            'job1_end': jobs[i].end_time,
                            'job2_start': jobs[i + 1].start_time
                        }
                    })
                    verification_report['valid'] = False
        
        # Check break time constraints
        for machine_id, jobs in machine_schedules.items():
            for job in jobs:
                # Check if job spans a break period
                job_duration = job.end_time - job.start_time
                if job_duration > self.break_interval:
                    # Job should have break time included
                    expected_breaks = int(job_duration / self.break_interval)
                    expected_duration_with_breaks = job_duration + (expected_breaks * self.break_duration)
                    
                    if abs(job.end_time - job.start_time - expected_duration_with_breaks) > 0.1:
                        verification_report['warnings'].append({
                            'type': 'break_time_missing',
                            'message': f"Job {job.job_id} may be missing break time",
                            'details': {
                                'duration': job_duration,
                                'expected_breaks': expected_breaks
                            }
                        })
        
        # Check LCD compliance for important jobs
        lcd_violations = []
        for sj in scheduled_jobs:
            if sj.job_id in job_dict:
                job = job_dict[sj.job_id]
                if job.is_important and sj.end_datetime > job.lcd_date:
                    lcd_violations.append({
                        'job_id': sj.job_id,
                        'end_time': sj.end_datetime,
                        'lcd_date': job.lcd_date,
                        'delay_hours': (sj.end_datetime - job.lcd_date).total_seconds() / 3600
                    })
        
        if lcd_violations:
            verification_report['constraint_violations'].append({
                'type': 'lcd_violation',
                'message': f"{len(lcd_violations)} important jobs miss their LCD",
                'violations': lcd_violations
            })
            if self.strict_mode:
                verification_report['valid'] = False
        
        # Calculate metrics
        if scheduled_jobs:
            makespan = max(sj.end_time for sj in scheduled_jobs)
            total_machine_time = sum(sj.end_time - sj.start_time for sj in scheduled_jobs)
            available_machine_time = makespan * len(machines)
            utilization = total_machine_time / available_machine_time if available_machine_time > 0 else 0
            
            verification_report['metrics'] = {
                'makespan': makespan,
                'utilization': utilization,
                'lcd_compliance_rate': 1 - (len(lcd_violations) / len(scheduled_jobs))
            }
        
        return verification_report
    
    def detect_anomalies(self, scheduled_jobs: List[ScheduledJob], 
                        metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect anomalies in the generated schedule.
        
        Args:
            scheduled_jobs: The generated schedule
            metrics: Schedule metrics
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Check makespan
        makespan = metrics.get('makespan', 0)
        if makespan > self.max_makespan_hours:
            anomalies.append({
                'type': 'excessive_makespan',
                'severity': 'high',
                'message': f"Makespan {makespan:.1f}h exceeds threshold {self.max_makespan_hours}h",
                'value': makespan,
                'threshold': self.max_makespan_hours
            })
        
        # Check utilization
        utilization = metrics.get('average_utilization', 0) / 100.0
        if utilization < self.min_utilization:
            anomalies.append({
                'type': 'low_utilization',
                'severity': 'medium',
                'message': f"Utilization {utilization:.1%} below threshold {self.min_utilization:.1%}",
                'value': utilization,
                'threshold': self.min_utilization
            })
        
        # Check completion rate
        completion_rate = metrics.get('completion_rate', 0) / 100.0
        if completion_rate < 0.95:
            anomalies.append({
                'type': 'low_completion_rate',
                'severity': 'high',
                'message': f"Completion rate {completion_rate:.1%} is below 95%",
                'value': completion_rate,
                'threshold': 0.95
            })
        
        # Check for machine imbalance
        machine_loads = defaultdict(float)
        for sj in scheduled_jobs:
            machine_loads[sj.machine_id] += (sj.end_time - sj.start_time)
        
        if machine_loads:
            avg_load = sum(machine_loads.values()) / len(machine_loads)
            max_load = max(machine_loads.values())
            min_load = min(machine_loads.values())
            
            if max_load > 2 * avg_load:
                anomalies.append({
                    'type': 'machine_imbalance',
                    'severity': 'medium',
                    'message': f"Machine load imbalance detected",
                    'details': {
                        'max_load': max_load,
                        'min_load': min_load,
                        'avg_load': avg_load
                    }
                })
        
        return anomalies
    
    def schedule(self, jobs: List[Job], machines: List[Machine], 
                schedule_start: datetime) -> Tuple[List[ScheduledJob], Dict[str, Any]]:
        """
        Generate a safe schedule with full validation and monitoring.
        
        Args:
            jobs: List of jobs to schedule
            machines: List of available machines
            schedule_start: Start time for the schedule
            
        Returns:
            Tuple of (scheduled_jobs, enhanced_metrics)
            
        Raises:
            ScheduleValidationError: If validation fails in strict mode
        """
        logger.info(f"SafeScheduler: Processing {len(jobs)} jobs on {len(machines)} machines")
        
        # Pre-scheduling validation
        input_validation = self.validate_inputs(jobs, machines)
        if not input_validation['valid']:
            logger.error(f"Input validation failed: {input_validation['errors']}")
            if self.strict_mode:
                raise ScheduleValidationError("Input validation failed")
        
        # Generate schedule using PPO
        try:
            scheduled_jobs, metrics = self.scheduler.schedule(jobs, machines, schedule_start)
        except Exception as e:
            logger.error(f"PPO scheduling failed: {str(e)}")
            raise
        
        # Post-scheduling verification
        verification = self.verify_schedule(scheduled_jobs, jobs, machines, schedule_start)
        if not verification['valid']:
            logger.error(f"Schedule verification failed: {verification['constraint_violations']}")
            if self.strict_mode:
                raise ScheduleValidationError("Schedule verification failed")
        
        # Anomaly detection
        anomalies = self.detect_anomalies(scheduled_jobs, metrics)
        if anomalies:
            logger.warning(f"Anomalies detected: {anomalies}")
            if self.strict_mode and any(a['severity'] == 'high' for a in anomalies):
                raise ScheduleValidationError("Critical anomalies detected in schedule")
        
        # Enhance metrics with safety information
        enhanced_metrics = metrics.copy()
        enhanced_metrics['safety_report'] = {
            'input_validation': input_validation,
            'schedule_verification': verification,
            'anomalies': anomalies,
            'safety_score': self._calculate_safety_score(input_validation, verification, anomalies)
        }
        
        logger.info(f"SafeScheduler: Schedule generated successfully with safety score "
                   f"{enhanced_metrics['safety_report']['safety_score']:.1f}%")
        
        return scheduled_jobs, enhanced_metrics
    
    def _calculate_safety_score(self, input_validation: Dict[str, Any],
                               verification: Dict[str, Any],
                               anomalies: List[Dict[str, Any]]) -> float:
        """
        Calculate an overall safety score for the schedule.
        
        Returns:
            Safety score from 0-100
        """
        score = 100.0
        
        # Deduct for validation warnings
        score -= len(input_validation.get('warnings', [])) * 2
        
        # Deduct for verification warnings
        score -= len(verification.get('warnings', [])) * 3
        
        # Deduct for anomalies based on severity
        for anomaly in anomalies:
            if anomaly['severity'] == 'high':
                score -= 10
            elif anomaly['severity'] == 'medium':
                score -= 5
            else:
                score -= 2
        
        # Ensure score stays in valid range
        return max(0.0, min(100.0, score))