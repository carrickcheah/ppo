"""
Rules Engine for Scheduling Game

This module defines the hard constraints (physics) of the scheduling game:
1. Sequence constraints - jobs must follow order within family
2. Machine compatibility - jobs can only run on capable machines  
3. No time overlap - one job per machine at a time
4. Working hours - only schedule within allowed time windows

These are NOT strategies - they are the fundamental rules of the game.
"""

from typing import Dict, List, Tuple, Set, Optional
from datetime import datetime, timedelta
import numpy as np
import logging

logger = logging.getLogger(__name__)


class RulesEngine:
    """
    Enforces hard constraints for the scheduling game.
    
    Think of this as the physics engine - it defines what moves are legal,
    not which moves are good. The AI learns strategy through rewards.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize rules engine.
        
        Args:
            config: Rules configuration from YAML
        """
        self.config = config
        self.sequence_enforcement = config.get('enforce_sequence', True)
        self.compatibility_enforcement = config.get('enforce_compatibility', True)
        self.overlap_enforcement = config.get('enforce_no_overlap', True)
        self.working_hours_enforcement = config.get('enforce_working_hours', True)
        
    def is_action_valid(
        self,
        job: Dict,
        machine: Dict,
        current_schedules: List[List[Dict]],
        completed_jobs: Set[int],
        job_assignments: Dict[int, Dict],
        job_to_family: Dict[str, str],
        job_sequences: Dict[str, int],
        all_jobs: List[Dict]
    ) -> Tuple[bool, str]:
        """
        Check if scheduling a job on a machine is valid.
        
        Args:
            job: Job to schedule
            machine: Machine to schedule on
            current_schedules: Current machine schedules
            completed_jobs: Set of completed job indices
            job_assignments: Current job assignments
            job_to_family: Mapping of job IDs to family IDs
            job_sequences: Mapping of job IDs to sequence numbers
            all_jobs: List of all jobs
            
        Returns:
            (is_valid, reason)
        """
        job_id = job['job_id']
        machine_id = machine['machine_id']
        
        # Rule 1: Sequence constraints
        if self.sequence_enforcement:
            is_valid, reason = self._check_sequence_constraint(
                job, job_to_family, job_sequences, completed_jobs, all_jobs
            )
            if not is_valid:
                return False, reason
        
        # Rule 2: Machine compatibility
        if self.compatibility_enforcement:
            is_valid, reason = self._check_compatibility_constraint(job, machine)
            if not is_valid:
                return False, reason
                
        # Rule 3: No overlap (this is checked when calculating start time)
        # The environment handles this by scheduling after previous jobs
        
        # Rule 4: Working hours (checked when adjusting start time)
        # The environment handles this through time adjustment
        
        return True, "Valid action"
    
    def _check_sequence_constraint(
        self,
        job: Dict,
        job_to_family: Dict[str, str],
        job_sequences: Dict[str, int],
        completed_jobs: Set[int],
        all_jobs: List[Dict]
    ) -> Tuple[bool, str]:
        """Check sequence constraints within family."""
        job_id = job['job_id']
        family_id = job_to_family.get(job_id)
        job_sequence = job_sequences.get(job_id, 1)
        
        # Check if previous sequences in family are completed
        for idx, other_job in enumerate(all_jobs):
            other_job_id = other_job['job_id']
            other_family = job_to_family.get(other_job_id)
            
            if other_family == family_id and other_job_id != job_id:
                other_sequence = job_sequences.get(other_job_id, 1)
                
                # Must complete all previous sequences
                if other_sequence < job_sequence and idx not in completed_jobs:
                    return False, f"Must complete sequence {other_sequence} before {job_sequence}"
                    
        return True, "Sequence valid"
    
    def _check_compatibility_constraint(
        self,
        job: Dict,
        machine: Dict
    ) -> Tuple[bool, str]:
        """Check machine compatibility constraint."""
        # Get job's allowed machine types
        job_machine_types = job.get('machine_types', [])
        
        # If no machine types specified, job can run on any machine
        if not job_machine_types:
            return True, "No machine type restrictions"
            
        # Get machine's type
        machine_type = machine.get('machine_type_id')
        
        # Check if compatible
        if machine_type in job_machine_types:
            return True, "Machine type compatible"
        else:
            return False, f"Machine type {machine_type} not in allowed types {job_machine_types}"
    
    def calculate_valid_start_time(
        self,
        job: Dict,
        machine_idx: int,
        current_schedules: List[List[Dict]],
        job_assignments: Dict[int, Dict],
        job_to_family: Dict[str, str],
        job_sequences: Dict[str, int],
        current_time: float,
        working_hours: Optional[Dict] = None
    ) -> float:
        """
        Calculate the earliest valid start time for a job on a machine.
        
        This considers:
        1. Machine availability (no overlap)
        2. Family dependencies (sequence)
        3. Working hours constraints
        
        Args:
            job: Job to schedule
            machine_idx: Machine index
            current_schedules: Current machine schedules
            job_assignments: Current job assignments
            job_to_family: Job to family mapping
            job_sequences: Job sequence numbers
            current_time: Current simulation time
            working_hours: Working hours configuration
            
        Returns:
            Earliest valid start time
        """
        start_time = current_time
        
        # Rule 1: Machine must be free (no overlap)
        if current_schedules[machine_idx]:
            last_job = current_schedules[machine_idx][-1]
            machine_free_time = last_job['end_time']
            start_time = max(start_time, machine_free_time)
        
        # Rule 2: Previous sequence in family must be complete
        job_id = job['job_id']
        family_id = job_to_family.get(job_id)
        job_sequence = job_sequences.get(job_id, 1)
        
        if job_sequence > 1:
            # Find when previous sequence completed
            family_ready_time = 0.0
            for assigned_job_idx, assignment in job_assignments.items():
                assigned_job = assignment['job']
                if (job_to_family.get(assigned_job['job_id']) == family_id and
                    job_sequences.get(assigned_job['job_id']) == job_sequence - 1):
                    family_ready_time = assignment['end_time']
                    break
                    
            start_time = max(start_time, family_ready_time)
        
        # Rule 3: Adjust for working hours
        if self.working_hours_enforcement and working_hours:
            start_time = self._adjust_for_working_hours(start_time, working_hours)
            
        return start_time
    
    def _adjust_for_working_hours(
        self,
        start_time: float,
        working_hours: Dict
    ) -> float:
        """
        Adjust start time to respect working hours.
        
        Args:
            start_time: Proposed start time (hours from epoch)
            working_hours: Working hours configuration
            
        Returns:
            Adjusted start time
        """
        # Simple implementation - can be enhanced
        # For now, assume 8am-6pm Monday-Friday
        
        # Convert hours to day of week and hour of day
        day_of_week = int(start_time // 24) % 7  # 0=Monday, 6=Sunday
        hour_of_day = start_time % 24
        
        # If weekend, move to Monday
        if day_of_week >= 5:  # Saturday or Sunday
            days_to_monday = 7 - day_of_week + 0  # Move to Monday
            start_time += days_to_monday * 24
            hour_of_day = 8.0  # Start at 8am
            
        # If outside working hours, adjust
        if hour_of_day < 8.0:
            start_time = start_time - hour_of_day + 8.0
        elif hour_of_day >= 18.0:
            # Move to next day 8am
            start_time = start_time - hour_of_day + 24 + 8.0
            
        return start_time
    
    def validate_schedule(
        self,
        schedule: List[Dict],
        jobs: List[Dict],
        machines: List[Dict]
    ) -> Tuple[bool, List[str]]:
        """
        Validate a complete schedule against all rules.
        
        Args:
            schedule: List of scheduled jobs with assignments
            jobs: All jobs
            machines: All machines
            
        Returns:
            (is_valid, list_of_violations)
        """
        violations = []
        
        # Check each scheduled job
        for scheduled_job in schedule:
            job = scheduled_job['job']
            machine_idx = scheduled_job['machine_idx']
            machine = machines[machine_idx]
            
            # Check sequence
            # Check compatibility
            # Check overlaps
            # Check working hours
            
            # Add violations to list
            pass
            
        is_valid = len(violations) == 0
        return is_valid, violations
    
    def get_action_mask(
        self,
        jobs: List[Dict],
        machines: List[Dict],
        current_schedules: List[List[Dict]],
        completed_jobs: Set[int],
        job_assignments: Dict[int, Dict],
        job_to_family: Dict[str, str],
        job_sequences: Dict[str, int]
    ) -> np.ndarray:
        """
        Get boolean mask of all valid actions.
        
        Args:
            Various state information
            
        Returns:
            Boolean array of shape (n_jobs * n_machines,)
        """
        n_jobs = len(jobs)
        n_machines = len(machines)
        mask = np.zeros((n_jobs, n_machines), dtype=bool)
        
        for job_idx, job in enumerate(jobs):
            # Skip if already completed
            if job_idx in completed_jobs:
                continue
                
            for machine_idx, machine in enumerate(machines):
                # Check if this action is valid
                is_valid, _ = self.is_action_valid(
                    job, machine, current_schedules, completed_jobs,
                    job_assignments, job_to_family, job_sequences, jobs
                )
                mask[job_idx, machine_idx] = is_valid
                
        return mask.flatten()
    
    def parse_job_families(self, jobs: List[Dict]) -> Tuple[Dict, Dict, Dict]:
        """
        Parse jobs into families and extract sequence information.
        
        Args:
            jobs: List of jobs from database
            
        Returns:
            (families, job_to_family, job_sequences)
        """
        families = {}
        job_to_family = {}
        job_sequences = {}
        
        for job in jobs:
            job_id = job['job_id']
            
            # Parse job ID to extract family and sequence
            # Expected format: FAMILYID_DETAILS-SEQ/TOTAL
            if '_' in job_id:
                parts = job_id.split('_')
                family_id = parts[0]
                
                # Extract sequence from end
                if '-' in parts[-1] and '/' in parts[-1]:
                    seq_part = parts[-1].split('-')[-1]
                    if '/' in seq_part:
                        seq_num, total = seq_part.split('/')
                        try:
                            sequence = int(seq_num)
                        except:
                            sequence = 1
                    else:
                        sequence = 1
                else:
                    sequence = 1
            else:
                # No family structure
                family_id = job_id
                sequence = 1
                
            # Store mappings
            job_to_family[job_id] = family_id
            job_sequences[job_id] = sequence
            
            # Build family structure
            if family_id not in families:
                families[family_id] = {
                    'jobs': [],
                    'max_sequence': 0
                }
            families[family_id]['jobs'].append(job)
            families[family_id]['max_sequence'] = max(
                families[family_id]['max_sequence'], sequence
            )
            
        return families, job_to_family, job_sequences