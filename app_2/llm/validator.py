"""
Constraint Validator

Validates schedules against hard constraints:
- Sequence constraints within families
- Machine compatibility  
- No time overlaps
- Multi-machine job requirements
"""

import logging
from typing import List, Dict, Any, Tuple, Set
from datetime import datetime
from collections import defaultdict

from .parser import ScheduledTask
from .data_adapter import DataAdapter

logger = logging.getLogger(__name__)


class ConstraintViolation:
    """Represents a constraint violation."""
    
    def __init__(self, violation_type: str, description: str, affected_jobs: List[str]):
        self.type = violation_type
        self.description = description
        self.affected_jobs = affected_jobs
        
    def __str__(self):
        return f"{self.type}: {self.description} (Jobs: {', '.join(self.affected_jobs)})"


class ScheduleValidator:
    """
    Validates schedules against all constraints.
    """
    
    def __init__(self, snapshot_path: str = None):
        """
        Initialize validator.
        
        Args:
            snapshot_path: Path to data snapshot for job/machine info
        """
        self.data_adapter = DataAdapter(snapshot_path)
        self.data_adapter.load_snapshot()
        
        # Build lookup structures
        self._build_job_info()
        self._build_machine_info()
        
    def _build_job_info(self):
        """Build job information lookup."""
        self.job_info = {}
        self.family_sequences = defaultdict(dict)
        
        for family_id, family_data in self.data_adapter.families.items():
            for task in family_data['tasks']:
                job_id = f"{family_id}-{task['process_name']}"
                
                self.job_info[job_id] = {
                    'family_id': family_id,
                    'sequence': task['sequence'],
                    'total_sequences': family_data['total_sequences'],
                    'processing_time': task['processing_time'],
                    'capable_machines': set(task['capable_machines']),
                    'lcd_date': family_data['lcd_date'],
                    'is_important': family_data['is_important']
                }
                
                self.family_sequences[family_id][task['sequence']] = job_id
                
    def _build_machine_info(self):
        """Build machine information lookup."""
        self.machine_info = {}
        
        # Handle list format from snapshot
        if isinstance(self.data_adapter.machines, list):
            for machine_data in self.data_adapter.machines:
                machine_id = machine_data['machine_id']
                self.machine_info[int(machine_id)] = {
                    'name': machine_data['machine_name'],  # Changed from 'name' to 'machine_name'
                    'type_id': machine_data['machine_type_id']
                }
        else:
            # Handle dict format (if any)
            for machine_id, machine_data in self.data_adapter.machines.items():
                self.machine_info[int(machine_id)] = {
                    'name': machine_data['name'],
                    'type_id': machine_data['machine_type_id']
                }
            
    def validate_schedule(
        self,
        scheduled_tasks: List[ScheduledTask],
        strict: bool = True
    ) -> Tuple[bool, List[ConstraintViolation]]:
        """
        Validate a complete schedule.
        
        Args:
            scheduled_tasks: List of scheduled tasks
            strict: If True, validate all constraints. If False, skip soft constraints.
            
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []
        
        # Check each constraint type
        violations.extend(self._check_sequence_constraints(scheduled_tasks))
        violations.extend(self._check_machine_compatibility(scheduled_tasks))
        violations.extend(self._check_time_overlaps(scheduled_tasks))
        violations.extend(self._check_multi_machine_availability(scheduled_tasks))
        
        if strict:
            violations.extend(self._check_processing_times(scheduled_tasks))
            violations.extend(self._check_lcd_compliance(scheduled_tasks))
        
        is_valid = len(violations) == 0
        
        if violations:
            logger.warning(f"Found {len(violations)} constraint violations")
            for v in violations[:5]:  # Log first 5
                logger.warning(f"  - {v}")
        else:
            logger.info("Schedule is valid - all constraints satisfied")
        
        return is_valid, violations
        
    def _check_sequence_constraints(self, tasks: List[ScheduledTask]) -> List[ConstraintViolation]:
        """Check that jobs within families are scheduled in sequence order."""
        violations = []
        
        # Group tasks by family
        family_tasks = defaultdict(list)
        for task in tasks:
            family_tasks[task.family_id].append(task)
        
        # Check each family
        for family_id, tasks_in_family in family_tasks.items():
            # Sort by sequence number
            tasks_in_family.sort(key=lambda t: t.sequence)
            
            # Check sequence order matches time order
            for i in range(1, len(tasks_in_family)):
                prev_task = tasks_in_family[i-1]
                curr_task = tasks_in_family[i]
                
                # Current task should start after previous ends
                if curr_task.start_time < prev_task.end_time:
                    violations.append(ConstraintViolation(
                        "SEQUENCE_VIOLATION",
                        f"Job {curr_task.job_id} (seq {curr_task.sequence}) starts before "
                        f"{prev_task.job_id} (seq {prev_task.sequence}) completes",
                        [prev_task.job_id, curr_task.job_id]
                    ))
                
                # Check sequence numbers are in order
                if curr_task.sequence != prev_task.sequence + 1:
                    violations.append(ConstraintViolation(
                        "SEQUENCE_GAP",
                        f"Missing sequence {prev_task.sequence + 1} between "
                        f"{prev_task.job_id} and {curr_task.job_id}",
                        [prev_task.job_id, curr_task.job_id]
                    ))
        
        return violations
        
    def _check_machine_compatibility(self, tasks: List[ScheduledTask]) -> List[ConstraintViolation]:
        """Check that jobs are scheduled on compatible machines."""
        violations = []
        
        for task in tasks:
            # Get job info
            if task.job_id not in self.job_info:
                violations.append(ConstraintViolation(
                    "UNKNOWN_JOB",
                    f"Job {task.job_id} not found in job data",
                    [task.job_id]
                ))
                continue
            
            job_info = self.job_info[task.job_id]
            capable_machines = job_info['capable_machines']
            
            # Check all assigned machines are capable
            for machine_id in task.machine_ids:
                if machine_id not in capable_machines:
                    violations.append(ConstraintViolation(
                        "MACHINE_INCOMPATIBLE",
                        f"Job {task.job_id} cannot run on machine {machine_id}. "
                        f"Capable machines: {sorted(capable_machines)}",
                        [task.job_id]
                    ))
            
            # Check multi-machine jobs have all required machines
            if len(capable_machines) > 1 and len(task.machine_ids) != len(capable_machines):
                violations.append(ConstraintViolation(
                    "MULTI_MACHINE_INCOMPLETE",
                    f"Multi-machine job {task.job_id} requires ALL machines "
                    f"{sorted(capable_machines)} but only assigned {sorted(task.machine_ids)}",
                    [task.job_id]
                ))
        
        return violations
        
    def _check_time_overlaps(self, tasks: List[ScheduledTask]) -> List[ConstraintViolation]:
        """Check for time overlaps on same machine."""
        violations = []
        
        # Build machine timeline
        machine_timeline = defaultdict(list)
        for task in tasks:
            for machine_id in task.machine_ids:
                machine_timeline[machine_id].append(task)
        
        # Check each machine for overlaps
        for machine_id, machine_tasks in machine_timeline.items():
            # Sort by start time
            machine_tasks.sort(key=lambda t: t.start_time)
            
            # Check for overlaps
            for i in range(1, len(machine_tasks)):
                prev_task = machine_tasks[i-1]
                curr_task = machine_tasks[i]
                
                if curr_task.start_time < prev_task.end_time:
                    violations.append(ConstraintViolation(
                        "TIME_OVERLAP",
                        f"Jobs {prev_task.job_id} and {curr_task.job_id} overlap "
                        f"on machine {machine_id} at {curr_task.start_time}",
                        [prev_task.job_id, curr_task.job_id]
                    ))
        
        return violations
        
    def _check_multi_machine_availability(self, tasks: List[ScheduledTask]) -> List[ConstraintViolation]:
        """Check that multi-machine jobs have all machines available simultaneously."""
        violations = []
        
        # Build complete timeline for all machines
        machine_busy_times = defaultdict(list)
        for task in tasks:
            for machine_id in task.machine_ids:
                machine_busy_times[machine_id].append((task.start_time, task.end_time, task.job_id))
        
        # Check multi-machine jobs
        for task in tasks:
            if len(task.machine_ids) > 1:
                # All machines should be free for exact same time period
                time_periods = []
                
                for machine_id in task.machine_ids:
                    # Find this job's time on this machine
                    machine_periods = [
                        (start, end) for start, end, job_id in machine_busy_times[machine_id]
                        if job_id == task.job_id
                    ]
                    
                    if not machine_periods:
                        violations.append(ConstraintViolation(
                            "MULTI_MACHINE_MISSING",
                            f"Multi-machine job {task.job_id} not scheduled on machine {machine_id}",
                            [task.job_id]
                        ))
                    else:
                        time_periods.extend(machine_periods)
                
                # All time periods should be identical
                if time_periods and not all(p == time_periods[0] for p in time_periods):
                    violations.append(ConstraintViolation(
                        "MULTI_MACHINE_MISALIGNED",
                        f"Multi-machine job {task.job_id} has different time periods "
                        f"on different machines",
                        [task.job_id]
                    ))
        
        return violations
        
    def _check_processing_times(self, tasks: List[ScheduledTask]) -> List[ConstraintViolation]:
        """Check that scheduled times match expected processing times."""
        violations = []
        
        for task in tasks:
            if task.job_id not in self.job_info:
                continue
            
            expected_hours = self.job_info[task.job_id]['processing_time']
            actual_hours = task.processing_hours
            
            # Allow 5% tolerance
            tolerance = 0.05 * expected_hours
            if abs(actual_hours - expected_hours) > tolerance:
                violations.append(ConstraintViolation(
                    "PROCESSING_TIME_MISMATCH",
                    f"Job {task.job_id} scheduled for {actual_hours:.1f}h "
                    f"but requires {expected_hours:.1f}h",
                    [task.job_id]
                ))
        
        return violations
        
    def _check_lcd_compliance(self, tasks: List[ScheduledTask]) -> List[ConstraintViolation]:
        """Check that jobs complete before LCD dates."""
        violations = []
        
        for task in tasks:
            if task.job_id not in self.job_info:
                continue
            
            lcd_str = self.job_info[task.job_id]['lcd_date']
            lcd_date = datetime.strptime(lcd_str + " 23:59", "%Y-%m-%d %H:%M")
            
            if task.end_time > lcd_date:
                days_late = (task.end_time - lcd_date).days
                violations.append(ConstraintViolation(
                    "LCD_VIOLATION",
                    f"Job {task.job_id} completes {days_late} days after LCD date {lcd_str}",
                    [task.job_id]
                ))
        
        return violations


def test_validator():
    """Test schedule validator."""
    from .parser import ScheduleParser
    
    # Create test schedule
    test_schedule = """
    JOTP25070237-CT10-013A-1/5 -> machines[80] @ 2025-01-24 08:00 - 2025-01-24 23:06
    JOTP25070237-CT10-013A-2/5 -> machines[80] @ 2025-01-25 08:00 - 2025-01-25 23:06
    JOTP25070237-CT10-013A-3/5 -> machines[80] @ 2025-01-24 10:00 - 2025-01-25 01:06
    """
    
    # Parse schedule
    parser = ScheduleParser()
    tasks = parser.parse_schedule(test_schedule)
    
    # Validate
    validator = ScheduleValidator()
    is_valid, violations = validator.validate_schedule(tasks)
    
    print(f"Schedule valid: {is_valid}")
    print(f"Found {len(violations)} violations:")
    for v in violations:
        print(f"  - {v}")


if __name__ == "__main__":
    test_validator()