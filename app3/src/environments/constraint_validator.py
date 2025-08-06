"""
Constraint validator for scheduling environment.
Validates sequence constraints, machine availability, and material arrival.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ConstraintValidator:
    """Validates scheduling constraints and generates action masks."""
    
    def __init__(self, loader):
        """
        Initialize validator with data loader.
        
        Args:
            loader: SnapshotLoader instance
        """
        self.loader = loader
        self.n_tasks = len(loader.tasks)
        self.n_machines = len(loader.machines)
        
    def get_action_mask(
        self,
        current_time: float,
        machine_schedules: Dict[str, List[Tuple[float, float]]],
        scheduled_tasks: Set[int]
    ) -> np.ndarray:
        """
        Generate action mask for valid tasks that can be scheduled.
        
        Args:
            current_time: Current simulation time in hours
            machine_schedules: Dict mapping machine_name -> [(start, end), ...]
            scheduled_tasks: Set of task indices already scheduled
            
        Returns:
            Boolean array of shape (n_tasks,) where True = valid action
        """
        mask = np.zeros(self.n_tasks, dtype=bool)
        
        for task_idx, task in enumerate(self.loader.tasks):
            if self.is_task_valid(task_idx, current_time, machine_schedules, scheduled_tasks):
                mask[task_idx] = True
                
        return mask
        
    def is_task_valid(
        self,
        task_idx: int,
        current_time: float,
        machine_schedules: Dict[str, List[Tuple[float, float]]],
        scheduled_tasks: Set[int]
    ) -> bool:
        """
        Check if a specific task can be scheduled now.
        
        Validates:
        1. Task not already scheduled
        2. Previous sequence in family completed
        3. Machine is available (assigned or any)
        
        Args:
            task_idx: Index of task to check
            current_time: Current simulation time
            machine_schedules: Machine availability
            scheduled_tasks: Already scheduled tasks
            
        Returns:
            True if task can be scheduled
        """
        task = self.loader.tasks[task_idx]
        family = self.loader.families[task.family_id]
        
        # Check 1: Not already scheduled
        if task_idx in scheduled_tasks or task.is_scheduled:
            return False
            
        # Check 2: Previous sequence completed
        if not self._check_sequence_constraint(task, family):
            return False
            
        # Check 3: Machine availability
        if not self._check_machine_availability(task, current_time, machine_schedules):
            return False
            
        return True
        
    def _check_sequence_constraint(self, task, family) -> bool:
        """
        Check if sequence constraints are satisfied.
        
        Args:
            task: Task to check
            family: Family the task belongs to
            
        Returns:
            True if sequence constraints satisfied
        """
        # First task in sequence is always ready
        if task.sequence == 1:
            return True
            
        # Check if previous sequence is completed
        prev_seq = task.sequence - 1
        prev_task_key = (family.family_id, prev_seq)
        
        if prev_task_key in self.loader.task_by_family_seq:
            prev_task = self.loader.task_by_family_seq[prev_task_key]
            return prev_task.is_scheduled
            
        # If previous sequence doesn't exist, allow scheduling
        return True
        
    def _check_machine_availability(
        self,
        task,
        current_time: float,
        machine_schedules: Dict[str, List[Tuple[float, float]]]
    ) -> bool:
        """
        Check if required machine is available.
        
        Args:
            task: Task to check
            current_time: Current simulation time
            machine_schedules: Machine availability
            
        Returns:
            True if machine is available
        """
        if task.assigned_machine:
            # Task has specific machine requirement
            return self._is_machine_free(task.assigned_machine, current_time, machine_schedules)
        else:
            # Task can use any available machine
            for machine in self.loader.machines:
                if self._is_machine_free(machine, current_time, machine_schedules):
                    return True
            return False
            
    def _is_machine_free(
        self,
        machine: str,
        current_time: float,
        machine_schedules: Dict[str, List[Tuple[float, float]]]
    ) -> bool:
        """
        Check if a specific machine is free at current time.
        
        Args:
            machine: Machine name
            current_time: Current simulation time
            machine_schedules: Machine schedules
            
        Returns:
            True if machine is free
        """
        if machine not in machine_schedules:
            return True  # Machine has no scheduled tasks
            
        schedule = machine_schedules[machine]
        if not schedule:
            return True
            
        # Check if current time overlaps with any scheduled period
        for start, end in schedule:
            if start <= current_time < end:
                return False  # Machine is busy
                
        # Check if we can start now (no future conflicts for task duration)
        # This is simplified - in full implementation would check task duration
        last_end = max(end for _, end in schedule)
        return current_time >= last_end
        
        
    def validate_action(
        self,
        action: int,
        current_time: float,
        machine_schedules: Dict[str, List[Tuple[float, float]]],
        scheduled_tasks: Set[int]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a proposed action with detailed error message.
        
        Args:
            action: Task index to schedule
            current_time: Current simulation time
            machine_schedules: Machine availability
            scheduled_tasks: Already scheduled tasks
            
        Returns:
            (is_valid, error_message)
        """
        if action < 0 or action >= self.n_tasks:
            return False, f"Invalid action index: {action}"
            
        task = self.loader.tasks[action]
        family = self.loader.families[task.family_id]
        
        # Check each constraint with specific error
        if action in scheduled_tasks or task.is_scheduled:
            return False, f"Task {task.family_id}-{task.sequence} already scheduled"
            
        if not self._check_sequence_constraint(task, family):
            return False, f"Previous sequence not completed for {task.family_id}-{task.sequence}"
            
        if not self._check_machine_availability(task, current_time, machine_schedules):
            if task.assigned_machine:
                return False, f"Machine {task.assigned_machine} not available"
            else:
                return False, "No machines available for unassigned task"
                
        if not self._check_material_arrival(family, current_time):
            return False, f"Material not arrived for family {family.family_id}"
            
        return True, None
        
    def get_machine_for_task(
        self,
        task_idx: int,
        current_time: float,
        machine_schedules: Dict[str, List[Tuple[float, float]]]
    ) -> Optional[str]:
        """
        Get available machine for task.
        
        Args:
            task_idx: Task index
            current_time: Current time
            machine_schedules: Machine schedules
            
        Returns:
            Machine name if available, None otherwise
        """
        task = self.loader.tasks[task_idx]
        
        if task.assigned_machine:
            # Use pre-assigned machine
            if self._is_machine_free(task.assigned_machine, current_time, machine_schedules):
                return task.assigned_machine
            return None
        else:
            # Find any available machine
            for machine in self.loader.machines:
                if self._is_machine_free(machine, current_time, machine_schedules):
                    return machine
            return None
            
    def check_all_constraints_satisfied(
        self,
        scheduled_tasks: List[int],
        task_schedules: Dict[int, Tuple[float, float, str]]
    ) -> Tuple[bool, List[str]]:
        """
        Verify all constraints are satisfied in final schedule.
        
        Args:
            scheduled_tasks: List of scheduled task indices
            task_schedules: Dict of task_idx -> (start, end, machine)
            
        Returns:
            (all_satisfied, list_of_violations)
        """
        violations = []
        
        # Check sequence constraints
        for family in self.loader.families.values():
            task_times = {}
            for task in family.tasks:
                if task.global_idx in task_schedules:
                    start, end, _ = task_schedules[task.global_idx]
                    task_times[task.sequence] = (start, end)
                    
            # Verify sequence order
            sequences = sorted(task_times.keys())
            for i in range(len(sequences) - 1):
                curr_seq = sequences[i]
                next_seq = sequences[i + 1]
                
                if curr_seq + 1 != next_seq:
                    continue  # Not consecutive sequences
                    
                curr_end = task_times[curr_seq][1]
                next_start = task_times[next_seq][0]
                
                if next_start < curr_end:
                    violations.append(
                        f"Sequence violation: {family.family_id}-{next_seq} "
                        f"starts before {family.family_id}-{curr_seq} ends"
                    )
                    
        # Check machine conflicts
        machine_tasks = {}
        for task_idx, (start, end, machine) in task_schedules.items():
            if machine not in machine_tasks:
                machine_tasks[machine] = []
            machine_tasks[machine].append((start, end, task_idx))
            
        for machine, tasks in machine_tasks.items():
            # Sort by start time
            tasks.sort(key=lambda x: x[0])
            
            # Check for overlaps
            for i in range(len(tasks) - 1):
                curr_end = tasks[i][1]
                next_start = tasks[i + 1][0]
                
                if curr_end > next_start:
                    curr_task = self.loader.tasks[tasks[i][2]]
                    next_task = self.loader.tasks[tasks[i + 1][2]]
                    violations.append(
                        f"Machine conflict on {machine}: "
                        f"{curr_task.family_id}-{curr_task.sequence} and "
                        f"{next_task.family_id}-{next_task.sequence} overlap"
                    )
                    
        return len(violations) == 0, violations


if __name__ == "__main__":
    # Test validator
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from data.snapshot_loader import SnapshotLoader
    
    loader = SnapshotLoader("data/10_jobs.json")
    validator = ConstraintValidator(loader)
    
    # Test action mask generation
    mask = validator.get_action_mask(
        current_time=0,
        machine_schedules={},
        scheduled_tasks=set()
    )
    
    print(f"Valid actions at start: {mask.sum()} out of {len(mask)}")
    print(f"First 10 valid actions: {np.where(mask[:10])[0]}")
    
    # Test validation
    if mask.sum() > 0:
        first_valid = np.where(mask)[0][0]
        is_valid, error = validator.validate_action(
            first_valid, 0, {}, set()
        )
        print(f"\nValidating action {first_valid}: {is_valid}")
        if error:
            print(f"Error: {error}")