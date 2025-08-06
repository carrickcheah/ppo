"""
Data loader for JSON snapshots containing real production scheduling data.
Handles families, tasks, machines, and constraints.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)


class Task:
    """Represents a single production task."""
    
    def __init__(
        self,
        family_id: str,
        sequence: int,
        process_name: str,
        processing_time: float,
        assigned_machine: Optional[str],
        balance_quantity: float,
        original_quantity: float,
        global_idx: int
    ):
        self.family_id = family_id
        self.sequence = sequence
        self.process_name = process_name
        self.processing_time = processing_time
        self.assigned_machine = assigned_machine
        self.balance_quantity = balance_quantity
        self.original_quantity = original_quantity
        self.global_idx = global_idx  # Index in flat task list
        self.is_scheduled = False
        self.start_time = None
        self.end_time = None
        self.machine_used = None
        
    def __repr__(self):
        return f"Task({self.family_id}-{self.sequence}/{self.process_name}, machine={self.assigned_machine})"


class Family:
    """Represents a job family with multiple sequential tasks."""
    
    def __init__(
        self,
        family_id: str,
        lcd_date: Optional[str],
        lcd_days_remaining: int,
        is_urgent: bool,
        material_arrival: Optional[str]
    ):
        self.family_id = family_id
        self.lcd_date = lcd_date
        self.lcd_days_remaining = lcd_days_remaining
        self.is_urgent = is_urgent
        self.material_arrival = material_arrival
        self.tasks: List[Task] = []
        self.completed_sequences = set()
        
    def add_task(self, task: Task):
        """Add a task to this family."""
        self.tasks.append(task)
        
    def get_next_sequence(self) -> Optional[int]:
        """Get the next sequence number that should be scheduled."""
        if not self.tasks:
            return None
        max_seq = max(t.sequence for t in self.tasks)
        for seq in range(1, max_seq + 1):
            if seq not in self.completed_sequences:
                return seq
        return None
        
    def is_sequence_ready(self, sequence: int) -> bool:
        """Check if a sequence is ready to be scheduled."""
        if sequence == 1:
            return True
        return (sequence - 1) in self.completed_sequences
        
    def mark_sequence_complete(self, sequence: int):
        """Mark a sequence as completed."""
        self.completed_sequences.add(sequence)
        
    def is_complete(self) -> bool:
        """Check if all tasks in family are complete."""
        return len(self.completed_sequences) == len(self.tasks)
        
    def __repr__(self):
        return f"Family({self.family_id}, tasks={len(self.tasks)}, urgent={self.is_urgent})"


class SnapshotLoader:
    """Loads and manages production data from JSON snapshots."""
    
    def __init__(self, snapshot_path: str):
        """
        Initialize loader with a JSON snapshot file.
        
        Args:
            snapshot_path: Path to JSON snapshot file
        """
        self.snapshot_path = Path(snapshot_path)
        if not self.snapshot_path.exists():
            raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")
            
        # Load data
        with open(self.snapshot_path, 'r') as f:
            self.data = json.load(f)
            
        # Parse components
        self.metadata = self.data.get('metadata', {})
        self.machines = self.data.get('machines', [])
        self.families: Dict[str, Family] = {}
        self.tasks: List[Task] = []
        
        # Parse families and tasks
        self._parse_families()
        
        # Create quick lookups
        self.machine_to_idx = {m: i for i, m in enumerate(self.machines)}
        self.idx_to_machine = {i: m for i, m in enumerate(self.machines)}
        self.task_by_family_seq: Dict[Tuple[str, int], Task] = {}
        
        for task in self.tasks:
            self.task_by_family_seq[(task.family_id, task.sequence)] = task
            
        logger.info(f"Loaded {len(self.families)} families with {len(self.tasks)} tasks and {len(self.machines)} machines")
        
    def _parse_families(self):
        """Parse families and tasks from JSON data."""
        global_task_idx = 0
        
        for family_id, family_data in self.data.get('families', {}).items():
            # Create family
            family = Family(
                family_id=family_id,
                lcd_date=family_data.get('lcd_date'),
                lcd_days_remaining=family_data.get('lcd_days_remaining', 30),
                is_urgent=family_data.get('is_urgent', False),
                material_arrival=family_data.get('material_arrival')
            )
            
            # Add tasks to family
            for task_data in family_data.get('tasks', []):
                task = Task(
                    family_id=family_id,
                    sequence=task_data.get('sequence', 1),
                    process_name=task_data.get('process_name', ''),
                    processing_time=task_data.get('processing_time', 1.0),
                    assigned_machine=task_data.get('assigned_machine'),
                    balance_quantity=task_data.get('balance_quantity', 0),
                    original_quantity=task_data.get('original_quantity', 0),
                    global_idx=global_task_idx
                )
                
                family.add_task(task)
                self.tasks.append(task)
                global_task_idx += 1
                
            self.families[family_id] = family
            
    def get_task_features(self, current_time: float = 0) -> np.ndarray:
        """
        Get feature vector for all tasks.
        
        Features per task:
        - is_available (0/1): Can be scheduled now?
        - urgency_score (0-1): How urgent based on LCD
        - processing_time_norm (0-1): Normalized processing time
        - has_assigned_machine (0/1): Has pre-assigned machine?
        - sequence_progress (0-1): Progress within family
        - is_scheduled (0/1): Already scheduled?
        
        Returns:
            Array of shape (n_tasks, 6)
        """
        n_tasks = len(self.tasks)
        features = np.zeros((n_tasks, 6))
        
        max_processing = max(t.processing_time for t in self.tasks) if self.tasks else 1.0
        
        for i, task in enumerate(self.tasks):
            family = self.families[task.family_id]
            
            # Is available to schedule?
            is_available = (
                not task.is_scheduled and 
                family.is_sequence_ready(task.sequence) and
                self._check_material_arrival(family, current_time)
            )
            
            # Urgency based on LCD
            urgency = 1.0 - (family.lcd_days_remaining / 30.0) if family.lcd_days_remaining else 0.5
            
            # Normalized processing time
            proc_norm = task.processing_time / max_processing
            
            # Has assigned machine?
            has_machine = 1.0 if task.assigned_machine else 0.0
            
            # Sequence progress in family
            seq_progress = task.sequence / len(family.tasks)
            
            # Is scheduled?
            is_scheduled = 1.0 if task.is_scheduled else 0.0
            
            features[i] = [is_available, urgency, proc_norm, has_machine, seq_progress, is_scheduled]
            
        return features
        
    def get_machine_features(self, machine_schedules: Dict[str, List[Tuple[float, float]]]) -> np.ndarray:
        """
        Get feature vector for all machines.
        
        Features per machine:
        - is_busy (0/1): Currently processing?
        - utilization (0-1): Fraction of time utilized
        - next_free_time_norm (0-1): When will be free?
        
        Returns:
            Array of shape (n_machines, 3)
        """
        n_machines = len(self.machines)
        features = np.zeros((n_machines, 3))
        
        horizon = 720  # 30 days in hours
        
        for i, machine in enumerate(self.machines):
            schedule = machine_schedules.get(machine, [])
            
            # Is currently busy?
            current_time = 0  # This should be passed as parameter
            is_busy = any(start <= current_time < end for start, end in schedule)
            
            # Calculate utilization
            total_busy = sum(end - start for start, end in schedule)
            utilization = min(total_busy / horizon, 1.0)
            
            # Next free time
            if schedule:
                next_free = max(end for _, end in schedule)
                next_free_norm = min(next_free / horizon, 1.0)
            else:
                next_free_norm = 0.0
                
            features[i] = [is_busy, utilization, next_free_norm]
            
        return features
        
    def _check_material_arrival(self, family: Family, current_time: float) -> bool:
        """Check if material has arrived for a family."""
        if not family.material_arrival:
            return True  # No material constraint
            
        # Convert material arrival date to hours from start
        # For simplicity, assume start date is today
        # In production, this would be properly calculated
        return True  # Simplified for now
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the loaded data."""
        stats = {
            'total_families': len(self.families),
            'total_tasks': len(self.tasks),
            'total_machines': len(self.machines),
            'urgent_families': sum(1 for f in self.families.values() if f.is_urgent),
            'tasks_with_machines': sum(1 for t in self.tasks if t.assigned_machine),
            'tasks_without_machines': sum(1 for t in self.tasks if not t.assigned_machine),
            'avg_tasks_per_family': len(self.tasks) / len(self.families) if self.families else 0,
            'avg_processing_time': np.mean([t.processing_time for t in self.tasks]) if self.tasks else 0,
        }
        return stats
        
    def reset(self):
        """Reset all scheduling state."""
        for task in self.tasks:
            task.is_scheduled = False
            task.start_time = None
            task.end_time = None
            task.machine_used = None
            
        for family in self.families.values():
            family.completed_sequences.clear()


if __name__ == "__main__":
    # Test loading
    import sys
    
    if len(sys.argv) > 1:
        snapshot_path = sys.argv[1]
    else:
        snapshot_path = "data/10_jobs.json"
        
    loader = SnapshotLoader(snapshot_path)
    stats = loader.get_statistics()
    
    print("\nSnapshot Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
        
    print("\nFirst 5 families:")
    for i, (fid, family) in enumerate(list(loader.families.items())[:5]):
        print(f"  {family}")
        for task in family.tasks[:3]:
            print(f"    - {task}")
            
    print("\nTask features shape:", loader.get_task_features().shape)
    print("Machine features shape:", loader.get_machine_features({}).shape)