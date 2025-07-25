"""
Data Adapter for LLM Scheduler

Converts between PPO data format and LLM-friendly text format.
Loads data from snapshots and formats for prompts.
"""

import json
import logging
from typing import Dict, List, Any, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class DataAdapter:
    """
    Adapts scheduling data between different formats.
    """
    
    def __init__(self, snapshot_path: str = None):
        """
        Initialize data adapter.
        
        Args:
            snapshot_path: Path to snapshot file. If None, uses default.
        """
        if snapshot_path is None:
            # Default to normal snapshot
            snapshot_path = "/Users/carrickcheah/Project/ppo/app_2/phase3/snapshots/snapshot_normal.json"
        
        self.snapshot_path = Path(snapshot_path)
        self.data = None
        self.families = None
        self.machines = None
        
    def load_snapshot(self) -> Dict[str, Any]:
        """Load data from snapshot file."""
        if not self.snapshot_path.exists():
            raise FileNotFoundError(f"Snapshot not found: {self.snapshot_path}")
        
        with open(self.snapshot_path, 'r') as f:
            self.data = json.load(f)
        
        self.families = self.data.get('families', {})
        self.machines = self.data.get('machines', [])  # machines is a list, not dict
        
        logger.info(f"Loaded snapshot: {self.data['metadata']['total_tasks']} tasks, "
                   f"{self.data['metadata']['total_families']} families, "
                   f"{self.data['metadata']['total_machines']} machines")
        
        return self.data
        
    def format_for_llm(self, max_jobs: int = None) -> Tuple[str, str]:
        """
        Format job and machine data for LLM prompt.
        
        Args:
            max_jobs: Maximum number of jobs to include (for testing)
            
        Returns:
            Tuple of (jobs_text, machines_text)
        """
        if self.data is None:
            self.load_snapshot()
        
        # Format jobs
        jobs_text = self._format_jobs(max_jobs)
        
        # Format machines
        machines_text = self._format_machines()
        
        return jobs_text, machines_text
        
    def _format_jobs(self, max_jobs: int = None) -> str:
        """Format job data as readable text."""
        lines = []
        job_count = 0
        
        # Group by urgency for better organization
        urgent_jobs = []
        normal_jobs = []
        
        for family_id, family_data in self.families.items():
            for task in family_data['tasks']:
                if task['status'] != 'pending':
                    continue
                    
                job_info = {
                    'id': f"{family_id}-{task['process_name']}",
                    'family': family_id,
                    'sequence': f"{task['sequence']}/{family_data['total_sequences']}",
                    'processing_time': task['processing_time'],
                    'machines': task['capable_machines'],
                    'lcd_date': family_data['lcd_date'],
                    'days_remaining': family_data['lcd_days_remaining'],
                    'important': family_data['is_important']
                }
                
                if family_data['lcd_days_remaining'] <= 7:
                    urgent_jobs.append(job_info)
                else:
                    normal_jobs.append(job_info)
                
                job_count += 1
                if max_jobs and job_count >= max_jobs:
                    break
            
            if max_jobs and job_count >= max_jobs:
                break
        
        # Format urgent jobs first
        if urgent_jobs:
            lines.append("=== URGENT JOBS (Due within 7 days) ===")
            for job in urgent_jobs:
                lines.append(self._format_single_job(job))
            lines.append("")
        
        # Then normal jobs
        if normal_jobs:
            lines.append("=== NORMAL JOBS ===")
            for job in normal_jobs:
                lines.append(self._format_single_job(job))
        
        return "\n".join(lines)
        
    def _format_single_job(self, job: Dict[str, Any]) -> str:
        """Format a single job as text."""
        parts = [
            f"Job: {job['id']}",
            f"Family: {job['family']}",
            f"Sequence: {job['sequence']}",
            f"Time: {job['processing_time']:.1f}h",
            f"Machines: {','.join(map(str, job['machines']))}",
            f"LCD: {job['lcd_date']} ({job['days_remaining']}d)",
        ]
        
        if job['important']:
            parts.append("IMPORTANT")
        
        return " | ".join(parts)
        
    def _format_machines(self) -> str:
        """Format machine data as readable text."""
        lines = []
        
        # Group machines by type
        machines_by_type = defaultdict(list)
        
        # Handle list format from snapshot
        if isinstance(self.machines, list):
            for machine_data in self.machines:
                machine_id = machine_data['machine_id']
                type_id = machine_data['machine_type_id']
                machines_by_type[type_id].append({
                    'id': machine_id,
                    'name': machine_data['machine_name']  # Changed from 'name' to 'machine_name'
                })
        else:
            # Handle dict format (if any)
            for machine_id, machine_data in self.machines.items():
                type_id = machine_data['machine_type_id']
                machines_by_type[type_id].append({
                    'id': machine_id,
                    'name': machine_data['name']
                })
        
        # Format by type
        for type_id, machines in sorted(machines_by_type.items()):
            machine_list = ", ".join([f"{m['id']}({m['name']})" for m in machines])
            lines.append(f"Type {type_id}: {machine_list}")
        
        return "\n".join(lines)
        
    def parse_llm_schedule(self, llm_output: str, start_date: str = None) -> List[Dict[str, Any]]:
        """
        Parse LLM output into structured schedule format.
        
        Args:
            llm_output: Raw text from LLM
            start_date: Start date for scheduling (default: now)
            
        Returns:
            List of scheduled tasks
        """
        if start_date is None:
            start_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        base_time = datetime.strptime(start_date, "%Y-%m-%d %H:%M")
        schedule = []
        
        # Parse each line looking for schedule entries
        lines = llm_output.strip().split('\n')
        for line in lines:
            task = self._parse_schedule_line(line, base_time)
            if task:
                schedule.append(task)
        
        return schedule
        
    def _parse_schedule_line(self, line: str, base_time: datetime) -> Dict[str, Any]:
        """
        Parse a single schedule line.
        
        Expected formats:
        - "JOAW25060101-CP01-123 -> machines[57,64] @ 2025-01-24 08:00 - 2025-01-24 10:30"
        - "Job: JOAW25060101-CP01-123 | Machine: 57 | Start: 08:00 | End: 10:30"
        """
        # Try different parsing patterns
        import re
        
        # Pattern 1: Arrow format
        pattern1 = r"(\S+)\s*->\s*machines?\[?([\d,\s]+)\]?\s*@\s*([\d-]+\s+[\d:]+)\s*-\s*([\d-]+\s+[\d:]+)"
        match = re.search(pattern1, line)
        if match:
            job_id = match.group(1)
            machines = [int(m.strip()) for m in match.group(2).split(',')]
            start_str = match.group(3)
            end_str = match.group(4)
            
            return {
                'job_id': job_id,
                'machine_ids': machines,
                'start_time': start_str,
                'end_time': end_str
            }
        
        # Pattern 2: Pipe format
        pattern2 = r"Job:\s*(\S+)\s*\|\s*Machine:\s*([\d,\s]+)\s*\|\s*Start:\s*([\d:]+)\s*\|\s*End:\s*([\d:]+)"
        match = re.search(pattern2, line)
        if match:
            job_id = match.group(1)
            machines = [int(m.strip()) for m in match.group(2).split(',')]
            start_time = base_time.replace(
                hour=int(match.group(3).split(':')[0]),
                minute=int(match.group(3).split(':')[1])
            )
            end_time = base_time.replace(
                hour=int(match.group(4).split(':')[0]),
                minute=int(match.group(4).split(':')[1])
            )
            
            # Handle day overflow
            if end_time < start_time:
                end_time += timedelta(days=1)
            
            return {
                'job_id': job_id,
                'machine_ids': machines,
                'start_time': start_time.strftime("%Y-%m-%d %H:%M"),
                'end_time': end_time.strftime("%Y-%m-%d %H:%M")
            }
        
        return None
        
    def format_schedule_output(self, schedule: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Format schedule for API compatibility.
        
        Args:
            schedule: List of scheduled tasks
            
        Returns:
            API-compatible schedule format
        """
        # Group by job family
        jobs_output = defaultdict(list)
        
        for task in schedule:
            job_id = task['job_id']
            family_id = job_id.split('-')[0]
            
            jobs_output[family_id].append({
                'task_id': job_id,
                'machine_ids': task['machine_ids'],
                'start_time': task['start_time'],
                'end_time': task['end_time'],
                'status': 'scheduled'
            })
        
        # Calculate metrics
        if schedule:
            start_times = [datetime.strptime(t['start_time'], "%Y-%m-%d %H:%M") for t in schedule]
            end_times = [datetime.strptime(t['end_time'], "%Y-%m-%d %H:%M") for t in schedule]
            makespan = (max(end_times) - min(start_times)).total_seconds() / 3600
        else:
            makespan = 0
        
        return {
            'status': 'success',
            'schedule': dict(jobs_output),
            'metrics': {
                'total_jobs': len(schedule),
                'makespan_hours': makespan,
                'families_scheduled': len(jobs_output)
            },
            'timestamp': datetime.now().isoformat()
        }


def test_data_adapter():
    """Test data adapter functionality."""
    adapter = DataAdapter()
    
    # Load and format data
    jobs_text, machines_text = adapter.format_for_llm(max_jobs=10)
    
    print("=== JOBS ===")
    print(jobs_text)
    print("\n=== MACHINES ===")
    print(machines_text)
    
    # Test parsing
    sample_output = """
    JOTP25070237-CT10-013A-1/5 -> machines[80] @ 2025-01-24 08:00 - 2025-01-24 23:06
    JOAW25060139-CP01-123-1/3 -> machines[57,64] @ 2025-01-24 08:00 - 2025-01-24 10:30
    """
    
    schedule = adapter.parse_llm_schedule(sample_output)
    print("\n=== PARSED SCHEDULE ===")
    for task in schedule:
        print(task)
    
    # Test output formatting
    output = adapter.format_schedule_output(schedule)
    print("\n=== API OUTPUT ===")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    test_data_adapter()