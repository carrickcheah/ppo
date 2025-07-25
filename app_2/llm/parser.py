"""
Schedule Parser

Advanced parsing for LLM-generated schedules with error recovery.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ScheduledTask:
    """Structured representation of a scheduled task."""
    job_id: str
    family_id: str
    sequence: int
    total_sequences: int
    machine_ids: List[int]
    start_time: datetime
    end_time: datetime
    processing_hours: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'job_id': self.job_id,
            'family_id': self.family_id,
            'sequence': f"{self.sequence}/{self.total_sequences}",
            'machine_ids': self.machine_ids,
            'start_time': self.start_time.strftime("%Y-%m-%d %H:%M"),
            'end_time': self.end_time.strftime("%Y-%m-%d %H:%M"),
            'processing_hours': self.processing_hours
        }


class ScheduleParser:
    """
    Advanced parser for LLM schedule outputs.
    """
    
    def __init__(self):
        """Initialize parser with regex patterns."""
        # Common patterns for schedule parsing
        self.patterns = {
            # Format: JobID -> machines[1,2,3] @ 2025-01-24 08:00 - 2025-01-24 10:00
            'arrow_format': re.compile(
                r'(\S+)\s*->\s*machines?\[?([\d,\s]+)\]?\s*@\s*'
                r'([\d-]+\s+[\d:]+)\s*-\s*([\d-]+\s+[\d:]+)'
            ),
            
            # Format: Job: JobID | Machine: 1,2,3 | Start: 08:00 | End: 10:00
            'pipe_format': re.compile(
                r'Job:\s*(\S+)\s*\|\s*Machines?:\s*([\d,\s]+)\s*\|\s*'
                r'Start:\s*([\d:]+)\s*\|\s*End:\s*([\d:]+)'
            ),
            
            # Format: Schedule JobID on machine(s) 1,2,3 from 08:00 to 10:00
            'text_format': re.compile(
                r'Schedule\s+(\S+)\s+on\s+machines?\s*([\d,\s]+)\s+'
                r'from\s+([\d:]+)\s+to\s+([\d:]+)'
            ),
            
            # Extract job components
            'job_id_pattern': re.compile(
                r'([A-Z]+\d+)-([A-Z0-9-]+)-(\d+)/(\d+)'
            ),
            
            # Time patterns
            'full_datetime': re.compile(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})'),
            'time_only': re.compile(r'(\d{1,2}:\d{2})'),
        }
        
    def parse_schedule(
        self,
        llm_output: str,
        base_date: Optional[str] = None
    ) -> List[ScheduledTask]:
        """
        Parse LLM output into structured schedule.
        
        Args:
            llm_output: Raw text from LLM
            base_date: Base date for relative times (YYYY-MM-DD)
            
        Returns:
            List of ScheduledTask objects
        """
        if base_date is None:
            base_date = datetime.now().strftime("%Y-%m-%d")
        
        base_datetime = datetime.strptime(base_date + " 00:00", "%Y-%m-%d %H:%M")
        
        # Extract schedule section if present
        schedule_text = self._extract_schedule_section(llm_output)
        
        # Parse each line
        scheduled_tasks = []
        lines = schedule_text.strip().split('\n')
        
        for line in lines:
            task = self._parse_line(line, base_datetime)
            if task:
                scheduled_tasks.append(task)
        
        logger.info(f"Parsed {len(scheduled_tasks)} tasks from LLM output")
        
        return scheduled_tasks
        
    def _extract_schedule_section(self, text: str) -> str:
        """Extract the schedule section from LLM output."""
        # Look for common section markers
        markers = [
            "=== FINAL SCHEDULE ===",
            "=== SCHEDULE ===",
            "Schedule:",
            "Final Schedule:",
            "Here is the schedule:",
            "Optimal schedule:"
        ]
        
        for marker in markers:
            if marker in text:
                parts = text.split(marker)
                if len(parts) > 1:
                    return parts[-1]
        
        # No marker found, assume entire text is schedule
        return text
        
    def _parse_line(self, line: str, base_datetime: datetime) -> Optional[ScheduledTask]:
        """Parse a single line into a scheduled task."""
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('==='):
            return None
        
        # Try each format pattern
        for format_name, pattern in [
            ('arrow', self.patterns['arrow_format']),
            ('pipe', self.patterns['pipe_format']),
            ('text', self.patterns['text_format'])
        ]:
            match = pattern.search(line)
            if match:
                return self._parse_match(match, format_name, base_datetime)
        
        # If no pattern matches, try flexible parsing
        return self._flexible_parse(line, base_datetime)
        
    def _parse_match(
        self,
        match: re.Match,
        format_name: str,
        base_datetime: datetime
    ) -> Optional[ScheduledTask]:
        """Parse a regex match into a scheduled task."""
        try:
            if format_name == 'arrow':
                job_id = match.group(1)
                machines = [int(m.strip()) for m in match.group(2).split(',')]
                start_str = match.group(3)
                end_str = match.group(4)
                
                # Parse times
                start_time = self._parse_datetime(start_str, base_datetime)
                end_time = self._parse_datetime(end_str, base_datetime)
                
            elif format_name == 'pipe':
                job_id = match.group(1)
                machines = [int(m.strip()) for m in match.group(2).split(',')]
                
                # Times are relative to base date
                start_time = self._parse_time_only(match.group(3), base_datetime)
                end_time = self._parse_time_only(match.group(4), base_datetime)
                
            elif format_name == 'text':
                job_id = match.group(1)
                machines = [int(m.strip()) for m in match.group(2).split(',')]
                
                # Times are relative to base date
                start_time = self._parse_time_only(match.group(3), base_datetime)
                end_time = self._parse_time_only(match.group(4), base_datetime)
            
            # Handle day rollover
            if end_time < start_time:
                end_time += timedelta(days=1)
            
            # Extract job components
            family_id, sequence, total_sequences = self._parse_job_id(job_id)
            
            # Calculate processing hours
            processing_hours = (end_time - start_time).total_seconds() / 3600
            
            return ScheduledTask(
                job_id=job_id,
                family_id=family_id,
                sequence=sequence,
                total_sequences=total_sequences,
                machine_ids=machines,
                start_time=start_time,
                end_time=end_time,
                processing_hours=processing_hours
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse match: {e}")
            return None
            
    def _flexible_parse(self, line: str, base_datetime: datetime) -> Optional[ScheduledTask]:
        """Flexible parsing for non-standard formats."""
        # Extract key components using flexible patterns
        
        # Find job ID (looks like XXXX12345-YY-1/3)
        job_match = re.search(r'([A-Z]+\d+[-][A-Z0-9-]+[-]\d+/\d+)', line)
        if not job_match:
            return None
        
        job_id = job_match.group(1)
        
        # Find machine numbers
        machine_matches = re.findall(r'machines?\s*[:=]?\s*\[?([\d,\s]+)\]?', line, re.IGNORECASE)
        if not machine_matches:
            # Try alternative patterns
            machine_matches = re.findall(r'on\s+machines?\s+([\d,\s]+)', line, re.IGNORECASE)
        
        if not machine_matches:
            return None
        
        machines = [int(m.strip()) for m in machine_matches[0].split(',') if m.strip()]
        
        # Find times
        time_matches = self.patterns['full_datetime'].findall(line)
        if len(time_matches) >= 2:
            start_time = datetime.strptime(time_matches[0], "%Y-%m-%d %H:%M")
            end_time = datetime.strptime(time_matches[1], "%Y-%m-%d %H:%M")
        else:
            # Try time-only format
            time_matches = self.patterns['time_only'].findall(line)
            if len(time_matches) >= 2:
                start_time = self._parse_time_only(time_matches[0], base_datetime)
                end_time = self._parse_time_only(time_matches[1], base_datetime)
                if end_time < start_time:
                    end_time += timedelta(days=1)
            else:
                return None
        
        # Extract job components
        family_id, sequence, total_sequences = self._parse_job_id(job_id)
        
        # Calculate processing hours
        processing_hours = (end_time - start_time).total_seconds() / 3600
        
        return ScheduledTask(
            job_id=job_id,
            family_id=family_id,
            sequence=sequence,
            total_sequences=total_sequences,
            machine_ids=machines,
            start_time=start_time,
            end_time=end_time,
            processing_hours=processing_hours
        )
        
    def _parse_job_id(self, job_id: str) -> Tuple[str, int, int]:
        """Parse job ID into components."""
        # First try the pattern matcher
        match = self.patterns['job_id_pattern'].match(job_id)
        if match:
            family_id = match.group(1) + match.group(2)
            sequence = int(match.group(3))
            total_sequences = int(match.group(4))
            return family_id, sequence, total_sequences
        
        # Fallback parsing for different formats
        parts = job_id.split('-')
        if len(parts) >= 2:
            # Family ID is usually the first part
            family_id = parts[0]
            
            # Look for sequence in the last part
            last_part = parts[-1]
            if '/' in last_part:
                seq_parts = last_part.split('/')
                # Extract sequence number (might have additional text)
                seq_text = seq_parts[0]
                # Remove any non-digit characters from the end
                seq_num = ''.join(c for c in seq_text if c.isdigit())
                if seq_num:
                    sequence = int(seq_num)
                else:
                    sequence = 1
                total_sequences = int(seq_parts[1])
                return family_id, sequence, total_sequences
        
        # Default fallback
        return job_id.split('-')[0] if '-' in job_id else job_id, 1, 1
        
    def _parse_datetime(self, datetime_str: str, base: datetime) -> datetime:
        """Parse datetime string."""
        datetime_str = datetime_str.strip()
        
        # Try full datetime format
        try:
            return datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
        except:
            pass
        
        # Try date with different separators
        for fmt in ["%Y/%m/%d %H:%M", "%d-%m-%Y %H:%M", "%d/%m/%Y %H:%M"]:
            try:
                return datetime.strptime(datetime_str, fmt)
            except:
                pass
        
        # If only time, use base date
        return self._parse_time_only(datetime_str, base)
        
    def _parse_time_only(self, time_str: str, base: datetime) -> datetime:
        """Parse time-only string using base date."""
        time_str = time_str.strip()
        
        # Handle HH:MM format
        if ':' in time_str:
            parts = time_str.split(':')
            hour = int(parts[0])
            minute = int(parts[1])
            
            return base.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        # Handle HHMM format
        if len(time_str) == 4 and time_str.isdigit():
            hour = int(time_str[:2])
            minute = int(time_str[2:])
            return base.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        raise ValueError(f"Cannot parse time: {time_str}")


def test_parser():
    """Test schedule parser."""
    parser = ScheduleParser()
    
    # Test various formats
    test_outputs = [
        # Arrow format
        "JOTP25070237-CT10-013A-1/5 -> machines[80] @ 2025-01-24 08:00 - 2025-01-24 23:06",
        
        # Pipe format
        "Job: JOAW25060139-CP01-123-1/3 | Machines: 57,64 | Start: 08:00 | End: 10:30",
        
        # Text format with reasoning
        """Let me schedule these jobs:
        First, I'll schedule the urgent job:
        Schedule JOTP25070237-CT10-013A-1/5 on machine 80 from 08:00 to 23:06
        
        Next, the multi-machine job:
        JOAW25060139-CP01-123-1/3 needs machines 57 and 64, scheduling from 08:00 to 10:30
        """,
        
        # Mixed format
        """=== FINAL SCHEDULE ===
        JOTP25070237-CT10-013A-1/5 -> machines[80] @ 2025-01-24 08:00 - 2025-01-24 23:06
        Job: JOAW25060139-CP01-123-1/3 | Machines: 57,64 | Start: 08:00 | End: 10:30
        JOEX25070123-TP01-1/1 on machines 15,16,17 from 10:30 to 12:00
        """
    ]
    
    for i, output in enumerate(test_outputs):
        print(f"\n=== TEST {i+1} ===")
        print(f"Input:\n{output}")
        
        tasks = parser.parse_schedule(output, "2025-01-24")
        
        print(f"\nParsed {len(tasks)} tasks:")
        for task in tasks:
            print(f"  {task.job_id}: machines {task.machine_ids} "
                  f"@ {task.start_time.strftime('%H:%M')} - {task.end_time.strftime('%H:%M')} "
                  f"({task.processing_hours:.1f}h)")


if __name__ == "__main__":
    test_parser()