"""
Enhanced break time constraints that support job splitting.
Allows jobs to be paused during breaks and resumed after.
"""

from datetime import datetime, time, timedelta
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

from .break_time_constraints import BreakTimeConstraints, BreakPeriod

logger = logging.getLogger(__name__)


@dataclass
class WorkSegment:
    """Represents a work segment between breaks."""
    start_time: datetime
    end_time: datetime
    duration_hours: float


class BreakTimeConstraintsWithSplitting(BreakTimeConstraints):
    """
    Extended break time constraints that support job splitting.
    Jobs can be paused during breaks and resumed after.
    """
    
    def get_work_segments_for_job(self, start_time: datetime, duration_hours: float) -> List[WorkSegment]:
        """
        Split a job into work segments that avoid breaks.
        
        Args:
            start_time: Proposed start time
            duration_hours: Total job duration in hours
            
        Returns:
            List of work segments that complete the job
        """
        segments = []
        current_time = start_time
        remaining_hours = duration_hours
        max_days = 7  # Prevent infinite loops
        
        for _ in range(max_days * 24):  # Max iterations
            if remaining_hours <= 0:
                break
                
            # Find next available work window
            work_start = self._find_next_work_start(current_time)
            work_end = self._find_next_break_start(work_start)
            
            # Calculate how much work can be done in this window
            available_hours = (work_end - work_start).total_seconds() / 3600
            work_hours = min(remaining_hours, available_hours)
            
            if work_hours > 0:
                segment_end = work_start + timedelta(hours=work_hours)
                segments.append(WorkSegment(
                    start_time=work_start,
                    end_time=segment_end,
                    duration_hours=work_hours
                ))
                remaining_hours -= work_hours
                
            # Move to next potential start time (after the break)
            current_time = work_end
            
        if remaining_hours > 0:
            logger.warning(f"Could not schedule full job duration. Remaining: {remaining_hours:.2f}h")
            
        return segments
    
    def _find_next_work_start(self, from_time: datetime) -> datetime:
        """Find the next time work can start (not in a break)."""
        current = from_time
        max_attempts = 100
        
        for _ in range(max_attempts):
            if not self._is_in_break(current):
                return current
                
            # Find which break we're in and jump to its end
            for break_period in self.breaks:
                if self._is_time_in_break(current, break_period):
                    break_end = self._get_break_end_datetime(current, break_period)
                    current = break_end + timedelta(minutes=1)
                    break
                    
        logger.warning("Could not find valid work start time")
        return from_time
    
    def _find_next_break_start(self, from_time: datetime) -> datetime:
        """Find when the next break starts."""
        # Check breaks for the next 24 hours
        for hours_ahead in range(24):
            check_time = from_time + timedelta(hours=hours_ahead)
            
            for break_period in self.breaks:
                break_start = self._get_break_start_datetime(check_time, break_period)
                
                # If this break starts after our current time and before any found so far
                if break_start > from_time:
                    # Check if we would hit this break
                    if check_time.date() == break_start.date() or (check_time.date() + timedelta(days=1)) == break_start.date():
                        return break_start
                        
        # No break found in next 24 hours, return end of day
        return from_time + timedelta(hours=24)
    
    def _is_in_break(self, check_time: datetime) -> bool:
        """Check if the given time is during any break."""
        day_of_week = (check_time.weekday() + 1) % 7
        
        for break_period in self.breaks:
            if break_period.applies_to_day(day_of_week):
                if self._is_time_in_break(check_time, break_period):
                    return True
        return False
    
    def _is_time_in_break(self, check_time: datetime, break_period: BreakPeriod) -> bool:
        """Check if a specific time falls within a break period."""
        break_start = check_time.replace(
            hour=break_period.start_time.hour,
            minute=break_period.start_time.minute,
            second=0,
            microsecond=0
        )
        break_end = check_time.replace(
            hour=break_period.end_time.hour,
            minute=break_period.end_time.minute,
            second=0,
            microsecond=0
        )
        
        # Handle breaks spanning midnight
        if break_period.end_time < break_period.start_time:
            # Break spans midnight
            return check_time >= break_start or check_time <= break_end
        else:
            # Normal break
            return break_start <= check_time <= break_end
    
    def _get_break_start_datetime(self, reference_time: datetime, break_period: BreakPeriod) -> datetime:
        """Get the datetime when a break starts."""
        return reference_time.replace(
            hour=break_period.start_time.hour,
            minute=break_period.start_time.minute,
            second=0,
            microsecond=0
        )
    
    def _get_break_end_datetime(self, reference_time: datetime, break_period: BreakPeriod) -> datetime:
        """Get the datetime when a break ends."""
        end_dt = reference_time.replace(
            hour=break_period.end_time.hour,
            minute=break_period.end_time.minute,
            second=0,
            microsecond=0
        )
        
        # Handle breaks spanning midnight
        if break_period.end_time < break_period.start_time:
            end_dt += timedelta(days=1)
            
        return end_dt
    
    def get_next_valid_start_quick(self, proposed_start: datetime) -> datetime:
        """
        Quickly find the next valid start time (for starting or resuming work).
        This doesn't check if the entire job fits - just finds the next work window.
        
        Args:
            proposed_start: Proposed start time
            
        Returns:
            Next time when work can begin
        """
        return self._find_next_work_start(proposed_start)