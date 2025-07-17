"""
Break time constraints for production scheduling.
Implements hard constraints that prevent scheduling during break times.
"""

try:
    import mysql.connector
    from mysql.connector import Error
except ImportError:
    # If mysql.connector not available, define placeholder
    mysql = None
    class Error(Exception):
        pass
from datetime import datetime, time, timedelta
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class BreakPeriod:
    """Represents a break period with time and day constraints."""
    name: str
    start_time: time
    end_time: time
    day_of_week: Optional[int]  # None means every day, 0=Sunday, 6=Saturday
    duration_minutes: int
    
    def __post_init__(self):
        """Convert timedelta to time if needed."""
        # Handle timedelta from database
        if isinstance(self.start_time, timedelta):
            total_seconds = int(self.start_time.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            self.start_time = time(hours, minutes, seconds)
            
        if isinstance(self.end_time, timedelta):
            total_seconds = int(self.end_time.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            self.end_time = time(hours, minutes, seconds)
    
    def applies_to_day(self, day_of_week: int) -> bool:
        """Check if this break applies to given day of week."""
        if self.day_of_week is None:
            return True  # Applies every day
        return self.day_of_week == day_of_week
    
    def overlaps_time_range(self, start_time: time, end_time: time) -> bool:
        """Check if break overlaps with given time range."""
        # Handle breaks that span midnight
        if self.end_time < self.start_time:
            # Break spans midnight (e.g., 23:00 - 02:00)
            return (start_time >= self.start_time or start_time < self.end_time or
                    end_time >= self.start_time or end_time < self.end_time)
        else:
            # Normal break within same day
            return not (end_time <= self.start_time or start_time >= self.end_time)


class BreakTimeConstraints:
    """Manages break time constraints from database."""
    
    def __init__(self, db_config: Optional[Dict] = None):
        """
        Initialize break time constraints.
        
        Args:
            db_config: Database configuration dict with keys:
                - host, user, password, database
        """
        self.db_config = db_config or {
            'host': 'localhost',
            'user': 'myuser',
            'password': 'mypassword',
            'database': 'nex_valiant'
        }
        self.breaks: List[BreakPeriod] = []
        self.load_breaks_from_db()
        
    def load_breaks_from_db(self) -> None:
        """Load active break times from database."""
        if mysql is None or mysql.connector is None:
            logger.warning("mysql.connector not available, using default breaks")
            self._load_default_breaks()
            return
            
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor(dictionary=True)
            
            query = """
                SELECT 
                    name,
                    start_time,
                    end_time,
                    day_of_week,
                    duration_minutes
                FROM ai_breaktimes
                WHERE is_active = 1
                ORDER BY 
                    CASE WHEN day_of_week IS NULL THEN -1 ELSE day_of_week END,
                    start_time
            """
            
            cursor.execute(query)
            breaks_data = cursor.fetchall()
            
            self.breaks = []
            for break_data in breaks_data:
                break_period = BreakPeriod(
                    name=break_data['name'],
                    start_time=break_data['start_time'],
                    end_time=break_data['end_time'],
                    day_of_week=break_data['day_of_week'],
                    duration_minutes=break_data['duration_minutes']
                )
                self.breaks.append(break_period)
                
            logger.info(f"Loaded {len(self.breaks)} active break periods")
            for b in self.breaks:
                day_desc = "Every day" if b.day_of_week is None else f"Day {b.day_of_week}"
                logger.debug(f"  {b.name}: {b.start_time}-{b.end_time} ({day_desc})")
                
        except Error as e:
            logger.error(f"Error loading breaks from database: {e}")
            # Fall back to minimal breaks if DB fails
            self._load_default_breaks()
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
    
    def _load_default_breaks(self) -> None:
        """Load default breaks if database unavailable."""
        logger.warning("Using default break times")
        self.breaks = [
            # Daily breaks (day_of_week = None means every day)
            BreakPeriod("Machine Off", time(1, 0), time(6, 30), None, 330),
            BreakPeriod("Morning Tea Break", time(9, 45), time(10, 0), None, 15),
            BreakPeriod("Lunch Break", time(12, 45), time(13, 30), None, 45),
            BreakPeriod("Afternoon Tea Break", time(15, 15), time(15, 30), None, 15),
            BreakPeriod("Dinner Break", time(18, 0), time(19, 0), None, 60),
            BreakPeriod("Supper Break", time(23, 0), time(23, 30), None, 30),
            # Weekly breaks (0=Sunday, 6=Saturday)
            BreakPeriod("Sunday Off", time(0, 0), time(23, 59, 59), 0, 1440),
            BreakPeriod("Saturday Half Day", time(13, 0), time(23, 59, 59), 6, 660),
        ]
    
    def is_valid_time_slot(self, start_datetime: datetime, duration_hours: float) -> bool:
        """
        Check if a time slot is valid (doesn't overlap any breaks).
        
        Args:
            start_datetime: Start time as datetime
            duration_hours: Duration in hours
            
        Returns:
            True if entire duration avoids all breaks
        """
        end_datetime = start_datetime + timedelta(hours=duration_hours)
        
        # Check each day that the job spans
        current_dt = start_datetime
        while current_dt < end_datetime:
            day_of_week = current_dt.weekday()  # Monday=0, Sunday=6
            # Convert to our format (Sunday=0, Saturday=6)
            day_of_week = (day_of_week + 1) % 7
            
            # Get start and end times for this day
            day_start = current_dt.time()
            next_day = current_dt.replace(hour=0, minute=0, second=0) + timedelta(days=1)
            
            if next_day > end_datetime:
                day_end = end_datetime.time()
            else:
                day_end = time(23, 59, 59)
            
            # Check breaks for this day
            for break_period in self.breaks:
                if break_period.applies_to_day(day_of_week):
                    if break_period.overlaps_time_range(day_start, day_end):
                        logger.debug(f"Time slot overlaps with {break_period.name}")
                        return False
            
            # Move to next day
            current_dt = next_day
            
        return True
    
    def get_next_valid_start(self, proposed_start: datetime, duration_hours: float) -> datetime:
        """
        Find the next valid start time after breaks.
        
        Args:
            proposed_start: Proposed start time
            duration_hours: Job duration in hours
            
        Returns:
            Next valid start time that avoids all breaks
        """
        current_start = proposed_start
        max_attempts = 100  # Prevent infinite loops
        
        # For long jobs, just find next work window
        if duration_hours > 8.0:
            return self._find_next_work_window(current_start)
        
        for _ in range(max_attempts):
            if self.is_valid_time_slot(current_start, duration_hours):
                return current_start
            
            # Find the break that's blocking us
            blocking_break_end = None
            day_of_week = (current_start.weekday() + 1) % 7
            
            for break_period in self.breaks:
                if break_period.applies_to_day(day_of_week):
                    break_start_dt = current_start.replace(
                        hour=break_period.start_time.hour,
                        minute=break_period.start_time.minute,
                        second=0,
                        microsecond=0
                    )
                    break_end_dt = current_start.replace(
                        hour=break_period.end_time.hour,
                        minute=break_period.end_time.minute,
                        second=0,
                        microsecond=0
                    )
                    
                    # Handle breaks spanning midnight
                    if break_period.end_time < break_period.start_time:
                        break_end_dt += timedelta(days=1)
                    
                    # Check if current start overlaps with this break
                    job_end = current_start + timedelta(hours=duration_hours)
                    
                    # If job would overlap with break
                    if not (job_end <= break_start_dt or current_start >= break_end_dt):
                        if blocking_break_end is None or break_end_dt > blocking_break_end:
                            blocking_break_end = break_end_dt
            
            if blocking_break_end:
                # Jump to after the break
                current_start = blocking_break_end + timedelta(minutes=1)
            else:
                # No blocking break found, move to next hour
                current_start += timedelta(hours=1)
        
        logger.warning(f"Could not find valid start time after {max_attempts} attempts")
        return proposed_start
    
    def _find_next_work_window(self, from_time: datetime) -> datetime:
        """
        Find the next available work window for job splitting.
        This doesn't check if entire job fits - just finds next work start.
        
        Args:
            from_time: Time to start searching from
            
        Returns:
            Next datetime when work can begin
        """
        current = from_time
        max_search_hours = 48  # Search up to 48 hours ahead
        
        for h in range(int(max_search_hours * 60)):  # Check every minute
            check_time = current + timedelta(minutes=h)
            day_of_week = (check_time.weekday() + 1) % 7
            
            # Check if this time is during any break
            is_break_time = False
            for break_period in self.breaks:
                if break_period.applies_to_day(day_of_week):
                    # Check if current time falls within break
                    break_start_dt = check_time.replace(
                        hour=break_period.start_time.hour,
                        minute=break_period.start_time.minute,
                        second=0,
                        microsecond=0
                    )
                    break_end_dt = check_time.replace(
                        hour=break_period.end_time.hour,
                        minute=break_period.end_time.minute,
                        second=0,
                        microsecond=0
                    )
                    
                    # Handle breaks spanning midnight
                    if break_period.end_time < break_period.start_time:
                        break_end_dt += timedelta(days=1)
                    
                    # Check if we're in this break
                    if break_start_dt <= check_time < break_end_dt:
                        is_break_time = True
                        break
            
            # If not in a break, we found a valid work window
            if not is_break_time:
                return check_time
        
        # If no valid time found, return original (shouldn't happen)
        logger.warning(f"Could not find work window within {max_search_hours} hours")
        return from_time
    
    @lru_cache(maxsize=128)
    def get_breaks_for_day(self, day_of_week: int) -> List[BreakPeriod]:
        """
        Get all breaks that apply to a specific day.
        
        Args:
            day_of_week: 0=Sunday, 6=Saturday
            
        Returns:
            List of breaks for that day
        """
        return [b for b in self.breaks if b.applies_to_day(day_of_week)]
    
    def hours_to_datetime(self, hours: float, base_date: Optional[datetime] = None) -> datetime:
        """
        Convert hours (float) to datetime.
        
        Args:
            hours: Hours since start (e.g., 10.5 = 10:30 AM on day 1)
            base_date: Base date to start from (default: today at midnight)
            
        Returns:
            Datetime object
        """
        if base_date is None:
            base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Convert to float to handle numpy types
        return base_date + timedelta(hours=float(hours))
    
    def datetime_to_hours(self, dt: datetime, base_date: Optional[datetime] = None) -> float:
        """
        Convert datetime back to hours (float).
        
        Args:
            dt: Datetime to convert
            base_date: Base date (default: today at midnight)
            
        Returns:
            Hours since base date
        """
        if base_date is None:
            base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        delta = dt - base_date
        return delta.total_seconds() / 3600