# Job Splitting Fix for Break Constraints

## Problem
The current implementation tries to find continuous time blocks for entire job durations (e.g., 20 hours), which is impossible with frequent breaks.

## Solution
Implement job splitting as shown in the visualization:
- Jobs can be paused during breaks
- Jobs resume after breaks end
- Track partial completion

## Implementation Steps

1. **Modify `find_next_valid_start_time`**:
   - Instead of checking full duration, check for any valid work window
   - Return the next available work slot, even if it's just 1 hour

2. **Add job state tracking**:
   - `remaining_hours`: Track how much work is left
   - `segments`: List of (start_time, end_time) for each work segment
   - `is_paused`: Whether job is currently paused for break

3. **Update scheduling logic**:
   ```python
   def schedule_job_with_breaks(job, machine, start_time):
       remaining = job.duration
       segments = []
       current_time = start_time
       
       while remaining > 0:
           # Find next work window
           work_start = find_next_work_window(current_time)
           work_end = find_next_break_start(work_start)
           
           # Work for available duration
           work_duration = min(remaining, hours_between(work_start, work_end))
           segments.append((work_start, work_start + work_duration))
           
           remaining -= work_duration
           current_time = work_end
       
       return segments
   ```

4. **Benefits**:
   - Matches real production behavior
   - Eliminates "Could not find valid start time" errors
   - Better machine utilization
   - More realistic scheduling