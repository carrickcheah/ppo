"""
Prompt Templates for LLM Scheduler

Contains various prompt strategies for job scheduling.
"""

from typing import Dict, List, Any
from datetime import datetime


class SchedulingPrompts:
    """
    Collection of prompt templates for different scheduling strategies.
    """
    
    @staticmethod
    def get_system_prompt() -> str:
        """Base system prompt for all scheduling tasks."""
        return """You are an expert production scheduler with deep knowledge of manufacturing operations.
Your task is to create optimal job schedules while respecting all constraints.

Key principles:
1. SEQUENCE CONSTRAINTS: Jobs within a family MUST be scheduled in order (1/3 before 2/3 before 3/3)
2. MACHINE COMPATIBILITY: Jobs can only run on specified machines
3. NO OVERLAPS: Each machine can only process one job at a time
4. MULTI-MACHINE JOBS: When a job requires multiple machines (e.g., machines[57,64,65]), ALL machines must be available and occupied simultaneously
5. MINIMIZE LATENESS: Prioritize jobs with tight deadlines (LCD dates)
6. MAXIMIZE EFFICIENCY: Minimize idle time and maximize machine utilization

Output format: Each scheduled job on a new line as:
JobID -> machines[machine_ids] @ YYYY-MM-DD HH:MM - YYYY-MM-DD HH:MM

Time should be in 24-hour format. Be precise with timings."""

    @staticmethod
    def direct_scheduling_prompt(jobs_text: str, machines_text: str, start_time: str) -> str:
        """Direct scheduling prompt - single shot approach."""
        return f"""Schedule the following jobs starting from {start_time}:

{jobs_text}

Available machines:
{machines_text}

Create an optimal schedule respecting all constraints. Prioritize urgent jobs and important jobs.
Output the complete schedule with precise timings."""

    @staticmethod
    def chain_of_thought_prompt(jobs_text: str, machines_text: str, start_time: str) -> str:
        """Chain of thought prompt - step by step reasoning."""
        return f"""Let's schedule these jobs step by step, starting from {start_time}.

{jobs_text}

Available machines:
{machines_text}

Think through this systematically:
1. First, identify all job families and their sequences
2. Identify urgent jobs (due within 7 days) and important jobs
3. Check which jobs are ready to schedule (first in sequence or previous completed)
4. For each ready job, find the earliest available time on compatible machines
5. For multi-machine jobs, ensure ALL required machines are free simultaneously
6. Schedule jobs considering their urgency and importance
7. Continue until all jobs are scheduled

Show your reasoning for each scheduling decision, then provide the final schedule."""

    @staticmethod
    def constraint_focused_prompt(jobs_text: str, machines_text: str, start_time: str) -> str:
        """Constraint-focused prompt - emphasizes constraint checking."""
        return f"""Schedule these jobs with strict constraint validation, starting from {start_time}:

{jobs_text}

Available machines:
{machines_text}

For EVERY job you schedule:
1. CHECK: Is this job ready? (first in family OR previous job completed)
2. CHECK: Are ALL required machines available at the proposed time?
3. CHECK: Will this complete before the LCD date?
4. VERIFY: No time conflicts with already scheduled jobs

Provide the schedule with explicit constraint validation for each job."""

    @staticmethod
    def iterative_prompt(current_schedule: str, remaining_jobs: str, machines_text: str, 
                        issues: List[str] = None) -> str:
        """Iterative refinement prompt - improve existing schedule."""
        prompt = f"""Improve this schedule by addressing the identified issues:

Current schedule:
{current_schedule}

Remaining unscheduled jobs:
{remaining_jobs}

Available machines:
{machines_text}
"""
        
        if issues:
            prompt += "\nIssues to fix:\n"
            for issue in issues:
                prompt += f"- {issue}\n"
        
        prompt += "\nProvide an improved schedule that fixes these issues while maintaining all constraints."
        
        return prompt

    @staticmethod
    def example_based_prompt(jobs_text: str, machines_text: str, start_time: str) -> str:
        """Few-shot learning prompt with examples."""
        return f"""Here's how to schedule jobs correctly:

EXAMPLE:
Given jobs:
- JOAW001-CP01-1/3: 2h on machines[57,64], LCD: 2025-01-26 (2 days)
- JOAW001-CP02-2/3: 3h on machine[80], LCD: 2025-01-26 (2 days)
- JOAW002-TP01-1/1: 1h on machine[57], LCD: 2025-01-28 (4 days)

Correct schedule:
JOAW001-CP01-1/3 -> machines[57,64] @ 2025-01-24 08:00 - 2025-01-24 10:00
JOAW001-CP02-2/3 -> machines[80] @ 2025-01-24 10:00 - 2025-01-24 13:00
JOAW002-TP01-1/1 -> machines[57] @ 2025-01-24 10:00 - 2025-01-24 11:00

Note: JOAW001-CP01 uses machines 57 AND 64 simultaneously. JOAW002-TP01 can start after JOAW001-CP01 finishes on machine 57.

NOW YOUR TURN:
Schedule these jobs starting from {start_time}:

{jobs_text}

Available machines:
{machines_text}

Create the schedule following the example format."""

    @staticmethod
    def decomposition_prompt(jobs_text: str, machines_text: str, start_time: str) -> str:
        """Decomposition prompt - break problem into subproblems."""
        return f"""Let's solve this scheduling problem by decomposition, starting from {start_time}:

{jobs_text}

Available machines:
{machines_text}

Step 1: Group jobs by urgency
- Critical (due in 1-3 days)
- Urgent (due in 4-7 days)  
- Normal (due in 8+ days)

Step 2: Within each group, identify job families and dependencies

Step 3: Schedule each group in priority order:
- First schedule all critical jobs
- Then urgent jobs in remaining slots
- Finally normal jobs

Step 4: For each job, find the earliest slot on compatible machines

Provide the grouped analysis and then the final integrated schedule."""

    @staticmethod
    def format_reasoning_output(reasoning: str, schedule: str) -> str:
        """Format output with reasoning and schedule."""
        return f"""=== SCHEDULING REASONING ===
{reasoning}

=== FINAL SCHEDULE ===
{schedule}"""

    @staticmethod
    def create_validation_prompt(schedule: str, jobs_text: str, machines_text: str) -> str:
        """Create prompt to validate a generated schedule."""
        return f"""Validate this schedule for constraint violations:

{schedule}

Original jobs:
{jobs_text}

Available machines:
{machines_text}

Check for:
1. Sequence violations (jobs scheduled out of order)
2. Machine compatibility violations
3. Time overlaps on same machine
4. Multi-machine jobs not having all machines available
5. Jobs scheduled after their LCD date

List any violations found or confirm the schedule is valid."""


def test_prompts():
    """Test prompt generation."""
    prompts = SchedulingPrompts()
    
    # Sample data
    jobs = """=== URGENT JOBS (Due within 7 days) ===
Job: JOTP25070237-CT10-013A-1/5 | Family: JOTP25070237 | Sequence: 1/5 | Time: 15.1h | Machines: 80 | LCD: 2025-08-01 (8d)"""
    
    machines = """Type 1: 1(AD01-75HP), 2(AD02-50HP)
Type 2: 3(BM01), 4(BM02)"""
    
    start = "2025-01-24 08:00"
    
    # Test different prompts
    print("=== DIRECT PROMPT ===")
    print(prompts.direct_scheduling_prompt(jobs, machines, start)[:500] + "...")
    
    print("\n=== CHAIN OF THOUGHT ===")
    print(prompts.chain_of_thought_prompt(jobs, machines, start)[:500] + "...")
    
    print("\n=== SYSTEM PROMPT ===")
    print(prompts.get_system_prompt()[:500] + "...")


if __name__ == "__main__":
    test_prompts()