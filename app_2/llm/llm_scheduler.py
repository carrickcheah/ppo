"""
LLM Scheduler Core

Main scheduler class that orchestrates the scheduling process using LLMs.
"""

import logging
import yaml
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

from .deepseek_client import DeepSeekClient
from .data_adapter import DataAdapter
from .prompts import SchedulingPrompts

logger = logging.getLogger(__name__)


class LLMScheduler:
    """
    Main LLM-based scheduler implementation.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize LLM scheduler.
        
        Args:
            config_path: Path to config.yaml. If None, uses default.
        """
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.client = DeepSeekClient()
        self.prompts = SchedulingPrompts()
        self.data_adapter = None
        
        # Scheduling settings
        self.max_jobs_per_prompt = self.config['scheduling']['max_jobs_per_prompt']
        self.default_strategy = self.config['scheduling']['default_strategy']
        
        logger.info(f"LLM Scheduler initialized with strategy: {self.default_strategy}")
        
    def schedule(
        self,
        snapshot_path: str = None,
        strategy: str = None,
        max_jobs: int = None,
        start_time: str = None
    ) -> Dict[str, Any]:
        """
        Generate schedule using LLM.
        
        Args:
            snapshot_path: Path to data snapshot
            strategy: Scheduling strategy (direct, chain_of_thought, etc.)
            max_jobs: Maximum jobs to schedule (for testing)
            start_time: Start time for scheduling
            
        Returns:
            Schedule dictionary with results and metrics
        """
        # Initialize data adapter
        self.data_adapter = DataAdapter(snapshot_path)
        
        # Use defaults if not specified
        strategy = strategy or self.default_strategy
        start_time = start_time or datetime.now().strftime("%Y-%m-%d %H:%M")
        
        logger.info(f"Starting {strategy} scheduling from {start_time}")
        
        # Load and format data
        jobs_text, machines_text = self.data_adapter.format_for_llm(max_jobs)
        
        # Check if we need to batch
        job_count = len(jobs_text.split('\n'))
        if job_count > self.max_jobs_per_prompt and self.config['scheduling']['enable_batching']:
            return self._batch_schedule(jobs_text, machines_text, start_time, strategy)
        
        # Single prompt scheduling
        return self._single_schedule(jobs_text, machines_text, start_time, strategy)
        
    def _single_schedule(
        self,
        jobs_text: str,
        machines_text: str,
        start_time: str,
        strategy: str
    ) -> Dict[str, Any]:
        """
        Schedule jobs with a single LLM prompt.
        """
        # Get appropriate prompt
        system_prompt = self.prompts.get_system_prompt()
        
        if strategy == "direct":
            user_prompt = self.prompts.direct_scheduling_prompt(jobs_text, machines_text, start_time)
        elif strategy == "chain_of_thought":
            user_prompt = self.prompts.chain_of_thought_prompt(jobs_text, machines_text, start_time)
        elif strategy == "constraint_focused":
            user_prompt = self.prompts.constraint_focused_prompt(jobs_text, machines_text, start_time)
        elif strategy == "example_based":
            user_prompt = self.prompts.example_based_prompt(jobs_text, machines_text, start_time)
        elif strategy == "decomposition":
            user_prompt = self.prompts.decomposition_prompt(jobs_text, machines_text, start_time)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Get LLM response
        logger.info("Sending request to LLM...")
        response = self.client.schedule_jobs(system_prompt, user_prompt)
        
        # Parse response
        from .parser import ScheduleParser
        parser = ScheduleParser()
        schedule_tasks = parser.parse_schedule(response['content'], start_time.split()[0])
        
        # Convert to dict format for compatibility
        schedule = [task.to_dict() for task in schedule_tasks]
        
        # Format output
        output = self.data_adapter.format_schedule_output(schedule)
        
        # Add LLM metadata
        output['llm_metadata'] = {
            'strategy': strategy,
            'model': response['model'],
            'tokens_used': response['usage']['total_tokens'],
            'response_time': response['elapsed_time'],
            'cost_estimate': self.client.get_cost_estimate()
        }
        
        # Include reasoning if configured
        if self.config['output']['include_reasoning']:
            output['reasoning'] = self._extract_reasoning(response['content'])
        
        logger.info(f"Scheduling complete: {len(schedule)} jobs scheduled")
        
        return output
        
    def _batch_schedule(
        self,
        jobs_text: str,
        machines_text: str,
        start_time: str,
        strategy: str
    ) -> Dict[str, Any]:
        """
        Schedule large job sets in batches.
        """
        # Split jobs into batches
        job_lines = jobs_text.split('\n')
        batch_size = self.config['scheduling']['batch_size']
        batches = []
        
        current_batch = []
        for line in job_lines:
            if line.strip() and not line.startswith('==='):
                current_batch.append(line)
                if len(current_batch) >= batch_size:
                    batches.append('\n'.join(current_batch))
                    current_batch = []
        
        if current_batch:
            batches.append('\n'.join(current_batch))
        
        logger.info(f"Processing {len(batches)} batches")
        
        # Process each batch
        all_schedules = []
        current_end_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M")
        
        for i, batch_jobs in enumerate(batches):
            logger.info(f"Processing batch {i+1}/{len(batches)}")
            
            batch_start = current_end_time.strftime("%Y-%m-%d %H:%M")
            result = self._single_schedule(batch_jobs, machines_text, batch_start, strategy)
            
            # Extract schedules
            batch_schedule = result.get('schedule', {})
            for family_jobs in batch_schedule.values():
                for job in family_jobs:
                    all_schedules.append(job)
            
            # Update end time for next batch
            if all_schedules:
                end_times = [datetime.strptime(job['end_time'], "%Y-%m-%d %H:%M") 
                           for job in all_schedules]
                current_end_time = max(end_times)
        
        # Combine results
        return self.data_adapter.format_schedule_output(all_schedules)
        
    def _parse_response(self, llm_output: str, start_time: str) -> List[Dict[str, Any]]:
        """
        Parse LLM response into schedule.
        """
        # Extract schedule section if reasoning is included
        if "=== FINAL SCHEDULE ===" in llm_output:
            schedule_section = llm_output.split("=== FINAL SCHEDULE ===")[1]
        else:
            schedule_section = llm_output
        
        # Parse using data adapter
        return self.data_adapter.parse_llm_schedule(schedule_section, start_time)
        
    def _extract_reasoning(self, llm_output: str) -> str:
        """
        Extract reasoning from LLM output.
        """
        if "=== SCHEDULING REASONING ===" in llm_output:
            reasoning = llm_output.split("=== SCHEDULING REASONING ===")[1]
            if "=== FINAL SCHEDULE ===" in reasoning:
                reasoning = reasoning.split("=== FINAL SCHEDULE ===")[0]
            return reasoning.strip()
        
        # For chain of thought, extract the thinking process
        if "Think through this systematically:" in llm_output:
            lines = llm_output.split('\n')
            reasoning_lines = []
            in_reasoning = False
            
            for line in lines:
                if "Think through this systematically:" in line:
                    in_reasoning = True
                elif line.strip().startswith("Job:") or "->" in line:
                    in_reasoning = False
                elif in_reasoning:
                    reasoning_lines.append(line)
            
            return '\n'.join(reasoning_lines).strip()
        
        return "No explicit reasoning provided"
        
    def validate_schedule(self, schedule: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Validate generated schedule.
        
        Args:
            schedule: List of scheduled tasks
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # TODO: Implement validation logic
        # For now, basic validation
        if not schedule:
            issues.append("No jobs scheduled")
        
        # Check for required fields
        for task in schedule:
            if 'job_id' not in task:
                issues.append(f"Missing job_id in task: {task}")
            if 'machine_ids' not in task:
                issues.append(f"Missing machine_ids in task: {task}")
            if 'start_time' not in task:
                issues.append(f"Missing start_time in task: {task}")
            if 'end_time' not in task:
                issues.append(f"Missing end_time in task: {task}")
        
        is_valid = len(issues) == 0
        return is_valid, issues
        
    def iterative_improve(
        self,
        initial_schedule: Dict[str, Any],
        max_iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Iteratively improve a schedule.
        
        Args:
            initial_schedule: Initial schedule to improve
            max_iterations: Maximum improvement iterations
            
        Returns:
            Improved schedule
        """
        current_schedule = initial_schedule
        
        for i in range(max_iterations):
            # Validate current schedule
            schedule_list = []
            for family_jobs in current_schedule.get('schedule', {}).values():
                schedule_list.extend(family_jobs)
            
            is_valid, issues = self.validate_schedule(schedule_list)
            
            if is_valid:
                logger.info(f"Schedule valid after {i} iterations")
                break
            
            logger.info(f"Iteration {i+1}: Found {len(issues)} issues")
            
            # TODO: Implement iterative improvement
            # For now, just return the initial schedule
            break
        
        return current_schedule


def test_scheduler():
    """Test LLM scheduler."""
    scheduler = LLMScheduler()
    
    # Test with small dataset
    result = scheduler.schedule(
        max_jobs=5,
        strategy="chain_of_thought"
    )
    
    print("=== SCHEDULE RESULT ===")
    print(f"Status: {result['status']}")
    print(f"Jobs scheduled: {result['metrics']['total_jobs']}")
    print(f"Makespan: {result['metrics']['makespan_hours']:.1f} hours")
    
    if 'llm_metadata' in result:
        print(f"\n=== LLM METADATA ===")
        print(f"Strategy: {result['llm_metadata']['strategy']}")
        print(f"Tokens: {result['llm_metadata']['tokens_used']}")
        print(f"Time: {result['llm_metadata']['response_time']:.2f}s")
        print(f"Cost: ${result['llm_metadata']['cost_estimate']['total_cost']:.4f}")
    
    if 'reasoning' in result:
        print(f"\n=== REASONING ===")
        print(result['reasoning'][:500] + "...")


if __name__ == "__main__":
    test_scheduler()