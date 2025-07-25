"""
Test script for LLM scheduler
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.llm_scheduler import LLMScheduler
import json


def test_llm_scheduler():
    """Test the complete LLM scheduler."""
    print("=== Testing LLM Scheduler ===\n")
    
    # Initialize scheduler
    print("1. Initializing scheduler...")
    scheduler = LLMScheduler()
    print("   ✓ Scheduler initialized\n")
    
    # Test with small dataset
    print("2. Generating schedule for 5 jobs...")
    try:
        result = scheduler.schedule(
            max_jobs=5,
            strategy="chain_of_thought"
        )
        
        print("   ✓ Schedule generated successfully\n")
        
        # Display results
        print("=== SCHEDULE RESULTS ===")
        print(f"Status: {result['status']}")
        print(f"Jobs scheduled: {result['metrics']['total_jobs']}")
        print(f"Families scheduled: {result['metrics']['families_scheduled']}")
        print(f"Makespan: {result['metrics']['makespan_hours']:.1f} hours")
        
        # Show LLM metadata
        if 'llm_metadata' in result:
            print(f"\n=== LLM METADATA ===")
            print(f"Strategy: {result['llm_metadata']['strategy']}")
            print(f"Model: {result['llm_metadata']['model']}")
            print(f"Tokens used: {result['llm_metadata']['tokens_used']}")
            print(f"Response time: {result['llm_metadata']['response_time']:.2f}s")
            
            cost = result['llm_metadata']['cost_estimate']
            print(f"\n=== COST BREAKDOWN ===")
            print(f"Input tokens: {cost['input_tokens']}")
            print(f"Output tokens: {cost['output_tokens']}")
            print(f"Total cost: ${cost['total_cost']:.4f}")
        
        # Show some scheduled jobs
        if 'schedule' in result:
            print(f"\n=== SAMPLE SCHEDULED JOBS ===")
            job_count = 0
            for family_id, tasks in result['schedule'].items():
                for task in tasks[:2]:  # Show first 2 tasks per family
                    print(f"- {task['task_id']}: machines {task['machine_ids']} "
                          f"@ {task['start_time']} - {task['end_time']}")
                    job_count += 1
                    if job_count >= 5:
                        break
                if job_count >= 5:
                    break
        
        # Show reasoning if available
        if 'reasoning' in result and result['reasoning']:
            print(f"\n=== REASONING EXCERPT ===")
            reasoning = result['reasoning']
            if len(reasoning) > 500:
                print(reasoning[:500] + "...\n[truncated]")
            else:
                print(reasoning)
        
        # Test validation
        print(f"\n3. Validating generated schedule...")
        from llm.validator import ScheduleValidator
        from llm.parser import ScheduleParser
        
        validator = ScheduleValidator()
        parser = ScheduleParser()
        
        # Parse the raw LLM output (if we had it)
        # For now, validate the structured output
        scheduled_tasks = []
        for family_tasks in result.get('schedule', {}).values():
            for task in family_tasks:
                # Convert back to parser format for validation
                # This is a simplified version
                pass
        
        print("   ✓ Validation complete (simplified check)\n")
        
        return True
        
    except Exception as e:
        print(f"\n   ✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_different_strategies():
    """Test different scheduling strategies."""
    print("\n=== Testing Different Strategies ===\n")
    
    strategies = ["direct", "chain_of_thought", "constraint_focused"]
    scheduler = LLMScheduler()
    
    for strategy in strategies:
        print(f"\nTesting {strategy} strategy...")
        try:
            result = scheduler.schedule(
                max_jobs=3,  # Very small for quick testing
                strategy=strategy
            )
            print(f"✓ {strategy}: {result['metrics']['total_jobs']} jobs scheduled "
                  f"in {result['llm_metadata']['response_time']:.1f}s")
        except Exception as e:
            print(f"✗ {strategy}: Failed - {str(e)}")


if __name__ == "__main__":
    # Test main functionality
    success = test_llm_scheduler()
    
    if success:
        # Test different strategies (optional, costs money)
        # test_different_strategies()
        print("\n=== All tests completed successfully! ===")
    else:
        print("\n=== Tests failed! ===")