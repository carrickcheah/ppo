"""
Clean the real data to only include schedulable tasks
Remove completed and in-progress tasks for clean training
"""

import json
import os
from datetime import datetime

def clean_stage_data(stage_name: str):
    """Clean data for a stage to only include pending tasks."""
    
    # Load original data
    input_path = f"/Users/carrickcheah/Project/ppo/app_2/data/stage_{stage_name}_real_data.json"
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    print(f"\nCleaning {stage_name}...")
    
    # Clean each family
    cleaned_families = {}
    total_original = 0
    total_cleaned = 0
    
    for fid, family in data['families'].items():
        # Only include pending tasks
        pending_tasks = []
        
        for i, task in enumerate(family['tasks']):
            total_original += 1
            
            # Only include if status is pending or not specified
            status = task.get('status', 'pending')
            if status == 'pending' or status == 'unknown':
                # Renumber sequences starting from 1
                clean_task = task.copy()
                clean_task['sequence'] = len(pending_tasks) + 1
                clean_task['status'] = 'pending'
                pending_tasks.append(clean_task)
                total_cleaned += 1
        
        # Only include family if it has pending tasks
        if pending_tasks:
            cleaned_family = family.copy()
            cleaned_family['tasks'] = pending_tasks
            cleaned_family['total_sequences'] = len(pending_tasks)
            cleaned_families[fid] = cleaned_family
    
    # Create cleaned data
    cleaned_data = {
        'families': cleaned_families,
        'machines': data['machines'],
        'metadata': {
            'cleaned_at': datetime.now().isoformat(),
            'original_tasks': total_original,
            'cleaned_tasks': total_cleaned,
            'families_with_tasks': len(cleaned_families)
        }
    }
    
    # Save cleaned data
    output_path = f"/Users/carrickcheah/Project/ppo/app_2/data/stage_{stage_name}_clean_data.json"
    with open(output_path, 'w') as f:
        json.dump(cleaned_data, f, indent=2)
    
    print(f"  Original: {len(data['families'])} families, {total_original} tasks")
    print(f"  Cleaned: {len(cleaned_families)} families, {total_cleaned} tasks")
    print(f"  Saved to: {output_path}")
    
    return cleaned_data


def main():
    """Clean all foundation stage data."""
    stages = ['toy_easy', 'toy_normal', 'toy_hard', 'toy_multi']
    
    print("Cleaning stage data for training...")
    print("="*60)
    
    for stage in stages:
        clean_stage_data(stage)
    
    print("\nDone! Use stage_XXX_clean_data.json files for training.")


if __name__ == "__main__":
    main()