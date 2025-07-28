"""
Analyze job feasibility across different samples to understand deadline pressure
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from phase3.environments.curriculum_env_truly_fixed import CurriculumEnvironmentTrulyFixed


def analyze_multiple_samples(stage_name: str, n_samples: int = 10):
    """Analyze feasibility across multiple job samples."""
    
    print(f"Analyzing {n_samples} samples from {stage_name}")
    print("=" * 60)
    
    feasibility_stats = {
        'all_feasible': 0,
        'some_infeasible': 0,
        'many_infeasible': 0,
        'total_jobs': [],
        'infeasible_jobs': [],
        'feasibility_rates': []
    }
    
    for sample in range(n_samples):
        env = CurriculumEnvironmentTrulyFixed(stage_name, verbose=False)
        env.reset()
        
        total_families = len(env.families)
        infeasible_families = 0
        total_tasks = 0
        
        print(f"\nSample {sample + 1}:")
        print("-" * 40)
        
        # Analyze each family
        family_details = []
        for fid, family in env.families.items():
            lcd_days = family.get('lcd_days_remaining', 0)
            deadline_hours = lcd_days * 24
            
            # Calculate total processing time
            total_processing = sum(task['processing_time'] for task in family['tasks'])
            n_tasks = len(family['tasks'])
            total_tasks += n_tasks
            
            is_feasible = total_processing <= deadline_hours
            if not is_feasible:
                infeasible_families += 1
            
            family_details.append({
                'id': fid,
                'tasks': n_tasks,
                'processing': total_processing,
                'deadline': deadline_hours,
                'feasible': is_feasible,
                'slack': deadline_hours - total_processing
            })
        
        # Sort by slack time (most urgent first)
        family_details.sort(key=lambda x: x['slack'])
        
        # Show most urgent families
        print("Most urgent families:")
        for i, fam in enumerate(family_details[:3]):
            status = "OK" if fam['feasible'] else "IMPOSSIBLE"
            print(f"  {fam['id']}: {fam['processing']:.1f}h work, {fam['deadline']:.1f}h deadline ({status})")
            
        # Calculate statistics
        feasibility_rate = (total_families - infeasible_families) / total_families
        feasibility_stats['feasibility_rates'].append(feasibility_rate)
        feasibility_stats['total_jobs'].append(total_tasks)
        feasibility_stats['infeasible_jobs'].append(infeasible_families)
        
        if infeasible_families == 0:
            feasibility_stats['all_feasible'] += 1
        elif infeasible_families <= 1:
            feasibility_stats['some_infeasible'] += 1
        else:
            feasibility_stats['many_infeasible'] += 1
            
        print(f"\nSummary: {infeasible_families}/{total_families} families infeasible")
        print(f"Total tasks: {total_tasks}")
        print(f"Feasibility rate: {feasibility_rate:.1%}")
    
    # Overall statistics
    print("\n" + "=" * 60)
    print("OVERALL STATISTICS")
    print("=" * 60)
    print(f"Samples with all families feasible: {feasibility_stats['all_feasible']}/{n_samples}")
    print(f"Samples with 1 infeasible family: {feasibility_stats['some_infeasible']}/{n_samples}")
    print(f"Samples with 2+ infeasible families: {feasibility_stats['many_infeasible']}/{n_samples}")
    print(f"\nAverage feasibility rate: {np.mean(feasibility_stats['feasibility_rates']):.1%}")
    print(f"Min feasibility rate: {np.min(feasibility_stats['feasibility_rates']):.1%}")
    print(f"Max feasibility rate: {np.max(feasibility_stats['feasibility_rates']):.1%}")
    
    return feasibility_stats


def compare_stages():
    """Compare feasibility across different stages."""
    print("\nCOMPARING FEASIBILITY ACROSS STAGES")
    print("=" * 70)
    
    stages = ['toy_normal', 'toy_hard', 'toy_multi']
    
    for stage in stages:
        print(f"\n{stage.upper()}:")
        stats = analyze_multiple_samples(stage, n_samples=5)
        
        # Check if this explains performance differences
        avg_feasibility = np.mean(stats['feasibility_rates'])
        print(f"\nCorrelation with performance:")
        print(f"  Average feasibility: {avg_feasibility:.1%}")
        
        if stage == 'toy_normal':
            print(f"  Best model performance: 56.2%")
            print(f"  With action masking: 25%")
        elif stage == 'toy_hard':
            print(f"  Best model performance: 30%")
        elif stage == 'toy_multi':
            print(f"  Best model performance: 36.4%")


if __name__ == "__main__":
    # First analyze toy_normal in detail
    analyze_multiple_samples('toy_normal', n_samples=10)
    
    # Then compare across stages
    compare_stages()