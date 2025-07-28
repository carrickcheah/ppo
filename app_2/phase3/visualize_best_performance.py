"""Visualize best performance achieved across all training attempts"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Collect best performances from all attempts
best_performances = {
    'toy_easy': 1.0,  # Already perfect
    'toy_normal': 0.0,
    'toy_hard': 0.0,
    'toy_multi': 0.0
}

# Check all result files
result_locations = [
    ('models_100_percent', ['toy_normal_results.json', 'toy_hard_results.json', 'toy_multi_results.json']),
    ('models_80_percent', ['toy_normal_results.json', 'toy_hard_results.json', 'toy_multi_results.json']),
    ('phased_models', ['toy_normal_results.json', 'toy_hard_results.json', 'toy_multi_results.json']),
    ('curriculum_models/toy_normal', ['results.json']),
    ('curriculum_models/toy_hard', ['results.json']),
    ('curriculum_models/toy_multi', ['results.json'])
]

for dir_name, files in result_locations:
    for file_name in files:
        path = f"/Users/carrickcheah/Project/ppo/app_2/phase3/{dir_name}/{file_name}"
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                stage = data.get('stage')
                rate = data.get('final_rate', data.get('avg_rate', 0))
                if stage and rate > best_performances.get(stage, 0):
                    best_performances[stage] = rate

# Also include manually tested results
manual_results = {
    'toy_normal': 0.562,  # 56.2% from our testing
    'toy_hard': 0.30,     # 30% from our testing
    'toy_multi': 0.364    # 36.4% from our testing
}

for stage, rate in manual_results.items():
    if rate > best_performances.get(stage, 0):
        best_performances[stage] = rate

# Known achievable rates from analysis
achievable_rates = {
    'toy_easy': 1.0,
    'toy_normal': 1.0,
    'toy_hard': 1.0,
    'toy_multi': 0.955
}

# Create comprehensive visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle('PPO Toy Stages Performance Analysis', fontsize=16, fontweight='bold')

# Chart 1: Performance vs Targets
stages = list(best_performances.keys())
y_pos = np.arange(len(stages))

# Plot bars
bars = ax1.barh(y_pos, [best_performances[s] for s in stages], 
                 color=['green' if best_performances[s] >= 0.8 else 'orange' if best_performances[s] >= 0.5 else 'red' for s in stages],
                 alpha=0.8, label='Current Best')

# Add 80% target line
ax1.axvline(x=0.8, color='blue', linestyle='--', linewidth=2, label='80% Target')

# Add achievable rates
for i, stage in enumerate(stages):
    achievable = achievable_rates[stage]
    ax1.plot(achievable, i, 'g*', markersize=15, label='Achievable' if i == 0 else '')
    
    # Add value labels
    current = best_performances[stage]
    ax1.text(current + 0.02, i, f'{current:.1%}', va='center', fontweight='bold')
    
    if achievable < 1.0:
        ax1.text(achievable + 0.02, i, f'({achievable:.1%} possible)', va='center', fontsize=9, style='italic')

ax1.set_yticks(y_pos)
ax1.set_yticklabels(stages)
ax1.set_xlabel('Completion Rate', fontsize=12)
ax1.set_title('Current Best Performance vs Targets', fontsize=14)
ax1.set_xlim(0, 1.1)
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)

# Chart 2: Gap Analysis
gaps_to_80 = {s: max(0, 0.8 - best_performances[s]) for s in stages}
gaps_to_achievable = {s: max(0, achievable_rates[s] - best_performances[s]) for s in stages}

bar_width = 0.35
y_pos2 = np.arange(len(stages))

bars1 = ax2.barh(y_pos2 - bar_width/2, [gaps_to_80[s] for s in stages], 
                 bar_width, label='Gap to 80%', color='skyblue', alpha=0.8)
bars2 = ax2.barh(y_pos2 + bar_width/2, [gaps_to_achievable[s] for s in stages], 
                 bar_width, label='Gap to Achievable', color='lightcoral', alpha=0.8)

# Add value labels
for i, stage in enumerate(stages):
    gap_80 = gaps_to_80[stage]
    gap_ach = gaps_to_achievable[stage]
    
    if gap_80 > 0:
        ax2.text(gap_80 + 0.01, i - bar_width/2, f'{gap_80:.1%}', va='center', fontsize=9)
    if gap_ach > 0:
        ax2.text(gap_ach + 0.01, i + bar_width/2, f'{gap_ach:.1%}', va='center', fontsize=9)

ax2.set_yticks(y_pos2)
ax2.set_yticklabels(stages)
ax2.set_xlabel('Performance Gap', fontsize=12)
ax2.set_title('Gap Analysis: How Far From Targets?', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Save the plot
output_dir = "/Users/carrickcheah/Project/ppo/app_2/visualizations/phase3"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'toy_stages_performance_analysis.png'), dpi=300, bbox_inches='tight')
print(f"Performance analysis saved to {output_dir}/toy_stages_performance_analysis.png")

# Create summary report
print("\n" + "="*60)
print("TOY STAGES PERFORMANCE SUMMARY")
print("="*60)
print(f"\n{'Stage':<15} {'Current Best':<15} {'80% Target':<15} {'Achievable':<15} {'Status'}")
print("-"*75)

for stage in stages:
    current = best_performances[stage]
    achievable = achievable_rates[stage]
    
    if current >= 0.8:
        status = "✓ Target Met"
    elif current >= achievable * 0.8:  # Within 80% of what's achievable
        status = "⚠ Good Progress"
    else:
        status = "✗ Needs Work"
    
    print(f"{stage:<15} {current:<15.1%} {'80%':<15} {achievable:<15.1%} {status}")

print("\nKEY FINDINGS:")
print("-"*40)
print("1. toy_easy: Already perfect at 100%")
print("2. toy_normal: Best achieved 56.2% (target 80%, 100% is possible)")
print("3. toy_hard: Best achieved 30.0% (target 80%, 100% is possible)")
print("4. toy_multi: Best achieved 36.4% (target 80%, 95.5% is possible)")
print("\nDespite proving 100% is achievable through random search,")
print("the RL models struggle due to conflicting reward signals")
print("and the complexity of the scheduling problem.")

# Save performance data
performance_data = {
    'best_performances': best_performances,
    'achievable_rates': achievable_rates,
    'gaps_to_80': gaps_to_80,
    'gaps_to_achievable': gaps_to_achievable,
    'target': 0.8
}

with open(os.path.join(output_dir, 'performance_summary.json'), 'w') as f:
    json.dump(performance_data, f, indent=2)

print(f"\nData saved to {output_dir}/performance_summary.json")