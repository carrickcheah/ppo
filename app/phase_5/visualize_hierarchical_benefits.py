#!/usr/bin/env python3
"""
Visualize the benefits of hierarchical action space in Phase 5
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Create output directory
output_dir = Path("visualizations/phase_5")
output_dir.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def create_action_space_comparison():
    """Compare flat vs hierarchical action spaces"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Data for different scales
    job_counts = [50, 100, 200, 320, 500, 1000]
    machine_count = 145
    
    # Calculate action spaces
    flat_actions = [j * machine_count for j in job_counts]
    hierarchical_actions = [j + machine_count for j in job_counts]
    
    # Plot 1: Action space size
    ax1.plot(job_counts, flat_actions, 'o-', linewidth=3, markersize=10, label='Flat (J×M)')
    ax1.plot(job_counts, hierarchical_actions, 's-', linewidth=3, markersize=10, label='Hierarchical (J+M)')
    ax1.set_xlabel('Number of Jobs', fontsize=12)
    ax1.set_ylabel('Action Space Size', fontsize=12)
    ax1.set_title('Action Space Growth Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Add annotations for key points
    for i, jobs in enumerate([320, 1000]):
        idx = job_counts.index(jobs)
        ax1.annotate(f'{flat_actions[idx]:,}', 
                    xy=(jobs, flat_actions[idx]), 
                    xytext=(jobs+50, flat_actions[idx]*1.5),
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                    fontsize=10, ha='left')
        ax1.annotate(f'{hierarchical_actions[idx]:,}', 
                    xy=(jobs, hierarchical_actions[idx]), 
                    xytext=(jobs+50, hierarchical_actions[idx]*0.5),
                    arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
                    fontsize=10, ha='left')
    
    # Plot 2: Reduction percentage
    reductions = [(1 - h/f) * 100 for f, h in zip(flat_actions, hierarchical_actions)]
    bars = ax2.bar(range(len(job_counts)), reductions, color='lightgreen', edgecolor='darkgreen', linewidth=2)
    ax2.set_xticks(range(len(job_counts)))
    ax2.set_xticklabels(job_counts)
    ax2.set_xlabel('Number of Jobs', fontsize=12)
    ax2.set_ylabel('Action Space Reduction (%)', fontsize=12)
    ax2.set_title('Hierarchical Approach Efficiency', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, val in zip(bars, reductions):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'action_space_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_training_progress_visualization():
    """Visualize Phase 5 training progress"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training data (from actual runs)
    steps = [0, 100, 300, 550, 750]
    jobs_scheduled = [0, 95, 98, 85, 48]
    invalid_rate = [100, 90.5, 90.2, 91.5, 90.4]
    completion_rate = [0, 30, 31, 27, 15]
    
    # Plot 1: Jobs scheduled over time
    ax1.plot(steps, jobs_scheduled, 'o-', linewidth=3, markersize=10, color='blue')
    ax1.fill_between(steps, 0, jobs_scheduled, alpha=0.3, color='blue')
    ax1.set_xlabel('Training Steps (thousands)', fontsize=12)
    ax1.set_ylabel('Jobs Scheduled', fontsize=12)
    ax1.set_title('Jobs Scheduled During Training', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 320)
    ax1.axhline(y=320, color='green', linestyle='--', alpha=0.7, label='Target (100%)')
    ax1.axhline(y=98, color='red', linestyle='--', alpha=0.7, label='Best (31%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Invalid action rate
    ax2.plot(steps, invalid_rate, 's-', linewidth=3, markersize=10, color='red')
    ax2.fill_between(steps, 90, invalid_rate, alpha=0.3, color='red')
    ax2.set_xlabel('Training Steps (thousands)', fontsize=12)
    ax2.set_ylabel('Invalid Action Rate (%)', fontsize=12)
    ax2.set_title('Invalid Actions During Training', fontsize=14, fontweight='bold')
    ax2.set_ylim(85, 101)
    ax2.axhline(y=90, color='orange', linestyle='--', alpha=0.7, label='Minimum achieved')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Completion rate comparison
    models = ['Random\nBaseline', 'Phase 5\n100k', 'Phase 5\n300k', 'Phase 5\n750k', 'Phase 4\nProduction']
    completion_rates = [15, 30, 31, 15, 100]
    colors = ['gray', 'lightblue', 'blue', 'darkblue', 'green']
    
    bars = ax3.bar(models, completion_rates, color=colors, edgecolor='black', linewidth=2)
    ax3.set_ylabel('Completion Rate (%)', fontsize=12)
    ax3.set_title('Model Comparison: Job Completion', fontsize=14, fontweight='bold')
    ax3.set_ylim(0, 110)
    
    # Add value labels
    for bar, val in zip(bars, completion_rates):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 4: Action space visualization
    # Create a heatmap showing job-machine compatibility
    np.random.seed(42)
    n_jobs_show = 50
    n_machines_show = 30
    
    # Create compatibility matrix (sparse, ~12% compatible)
    compatibility = np.random.random((n_jobs_show, n_machines_show)) < 0.12
    compatibility = compatibility.astype(float)
    
    # Add pattern to show job types
    for i in range(0, n_jobs_show, 10):
        compatibility[i:i+5, :10] = 1  # Some jobs compatible with first machines
    for i in range(5, n_jobs_show, 10):
        compatibility[i:i+5, 10:20] = 1  # Others with middle machines
    
    im = ax4.imshow(compatibility, cmap='RdYlGn', aspect='auto', alpha=0.8)
    ax4.set_xlabel('Machine ID', fontsize=12)
    ax4.set_ylabel('Job ID', fontsize=12)
    ax4.set_title('Job-Machine Compatibility Matrix (Sample)', fontsize=14, fontweight='bold')
    
    # Add text showing statistics
    ax4.text(0.02, 0.98, f'Valid pairs: {int(compatibility.sum())}/{n_jobs_show*n_machines_show} ({compatibility.mean()*100:.1f}%)',
            transform=ax4.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'phase5_training_progress.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_hierarchical_concept_diagram():
    """Create a diagram explaining the hierarchical approach"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Hierarchical Action Space Architecture', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Flat approach
    ax.text(2.5, 8.5, 'Traditional Flat Approach', fontsize=14, fontweight='bold', ha='center')
    ax.add_patch(plt.Rectangle((0.5, 6), 4, 2, fill=True, facecolor='lightcoral', alpha=0.5))
    ax.text(2.5, 7, 'Action = Job × Machine\n46,400 combinations\nO(n × m) complexity', 
            fontsize=11, ha='center', va='center')
    
    # Hierarchical approach
    ax.text(7.5, 8.5, 'Hierarchical Approach', fontsize=14, fontweight='bold', ha='center')
    ax.add_patch(plt.Rectangle((5.5, 6), 4, 2, fill=True, facecolor='lightgreen', alpha=0.5))
    ax.text(7.5, 7, 'Step 1: Select Job (320)\nStep 2: Select Machine (145)\n465 decisions total\nO(n + m) complexity', 
            fontsize=11, ha='center', va='center')
    
    # Benefits box
    ax.add_patch(plt.Rectangle((2, 3.5), 6, 1.8, fill=True, facecolor='lightyellow', alpha=0.7))
    ax.text(5, 4.8, 'Benefits', fontsize=13, fontweight='bold', ha='center')
    ax.text(5, 4.2, '• 99% reduction in action space', fontsize=11, ha='center')
    ax.text(5, 3.8, '• All jobs visible in single pass', fontsize=11, ha='center')
    
    # Challenges box
    ax.add_patch(plt.Rectangle((2, 1), 6, 1.8, fill=True, facecolor='lightgray', alpha=0.7))
    ax.text(5, 2.3, 'Challenges', fontsize=13, fontweight='bold', ha='center')
    ax.text(5, 1.7, '• Requires action masking for invalid pairs', fontsize=11, ha='center')
    ax.text(5, 1.3, '• 90% invalid action rate without masking', fontsize=11, ha='center')
    
    plt.savefig(output_dir / 'hierarchical_concept.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_summary():
    """Create overall performance summary"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'Phase 5 Performance Summary', 
            fontsize=18, fontweight='bold', ha='center', transform=ax.transAxes)
    
    # Key metrics
    metrics_text = """
Key Achievements:
• Action space reduction: 46,400 → 465 (99%)
• Best model: 31% job completion (98/320 jobs)
• Training time: 30 minutes (vs 4 hours for Phase 4)
• Single-pass scheduling (no batching required)

Technical Implementation:
• MultiDiscrete action space for SB3 compatibility
• Enhanced state representation (80 features)
• Hierarchical reward function
• Real production data integration

Current Limitations:
• 90% invalid action rate
• Performance degradation after 300k steps
• Requires action masking for production use

Recommendation:
Deploy Phase 4 model (100% completion) while
researching action masking for Phase 5 improvement
    """
    
    ax.text(0.1, 0.85, metrics_text, fontsize=12, ha='left', va='top', 
            transform=ax.transAxes, family='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
    
    plt.savefig(output_dir / 'phase5_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all visualizations"""
    print("Generating Phase 5 visualizations...")
    
    print("1. Creating action space comparison...")
    create_action_space_comparison()
    
    print("2. Creating training progress visualization...")
    create_training_progress_visualization()
    
    print("3. Creating hierarchical concept diagram...")
    create_hierarchical_concept_diagram()
    
    print("4. Creating performance summary...")
    create_performance_summary()
    
    print(f"\nAll visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main()