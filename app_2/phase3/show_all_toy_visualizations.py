"""
Display all toy stage visualizations in a combined view
"""

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle

def create_combined_visualization():
    """Create a combined view of all toy stage visualizations"""
    
    viz_dir = "/Users/carrickcheah/Project/ppo/app_2/visualizations/phase3"
    stages = ['toy_easy', 'toy_normal', 'toy_hard', 'toy_multi']
    
    # Create figure with subplots for all stages
    fig = plt.figure(figsize=(24, 20))
    fig.suptitle('All Toy Stage Schedules - AI Generated', fontsize=20, fontweight='bold')
    
    # Job view charts (top row)
    print("Loading job-view charts...")
    for i, stage in enumerate(stages):
        ax = plt.subplot(4, 2, i*2 + 1)
        
        job_view_path = os.path.join(viz_dir, f"{stage}_job_view.png")
        if os.path.exists(job_view_path):
            img = mpimg.imread(job_view_path)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"{stage.upper()} - Job View", fontsize=14, pad=10)
        else:
            ax.text(0.5, 0.5, f"No image found for\n{stage} job view", 
                   ha='center', va='center', fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
    
    # Machine view charts (bottom row)
    print("Loading machine-view charts...")
    for i, stage in enumerate(stages):
        ax = plt.subplot(4, 2, i*2 + 2)
        
        machine_view_path = os.path.join(viz_dir, f"{stage}_machine_view.png")
        if os.path.exists(machine_view_path):
            img = mpimg.imread(machine_view_path)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"{stage.upper()} - Machine View", fontsize=14, pad=10)
        else:
            ax.text(0.5, 0.5, f"No image found for\n{stage} machine view", 
                   ha='center', va='center', fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
    
    plt.tight_layout()
    
    # Save combined view
    output_path = os.path.join(viz_dir, "all_toy_stages_combined.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved combined visualization to: {output_path}")
    
    # Also create individual summary images for each stage
    for stage in stages:
        create_stage_summary(stage, viz_dir)
    
    plt.close()

def create_stage_summary(stage_name, viz_dir):
    """Create a summary image for a single stage with both views"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle(f'{stage_name.upper()} - AI Scheduling Results', fontsize=16, fontweight='bold')
    
    # Job view
    job_view_path = os.path.join(viz_dir, f"{stage_name}_job_view.png")
    if os.path.exists(job_view_path):
        img = mpimg.imread(job_view_path)
        ax1.imshow(img)
        ax1.axis('off')
    else:
        ax1.text(0.5, 0.5, "Job view not available", ha='center', va='center')
        ax1.axis('off')
    
    # Machine view
    machine_view_path = os.path.join(viz_dir, f"{stage_name}_machine_view.png")
    if os.path.exists(machine_view_path):
        img = mpimg.imread(machine_view_path)
        ax2.imshow(img)
        ax2.axis('off')
    else:
        ax2.text(0.5, 0.5, "Machine view not available", ha='center', va='center')
        ax2.axis('off')
    
    plt.tight_layout()
    
    output_path = os.path.join(viz_dir, f"{stage_name}_combined.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved {stage_name} combined view to: {output_path}")
    plt.close()

def create_grid_view():
    """Create a 2x2 grid view of job-view charts only"""
    
    viz_dir = "/Users/carrickcheah/Project/ppo/app_2/visualizations/phase3"
    stages = ['toy_easy', 'toy_normal', 'toy_hard', 'toy_multi']
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('All Toy Stages - Job Scheduling Views (AI Generated)', fontsize=18, fontweight='bold')
    
    for i, stage in enumerate(stages):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        job_view_path = os.path.join(viz_dir, f"{stage}_job_view.png")
        if os.path.exists(job_view_path):
            img = mpimg.imread(job_view_path)
            ax.imshow(img)
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, f"{stage} not available", ha='center', va='center', fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
        # Add stage description
        descriptions = {
            'toy_easy': 'Learn Sequence Rules',
            'toy_normal': 'Learn Deadlines', 
            'toy_hard': 'Learn Priorities',
            'toy_multi': 'Learn Multi-Machine'
        }
        ax.text(0.5, -0.05, descriptions.get(stage, ''), 
                transform=ax.transAxes, ha='center', fontsize=12, style='italic')
    
    plt.tight_layout()
    
    output_path = os.path.join(viz_dir, "all_toy_stages_grid.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved grid view to: {output_path}")
    plt.close()

def main():
    """Create all combined visualizations"""
    print("Creating combined visualizations for all toy stages...")
    print("=" * 60)
    
    # Create different views
    create_combined_visualization()
    create_grid_view()
    
    print("\n" + "=" * 60)
    print("All combined visualizations created!")
    print(f"Check the output directory: /Users/carrickcheah/Project/ppo/app_2/visualizations/phase3/")
    
    # List all generated files
    viz_dir = "/Users/carrickcheah/Project/ppo/app_2/visualizations/phase3"
    print("\nGenerated files:")
    for file in sorted(os.listdir(viz_dir)):
        if file.endswith('.png'):
            print(f"  - {file}")

if __name__ == "__main__":
    main()