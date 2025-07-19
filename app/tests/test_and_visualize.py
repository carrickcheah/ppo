"""
Simple test run with visualization for the trained PPO model.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from stable_baselines3 import PPO
from src.environments.full_production_env import FullProductionEnv

# Create visualization directory
viz_dir = Path("/Users/carrickcheah/Project/ppo/visualizations")
viz_dir.mkdir(parents=True, exist_ok=True)

def run_and_visualize():
    """Run the model and create visualizations."""
    print("=== PPO Production Scheduler Test ===")
    
    # Load model
    print("Loading trained model...")
    model = PPO.load("app/models/full_production/final_model.zip")
    
    # Create environment
    print("Setting up environment with 152 machines...")
    env = FullProductionEnv(
        n_machines=152,
        n_jobs=500,
        state_compression="hierarchical",
        use_break_constraints=True,
        use_holiday_constraints=True,
        seed=42
    )
    
    # Run one episode
    print("\nRunning production schedule...")
    obs, info = env.reset()
    print(f"Starting with {len(env.jobs)} jobs to schedule")
    
    terminated = False
    truncated = False
    step = 0
    rewards = []
    times = []
    
    # Track job assignments
    job_assignments = []
    
    while not (terminated or truncated) and step < 500:  # Limit steps for visualization
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        rewards.append(reward)
        times.append(env.current_time)
        
        # Track if a job was assigned
        if step < len(env.jobs):
            job_idx = step % len(env.jobs)
            machine_idx = action % env.n_machines
            job_assignments.append({
                'step': step,
                'job': job_idx,
                'machine': machine_idx,
                'time': env.current_time
            })
        
        step += 1
        
        if step % 50 == 0:
            print(f"Step {step}: Time={env.current_time:.1f}h")
    
    print(f"\nCompleted in {step} steps")
    print(f"Final time: {env.current_time:.1f} hours")
    
    # Create visualizations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Reward progression
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(rewards, 'b-', alpha=0.7)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Reward')
    ax1.set_title('Reward per Step')
    ax1.grid(True, alpha=0.3)
    
    # 2. Time progression
    ax2.plot(times, 'g-', linewidth=2)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Time (hours)')
    ax2.set_title('Schedule Time Progression')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    reward_path = viz_dir / f"reward_progression_{timestamp}.png"
    plt.savefig(reward_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved reward chart to: {reward_path}")
    
    # 3. Machine utilization heatmap
    if job_assignments:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create a simple utilization matrix
        n_machines_to_show = min(50, env.n_machines)  # Show first 50 machines
        n_time_slots = 50
        utilization = np.zeros((n_machines_to_show, n_time_slots))
        
        max_time = max(times) if times else 1
        for assignment in job_assignments[:200]:  # First 200 assignments
            machine = min(assignment['machine'], n_machines_to_show - 1)
            time_slot = int((assignment['time'] / max_time) * (n_time_slots - 1))
            utilization[machine, time_slot] += 1
        
        im = ax.imshow(utilization, cmap='YlOrRd', aspect='auto')
        ax.set_xlabel('Time Progress')
        ax.set_ylabel('Machine ID')
        ax.set_title('Machine Utilization Heatmap (First 50 Machines)')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Jobs Assigned')
        
        util_path = viz_dir / f"machine_utilization_{timestamp}.png"
        plt.savefig(util_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved utilization heatmap to: {util_path}")
    
    # 4. Summary statistics
    fig, ax = plt.subplots(figsize=(10, 6))
    
    summary_text = f"""
    PPO Production Scheduler Results
    ================================
    
    Environment Configuration:
    • Total Machines: {env.n_machines}
    • Total Jobs: {len(env.jobs)}
    • State Compression: Hierarchical (60 features)
    • Constraints: Breaks + Holidays enabled
    
    Performance Metrics:
    • Steps Taken: {step}
    • Final Makespan: {env.current_time:.1f} hours
    • Average Reward: {np.mean(rewards):.2f}
    • Jobs per Step: {len(env.jobs)/step:.2f}
    
    Model Details:
    • Algorithm: PPO (Proximal Policy Optimization)
    • Network: [256, 256, 256] with tanh activation
    • Training: 1M timesteps
    • Learning Rate: 1e-5
    """
    
    ax.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
            verticalalignment='center', 
            bbox=dict(boxstyle="round,pad=1", facecolor="lightblue", alpha=0.8))
    ax.axis('off')
    
    summary_path = viz_dir / f"test_summary_{timestamp}.png"
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved summary to: {summary_path}")
    
    print(f"\n✅ Test complete! All visualizations saved to:\n{viz_dir}")

if __name__ == "__main__":
    run_and_visualize()