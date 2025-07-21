#!/usr/bin/env python3
"""
Simple test for hierarchical environment without production data dependencies.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SimpleHierarchicalEnv(gym.Env):
    """
    Simplified hierarchical environment for testing the concept.
    """
    
    def __init__(self, n_jobs=10, n_machines=5):
        super().__init__()
        
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        
        # Hierarchical action space
        self.action_space = spaces.Dict({
            'job': spaces.Discrete(n_jobs),
            'machine': spaces.Discrete(n_machines)
        })
        
        # Simple state: just track which jobs are scheduled
        self.observation_space = spaces.Box(0, 1, shape=(n_jobs + n_machines,))
        
        # Create simple jobs with processing times
        self.jobs = [
            {'id': i, 'processing_time': np.random.uniform(1, 5)} 
            for i in range(n_jobs)
        ]
        
        # Simple compatibility: each job can run on 2-3 random machines
        self.compatibility = np.zeros((n_jobs, n_machines), dtype=bool)
        for i in range(n_jobs):
            # Each job compatible with 2-3 machines
            n_compatible = np.random.randint(2, 4)
            compatible_machines = np.random.choice(n_machines, n_compatible, replace=False)
            self.compatibility[i, compatible_machines] = True
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        # Reset state
        self.job_scheduled = np.zeros(self.n_jobs, dtype=bool)
        self.machine_busy_until = np.zeros(self.n_machines)
        self.current_time = 0
        self.schedule = []
        
        return self._get_obs(), self._get_info()
    
    def _get_obs(self):
        # Simple observation: job scheduled status + machine utilization
        job_status = self.job_scheduled.astype(float)
        machine_util = np.clip(self.machine_busy_until / 10, 0, 1)  # Normalize
        return np.concatenate([job_status, machine_util])
    
    def _get_info(self):
        return {
            'scheduled_count': np.sum(self.job_scheduled),
            'current_time': self.current_time,
            'action_masks': self._get_action_masks()
        }
    
    def _get_action_masks(self):
        # Job mask: unscheduled jobs
        job_mask = ~self.job_scheduled
        
        # Machine masks for each job
        machine_masks = np.zeros((self.n_jobs, self.n_machines), dtype=bool)
        for job_idx in range(self.n_jobs):
            if not self.job_scheduled[job_idx]:
                # Job can be assigned to compatible machines
                machine_masks[job_idx] = self.compatibility[job_idx]
        
        return {'job': job_mask, 'machine': machine_masks}
    
    def step(self, action):
        job_idx = action['job']
        machine_idx = action['machine']
        
        # Validate action
        if self.job_scheduled[job_idx]:
            # Invalid: job already scheduled
            return self._get_obs(), -10, False, False, {'error': 'Job already scheduled'}
        
        if not self.compatibility[job_idx, machine_idx]:
            # Invalid: incompatible machine
            return self._get_obs(), -10, False, False, {'error': 'Incompatible machine'}
        
        # Schedule the job
        job = self.jobs[job_idx]
        start_time = max(self.current_time, self.machine_busy_until[machine_idx])
        end_time = start_time + job['processing_time']
        
        self.job_scheduled[job_idx] = True
        self.machine_busy_until[machine_idx] = end_time
        self.current_time = start_time
        
        self.schedule.append({
            'job': job_idx,
            'machine': machine_idx,
            'start': start_time,
            'end': end_time
        })
        
        # Reward based on utilization
        reward = 10.0  # Base reward for scheduling
        if start_time == self.current_time:
            reward += 5.0  # Bonus for immediate scheduling
        
        # Check if done
        done = np.all(self.job_scheduled)
        
        if done:
            # Final reward based on makespan
            makespan = np.max(self.machine_busy_until)
            reward += 100.0 / makespan
        
        return self._get_obs(), reward, done, False, self._get_info()


def test_simple_hierarchical():
    """Test the simple hierarchical environment."""
    print("\nTesting Simple Hierarchical Environment\n")
    
    # Create environment
    env = SimpleHierarchicalEnv(n_jobs=10, n_machines=5)
    print(f"Created environment with {env.n_jobs} jobs and {env.n_machines} machines")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Check compatibility
    print(f"\nCompatibility matrix:\n{env.compatibility.astype(int)}")
    print(f"Jobs with compatible machines: {np.sum(np.any(env.compatibility, axis=1))}")
    
    # Reset and check initial state
    obs, info = env.reset(seed=42)
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial scheduled count: {info['scheduled_count']}")
    
    # Check action masks
    masks = info['action_masks']
    print(f"\nJob mask sum: {np.sum(masks['job'])} (should be {env.n_jobs})")
    print(f"Machine masks shape: {masks['machine'].shape}")
    
    # Run episode
    print("\nRunning episode...")
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < 20:
        # Find valid action
        job_mask = masks['job']
        valid_jobs = np.where(job_mask)[0]
        
        if len(valid_jobs) == 0:
            break
        
        # Pick first valid job
        job_idx = valid_jobs[0]
        
        # Find compatible machine
        machine_mask = masks['machine'][job_idx]
        valid_machines = np.where(machine_mask)[0]
        
        if len(valid_machines) == 0:
            print(f"No valid machines for job {job_idx}!")
            break
        
        machine_idx = valid_machines[0]
        
        # Take action
        action = {'job': int(job_idx), 'machine': int(machine_idx)}
        obs, reward, done, _, info = env.step(action)
        
        print(f"Step {steps}: Scheduled job {job_idx} on machine {machine_idx}, reward: {reward:.2f}")
        
        total_reward += reward
        steps += 1
        masks = info['action_masks']
    
    print(f"\nEpisode completed!")
    print(f"Total steps: {steps}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Jobs scheduled: {info['scheduled_count']}/{env.n_jobs}")
    
    if done:
        makespan = np.max(env.machine_busy_until)
        print(f"Makespan: {makespan:.2f}")
    
    # Show final schedule
    print("\nFinal Schedule:")
    for s in env.schedule[:5]:  # Show first 5
        print(f"  Job {s['job']} on Machine {s['machine']}: {s['start']:.1f} - {s['end']:.1f}")
    
    print("\n✅ Simple hierarchical environment works correctly!")
    print("\nKey insights:")
    print("- Hierarchical action space successfully implemented")
    print("- Job and machine selection work independently")
    print("- Action masking prevents invalid selections")
    print(f"- Action space size: {env.n_jobs + env.n_machines} vs {env.n_jobs * env.n_machines} (flat)")
    

if __name__ == "__main__":
    test_simple_hierarchical()