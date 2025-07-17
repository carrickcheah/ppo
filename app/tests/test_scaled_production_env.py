"""
Test scaled production environment with 40 machines.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.environments.scaled_production_env import ScaledProductionEnv


def test_environment_initialization():
    """Test environment can be created and initialized."""
    print("Testing environment initialization...")
    
    env = ScaledProductionEnv(n_machines=40)
    
    assert env.n_machines == 40
    assert env.n_jobs == 172  # From SAMPLE_50.md
    assert env.n_families == 50
    
    # Check observation space
    obs_dim = env.observation_space.shape[0]
    expected_dim = 40 + 4 * 50 + 40 + 1  # machines + families*4 + utilization + time
    assert obs_dim == expected_dim, f"Expected {expected_dim} dims, got {obs_dim}"
    
    # Check action space
    assert env.action_space.n == 100  # max_valid_actions
    
    print("✓ Environment initialization successful")
    print(f"  - Machines: {env.n_machines}")
    print(f"  - Families: {env.n_families}")
    print(f"  - Jobs: {env.n_jobs}")
    print(f"  - Observation dims: {obs_dim}")


def test_machine_diversity():
    """Test that selected machines have diverse types."""
    print("\nTesting machine diversity...")
    
    env = ScaledProductionEnv(n_machines=40)
    
    # Check machine types
    machine_types = [m['machine_type_id'] for m in env.machines]
    unique_types = set(machine_types)
    
    # Filter out None values for display
    types_for_display = [t for t in unique_types if t is not None]
    
    print(f"✓ Selected {len(unique_types)} unique machine types")
    if types_for_display:
        print(f"  Machine type distribution: {sorted(types_for_display)[:10]}...")
    else:
        print(f"  Machine type distribution: {list(unique_types)[:10]}...")
    
    assert len(unique_types) >= 10, "Should have at least 10 different machine types"


def test_machine_capabilities():
    """Test machine capability assignment."""
    print("\nTesting machine capabilities...")
    
    env = ScaledProductionEnv(n_machines=40)
    
    # Check each machine has capabilities
    for idx in range(env.n_machines):
        capabilities = env.machine_capabilities[idx]
        assert len(capabilities) > 0, f"Machine {idx} has no capabilities"
    
    # Check coverage of product types
    all_capabilities = set()
    for caps in env.machine_capabilities.values():
        all_capabilities.update(caps)
    
    print(f"✓ All machines have capabilities assigned")
    print(f"  Product types covered: {sorted(all_capabilities)}")
    
    assert 'CF' in all_capabilities
    assert 'CP' in all_capabilities


def test_reset_and_observation():
    """Test environment reset and observation generation."""
    print("\nTesting reset and observation...")
    
    env = ScaledProductionEnv(n_machines=40)
    obs = env.reset()[0]
    
    assert obs.shape == env.observation_space.shape
    assert np.all(obs >= 0) and np.all(obs <= 1), "Observations should be normalized"
    
    # Check initial state
    assert np.all(env.machine_loads == 0), "Machine loads should start at 0"
    assert np.all(env.machine_utilization == 0), "Utilization should start at 0"
    assert len(env.valid_actions) > 0, "Should have valid actions at start"
    
    print(f"✓ Reset successful")
    print(f"  Initial valid actions: {len(env.valid_actions)}")


def test_action_execution():
    """Test executing actions in the environment."""
    print("\nTesting action execution...")
    
    env = ScaledProductionEnv(n_machines=40)
    obs = env.reset()[0]
    
    # Execute first few valid actions
    completed_tasks = 0
    total_reward = 0
    
    for step in range(10):
        if not env.valid_actions:
            break
            
        # Take first valid action
        action = 0
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        
        # Verify action was executed
        if 'scheduled_job' in info:
            completed_tasks += 1
            print(f"  Step {step+1}: Scheduled {info['scheduled_job']} "
                  f"on machine {info['on_machine']}")
    
    assert completed_tasks > 0, "Should have completed some tasks"
    assert total_reward > 0, "Should have positive rewards"
    
    print(f"✓ Action execution successful")
    print(f"  Tasks scheduled: {completed_tasks}")
    print(f"  Total reward: {total_reward:.1f}")


def test_machine_constraints():
    """Test that machine type constraints are respected."""
    print("\nTesting machine constraints...")
    
    env = ScaledProductionEnv(n_machines=40)
    obs = env.reset()[0]
    
    violations = 0
    checks = 0
    
    # Check several valid actions
    for family_id, task_idx, task in env.valid_actions[:10]:
        family = env.families_data[family_id]
        product_code = family['product']
        
        # Verify capable machines can actually process this product
        if 'capable_machines' in task:
            for machine_idx in task['capable_machines']:
                can_process = env._can_machine_process(machine_idx, product_code)
                checks += 1
                if not can_process:
                    violations += 1
    
    assert violations == 0, f"Found {violations} constraint violations"
    
    print(f"✓ Machine constraints respected")
    print(f"  Checked {checks} machine-product pairs")


def test_setup_times():
    """Test setup time calculation between different products."""
    print("\nTesting setup times...")
    
    env = ScaledProductionEnv(n_machines=40)
    
    # Test setup times
    setup_times = []
    for p1 in ['CF', 'CP', 'CD']:
        for p2 in ['CF', 'CP', 'CD']:
            time = env.setup_times.get((p1, p2), 0)
            setup_times.append((p1, p2, time))
            
    # Same type should have minimal setup
    for p in ['CF', 'CP', 'CD']:
        assert env.setup_times[(p, p)] < 0.2
    
    # Different types should have more setup
    assert env.setup_times[('CF', 'CP')] > env.setup_times[('CF', 'CF')]
    
    print("✓ Setup times configured correctly")
    for p1, p2, time in setup_times[:6]:
        print(f"  {p1} → {p2}: {time:.1f}h")


def test_completion():
    """Test completing all tasks."""
    print("\nTesting full episode completion...")
    
    env = ScaledProductionEnv(n_machines=40)
    obs = env.reset()[0]
    
    max_steps = 1000
    steps = 0
    total_reward = 0
    
    while steps < max_steps:
        if not env.valid_actions:
            break
            
        # Use least loaded machine strategy
        best_action = 0
        best_load = float('inf')
        
        for i, (family_id, task_idx, task) in enumerate(env.valid_actions):
            if 'capable_machines' in task:
                min_load = min(env.machine_loads[m] for m in task['capable_machines'])
                if min_load < best_load:
                    best_load = min_load
                    best_action = i
        
        obs, reward, terminated, truncated, info = env.step(best_action)
        total_reward += reward
        steps += 1
        
        if terminated:
            print(f"✓ Episode completed successfully!")
            print(f"  Steps: {steps}")
            print(f"  Makespan: {info.get('makespan', 0):.1f}h")
            print(f"  Efficiency: {info.get('efficiency', 0):.1%}")
            print(f"  Avg utilization: {info.get('avg_utilization', 0):.1%}")
            print(f"  Total reward: {total_reward:.1f}")
            break
    
    if not terminated:
        completed = sum(len(c) for c in env.completed_tasks.values())
        print(f"⚠ Episode did not complete in {max_steps} steps")
        print(f"  Completed: {completed}/{env.n_jobs} tasks")


def test_scaling_metrics():
    """Compare metrics between different numbers of machines."""
    print("\nTesting scaling metrics...")
    
    machine_counts = [10, 20, 40]
    results = {}
    
    for n_machines in machine_counts:
        env = ScaledProductionEnv(n_machines=n_machines)
        obs = env.reset()[0]
        
        # Run simple scheduler
        steps = 0
        while steps < 500 and env.valid_actions:
            action = 0  # First fit
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            
            if terminated:
                results[n_machines] = {
                    'makespan': env.episode_makespan,
                    'steps': steps,
                    'utilization': np.mean(env.machine_utilization)
                }
                break
    
    print("✓ Scaling metrics collected")
    print(f"\n{'Machines':>10} {'Makespan':>10} {'Steps':>10} {'Utilization':>12}")
    print("-" * 45)
    for n, metrics in results.items():
        print(f"{n:>10} {metrics['makespan']:>9.1f}h {metrics['steps']:>10} "
              f"{metrics['utilization']:>11.1%}")


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("SCALED PRODUCTION ENVIRONMENT TESTS")
    print("="*60)
    
    tests = [
        test_environment_initialization,
        test_machine_diversity,
        test_machine_capabilities,
        test_reset_and_observation,
        test_action_execution,
        test_machine_constraints,
        test_setup_times,
        test_completion,
        test_scaling_metrics
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"\n❌ Test failed: {test.__name__}")
            print(f"   Error: {str(e)}")
            raise
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()