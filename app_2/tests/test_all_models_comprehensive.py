#!/usr/bin/env python3
"""
Comprehensive Model Testing for Phase 4 Production Planning
Tests ALL available PPO models to find the best performing one
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to Python path
sys.path.append('/Users/carrickcheah/Project/ppo/app_2/src')
sys.path.append('/Users/carrickcheah/Project/ppo/app_2/phase3')
sys.path.append('/Users/carrickcheah/Project/ppo/app_2/phase4')

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def create_test_environment():
    """Create a test environment for model evaluation"""
    try:
        # Try to import from phase4 first
        from environments.base_strategy_env import BaseStrategyEnv
        data_path = '/Users/carrickcheah/Project/ppo/app_2/phase4/data/small_balanced_data.json'
        env = BaseStrategyEnv(data_path=data_path)
        return env, "phase4_base_strategy"
    except:
        try:
            # Fallback to phase3 environment
            from environments.curriculum_env_real import CurriculumEnvReal
            data_path = '/Users/carrickcheah/Project/ppo/app_2/data/stage_toy_normal_real_data.json'
            env = CurriculumEnvReal(data_path=data_path, stage="toy_normal")
            return env, "phase3_curriculum"
        except Exception as e:
            print(f"Error creating environment: {e}")
            return None, None

def test_model_performance(model_path, env, model_type):
    """Test a single model and return performance metrics"""
    try:
        print(f"Testing model: {model_path}")
        
        # Load model
        model = PPO.load(model_path)
        
        # Reset environment
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        total_reward = 0
        steps_taken = 0
        scheduled_jobs = set()
        scheduled_tasks = []
        
        # Run episode
        for step in range(500):  # Increased step limit
            try:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                
                if isinstance(obs, tuple):
                    obs = obs[0]
                
                total_reward += reward
                steps_taken += 1
                
                # Track scheduled jobs
                if hasattr(env, 'scheduled_jobs'):
                    for job_info in env.scheduled_jobs:
                        scheduled_jobs.add(job_info.get('job_reference', ''))
                        scheduled_tasks.append(job_info)
                
                if done:
                    break
                    
            except Exception as e:
                print(f"Error during step {step}: {e}")
                break
        
        # Get final schedule information
        schedule_info = {}
        total_jobs = 0
        
        if hasattr(env, 'jobs'):
            total_jobs = len(env.jobs)
            for job in env.jobs:
                job_ref = job.get('job_reference', '')
                schedule_info[job_ref] = {
                    'job_reference': job_ref,
                    'lcd_date': job.get('lcd_date', ''),
                    'total_sequences': len(job.get('sequences', [])),
                    'scheduled_tasks': []
                }
        
        # Add scheduled tasks to schedule info
        for task in scheduled_tasks:
            job_ref = task.get('job_reference', '')
            if job_ref in schedule_info:
                schedule_info[job_ref]['scheduled_tasks'].append(task)
        
        completion_rate = len(scheduled_jobs) / max(total_jobs, 1) * 100
        
        return {
            'model_path': model_path,
            'model_type': model_type,
            'total_reward': total_reward,
            'steps_taken': steps_taken,
            'scheduled_jobs_count': len(scheduled_jobs),
            'total_jobs': total_jobs,
            'completion_rate': completion_rate,
            'scheduled_jobs': list(scheduled_jobs),
            'schedule_info': schedule_info
        }
        
    except Exception as e:
        print(f"Error testing model {model_path}: {e}")
        return {
            'model_path': model_path,
            'model_type': model_type,
            'error': str(e),
            'completion_rate': 0,
            'scheduled_jobs_count': 0
        }

def find_all_models():
    """Find all available PPO models"""
    models = []
    base_path = Path('/Users/carrickcheah/Project/ppo/app_2')
    
    # Phase 4 models
    phase4_results = base_path / 'phase4' / 'results'
    if phase4_results.exists():
        for scenario_dir in phase4_results.iterdir():
            if scenario_dir.is_dir():
                # Check for final model
                final_model = scenario_dir / f'{scenario_dir.name}_final.zip'
                if final_model.exists():
                    models.append((str(final_model), f'phase4_{scenario_dir.name}_final'))
                
                # Check checkpoints
                checkpoints_dir = scenario_dir / 'checkpoints'
                if checkpoints_dir.exists():
                    for checkpoint in checkpoints_dir.glob('*.zip'):
                        models.append((str(checkpoint), f'phase4_{scenario_dir.name}_{checkpoint.stem}'))
    
    # Phase 3 models - different categories
    phase3_checkpoints = base_path / 'phase3' / 'checkpoints'
    categories = [
        '80percent', 'better_rewards', 'fixed_models', 'foundation', 
        'masked', 'perfect', 'schedule_all', 'simple_fix', 'truly_fixed'
    ]
    
    for category in categories:
        category_path = phase3_checkpoints / category
        if category_path.exists():
            for sub_dir in category_path.iterdir():
                if sub_dir.is_dir():
                    for model_file in sub_dir.glob('*.zip'):
                        models.append((str(model_file), f'phase3_{category}_{sub_dir.name}_{model_file.stem}'))
                elif sub_dir.suffix == '.zip':
                    models.append((str(sub_dir), f'phase3_{category}_{sub_dir.stem}'))
    
    # Phase 3 curriculum models
    phase3_curriculum = base_path / 'phase3' / 'curriculum_models'
    if phase3_curriculum.exists():
        for scenario_dir in phase3_curriculum.iterdir():
            if scenario_dir.is_dir():
                for model_file in scenario_dir.glob('*.zip'):
                    models.append((str(model_file), f'phase3_curriculum_{scenario_dir.name}_{model_file.stem}'))
    
    # Additional model directories
    additional_dirs = [
        'models_100_percent', 'models_80_percent', 'models_schedule_all',
        'final_models', 'truly_fixed_models'
    ]
    
    for dir_name in additional_dirs:
        models_dir = base_path / 'phase3' / dir_name
        if models_dir.exists():
            for model_file in models_dir.glob('*.zip'):
                models.append((str(model_file), f'phase3_{dir_name}_{model_file.stem}'))
    
    print(f"Found {len(models)} models to test")
    return models

def main():
    """Main function to test all models and find the best one"""
    print("Starting comprehensive model testing...")
    
    # Create test environment
    env, env_type = create_test_environment() 
    if env is None:
        print("Failed to create test environment")
        return
    
    print(f"Using environment: {env_type}")
    
    # Find all models
    models = find_all_models()
    
    # Test all models
    results = []
    best_model = None
    best_completion_rate = 0
    
    print(f"\nTesting {len(models)} models...")
    
    for i, (model_path, model_type) in enumerate(models):
        print(f"\nProgress: {i+1}/{len(models)}")
        result = test_model_performance(model_path, env, model_type)
        results.append(result)
        
        # Track best model
        if result.get('completion_rate', 0) > best_completion_rate:
            best_completion_rate = result['completion_rate']
            best_model = result
        
        # Print progress
        if 'error' not in result:
            print(f"  Completion Rate: {result['completion_rate']:.1f}% ({result['scheduled_jobs_count']}/{result.get('total_jobs', 0)} jobs)")
        else:
            print(f"  Error: {result['error']}")
    
    # Sort results by completion rate
    results.sort(key=lambda x: x.get('completion_rate', 0), reverse=True)
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'/Users/carrickcheah/Project/ppo/app_2/tests/comprehensive_model_test_results_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'environment_type': env_type,
            'total_models_tested': len(models),
            'best_model': best_model,
            'all_results': results
        }, f, indent=2, default=str)
    
    # Print summary
    print(f"\n{'='*60}")
    print("COMPREHENSIVE MODEL TEST RESULTS")
    print(f"{'='*60}")
    print(f"Total models tested: {len(models)}")
    print(f"Environment used: {env_type}")
    
    print(f"\nTOP 10 PERFORMING MODELS:")
    for i, result in enumerate(results[:10]):
        if 'error' not in result:
            print(f"{i+1:2d}. {result['model_type'][:60]:<60} | {result['completion_rate']:5.1f}% | {result['scheduled_jobs_count']:2d} jobs")
        else:
            print(f"{i+1:2d}. {result['model_type'][:60]:<60} | ERROR: {result['error'][:20]}")
    
    if best_model:
        print(f"\nBEST MODEL FOUND:")
        print(f"Path: {best_model['model_path']}")
        print(f"Type: {best_model['model_type']}")
        print(f"Completion Rate: {best_model['completion_rate']:.1f}%")
        print(f"Jobs Scheduled: {best_model['scheduled_jobs_count']}/{best_model.get('total_jobs', 0)}")
        print(f"Total Reward: {best_model.get('total_reward', 0):.1f}")
    
    print(f"\nDetailed results saved to: {results_file}")
    
    return best_model, results

if __name__ == "__main__":
    best_model, all_results = main()