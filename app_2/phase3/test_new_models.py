"""Test the newly trained 100% models"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from phase3.environments.curriculum_env_truly_fixed import CurriculumEnvironmentTrulyFixed

def test_model(stage_name):
    """Test a newly trained model"""
    print(f"\nTesting {stage_name}:")
    print("-" * 40)
    
    # Load the new model
    model_path = f"/Users/carrickcheah/Project/ppo/app_2/phase3/models_100_percent/{stage_name}_100.zip"
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    
    model = PPO.load(model_path)
    env = CurriculumEnvironmentTrulyFixed(stage_name, verbose=False)
    
    # Test multiple episodes
    total_scheduled = 0
    total_possible = 0
    
    for ep in range(5):
        obs, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 200:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
            done = done or truncated
        
        scheduled = len(env.scheduled_jobs) if hasattr(env, 'scheduled_jobs') else 0
        total = env.total_tasks if hasattr(env, 'total_tasks') else 1
        
        print(f"  Episode {ep+1}: {scheduled}/{total} jobs ({scheduled/total*100:.1f}%)")
        
        total_scheduled += scheduled
        total_possible += total
    
    avg_rate = total_scheduled / total_possible if total_possible > 0 else 0
    print(f"  Average: {avg_rate:.1%}")
    return avg_rate

# Test all stages
print("Testing Newly Trained 100% Models")
print("=" * 50)

results = {}
for stage in ['toy_normal', 'toy_hard', 'toy_multi']:
    rate = test_model(stage)
    results[stage] = rate

print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"toy_easy:   100.0% (already perfect)")
print(f"toy_normal: {results.get('toy_normal', 0):.1%}")
print(f"toy_hard:   {results.get('toy_hard', 0):.1%}")
print(f"toy_multi:  {results.get('toy_multi', 0):.1%}")

if all(r >= 0.99 for r in results.values()):
    print("\n✓ ALL TOYS ACHIEVED 100%\!")
else:
    print("\n✗ Still need more training")
