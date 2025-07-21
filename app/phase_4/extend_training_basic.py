#!/usr/bin/env python3
"""
Most basic extended training - just updates the model
"""

import os
from stable_baselines3 import PPO

print("PHASE 4 EXTENDED TRAINING - BASIC")
print("="*40)

# Just update the model hyperparameters and save it
model_path = "models/full_production/final_model.zip"

if os.path.exists(model_path):
    print("1. Loading model...")
    model = PPO.load(model_path)
    
    print("\n2. Current hyperparameters:")
    print(f"   Learning rate: {model.learning_rate}")
    print(f"   Batch size: {model.batch_size}")
    print(f"   Entropy: {model.ent_coef}")
    
    print("\n3. Updating hyperparameters...")
    model.learning_rate = 3e-5  # 3x
    model.batch_size = 1024      # 2x
    model.ent_coef = 0.005       # half
    
    print(f"   Learning rate: {model.learning_rate} (3x)")
    print(f"   Batch size: {model.batch_size} (2x)")
    print(f"   Entropy: {model.ent_coef} (half)")
    
    print("\n4. Saving optimized model...")
    os.makedirs("models/full_production/extended", exist_ok=True)
    model.save("models/full_production/extended/optimized_model")
    
    print("\nDone! Model saved with optimized hyperparameters.")
    print("\nTo train with the original environment setup:")
    print("- The environment expects generated data, not real snapshot")
    print("- Use the same setup as the original phase 4 training")
    print("- Or modify the environment to handle real data format")
else:
    print(f"ERROR: Model not found at {model_path}")