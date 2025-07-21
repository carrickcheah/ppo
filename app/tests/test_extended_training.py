#!/usr/bin/env python3
"""
Test extended training - minimal setup
"""

import os
os.environ['PYTHONPATH'] = '/Users/carrickcheah/Project/ppo/app'

from stable_baselines3 import PPO

# Just test loading and updating the model
print("Testing model loading and hyperparameter update...")

model_path = "models/full_production/final_model.zip"
if not os.path.exists(model_path):
    print(f"ERROR: Model not found at {model_path}")
else:
    print(f"Loading model from {model_path}")
    
    # Load without environment
    model = PPO.load(model_path, device="cpu")
    
    print("\nCurrent hyperparameters:")
    print(f"- Learning rate: {model.learning_rate}")
    print(f"- Batch size: {model.batch_size}")
    print(f"- Entropy coefficient: {model.ent_coef}")
    
    print("\nUpdating hyperparameters:")
    model.learning_rate = 3e-5
    model.batch_size = 1024
    model.ent_coef = 0.005
    
    print(f"- Learning rate: {model.learning_rate} (3x increase)")
    print(f"- Batch size: {model.batch_size} (2x increase)")
    print(f"- Entropy coefficient: {model.ent_coef} (half)")
    
    print("\nModel loaded and updated successfully!")
    print("\nTo run full training, the environment needs to be configured properly.")
    print("The data format mismatch needs to be resolved between:")
    print("- real_production_snapshot.json (new format)")
    print("- parsed_production_data_boolean.json (old format)")